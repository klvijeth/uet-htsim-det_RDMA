# Circuit-switched RDMA baseline
#
# Assigns a static, fixed bandwidth to each flow for its entire lifetime.
# No time-slotting, no CQF cycle scheduling, no work-conserving slack
# redistribution.  Used as a comparison baseline against CQF_RDMA.
#
# Key intentional simplifications vs CQF:
#   FG flows  – bandwidth reserved at peak(profile).  Circuit-switching cannot
#               time-slice bursts, so it must hold the worst-case slot open at
#               all times.  CQF handles the same traffic more efficiently by
#               scheduling burst slots only when needed.
#   CG flows  – reserved at required_bw.  Freed bandwidth (when a flow leaves)
#               stays in the pool for new admissions but is NOT redistributed to
#               already-running flows.
#   Routing   – same widest-path-first algorithm; comparison is apples-to-apples.
#   Metrics   – identical dict keys to CQF_RDMA.get_performance_metrics().

import heapq
from collections import defaultdict

import numpy as np
import simpy

from network import RDMAFlow
from CQF_RDMA import CollectiveTrafficGenerator  # controller-agnostic; reuse directly


class CircuitSwitchedController:
    """
    Circuit-switched network controller for RDMA GPU cluster workloads.

    Parameters
    ----------
    network     : NetworkX graph with 'bandwidth' (Gbps) and 'length' (km) edge attrs
    packet_size : KB per packet (same value used in CQF_RDMA)
    batch_time  : time step for progress accounting (same units as flow deadlines)
    env         : SimPy Environment
    """

    def __init__(self, network, packet_size, batch_time, env=None):
        self.network     = network
        self.packet_size = packet_size  # KB
        self.batch_time  = batch_time   # time quantum (same units as deadlines / cycle_time)
        self.env         = env

        # Per-link flow lists
        self.coarse_flows = defaultdict(list)
        self.fine_flows   = defaultdict(list)

        # Counters and per-flow metrics
        self.total_flows           = 0
        self.accepted_flows        = 0
        self.rejected_flows        = 0
        self.total_bytes_delivered = 0  # bytes
        self.latency_list          = []
        self.slo_met               = 0
        self.slo_missed            = 0
        self.fg_slo_met            = 0
        self.fg_slo_missed         = 0
        self.cg_slo_met            = 0
        self.cg_slo_missed         = 0
        self.cg_completion_times   = []

        # Link-level utilization snapshots (mirrors CQF_RDMA API)
        self.link_bws            = defaultdict(dict)
        self.network_utilization = defaultdict(float)

        self._reserve_lock = simpy.Resource(env, capacity=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def normalize_link(self, u, v):
        return (min(u, v), max(u, v))

    def fg_reserved_bw(self, flow):
        """
        Static BW reserved for an FG flow: peak of its profile vector.

        Circuit-switching cannot time-slice bursts, so it must keep the
        worst-case slot capacity open for the entire flow lifetime.
        CQF handles the same traffic more efficiently because it can
        schedule the burst slots only in the cycles where they are needed.
        """
        if flow.profile:
            return max(flow.profile)
        return flow.required_bw

    # ------------------------------------------------------------------
    # Bandwidth accounting  (no overhead penalty — circuit switch doesn't
    # have a cycle boundary to waste; propagation is just latency)
    # ------------------------------------------------------------------

    def compute_link_bandwidths(self, link):
        """Sum of all statically reserved bandwidths on link."""
        cg_bw = sum(f.required_bw          for f in self.coarse_flows.get(link, []))
        fg_bw = sum(self.fg_reserved_bw(f)  for f in self.fine_flows.get(link, []))
        return cg_bw + fg_bw

    def calc_link_available_bandwidth(self, link):
        norm     = self.normalize_link(link[0], link[1])
        used     = self.compute_link_bandwidths(norm)
        capacity = self.network[link[0]][link[1]]['bandwidth']
        return max(capacity - used, 0.0)

    # ------------------------------------------------------------------
    # Reservation  (no re-adjustment on departure — circuit-switch semantics)
    # ------------------------------------------------------------------

    def reserve_bandwidth(self, path, flow):
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            if flow.type == "coarse":
                self.coarse_flows[link].append(flow)
            else:
                self.fine_flows[link].append(flow)

    def release_bandwidth(self, path, flow):
        """
        Remove the flow's static reservation.  No slack redistribution:
        freed bandwidth becomes available for newly arriving flows only.
        """
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            if flow.type == "coarse":
                if flow in self.coarse_flows[link]:
                    self.coarse_flows[link].remove(flow)
            else:
                if flow in self.fine_flows[link]:
                    self.fine_flows[link].remove(flow)

    # ------------------------------------------------------------------
    # Routing  (identical widest-path algorithm as CQF_RDMA)
    # ------------------------------------------------------------------

    def widest_path_dynamic(self, G, src, dst):
        bottleneck = {n: 0.0 for n in G.nodes()}
        bottleneck[src] = float('inf')
        parent = {}
        seen   = set()
        heap   = [(-bottleneck[src], src)]

        while heap:
            neg_cap, u = heapq.heappop(heap)
            cap_u = -neg_cap
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                break
            for v in G.neighbors(u):
                cap_uv   = self.calc_link_available_bandwidth(tuple(sorted((u, v))))
                path_cap = min(cap_u, cap_uv)
                if path_cap > bottleneck[v]:
                    bottleneck[v] = path_cap
                    parent[v]     = u
                    heapq.heappush(heap, (-path_cap, v))

        if dst not in seen:
            return [], 0.0

        path, cur = [], dst
        while cur != src:
            path.append(cur)
            cur = parent[cur]
        path.append(src)
        path.reverse()
        return path, bottleneck[dst]

    # ------------------------------------------------------------------
    # Admission + transmission
    # ------------------------------------------------------------------

    def route_flow(self, env, flow):
        with self._reserve_lock.request() as req:
            yield req
            path, bottleneck = self.widest_path_dynamic(
                self.network, flow.source, flow.destination
            )

        if not path:
            self.rejected_flows += 1
            self.slo_missed += 1
            if flow.type == "coarse": self.cg_slo_missed += 1
            else:                     self.fg_slo_missed += 1
            return

        needed = self.fg_reserved_bw(flow) if flow.type == "fine" else flow.required_bw
        if needed > bottleneck:
            self.rejected_flows += 1
            self.slo_missed += 1
            if flow.type == "coarse": self.cg_slo_missed += 1
            else:                     self.fg_slo_missed += 1
            return

        self.accepted_flows += 1
        flow.path        = path
        flow.assigned_bw = needed  # locked in for the flow's lifetime
        self.reserve_bandwidth(path, flow)

        if flow.type == "fine":
            yield env.process(self.transmit_fg_flow(env, flow))
        else:
            yield env.process(self.transmit_cg_flow(env, flow))

    def _pkts_per_step(self, bw):
        """Packets sent in one batch_time at bandwidth bw (Gbps).
        Uses the same formula as CQF_RDMA required_packets so simulations
        use identical packet-accounting arithmetic."""
        return int(bw * self.batch_time * 1000 // self.packet_size)

    def transmit_fg_flow(self, env, flow):
        """
        FG circuit-switched transmission using the same duty-cycle profile as
        CQF_RDMA, so both controllers spend the same wall-clock time on FG flows
        and comparisons are apples-to-apples.  The peak bandwidth is still
        reserved for the full flow lifetime (circuit-switch semantics); the
        profile only controls how many packets are sent each step.
        """
        flow.start_time = env.now
        bytes_remaining = flow.total_bytes   # KB
        cycle_idx       = 0

        while bytes_remaining > 0:
            bw   = flow.profile[cycle_idx % flow.period] if flow.profile else flow.assigned_bw
            pkts = self._pkts_per_step(bw)
            bytes_remaining -= pkts * self.packet_size
            cycle_idx += 1
            yield env.timeout(self.batch_time)

        flow.completion_time = env.now - flow.start_time
        flow.met_deadline    = flow.completion_time <= flow.deadline

        self.release_bandwidth(flow.path, flow)
        self.total_bytes_delivered += flow.total_bytes * 1024
        self.latency_list.append(flow.completion_time)

        if flow.met_deadline:
            self.slo_met    += 1
            self.fg_slo_met += 1
        else:
            self.slo_missed    += 1
            self.fg_slo_missed += 1

    def transmit_cg_flow(self, env, flow):
        """
        CG circuit-switched transmission.  Sends at required_bw — no slack.

        Persistent mode (flow.deadline == inf): loops forever, recording each
        request without releasing bandwidth.  Matches CQF's persistent model
        so both controllers have stable bandwidth accounting throughout.

        One-shot mode (finite deadline): original single-request behaviour.
        """
        persistent = (flow.deadline == float('inf'))
        pkts       = self._pkts_per_step(flow.assigned_bw)

        if pkts <= 0:
            self.release_bandwidth(flow.path, flow)
            self.slo_missed    += 1
            self.cg_slo_missed += 1
            return

        while True:
            flow.start_time   = env.now
            bytes_remaining   = flow.total_bytes

            while bytes_remaining > 0:
                yield env.timeout(self.batch_time)
                bytes_remaining -= pkts * self.packet_size

            flow.completion_time = env.now - flow.start_time
            flow.met_deadline    = flow.completion_time <= flow.deadline

            self.total_bytes_delivered += flow.total_bytes * 1024
            self.latency_list.append(flow.completion_time)
            self.cg_completion_times.append(flow.completion_time)

            if flow.met_deadline:
                self.slo_met    += 1
                self.cg_slo_met += 1
            else:
                self.slo_missed    += 1
                self.cg_slo_missed += 1

            if not persistent:
                break

        self.release_bandwidth(flow.path, flow)

    # ------------------------------------------------------------------
    # Utilization snapshots
    # ------------------------------------------------------------------

    def set_bandwidth_utilization(self, time):
        total_util = 0.0
        used_links = 0
        for link in self.network.edges():
            norm = self.normalize_link(link[0], link[1])
            if not self.coarse_flows[norm] and not self.fine_flows[norm]:
                continue
            used     = self.compute_link_bandwidths(norm)
            capacity = self.network[link[0]][link[1]]['bandwidth']
            self.link_bws[norm][time] = used
            total_util += used / capacity
            used_links += 1
        self.network_utilization[time] = (total_util / used_links) if used_links > 0 else 0.0

    # ------------------------------------------------------------------
    # Metrics  (same dict keys as CQF_RDMA.get_performance_metrics)
    # ------------------------------------------------------------------

    def get_performance_metrics(self, simulation_time):
        avg_latency     = float(np.mean(self.latency_list))          if self.latency_list else 0
        throughput_gbps = (self.total_bytes_delivered * 8 / 1e9) / simulation_time \
                          if simulation_time > 0 else 0
        total_decided   = self.slo_met + self.slo_missed  # includes rejections
        slo_rate        = (self.slo_met / total_decided) * 100 if total_decided > 0 else 0

        cg_times = self.cg_completion_times
        cg_mean  = float(np.mean(cg_times))            if cg_times else 0
        cg_p50   = float(np.percentile(cg_times, 50))  if cg_times else 0
        cg_p95   = float(np.percentile(cg_times, 95))  if cg_times else 0
        cg_p99   = float(np.percentile(cg_times, 99))  if cg_times else 0

        fg_total    = self.fg_slo_met + self.fg_slo_missed
        cg_total    = self.cg_slo_met + self.cg_slo_missed
        cg_slo_rate = (self.cg_slo_met / cg_total) * 100 if cg_total > 0 else 0
        fg_slo_rate = (self.fg_slo_met / fg_total) * 100 if fg_total > 0 else 0

        return {
            "Total Flows":             self.total_flows,
            "Accepted Flows":          self.accepted_flows,
            "Rejected Flows":          self.rejected_flows,
            "SLO Attainment (%)":      slo_rate,
            "FG SLO Attainment (%)":   fg_slo_rate,
            "CG SLO Attainment (%)":   cg_slo_rate,
            "Average Latency":         avg_latency,
            "Throughput (Gbps)":       throughput_gbps,
            # FG-specific
            "FG Flows Completed":      fg_total,
            "FG SLO Met":              self.fg_slo_met,
            "FG SLO Missed":           self.fg_slo_missed,
            # CG-specific
            "CG Ops Completed":        cg_total,
            "CG SLO Met":              self.cg_slo_met,
            "CG SLO Missed":           self.cg_slo_missed,
            "CG Mean Completion Time": cg_mean,
            "CG p50 Completion Time":  cg_p50,
            "CG p95 Completion Time":  cg_p95,
            "CG p99 Completion Time":  cg_p99,
        }
