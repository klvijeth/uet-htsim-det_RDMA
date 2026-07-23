#CQF code
import networkx as nx
from collections import defaultdict
import heapq
from tqdm import tqdm
import simpy
import math
from functools import reduce
import numpy as np
import random
from network import RDMAFlow

"""
Class: CQF controller
Implements the CQF Central Controller for bandwidth reservation and flow management.
"""
class CQFCentralController:
    # Initialization of the CQF controller
    def __init__(self, network, cycle_time, packet_size, max_paths=1, env=None):
        self.network      = network # NetworkX graph representing the network
        self.packet_size  = packet_size # Size of each packet in kilobytes
        self.cycle_time   = cycle_time # Duration of each CQF cycle in seconds
        self.max_paths    = max_paths # Maximum number of paths for multipath routing
        self.env          = env
        self.buffer_a = 0.1 # Buffer margin in Gbps to prevent oversubscription (tunable parameter)

        self.completed_flows = 0 # Counter for completed flows
        self.total_flows = 0  # Total number of flows
        self.progress_bar = None

        #flow tracking
        self.coarse_flows=defaultdict(list) #Flows on each link without coarse grained reservation
        self.fine_flows=defaultdict(list) #Flows on each link with fine grained reservation
        
        #metrics
        self.accepted_flows=0 # Counter for accepted flows
        self.rejected_flows=0 # Counter for rejected flows
        self.latency_list=[] # List to store latency of each accepted flow
        self.total_frer_flows=0 # Total number of FRER flows
        self.link_bws=defaultdict(dict) # Bandwidth utilization on each link at different timestamp
        self.network_utilization=defaultdict(float) # Network utilization metrics
        
        # Performance Metrics
        self.slo_met = 0
        self.slo_missed = 0
        self.total_bytes_delivered = 0
        self.fg_slo_met    = 0
        self.fg_slo_missed = 0
        self.cg_slo_met    = 0
        self.cg_slo_missed = 0
        self.cg_completion_times = []  # per-op CG completion times


        self._reserve_lock = simpy.Resource(self.env, capacity=1)

    def set_progress_bar(self, total):
        self.progress_bar = tqdm(total=total, desc="Completed Flows")
    
    # normalize link representation (a,b) and (b,a) should be the same
    def normalize_link(self, u, v):
        return (min(u, v), max(u, v))

    def compute_link_bw_vectors(self, link):
        # 1. Calculate the standard baseline Coarse-Grained required bandwidth
        coarse_bandwidth = sum(s.required_bw for s in self.coarse_flows.get(link, []))
        
        # 2. Inject physical overhead penalties (Propagation + Processing Delay)
        distance = self.network[link[0]][link[1]]['length']
        propagation_speed = 200000  # speed of light in fiber (km/s)
        propagation_delay = distance / propagation_speed
        processing_delay = 0        # Editable placeholder for switch transit overheads
        
        total_overhead_delay = propagation_delay + processing_delay
        
        # If the physical delay is greater than or equal to the cycle time window,
        # the link is completely unusable for standard single-cycle CQF pipelining.
        if total_overhead_delay >= self.cycle_time:
            return np.array([float('inf')])
            
        # Calculate the fraction of the cycle window lost completely to transit overhead
        wasted_time_fraction = total_overhead_delay / self.cycle_time
        
        # The total physical capacity of the link hardware
        physical_link_capacity = self.network[link[0]][link[1]]['bandwidth']
        
        # Convert that dead-time fraction into a constant bandwidth loss metric
        overhead_bandwidth_penalty = physical_link_capacity * wasted_time_fraction
        
        # Add the overhead penalty directly to the baseline coarse floor
        coarse_bandwidth += overhead_bandwidth_penalty
        
        # 3. Get the list of fine-grained flows on this link
        fg_flows = self.fine_flows.get(link, [])
        
        # If there are no fine-grained flows, return scalar coarse floor as length-1 array
        if not fg_flows:
            return np.array([coarse_bandwidth])

        # 4. Find the global hyperperiod (TL) using LCM of all FG flow periods
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        
        periods = [f.period for f in fg_flows]
        T_L = reduce(lcm, periods)
        
        # 5. Create a master vector of length T_L to aggregate instantaneous traffic
        aggregate_fg_vector = np.zeros(T_L)
        
        # 6. Tile (repeat) each flow's profile vector to fill the global hyperperiod
        for f in fg_flows:
            profile = np.array(f.profile)
            reps = T_L // f.period
            extended_profile = np.tile(profile, reps)
            aggregate_fg_vector += extended_profile
        
        total_profile_vector = aggregate_fg_vector + coarse_bandwidth
        return total_profile_vector

    def compute_link_bandwidths(self, link):
        # 7. Find the worst-case peak of the fine-grained aggregate envelope
        total_profile_vector = self.compute_link_bw_vectors(link)
        total_used_bandwidth = np.max(total_profile_vector)
        
        return total_used_bandwidth

    # Reserve bandwidth for a flow on its path
    def reserve_bandwidth(self, path, flow):
        if(flow.type=="coarse"):
            for i in range(len(path) - 1):
                link = self.normalize_link(path[i], path[i + 1])
                self.coarse_flows[link].append(flow) #coarse flow reservation
        else:
            for i in range(len(path) - 1):
                link = self.normalize_link(path[i], path[i + 1])
                self.fine_flows[link].append(flow) #fine flow reservation

        # Redistribute slack to all CG flows that now share links with this flow.
        # Must happen after reserve_bandwidth so the new flow is visible in coarse_flows.
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            for cg_flow in list(self.coarse_flows.get(link, [])):
                self.adjust_flow_bandwidth(cg_flow, cg_flow.path)

    # Release bandwidth for a flow on its path after completion
    def release_bandwidth(self, path, flow):
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            if flow.type == "coarse":
                if flow in self.coarse_flows[link]:
                    self.coarse_flows[link].remove(flow)
            else:
                if flow in self.fine_flows[link]:
                    self.fine_flows[link].remove(flow)

        # Re-distribute freed bandwidth to all remaining CG flows on this path.
        # This is safe to do here because release_bandwidth is only called from transmit
        # methods after a yield, so SimPy won't interleave other events mid-loop.
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            for cg_flow in list(self.coarse_flows.get(link, [])):
                if cg_flow.path:
                    self.adjust_flow_bandwidth(cg_flow, cg_flow.path)


    def adjust_flow_bandwidth(self, flow, path):
        min_extra_packets = float('inf')
        
        # 1. Loop through the path to find the bottleneck "slack" capacity
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            
            # compute_link_bandwidths now natively includes the overhead bandwidth penalty!
            total_required_bw = self.compute_link_bandwidths(link)
            
            # Total link capacity
            link_capacity = self.network[link[0]][link[1]]['bandwidth']
            
            # The remaining bandwidth pool is now purely unallocated slack
            slack_bandwidth = link_capacity - total_required_bw
            slack_bandwidth = max(slack_bandwidth, 0.0) 
            
            if slack_bandwidth == 0:
                link_extra_packets_per_flow = 0
            else:
                # Convert the remaining slack bandwidth into absolute packet units
                # The timing/delay reductions are already absorbed into the slack_bandwidth
                packet_delay = (self.packet_size / (slack_bandwidth * 1000))
                total_slack_packets_on_link = self.cycle_time // packet_delay
                
                # Divide this extra packet slack pool equally among registered CG flows
                num_cg_flows = len(self.coarse_flows[link])
                link_extra_packets_per_flow = total_slack_packets_on_link // num_cg_flows
                
            # Track the absolute worst-case hop across the entire end-to-end path
            if link_extra_packets_per_flow < min_extra_packets:
                min_extra_packets = link_extra_packets_per_flow

        # 2. Assign the baseline required packets PLUS the fair-share extra slack packets
        required_packets = int((flow.required_bw * self.cycle_time * 1000) // self.packet_size)
        
        flow.cycle_packets = required_packets + int(min_extra_packets)
        flow.assigned_bw = (flow.cycle_packets * self.packet_size) / (self.cycle_time * 1000)

    # Function to update bandwidth and network utilization metrics at each time step
    def set_bandwidth_utilization(self, time):
        """
        Updates snapshot tracking vectors for link usage and global network saturation.
        """
        total_utilization = 0.0
        used_links = 0
        
        for link in self.network.edges():
            norm_link = self.normalize_link(link[0], link[1])
            
            # Skip evaluation if the link is totally quiet to save clock cycles
            if not self.coarse_flows[norm_link] and not self.fine_flows[norm_link]:
                continue
                
            # Get current baseline required floors and fine peaks
            total_allocated_peak = self.compute_link_bandwidths(norm_link)
            
            # Calculate saturation relative to the physical link capacity
            physical_capacity = self.network[link[0]][link[1]]['bandwidth']
            link_utilization = total_allocated_peak / physical_capacity
            
            # Log metric mapping histories
            self.link_bws[norm_link][time] = total_allocated_peak
            total_utilization += link_utilization
            used_links += 1
            
        # Update global aggregated system utilization metric
        if used_links > 0:
            self.network_utilization[time] = total_utilization / used_links
        else:
            self.network_utilization[time] = 0.0
    
    def calc_link_available_bandwidth(self, link):
        norm_link = self.normalize_link(link[0], link[1])
        total_used_bw = self.compute_link_bandwidths(norm_link)
        physical_capacity = self.network[link[0]][link[1]]['bandwidth']
        available_bw = physical_capacity - total_used_bw
        return max(available_bw, 0.0)


    def widest_path_dynamic(self, G, src, dst):
        """
        Finds a path from src to dst maximizing the minimum *dynamic* bandwidth
        along the path, by calling calc_bandwidth_path on each edge [u, v].
        Returns (path, bottleneck_capacity), or ([], 0) if no path exists.
        """
        # best possible min‑edge so far to reach each node
        bottleneck = {n: 0.0 for n in G.nodes()}
        bottleneck[src] = float('inf')
        parent     = {}
        seen       = set()
        # max‑heap ordered by current bottleneck
        heap = [(-bottleneck[src], src)]

        while heap:
            neg_cap, u = heapq.heappop(heap)
            cap_u = -neg_cap
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                break

            # explore neighbors dynamically
            for v in G.neighbors(u):
                # getting available bandwidth of the edge
                cap_uv = self.calc_link_available_bandwidth(tuple(sorted((u,v))))
                if cap_uv < 0:
                    continue
                path_cap = min(cap_u, cap_uv)
                if path_cap > bottleneck[v]:
                    bottleneck[v] = path_cap
                    parent[v]     = u
                    heapq.heappush(heap, (-path_cap, v))

        # if dst wasn’t reached
        if dst not in seen:
            return [], 0.0

        # reconstruct the path
        path = []
        cur  = dst
        while cur != src:
            path.append(cur)
            cur = parent[cur]
        path.append(src)
        path.reverse()

        return path, bottleneck[dst]

    def check_fg_flow_fits(self, path, flow):
        """
        Helper: Reuses compute_link_bw_profile and aligns hyperperiods to verify 
        if an incoming FG flow fits along every hop of a candidate path.
        """
        def lcm(a, b): 
            return abs(a * b) // math.gcd(a, b)

        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            link_capacity = self.network[link[0]][link[1]]['bandwidth']
            
            # 1. Get the current aggregate profile vector
            current_profile = self.compute_link_bw_vectors(link)
            
            # 2. Find the new common hyperperiod between the current timeline and the new flow
            current_TL = len(current_profile)
            new_TL = lcm(current_TL, flow.period)
            
            # 3. Tile BOTH profiles to the new unified hyperperiod length
            extended_current_profile = np.tile(current_profile, new_TL // current_TL)
            extended_candidate_profile = np.tile(flow.profile, new_TL // flow.period)
            
            # 4. Construct the prospective timeline mesh
            prospective_profile = extended_current_profile + extended_candidate_profile
            
            # 5. Enforce Whiteboard Condition
            if np.max(prospective_profile) > (link_capacity - self.buffer_a):
                return False  # Collision detected on this hop
                
        return True  # Flow fits perfectly across the entire path

    def route_flow(self, env, flow):
        with self._reserve_lock.request() as req:
            yield req
            path, bottleneck = self.widest_path_dynamic(self.network, flow.source, flow.destination)

# reject the flow if a path with sufficient bandwidth doesn't exist, or if it fails the FG flow fit check
        if not path:
            self.rejected_flows += 1
            self.slo_missed += 1
            if flow.type == "coarse": self.cg_slo_missed += 1
            else:                     self.fg_slo_missed += 1
            return

        if flow.type == "coarse" and flow.required_bw > bottleneck:
            self.rejected_flows += 1
            self.slo_missed     += 1
            self.cg_slo_missed  += 1
            return

        if flow.type == "fine" and not self.check_fg_flow_fits(path, flow):
            self.rejected_flows += 1
            self.slo_missed     += 1
            self.fg_slo_missed  += 1
            return

        self.accepted_flows += 1
        flow.path = path
        self.reserve_bandwidth(path, flow)

        if flow.type == "coarse":
            yield env.process(self.transmit_cg_flow(env, flow))
        else:
            yield env.process(self.transmit_fg_flow(env, flow))

    def transmit_fg_flow(self, env, flow):
        """
        FG flow transmission: run for exactly N complete period repetitions.

        bytes_per_period — KB delivered in one full pass through the profile.
        num_periods      — ceiling of (total_bytes / bytes_per_period); the flow
                           always finishes on a period boundary, matching the CQF
                           contract that bandwidth is reserved in whole-period windows.

        Each period repetition is one deadline unit: the flow is "on time" if it
        completes all num_periods within flow.deadline.
        """
        flow.start_time = env.now

        # How many KB does one full period deliver?
        bytes_per_period = sum(
            int(bw * self.cycle_time * 1000 // self.packet_size) * self.packet_size
            for bw in flow.profile
        )

        if bytes_per_period <= 0:
            # Profile carries no traffic — immediate rejection of transmission
            flow.completion_time = 0.0
            flow.met_deadline    = False
            self.slo_missed    += 1
            self.fg_slo_missed += 1
            self.release_bandwidth(flow.path, flow)
            return

        num_periods = math.ceil(flow.total_bytes / bytes_per_period)

        for rep in range(num_periods):
            for bw in flow.profile:
                yield env.timeout(self.cycle_time)

        flow.completion_time = env.now - flow.start_time
        flow.met_deadline    = flow.completion_time <= flow.deadline

        self.release_bandwidth(flow.path, flow)
        self.total_bytes_delivered += flow.total_bytes * 1024  # to bytes
        self.latency_list.append(flow.completion_time)

        if flow.met_deadline:
            self.slo_met     += 1
            self.fg_slo_met  += 1
        else:
            self.slo_missed    += 1
            self.fg_slo_missed += 1

    def transmit_cg_flow(self, env, flow):
        """
        CG flow transmission — two modes:

        Inference mode  (flow.request_rate is not None)
            Models ML inference traffic: Poisson request arrivals with log-normal
            payload sizes.  The bandwidth floor is reserved permanently; the link
            is idle between requests.  Each request is measured independently
            against flow.deadline (per-request latency SLO).

        Fixed mode  (flow.request_rate is None)
            One-shot (finite deadline) or persistent (deadline == inf) transfer
            of flow.total_bytes per iteration.  Legacy behaviour.
        """
        if flow.request_rate is not None:
            yield from self._transmit_cg_inference(env, flow)
        else:
            yield from self._transmit_cg_fixed(env, flow)

    def _transmit_cg_inference(self, env, flow):
        """
        Inference / random-burst CG mode.

        Bandwidth floor is reserved for the life of the simulation.  Between
        requests the link is idle.  Each request:
          1. Waits an exponentially-distributed inter-arrival gap.
          2. Draws a log-normal payload size.
          3. Sends it using whatever cycle_packets the controller has allocated,
             which varies dynamically as FG flows are admitted/released.
          4. Records the per-request completion time and SLO outcome.
        """
        rng  = np.random.default_rng(flow.rng_seed)
        mean = flow.request_size_mean_kb
        std  = flow.request_size_std_kb
        # Log-normal parameters
        sigma = math.sqrt(math.log1p((std / mean) ** 2))
        mu    = math.log(mean) - 0.5 * sigma ** 2

        while True:
            # Idle: wait for next inference request to arrive
            inter_arrival = rng.exponential(1.0 / flow.request_rate)
            yield env.timeout(inter_arrival)

            # Sample a random payload size for this request
            req_kb = max(self.packet_size, float(rng.lognormal(mu, sigma)))

            req_start     = env.now
            bytes_remaining = req_kb

            # Transmit using the dynamically-assigned cycle_packets
            while bytes_remaining > 0:
                yield env.timeout(self.cycle_time)
                if flow.cycle_packets > 0:
                    bytes_remaining -= flow.cycle_packets * self.packet_size

            req_ct = env.now - req_start
            flow.request_completion_times.append(req_ct)
            self.total_bytes_delivered += req_kb * 1024   # KB → bytes
            self.cg_completion_times.append(req_ct)

            met = req_ct <= flow.deadline
            flow.met_deadline = met
            if met:
                self.slo_met    += 1
                self.cg_slo_met += 1
            else:
                self.slo_missed    += 1
                self.cg_slo_missed += 1

        # Bandwidth reservation is never released — simulation termination cleans up.

    def _transmit_cg_fixed(self, env, flow):
        """
        Fixed-data CG mode: send flow.total_bytes per iteration.
        persistent=True (deadline==inf) loops immediately to the next request.
        """
        persistent = (flow.deadline == float('inf'))

        while True:
            flow.start_time   = env.now
            bytes_remaining   = flow.total_bytes

            while bytes_remaining > 0:
                yield env.timeout(self.cycle_time)
                if flow.cycle_packets > 0:
                    bytes_remaining -= flow.cycle_packets * self.packet_size

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
                
    def get_performance_metrics(self, simulation_time):
        """Returns a dictionary of key performance metrics."""
        avg_latency = sum(self.latency_list) / len(self.latency_list) if self.latency_list else 0
        throughput_gbps = (self.total_bytes_delivered * 8 / 1e9) / simulation_time if simulation_time > 0 else 0
        total_decided       = self.slo_met + self.slo_missed  # includes rejections
        slo_attainment_rate = (self.slo_met / total_decided) * 100 if total_decided > 0 else 0

        cg_times = self.cg_completion_times
        cg_p50  = float(np.percentile(cg_times, 50))  if cg_times else 0
        cg_p95  = float(np.percentile(cg_times, 95))  if cg_times else 0
        cg_p99  = float(np.percentile(cg_times, 99))  if cg_times else 0
        cg_mean = float(np.mean(cg_times))             if cg_times else 0

        fg_total    = self.fg_slo_met + self.fg_slo_missed
        cg_total    = self.cg_slo_met + self.cg_slo_missed
        cg_slo_rate = (self.cg_slo_met / cg_total) * 100 if cg_total > 0 else 0
        fg_slo_rate = (self.fg_slo_met / fg_total) * 100 if fg_total > 0 else 0

        return {
            "Total Flows":               self.total_flows,
            "Accepted Flows":            self.accepted_flows,
            "Rejected Flows":            self.rejected_flows,
            "SLO Attainment (%)":        slo_attainment_rate,
            "FG SLO Attainment (%)":     fg_slo_rate,
            "CG SLO Attainment (%)":     cg_slo_rate,
            "Average Latency":           avg_latency,
            "Throughput (Gbps)":         throughput_gbps,
            # FG-specific
            "FG Flows Completed":        fg_total,
            "FG SLO Met":                self.fg_slo_met,
            "FG SLO Missed":             self.fg_slo_missed,
            # CG-specific
            "CG Ops Completed":          cg_total,
            "CG SLO Met":                self.cg_slo_met,
            "CG SLO Missed":             self.cg_slo_missed,
            "CG Mean Completion Time":   cg_mean,
            "CG p50 Completion Time":    cg_p50,
            "CG p95 Completion Time":    cg_p95,
            "CG p99 Completion Time":    cg_p99,
        }


class CollectiveTrafficGenerator:
    """
    Drives a CQFCentralController with two classes of collective-communication flows:

      FG collectives – periodic AllReduce-style flows with a known bandwidth profile.
                       All are submitted at t=0 and run for the full simulation.

      CG collectives – randomly triggered ops (Poisson arrivals).  Each op is modelled
                       as a ring-AllReduce: n_gpus simultaneous point-to-point flows,
                       one per ring segment, each carrying data_size_kb / n_gpus data.
                       The op is considered complete when the last flow finishes.

    Parameters
    ----------
    env              : simpy.Environment
    controller       : CQFCentralController
    gpu_nodes        : list of node IDs that are GPU endpoints (ring participants)
    sim_duration     : simulation end time (same units as cycle_time)
    fg_flows         : list of pre-built RDMAFlow objects (type="fine")
    cg_arrival_rate  : mean CG ops per unit time (Poisson rate)
    cg_data_size_kb  : total gradient payload per CG op (KB); split across ring segments
    cg_required_bw   : minimum bandwidth per CG flow (Gbps)
    cg_deadline      : per-op completion deadline (same units as cycle_time)
    seed             : optional RNG seed for reproducibility
    """

    def __init__(self, env, controller, gpu_nodes, sim_duration,
                 fg_flows=None,
                 cg_arrival_rate=1.0,
                 cg_data_size_kb=1024,
                 cg_required_bw=1.0,
                 cg_deadline=100.0,
                 seed=None):
        self.env              = env
        self.controller       = controller
        self.gpu_nodes        = list(gpu_nodes)
        self.sim_duration     = sim_duration
        self.fg_flows         = fg_flows or []
        self.cg_arrival_rate  = cg_arrival_rate
        self.cg_data_size_kb  = cg_data_size_kb
        self.cg_required_bw   = cg_required_bw
        self.cg_deadline      = cg_deadline
        self._rng             = random.Random(seed)
        self._np_rng          = np.random.default_rng(seed)
        self._op_counter      = 0
        # per-op completion times (indexed by op_id); populated as ops finish
        self.op_completion_times = {}

    def run(self):
        return self.env.process(self._generate())

    def _generate(self):
        # Submit all FG flows immediately
        for flow in self.fg_flows:
            self.controller.total_flows += 1
            self.env.process(self.controller.route_flow(self.env, flow))

        # Poisson CG arrivals until sim_duration
        while self.env.now < self.sim_duration:
            inter_arrival = self._np_rng.exponential(1.0 / self.cg_arrival_rate)
            yield self.env.timeout(inter_arrival)
            if self.env.now >= self.sim_duration:
                break
            self._launch_cg_op()

    def _launch_cg_op(self):
        op_id = self._op_counter
        self._op_counter += 1

        n = len(self.gpu_nodes)
        # Each ring segment carries an equal share of the payload
        per_flow_bytes = max(1, self.cg_data_size_kb // n)

        # Build one flow per ring segment (GPU_i → GPU_{i+1 mod n})
        flows = []
        for i in range(n):
            src = self.gpu_nodes[i]
            dst = self.gpu_nodes[(i + 1) % n]
            f = RDMAFlow(
                flow_id=f"cg_{op_id}_{i}",
                source=src,
                destination=dst,
                flow_type="coarse",
                total_bytes=per_flow_bytes,
                deadline=self.cg_deadline,
                required_bw=self.cg_required_bw,
            )
            flows.append(f)
            self.controller.total_flows += 1

        start_time = self.env.now
        self.env.process(self._track_op(op_id, flows, start_time))

    def _track_op(self, op_id, flows, start_time):
        """Wait for every flow in the op to finish, then record the op completion time."""
        procs = [self.env.process(self.controller.route_flow(self.env, f)) for f in flows]
        for proc in procs:
            yield proc
        self.op_completion_times[op_id] = self.env.now - start_time

    # ------------------------------------------------------------------
    # Helpers for building FG flows outside this class
    # ------------------------------------------------------------------

    @staticmethod
    def make_fg_flow(flow_id, source, destination, total_bytes_kb, deadline,
                     active_bw_gbps, active_cycles, period_cycles):
        """
        Convenience factory for a bursty periodic FG collective flow.
        The flow is active (sending at active_bw_gbps) for the first active_cycles
        slots of each period, then idle for the remainder.

        Example: period=10, active=3 → [bw, bw, bw, 0, 0, 0, 0, 0, 0, 0]
        """
        profile = [active_bw_gbps] * active_cycles + [0.0] * (period_cycles - active_cycles)
        return RDMAFlow(
            flow_id=flow_id,
            source=source,
            destination=destination,
            flow_type="fine",
            total_bytes=total_bytes_kb,
            deadline=deadline,
            profile=profile,
            period=period_cycles,
        )

    def get_op_metrics(self):
        """Summary statistics over completed CG op completion times."""
        times = list(self.op_completion_times.values())
        if not times:
            return {"CG Ops Triggered": self._op_counter, "CG Ops Completed": 0}
        arr = np.array(times)
        return {
            "CG Ops Triggered":       self._op_counter,
            "CG Ops Completed":       len(times),
            "CG Op Mean (s)":         float(arr.mean()),
            "CG Op p50  (s)":         float(np.percentile(arr, 50)),
            "CG Op p95  (s)":         float(np.percentile(arr, 95)),
            "CG Op p99  (s)":         float(np.percentile(arr, 99)),
            "CG Op Max  (s)":         float(arr.max()),
        }


class RDMATrafficGenerator:
    """
    Co-located inference + training traffic for CQF vs circuit-switched comparison.

    CG (inference) – persistent ring streams.  One flow per ring edge cycles
                     continuously: when a request completes the next starts
                     immediately.  No deadline; metric is completion time.

    FG (training)  – Poisson arrivals of gradient-sync (AllReduce) jobs.  Each
                     job spawns n_ring simultaneous flows with a bursty profile.
                     Metric: SLO attainment (completed within fg_deadline).

    Compatible with CQFCentralController and CircuitSwitchedController.
    """

    def __init__(self, env, controller, gpu_nodes, sim_duration,
                 cg_data_kb, cg_req_bw,
                 fg_arrival_rate, fg_data_kb, fg_deadline,
                 fg_peak_bw, fg_active_cycles, fg_period,
                 step_time=1.0, seed=42):
        self.env             = env
        self.controller      = controller
        self.gpu_nodes       = list(gpu_nodes)
        self.n               = len(gpu_nodes)
        self.sim_duration    = sim_duration
        self.cg_data_kb      = cg_data_kb
        self.cg_req_bw       = cg_req_bw
        self.fg_arrival_rate  = fg_arrival_rate
        self.fg_data_kb       = fg_data_kb
        self.fg_deadline      = fg_deadline
        self.fg_peak_bw       = fg_peak_bw
        self.fg_active_cycles = fg_active_cycles
        self.fg_period        = fg_period
        # Number of non-overlapping slot windows in one period.
        # Jobs are assigned round-robin so CQF can pack them into different
        # windows; CS still reserves peak BW regardless of window offset.
        self._n_offsets = fg_period // fg_active_cycles  # e.g. 10//3 = 3
        self.step_time       = step_time
        self._rng            = np.random.default_rng(seed)
        self._fg_counter     = 0

    def run(self):
        for i in range(self.n):
            src = self.gpu_nodes[i]
            dst = self.gpu_nodes[(i + 1) % self.n]
            self.env.process(self._cg_loop(src, dst, i))
        self.env.process(self._fg_arrivals())

    def _cg_loop(self, src, dst, idx):
        # Submit the CG flow once.  transmit_cg_flow runs in persistent mode
        # (deadline==inf), looping internally without ever releasing bandwidth.
        # The flow stays in coarse_flows for the whole simulation so FG
        # admission always sees a stable bandwidth baseline — no re-admit gap.
        while self.env.now < self.sim_duration:
            flow = RDMAFlow(
                flow_id     = f"cg_{idx}",
                source      = src,
                destination = dst,
                flow_type   = "coarse",
                total_bytes = self.cg_data_kb,
                deadline    = float('inf'),
                required_bw = self.cg_req_bw,
            )
            self.controller.total_flows += 1
            yield self.env.process(self.controller.route_flow(self.env, flow))
            # If admitted, route_flow blocks forever (persistent transmit loop).
            # If rejected, it returns immediately — wait one step and retry.
            if not flow.path:
                yield self.env.timeout(self.step_time)

    def _fg_arrivals(self):
        while self.env.now < self.sim_duration:
            iat = self._rng.exponential(1.0 / self.fg_arrival_rate)
            yield self.env.timeout(iat)
            if self.env.now >= self.sim_duration:
                break
            self._launch_fg_job()

    def _launch_fg_job(self):
        job_id = self._fg_counter
        self._fg_counter += 1

        # Stagger burst window round-robin so CQF can pack multiple jobs into
        # the same period without slot collision.  Job 0 bursts in slots 0-2,
        # job 1 in slots 3-5, job 2 in slots 6-8, job 3 back to 0-2, etc.
        # CS reserves max(profile)=peak regardless of offset → same 10 Gbps/flow.
        offset  = (job_id % self._n_offsets) * self.fg_active_cycles
        profile = ([0.0] * offset +
                   [self.fg_peak_bw] * self.fg_active_cycles +
                   [0.0] * (self.fg_period - offset - self.fg_active_cycles))

        procs = []
        for i in range(self.n):
            src = self.gpu_nodes[i]
            dst = self.gpu_nodes[(i + 1) % self.n]
            f = RDMAFlow(
                flow_id     = f"fg_{job_id}_{i}",
                source      = src,
                destination = dst,
                flow_type   = "fine",
                total_bytes = self.fg_data_kb,
                deadline    = self.fg_deadline,
                profile     = profile,
                period      = self.fg_period,
            )
            self.controller.total_flows += 1
            procs.append(self.env.process(self.controller.route_flow(self.env, f)))
        self.env.process(self._await_fg_job(procs))

    def _await_fg_job(self, procs):
        for p in procs:
            yield p