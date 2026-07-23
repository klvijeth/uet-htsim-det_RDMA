"""
rdma_collectives.py — ML collective communication patterns for RDMA GPU clusters.

Simulates AllReduce (ring), ReduceScatter, AllGather, AllToAll, and Broadcast
as sequences of barrier-synchronized communication steps over a fat-tree fabric.

Each step is a set of N simultaneous point-to-point RDMA flows.  Congestion is
modelled with max-min fair sharing: flows that share a link split its bandwidth
equally.  The step completes when the slowest flow finishes (barrier sync).

Time units  : seconds (consistent with cqf_sim.py)
Bandwidth   : Mbps   (consistent with fat_tree.py graph attributes)

Typical usage
-------------
    from fat_tree import FatTreeTopology
    from rdma_collectives import RingAllReduce, RDMACollectiveSimulator

    topo  = FatTreeTopology(k=8)
    nodes = topo.select_gpu_group(8, strategy="spread_pods")
    sim   = RDMACollectiveSimulator(topo)
    result = sim.run_sync(RingAllReduce(), nodes, tensor_bytes=1_000_000_000)
    print(result.summary())

SimPy integration
-----------------
    import simpy
    env = simpy.Environment()
    proc = env.process(sim.run(env, RingAllReduce(), nodes, tensor_bytes=1e9))
    env.run()
    print(proc.value.summary())
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import simpy

from fat_tree import FatTreeTopology


# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# Cut-through switching latency per switch hop (conservative datacenter value)
DEFAULT_SWITCH_LATENCY_S = 400e-9   # 400 ns per switch


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class CommStep:
    """One barrier-synchronised communication step.

    All flows listed here start simultaneously and the step ends when the
    slowest flow finishes (i.e. a collective barrier).

    Attributes
    ----------
    flows : list of (src_gpu, dst_gpu, chunk_bytes)
    """
    flows: list[tuple[str, str, int]]


@dataclass
class FlowTrace:
    """Timing breakdown for one flow within a step."""
    src:          str
    dst:          str
    chunk_bytes:  int
    path:         list[str]      # node sequence through the fabric
    prop_delay_s: float          # propagation + switching latency
    tx_delay_s:   float          # transmission (serialisation) delay
    eff_bw_gbps:  float          # achieved bandwidth (after contention)

    @property
    def total_s(self) -> float:
        return self.prop_delay_s + self.tx_delay_s


@dataclass
class StepResult:
    """Outcome of one communication step."""
    step_idx:    int
    time_s:      float           # wall time for this step (max flow time)
    flow_traces: list[FlowTrace]

    @property
    def bottleneck_flow(self) -> FlowTrace:
        return max(self.flow_traces, key=lambda f: f.total_s)


@dataclass
class CollectiveResult:
    """Metrics for a completed collective operation."""
    collective:   str
    nodes:        list[str]
    tensor_bytes: int
    num_steps:    int
    total_time_s: float
    step_results: list[StepResult]

    # Bus-bandwidth scaling factor — overridden by subclasses
    _bus_factor: float = field(default=1.0, repr=False)

    @property
    def algo_bw_gbps(self) -> float:
        """algbw = tensor_size / total_time  (standard NCCL metric, per GPU)."""
        return (self.tensor_bytes / self.total_time_s) / 1e9

    @property
    def bus_bw_gbps(self) -> float:
        """busbw = algbw × bus_factor  (actual bytes transferred on the wire per GPU)."""
        return self.algo_bw_gbps * self._bus_factor

    def summary(self) -> str:
        n = len(self.nodes)
        step_times_us = [r.time_s * 1e6 for r in self.step_results]
        lines = [
            f"{'─'*55}",
            f"  {self.collective}",
            f"  Nodes       : {n}  |  Tensor : {self.tensor_bytes/1e9:.3f} GB",
            f"  Total time  : {self.total_time_s*1e3:.3f} ms",
            f"  Algo BW     : {self.algo_bw_gbps:.2f} Gbps/GPU",
            f"  Bus BW      : {self.bus_bw_gbps:.2f} Gbps/GPU",
            f"  Steps       : {self.num_steps}",
            f"  Step time   : min={min(step_times_us):.1f} µs  "
            f"max={max(step_times_us):.1f} µs  "
            f"avg={sum(step_times_us)/len(step_times_us):.1f} µs",
        ]
        # Bottleneck link across all steps
        all_traces = [t for r in self.step_results for t in r.flow_traces]
        slowest = min(all_traces, key=lambda t: t.eff_bw_gbps)
        lines.append(
            f"  Min flow BW : {slowest.eff_bw_gbps:.2f} Gbps  "
            f"({slowest.src} → {slowest.dst}, path {len(slowest.path)-1} hops)"
        )
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collective pattern generators
# ---------------------------------------------------------------------------

class CollectivePattern:
    """
    Base class for collective communication patterns.

    Subclasses implement `steps()` which returns the communication schedule
    as a list of CommStep objects (each = one barrier round).
    """

    name = "Collective"
    _bus_factor = 1.0

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        raise NotImplementedError

    def sort_nodes_by_locality(
        self, nodes: list[str], topo: FatTreeTopology
    ) -> list[str]:
        """
        Reorder nodes so that adjacent ring neighbours share a pod/rack where
        possible — reduces cross-fabric hops per ring step.
        """
        by_pod: dict[int, list[str]] = defaultdict(list)
        for n in nodes:
            pod = topo.pod_of(n)
            by_pod[pod if pod is not None else -1].append(n)
        ordered: list[str] = []
        for pod_nodes in by_pod.values():
            ordered.extend(pod_nodes)
        return ordered


class RingAllReduce(CollectivePattern):
    """
    Ring AllReduce = ReduceScatter + AllGather (bandwidth-optimal).

    2*(N-1) steps, each step N concurrent flows of size tensor_bytes/N.
    Every GPU sends a total of 2*(N-1)/N * tensor_bytes over the collective.

    Best for large tensors (gradient synchronisation in LLM training).
    """
    name = "AllReduceRing"
    _bus_factor_fn = staticmethod(lambda n: 2 * (n - 1) / n)

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        n = len(nodes)
        if n < 2:
            return []
        chunk = max(1, tensor_bytes // n)
        out: list[CommStep] = []
        # ReduceScatter phase
        for _ in range(n - 1):
            out.append(CommStep(
                [(nodes[i], nodes[(i + 1) % n], chunk) for i in range(n)]
            ))
        # AllGather phase
        for _ in range(n - 1):
            out.append(CommStep(
                [(nodes[i], nodes[(i + 1) % n], chunk) for i in range(n)]
            ))
        return out


class ReduceScatter(CollectivePattern):
    """
    ReduceScatter: each GPU ends with 1/N of the fully-reduced tensor.

    N-1 ring steps.  Used as the first half of ring AllReduce, and also
    standalone in Megatron-LM tensor parallelism.
    """
    name = "ReduceScatter"
    _bus_factor_fn = staticmethod(lambda n: (n - 1) / n)

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        n = len(nodes)
        if n < 2:
            return []
        chunk = max(1, tensor_bytes // n)
        return [
            CommStep([(nodes[i], nodes[(i + 1) % n], chunk) for i in range(n)])
            for _ in range(n - 1)
        ]


class AllGather(CollectivePattern):
    """
    AllGather: every GPU broadcasts its chunk so all end up with the full tensor.

    N-1 ring steps.  Used as the second half of ring AllReduce, and standalone
    in sequence-parallel and tensor-parallel LLM layers.
    """
    name = "AllGather"
    _bus_factor_fn = staticmethod(lambda n: (n - 1) / n)

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        n = len(nodes)
        if n < 2:
            return []
        chunk = max(1, tensor_bytes // n)
        return [
            CommStep([(nodes[i], nodes[(i + 1) % n], chunk) for i in range(n)])
            for _ in range(n - 1)
        ]


class AllToAll(CollectivePattern):
    """
    AllToAll (personalised exchange): each GPU sends a different chunk to
    every other GPU.

    Used in Mixture-of-Experts (MoE) expert dispatch and combine steps.
    tensor_bytes = total data each GPU sends, split evenly across N-1 peers.

    Two variants:
      structured=True  (default) — N-1 serialised rounds; in round r, node i
                                   communicates with node (i+r) % N.  Avoids
                                   simultaneous hot-spots on the fabric.
      structured=False           — all N*(N-1) flows in one single step.
    """
    name = "AllToAll"
    _bus_factor_fn = staticmethod(lambda n: (n - 1) / n)

    def __init__(self, structured: bool = True):
        self.structured = structured

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        n = len(nodes)
        if n < 2:
            return []
        chunk = max(1, tensor_bytes // (n - 1))

        if self.structured:
            return [
                CommStep([(nodes[i], nodes[(i + r) % n], chunk) for i in range(n)])
                for r in range(1, n)
            ]
        else:
            return [CommStep([
                (nodes[i], nodes[j], chunk)
                for i in range(n) for j in range(n) if i != j
            ])]


class Broadcast(CollectivePattern):
    """
    Broadcast: one root GPU sends the full tensor to all others.

    Uses a binary-tree schedule: ceil(log2(N)) rounds.
    Each round, current receivers become senders — optimal latency.
    Commonly used for parameter broadcast at start of training.
    """
    name = "Broadcast"
    _bus_factor_fn = staticmethod(lambda n: 1.0)

    def __init__(self, root_idx: int = 0):
        self.root_idx = root_idx

    def steps(self, nodes: list[str], tensor_bytes: int) -> list[CommStep]:
        n = len(nodes)
        if n < 2:
            return []
        active   = {self.root_idx}
        pending  = list(range(n))
        pending.remove(self.root_idx)
        out: list[CommStep] = []

        while pending:
            flows: list[tuple[str, str, int]] = []
            new_active: set[int] = set()
            senders = list(active)
            for sender_idx in senders:
                if pending:
                    recv_idx = pending.pop(0)
                    flows.append((nodes[sender_idx], nodes[recv_idx], tensor_bytes))
                    new_active.add(recv_idx)
            if flows:
                out.append(CommStep(flows))
            active |= new_active

        return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class RDMACollectiveSimulator:
    """
    SimPy-based simulator for RDMA collective communication on a fat-tree.

    Timing model per flow
    ---------------------
    total_time = prop_delay + switch_latency × num_switches + tx_delay

    where:
      prop_delay      = sum of per-link propagation delays (from graph 'prop_delay' attr)
      switch_latency  = configurable cut-through latency per intermediate switch node
      tx_delay        = chunk_bytes / effective_bandwidth
      eff_bw          = link_bw / num_concurrent_flows_on_link  (max-min fair sharing)

    Step time = max(flow times)  — barrier synchronisation.

    Parameters
    ----------
    topo : FatTreeTopology
    switch_latency_s : float
        Cut-through switching latency per intermediate switch node (default 400 ns).
    optimize_ring : bool
        If True, reorder ring nodes by pod for locality before generating steps.
    """

    def __init__(
        self,
        topo: FatTreeTopology,
        switch_latency_s: float = DEFAULT_SWITCH_LATENCY_S,
        optimize_ring: bool = True,
    ):
        self.topo = topo
        self.G = topo.graph
        self.switch_latency_s = switch_latency_s
        self.optimize_ring = optimize_ring
        self._path_cache: dict[tuple, list[str]] = {}

    # ------------------------------------------------------------------
    # Path computation — ECMP via flow-hash selection
    # ------------------------------------------------------------------

    def _path(self, src: str, dst: str, flow_hash: int = 0) -> list[str]:
        """Shortest-hop path, spread across ECMP alternatives by flow_hash."""
        key = (src, dst, flow_hash)
        if key in self._path_cache:
            return self._path_cache[key]

        try:
            paths = list(nx.all_shortest_paths(self.G, src, dst))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        path = paths[flow_hash % len(paths)]
        self._path_cache[key] = path
        return path

    # ------------------------------------------------------------------
    # Congestion model — max-min fair sharing per step
    # ------------------------------------------------------------------

    def _compute_step(self, step: CommStep, step_idx: int) -> StepResult:
        """
        Compute timing for one barrier-synchronised communication step.

        Algorithm
        ---------
        1. Find paths for all flows (ECMP-spread).
        2. Count concurrent flows per undirected link.
        3. Each flow's effective bandwidth = min over its links of
           (link_bw / num_flows_on_link).
        4. Flow time = prop_delay + switch_latency × switches + tx_delay.
        5. Step time = max(flow_times).
        """
        flows   = step.flows
        n_flows = len(flows)

        paths = [
            self._path(src, dst, step_idx * n_flows + fid)
            for fid, (src, dst, _) in enumerate(flows)
        ]

        # link → [flow_ids] using normalised (u, v) with u < v
        link_users: dict[tuple, list[int]] = defaultdict(list)
        for fid, path in enumerate(paths):
            for u, v in zip(path, path[1:]):
                link_users[(min(u, v), max(u, v))].append(fid)

        traces: list[FlowTrace] = []
        for fid, (src, dst, chunk_bytes) in enumerate(flows):
            path = paths[fid]
            if not path:
                traces.append(FlowTrace(src, dst, chunk_bytes, [], 0.0, float("inf"), 0.0))
                continue

            # Effective bandwidth: bottleneck under fair sharing
            eff_bw_mbps = float("inf")
            for u, v in zip(path, path[1:]):
                link = (min(u, v), max(u, v))
                bw = self.G[u][v]["bandwidth"]          # Mbps
                eff_bw_mbps = min(eff_bw_mbps, bw / len(link_users[link]))

            # Propagation delay (sum of link prop_delay attrs)
            prop_s = sum(self.G[u][v]["prop_delay"] for u, v in zip(path, path[1:]))

            # Switching latency: one per intermediate node (excludes src & dst GPUs)
            num_switches = len(path) - 2    # intermediate switch nodes
            switch_s = num_switches * self.switch_latency_s

            # Transmission delay
            bw_bytes_s = eff_bw_mbps * 1e6 / 8        # Mbps → bytes/s
            tx_s = chunk_bytes / bw_bytes_s if bw_bytes_s > 0 else float("inf")

            traces.append(FlowTrace(
                src=src, dst=dst,
                chunk_bytes=chunk_bytes,
                path=path,
                prop_delay_s=prop_s + switch_s,
                tx_delay_s=tx_s,
                eff_bw_gbps=eff_bw_mbps / 1e3,
            ))

        step_time = max((t.total_s for t in traces), default=0.0)
        return StepResult(step_idx=step_idx, time_s=step_time, flow_traces=traces)

    # ------------------------------------------------------------------
    # SimPy process interface
    # ------------------------------------------------------------------

    def run(
        self,
        env: simpy.Environment,
        pattern: CollectivePattern,
        nodes: list[str],
        tensor_bytes: int,
        optimize_ring: bool | None = None,
    ):
        """
        SimPy generator process — yields one env.timeout per collective step.

        Returns a CollectiveResult accessible via proc.value after env.run().
        """
        opt     = self.optimize_ring if optimize_ring is None else optimize_ring
        ordered = pattern.sort_nodes_by_locality(nodes, self.topo) if opt else nodes

        step_list    = pattern.steps(ordered, tensor_bytes)
        step_results: list[StepResult] = []

        for idx, step in enumerate(step_list):
            sr = self._compute_step(step, idx)
            yield env.timeout(sr.time_s)
            step_results.append(sr)

        n          = len(ordered)
        bus_factor = pattern._bus_factor_fn(n) if callable(getattr(pattern, "_bus_factor_fn", None)) else 1.0

        result = CollectiveResult(
            collective=pattern.name,
            nodes=ordered,
            tensor_bytes=tensor_bytes,
            num_steps=len(step_list),
            total_time_s=sum(r.time_s for r in step_results),
            step_results=step_results,
            _bus_factor=bus_factor,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience synchronous interface
    # ------------------------------------------------------------------

    def run_sync(
        self,
        pattern: CollectivePattern,
        nodes: list[str],
        tensor_bytes: int,
        optimize_ring: bool | None = None,
    ) -> CollectiveResult:
        """Run a collective without wiring up SimPy yourself."""
        env  = simpy.Environment()
        proc = env.process(self.run(env, pattern, nodes, tensor_bytes, optimize_ring))
        env.run()
        return proc.value

    def compare(
        self,
        patterns: list[CollectivePattern],
        nodes: list[str],
        tensor_bytes: int,
    ) -> list[CollectiveResult]:
        """Run several collectives on the same node set and return all results."""
        return [self.run_sync(p, nodes, tensor_bytes) for p in patterns]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    k            = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_gpus       = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tensor_gb    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tensor_bytes = int(tensor_gb * 1e9)

    print(f"\nBuilding k={k} fat-tree topology…")
    topo = FatTreeTopology(k=k)
    print(topo.summary())

    sim = RDMACollectiveSimulator(topo, switch_latency_s=400e-9)

    for strategy in ("spread_pods", "same_pod", "same_rack"):
        nodes = topo.select_gpu_group(n_gpus, strategy=strategy)
        print(f"\n{'═'*55}")
        print(f"  Strategy : {strategy}  |  {n_gpus} GPUs  |  {tensor_gb:.1f} GB tensor")
        print(f"  Nodes    : {nodes}")

        results = sim.compare(
            patterns=[
                RingAllReduce(),
                ReduceScatter(),
                AllGather(),
                AllToAll(),
                Broadcast(),
            ],
            nodes=nodes,
            tensor_bytes=tensor_bytes,
        )
        for r in results:
            print(r.summary())
