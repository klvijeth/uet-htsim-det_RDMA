"""
fat_tree.py — Fat-tree (CLOS) topology for RDMA datacenter simulation.

Models GPU nodes interconnected through a 3-tier switching fabric identical
to those used in large-scale RDMA (RoCE / InfiniBand) GPU clusters.

Tier 3  Core switches       : (k/2)^2 switches, fully meshed upward
Tier 2  Aggregation switches: k pods × k/2 switches per pod
Tier 1  ToR / Edge switches : k pods × k/2 switches per pod
Tier 0  GPU nodes           : k pods × k/2 ToR × k/2 hosts = k^3/4 nodes

For k=8  → 128 GPU nodes  (a typical 1-rack-unit scale test cluster)
For k=16 → 1024 GPU nodes (production-scale GPU cluster)

Edge attributes (consistent with existing cqf_sim.py conventions):
  bandwidth  : float  Mbps
  length     : float  km  (used for propagation delay: delay = length / 200_000 s)
  link_type  : str    'host_tor' | 'tor_aggr' | 'aggr_core'
  inverse_bw : float  1 / bandwidth  (used by widest-path routing)

Node attributes:
  node_type  : str    'gpu' | 'tor' | 'aggr' | 'core'
  tier       : int    0-3
  pod        : int    pod index (not set on core switches)
  tor        : str    ToR node id (only on gpu nodes)
"""

import itertools
import random
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------------
# Default link parameters
# ---------------------------------------------------------------------------

# Datacenter RDMA links — 100 Gbps is the current standard (RoCEv2 / HDR IB)
# These can be overridden per FatTreeTopology instance.
DEFAULT_LINK_BW_GBPS = {
    "host_tor":  100,   # GPU NIC ↔ ToR
    "tor_aggr":  100,   # ToR ↔ Aggregation (full bisection: equal uplinks)
    "aggr_core": 100,   # Aggregation ↔ Core
}

# Cable lengths in km (used with the 200 000 km/s fiber speed constant)
DEFAULT_CABLE_LEN_KM = {
    "host_tor":  0.003,   # ~3 m  within rack
    "tor_aggr":  0.010,   # ~10 m within pod
    "aggr_core": 0.030,   # ~30 m cross-pod
}

_GBPS_TO_MBPS = 1_000        # 1 Gbps = 1 000 Mbps
_FIBER_SPEED_KM_PER_S = 200_000


# ---------------------------------------------------------------------------
# FatTreeTopology
# ---------------------------------------------------------------------------

class FatTreeTopology:
    """
    k-port fat-tree (CLOS) topology for an RDMA GPU datacenter.

    Parameters
    ----------
    k : int
        Port count per switch (must be even).  Scales the cluster as k^3/4
        GPU nodes.
    link_bw_gbps : dict, optional
        Per-link-type bandwidth overrides (keys: 'host_tor', 'tor_aggr',
        'aggr_core').  Missing keys fall back to DEFAULT_LINK_BW_GBPS.
    cable_len_km : dict, optional
        Per-link-type cable-length overrides in km.
    oversubscription : float
        Oversubscription ratio applied to uplinks (tor_aggr, aggr_core).
        1.0 = non-blocking / full bisection bandwidth (default).
        2.0 = 2:1 oversubscription (common in production).
    """

    def __init__(
        self,
        k: int = 8,
        link_bw_gbps: dict | None = None,
        cable_len_km: dict | None = None,
        oversubscription: float = 1.0,
    ):
        if k % 2 != 0 or k < 2:
            raise ValueError("k must be an even integer ≥ 2")

        self.k = k
        self.oversubscription = oversubscription
        self._bw = {**DEFAULT_LINK_BW_GBPS, **(link_bw_gbps or {})}
        self._len = {**DEFAULT_CABLE_LEN_KM, **(cable_len_km or {})}

        self.graph: nx.Graph = nx.Graph()

        # Node lists (populated during _build)
        self.gpu_nodes:   list[str] = []
        self.tor_switches: list[str] = []
        self.aggr_switches: list[str] = []
        self.core_switches: list[str] = []
        self.pods: list[list[str]] = []   # pods[pod_id] = [gpu_node_ids]

        self._meta: dict[str, dict] = {}  # per-node metadata

        self._build()

    # ------------------------------------------------------------------
    # Private build helpers
    # ------------------------------------------------------------------

    def _add_node(self, name: str, node_type: str, **attrs):
        self.graph.add_node(name, node_type=node_type, **attrs)
        self._meta[name] = {"node_type": node_type, **attrs}

    def _add_link(self, u: str, v: str, link_type: str):
        bw_gbps = self._bw[link_type]
        # Apply oversubscription to uplinks only
        if link_type in ("tor_aggr", "aggr_core"):
            bw_gbps = bw_gbps / self.oversubscription
        bw_mbps = bw_gbps * _GBPS_TO_MBPS
        length_km = self._len[link_type]

        self.graph.add_edge(
            u, v,
            bandwidth=bw_mbps,
            bandwidth_gbps=bw_gbps,
            length=length_km,
            prop_delay=length_km / _FIBER_SPEED_KM_PER_S,
            link_type=link_type,
            inverse_bw=1.0 / bw_mbps,
        )

    def _build(self):
        k = self.k
        hk = k // 2   # half-k

        # ---- Core switches: (k/2)^2 ----
        # core_j corresponds to stride group (j // hk) and port (j % hk)
        for j in range(hk * hk):
            name = f"core_{j}"
            self._add_node(name, "core", tier=3)
            self.core_switches.append(name)

        # ---- Pods ----
        for pod in range(k):
            pod_gpus: list[str] = []

            # Aggregation switches
            for ai in range(hk):
                aggr = f"aggr_{pod}_{ai}"
                self._add_node(aggr, "aggr", tier=2, pod=pod)
                self.aggr_switches.append(aggr)
                # Each aggr switch connects to hk core switches in its stride group
                # Stride group ai owns cores [ai*hk .. ai*hk + hk - 1]
                for port in range(hk):
                    core = f"core_{ai * hk + port}"
                    self._add_link(aggr, core, "aggr_core")

            # ToR / Edge switches
            for ei in range(hk):
                tor = f"tor_{pod}_{ei}"
                self._add_node(tor, "tor", tier=1, pod=pod)
                self.tor_switches.append(tor)
                # hk uplinks to each aggregation switch in the same pod
                for ai in range(hk):
                    aggr = f"aggr_{pod}_{ai}"
                    self._add_link(tor, aggr, "tor_aggr")
                # hk downlinks to GPU nodes
                for hi in range(hk):
                    gpu = f"gpu_{pod}_{ei}_{hi}"
                    self._add_node(gpu, "gpu", tier=0, pod=pod, tor=tor, rack=f"{pod}_{ei}")
                    self.gpu_nodes.append(gpu)
                    pod_gpus.append(gpu)
                    self._add_link(gpu, tor, "host_tor")

            self.pods.append(pod_gpus)

        assert len(self.gpu_nodes) == k ** 3 // 4
        assert len(self.core_switches) == hk ** 2
        assert len(self.tor_switches) == k * hk
        assert len(self.aggr_switches) == k * hk

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    def node_type(self, node: str) -> str:
        return self._meta[node]["node_type"]

    def pod_of(self, node: str) -> int | None:
        return self._meta[node].get("pod")

    def tor_of(self, gpu: str) -> str:
        return self._meta[gpu]["tor"]

    def rack_of(self, gpu: str) -> str:
        """Returns the rack identifier string '{pod}_{edge_idx}'."""
        return self._meta[gpu]["rack"]

    def gpus_in_same_rack(self, a: str, b: str) -> bool:
        return self.rack_of(a) == self.rack_of(b)

    def gpus_in_same_pod(self, a: str, b: str) -> bool:
        return self.pod_of(a) == self.pod_of(b)

    def gpus_in_pod(self, pod_id: int) -> list[str]:
        return [n for n in self.gpu_nodes if self._meta[n]["pod"] == pod_id]

    def gpus_in_rack(self, rack_id: str) -> list[str]:
        return [n for n in self.gpu_nodes if self._meta[n]["rack"] == rack_id]

    # ------------------------------------------------------------------
    # GPU group selection (for collective communication patterns)
    # ------------------------------------------------------------------

    def select_gpu_group(self, num_gpus: int, strategy: str = "spread_pods") -> list[str]:
        """
        Return a list of GPU node IDs to participate in a collective operation.

        Strategies
        ----------
        spread_pods
            One GPU per pod in round-robin pod order.  Maximises cross-pod
            (bisection) traffic — worst-case stress for the fabric.
        same_pod
            All GPUs drawn from a single pod.  Intra-pod all-reduce pattern.
        same_rack
            All GPUs drawn from a single ToR domain.  Intra-rack pattern.
        random
            Uniformly random selection from all GPU nodes.
        """
        if strategy == "spread_pods":
            selected: list[str] = []
            for pod_id in itertools.cycle(range(self.k)):
                if len(selected) >= num_gpus:
                    break
                candidates = self.gpus_in_pod(pod_id)
                already = {self.tor_of(g) for g in selected if self.pod_of(g) == pod_id}
                remaining = [g for g in candidates if self.tor_of(g) not in already]
                pick = random.choice(remaining) if remaining else random.choice(candidates)
                selected.append(pick)
            return selected[:num_gpus]

        elif strategy == "same_pod":
            pod_id = random.randrange(self.k)
            candidates = self.gpus_in_pod(pod_id)
            return random.sample(candidates, min(num_gpus, len(candidates)))

        elif strategy == "same_rack":
            tor = random.choice(self.tor_switches)
            rack = f"{self._meta[tor]['pod']}_{tor.split('_')[-1]}"
            candidates = self.gpus_in_rack(rack)
            return random.sample(candidates, min(num_gpus, len(candidates)))

        elif strategy == "random":
            return random.sample(self.gpu_nodes, min(num_gpus, len(self.gpu_nodes)))

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        k, hk = self.k, self.k // 2
        lines = [
            f"Fat-tree CLOS topology  (k={k})",
            f"  GPU nodes    : {len(self.gpu_nodes):>6}  ({hk} per ToR × {hk} ToR/pod × {k} pods)",
            f"  ToR switches : {len(self.tor_switches):>6}  ({hk} per pod)",
            f"  Aggr switches: {len(self.aggr_switches):>6}  ({hk} per pod)",
            f"  Core switches: {len(self.core_switches):>6}  ({hk}²)",
            f"  Total links  : {self.graph.number_of_edges():>6}",
            "",
            f"  Link bandwidth (Gbps):",
            f"    Host↔ToR : {self._bw['host_tor']} Gbps",
            f"    ToR↔Aggr : {self._bw['tor_aggr'] / self.oversubscription:.1f} Gbps  (oversubscription {self.oversubscription}:1)",
            f"    Aggr↔Core: {self._bw['aggr_core'] / self.oversubscription:.1f} Gbps",
            "",
            f"  Cable length / propagation delay:",
            f"    Host↔ToR : {self._len['host_tor']*1000:.0f} m  → {self._len['host_tor']/_FIBER_SPEED_KM_PER_S*1e9:.0f} ns",
            f"    ToR↔Aggr : {self._len['tor_aggr']*1000:.0f} m  → {self._len['tor_aggr']/_FIBER_SPEED_KM_PER_S*1e9:.0f} ns",
            f"    Aggr↔Core: {self._len['aggr_core']*1000:.0f} m → {self._len['aggr_core']/_FIBER_SPEED_KM_PER_S*1e9:.0f} ns",
        ]
        return "\n".join(lines)

    def link_stats(self) -> dict:
        """Return per-link-type counts and aggregate bandwidth."""
        stats = defaultdict(lambda: {"count": 0, "total_bw_tbps": 0.0})
        for u, v, d in self.graph.edges(data=True):
            lt = d["link_type"]
            stats[lt]["count"] += 1
            stats[lt]["total_bw_tbps"] += d["bandwidth_gbps"] / 1000
        return dict(stats)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw(self, figsize: tuple = (20, 10), save_path: str | None = None):
        """
        Draw the fat-tree topology with a tiered hierarchical layout.

        For large k the figure becomes dense; consider save_path with high dpi.
        """
        G = self.graph
        k, hk = self.k, self.k // 2
        pos = {}

        # Core: evenly spaced across the top
        n_core = len(self.core_switches)
        for i, n in enumerate(self.core_switches):
            pos[n] = (i * (k * (hk + 1)) / (n_core - 1 or 1), 3.0)

        pod_width = hk + 1
        for pod in range(k):
            x0 = pod * pod_width * hk

            for ai in range(hk):
                n = f"aggr_{pod}_{ai}"
                pos[n] = (x0 + ai * pod_width, 2.0)

            for ei in range(hk):
                n = f"tor_{pod}_{ei}"
                pos[n] = (x0 + ei * pod_width, 1.0)
                for hi in range(hk):
                    gn = f"gpu_{pod}_{ei}_{hi}"
                    pos[gn] = (x0 + ei * pod_width + hi * (pod_width / hk), 0.0)

        _COLORS = {
            "core": "#e74c3c",
            "aggr": "#e67e22",
            "tor":  "#3498db",
            "gpu":  "#2ecc71",
        }
        node_colors = [_COLORS[self.node_type(n)] for n in G.nodes()]
        node_sizes  = [80 if self.node_type(n) == "gpu" else 150 for n in G.nodes()]

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
            edge_color="#bdc3c7",
            width=0.4,
            arrows=False,
        )

        legend = [mpatches.Patch(color=c, label=lbl) for lbl, c in [
            ("Core switch",        "#e74c3c"),
            ("Aggregation switch", "#e67e22"),
            ("ToR / Edge switch",  "#3498db"),
            ("GPU node",           "#2ecc71"),
        ]]
        ax.legend(handles=legend, loc="upper right", fontsize=9)
        ax.set_title(
            f"Fat-tree CLOS  k={k}  |  {len(self.gpu_nodes)} GPU nodes  |  "
            f"{len(self.tor_switches)} ToR  |  {len(self.aggr_switches)} Aggr  |  "
            f"{len(self.core_switches)} Core",
            fontsize=11,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved topology diagram to {save_path}")
        else:
            plt.show()
        return fig, ax


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for k in (4, 8, 16):
        topo = FatTreeTopology(k=k)
        print(topo.summary())
        print()
        stats = topo.link_stats()
        for lt, s in stats.items():
            print(f"  {lt:12s}  {s['count']:4d} links  {s['total_bw_tbps']:.1f} Tbps aggregate")
        print()

    # Demonstrate GPU group selection for collective communication
    topo = FatTreeTopology(k=8)
    print("=== GPU group selection (k=8, 128 GPUs) ===")
    for strategy in ("spread_pods", "same_pod", "same_rack", "random"):
        group = topo.select_gpu_group(8, strategy=strategy)
        pods = [topo.pod_of(g) for g in group]
        print(f"  {strategy:15s}: {group[:4]}...  pods={pods}")

    # Draw a small topology (k=4 is readable)
    small = FatTreeTopology(k=4)
    print("\n" + small.summary())
    small.draw(figsize=(14, 7), save_path="fat_tree_k4.png")
