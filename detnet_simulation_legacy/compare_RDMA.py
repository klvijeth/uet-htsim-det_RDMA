"""
compare_RDMA.py

Sweeps FG (training) workload intensity and compares CQF vs circuit-switched
on two metrics:

  FG SLO Attainment (%) : fraction of training AllReduce jobs that met their
                           synchronization-barrier deadline.

  CG Completion Time     : per-request inference latency (mean, p95, p99).
                           CG flows are persistent — one continuous stream per
                           ring edge, cycling as fast as the network allows.

Story:
  CG inference flows are always present and need low latency.
  FG training jobs arrive in bursts (Poisson) and compete for bandwidth.
  CQF redistributes FG idle-slot slack to CG flows → lower inference latency.
  Circuit-switched locks CG at required_bw regardless of FG activity.

Key difference the plots expose:
  FG SLO  — both controllers have the same admission capacity (same worst-case
             peak reservation), so FG SLO degrades identically.  This is a
             fairness sanity-check: CQF does not sacrifice training throughput.
  CG time — CQF gives CG flows up to (capacity − FG_peak − CG_floor) Gbps of
             slack; CS gives exactly CG_floor.  At 0 FG load the CQF speedup
             is ~20× (1 vs 10 cycles); it narrows as FG fills the link.
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import simpy

from CQF_RDMA import CQFCentralController, RDMATrafficGenerator
from circuit_RDMA import CircuitSwitchedController


# ──────────────────────────────────────────────────────────────────────────────
# Network topology
# ──────────────────────────────────────────────────────────────────────────────

def create_spine_leaf(n_spines=4, n_leaves=4, n_servers_per_leaf=2,
                      server_bw=100.0, spine_bw=400.0, link_km=0.1):
    G       = nx.Graph()
    n_srv   = n_leaves * n_servers_per_leaf
    srv_ids = list(range(n_srv))
    lv_ids  = list(range(n_srv, n_srv + n_leaves))
    sp_ids  = list(range(n_srv + n_leaves, n_srv + n_leaves + n_spines))

    for l_i, leaf in enumerate(lv_ids):
        for s_i in range(n_servers_per_leaf):
            srv = srv_ids[l_i * n_servers_per_leaf + s_i]
            G.add_edge(srv, leaf, bandwidth=server_bw, length=link_km)

    for leaf in lv_ids:
        for spine in sp_ids:
            G.add_edge(leaf, spine, bandwidth=spine_bw, length=link_km)

    return G, srv_ids, lv_ids, sp_ids


# ──────────────────────────────────────────────────────────────────────────────
# Single simulation run
# ──────────────────────────────────────────────────────────────────────────────

def run_one(controller_class, ctrl_kwargs, gpu_nodes, sim_duration,
            cg_data_kb, cg_req_bw,
            fg_arrival_rate, fg_data_kb, fg_deadline,
            fg_peak_bw, fg_active, fg_period,
            cycle_time=1.0, seed=42):
    env  = simpy.Environment()
    ctrl = controller_class(env=env, **ctrl_kwargs)

    gen = RDMATrafficGenerator(
        env              = env,
        controller       = ctrl,
        gpu_nodes        = gpu_nodes,
        sim_duration     = sim_duration,
        cg_data_kb       = cg_data_kb,
        cg_req_bw        = cg_req_bw,
        fg_arrival_rate  = fg_arrival_rate,
        fg_data_kb       = fg_data_kb,
        fg_deadline      = fg_deadline,
        fg_peak_bw       = fg_peak_bw,
        fg_active_cycles = fg_active,
        fg_period        = fg_period,
        step_time        = cycle_time,
        seed             = seed,
    )
    gen.run()
    # Run past sim_duration to let in-flight FG jobs finish
    env.run(until=sim_duration + fg_deadline)
    return ctrl.get_performance_metrics(sim_duration)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep(arrival_rates, controller_class, ctrl_kwargs, gpu_nodes, sim_duration,
          cg_data_kb, cg_req_bw, fg_data_kb, fg_deadline,
          fg_peak_bw, fg_active, fg_period, cycle_time, label, seed=42):
    rows = []
    for rate in arrival_rates:
        m = run_one(controller_class, ctrl_kwargs, gpu_nodes, sim_duration,
                    cg_data_kb, cg_req_bw,
                    rate, fg_data_kb, fg_deadline,
                    fg_peak_bw, fg_active, fg_period,
                    cycle_time, seed)
        rows.append(m)
        print(f"  [{label:10s}] λ={rate:.3f}  "
              f"FG_SLO={m['FG SLO Attainment (%)']:5.1f}%  "
              f"CG_mean={m['CG Mean Completion Time']:6.2f}  "
              f"CG_p95={m['CG p95 Completion Time']:6.2f}  "
              f"accepted={m['Accepted Flows']:4d}  rejected={m['Rejected Flows']:4d}")
        sys.stdout.flush()
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")
    plt.show()
    plt.close(fig)


def plot_comparison(arrival_rates, cqf_rows, cs_rows, out_prefix):
    def get(rows, key):
        return np.array([r[key] for r in rows])

    x     = arrival_rates
    xl    = "FG Training Job Arrival Rate  (jobs / cycle)"
    CQF_C = "#1f77b4"
    CS_C  = "#ff7f0e"

    FS = dict(title=15, label=13, tick=11, legend=12)

    def _style(ax, title, ylabel):
        ax.set_title(title,  fontsize=FS["title"], pad=10)
        ax.set_xlabel(xl,    fontsize=FS["label"])
        ax.set_ylabel(ylabel, fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.legend(fontsize=FS["legend"], loc="best")
        ax.grid(True, alpha=0.3)

    # ── figure 1 : FG SLO attainment ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, get(cqf_rows, "FG SLO Attainment (%)"), "o-",  color=CQF_C,
            label="CQF", linewidth=2, markersize=7)
    ax.plot(x, get(cs_rows,  "FG SLO Attainment (%)"), "s--", color=CS_C,
            label="Circuit-Switched", linewidth=2, markersize=7)
    ax.set_ylim(-5, 105)
    _style(ax, "Training AllReduce SLO vs. Load", "FG SLO Attainment (%)")
    plt.tight_layout()
    _save_fig(fig, f"{out_prefix}_fg_slo.png")

    # ── figure 2 : CG mean completion time ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, get(cqf_rows, "CG Mean Completion Time"), "o-",  color=CQF_C,
            label="CQF", linewidth=2, markersize=7)
    ax.plot(x, get(cs_rows,  "CG Mean Completion Time"), "s--", color=CS_C,
            label="Circuit-Switched", linewidth=2, markersize=7)
    _style(ax, "Inference Request Mean Latency vs. Load", "Completion Time (cycles)")
    plt.tight_layout()
    _save_fig(fig, f"{out_prefix}_cg_mean.png")

    # ── figure 3 : CG tail latency ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, get(cqf_rows, "CG p95 Completion Time"), "o-",  color=CQF_C,
            label="CQF p95", linewidth=2, markersize=7)
    ax.plot(x, get(cs_rows,  "CG p95 Completion Time"), "s--", color=CS_C,
            label="Circuit p95", linewidth=2, markersize=7)
    ax.plot(x, get(cqf_rows, "CG p99 Completion Time"), "^:",  color=CQF_C, alpha=0.7,
            label="CQF p99", linewidth=1.8, markersize=6)
    ax.plot(x, get(cs_rows,  "CG p99 Completion Time"), "v:",  color=CS_C,  alpha=0.7,
            label="Circuit p99", linewidth=1.8, markersize=6)
    _style(ax, "Inference Request Tail Latency vs. Load", "Completion Time (cycles)")
    plt.tight_layout()
    _save_fig(fig, f"{out_prefix}_cg_tail.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── constants ────────────────────────────────────────────────────────
    CYCLE_TIME   = 1.0    # abstract time unit
    PACKET_SIZE  = 1.0    # KB
    SERVER_BW    = 100.0  # Gbps (server ↔ leaf; bottleneck for ring traffic)
    SIM_DURATION = 500.0

    # ── topology ─────────────────────────────────────────────────────────
    G, server_ids, leaf_ids, _ = create_spine_leaf(
        n_spines=4, n_leaves=4, n_servers_per_leaf=2,
        server_bw=SERVER_BW, spine_bw=400.0, link_km=0.1,
    )
    n_leaves  = len(leaf_ids)
    gpu_nodes = [server_ids[i * 2] for i in range(n_leaves)]   # [0, 2, 4, 6]

    print(f"Topology : {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"GPU ring : {gpu_nodes}")

    # ── CG inference parameters ──────────────────────────────────────────
    # Persistent point-to-point requests (one stream per ring edge).
    # Service time at required_bw: ceil(50 000 KB / 5 000 KB·cycle⁻¹) = 10 cycles.
    # CQF gives CG flows slack from FG idle slots → completes in 1–5 cycles
    # depending on how many FG jobs are active; CS always takes 10 cycles.
    CG_DATA_KB = 50_000   # KB per inference request
    CG_REQ_BW  = 5.0      # Gbps guaranteed floor

    # ── FG training parameters ───────────────────────────────────────────
    # Bursty gradient-sync AllReduce: 10 Gbps for 3 of every 10 cycles.
    # Both controllers use the same duty-cycle profile for FG transmission so
    # service times are identical (10 cycles) and only CG handling differs.
    # Peak reservation: CQF slots the burst; CS holds 10 Gbps the whole time.
    # Admission capacity: both reject when (N_fg × 10 + 5) > 100 → max 9 jobs.
    FG_PEAK_BW  = 10.0    # Gbps burst
    FG_ACTIVE   = 3       # active slots per period
    FG_PERIOD   = 10      # period cycles
    # 30 000 KB per ring segment → 3 active slots × 10 000 KB = 30 000 KB → 1 period = 10 cycles
    FG_DATA_KB  = 30_000  # KB per ring-segment flow
    FG_DEADLINE = 50.0    # cycles (5× service time; both controllers always meet this when admitted)

    # ── workload intensity sweep ─────────────────────────────────────────
    # With staggered profiles (3 non-overlapping slot windows per period):
    #   CS  : max 9 concurrent FG (peak 10 Gbps reserved per flow regardless
    #          of window).  Avg service time = (3+6+9)/3 = 6 cycles → λ_sat ≈ 1.5
    #   CQF : max 9 flows per window × 3 windows = 27 concurrent FG.
    #          Same avg service time 6 cycles → λ_sat ≈ 4.5
    # Sweep from lightly loaded through both saturation points.
    arrival_rates = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0,
                               2.5, 3.0, 3.5, 4.0, 5.0])

    # ── controller kwargs ────────────────────────────────────────────────
    cqf_kw = dict(network=G, cycle_time=CYCLE_TIME, packet_size=PACKET_SIZE)
    cs_kw  = dict(network=G, packet_size=PACKET_SIZE, batch_time=CYCLE_TIME)

    # ── run sweeps ───────────────────────────────────────────────────────
    print("\n── CQF sweep ──────────────────────────────────────────────────")
    cqf_rows = sweep(arrival_rates, CQFCentralController, cqf_kw,
                     gpu_nodes, SIM_DURATION,
                     CG_DATA_KB, CG_REQ_BW,
                     FG_DATA_KB, FG_DEADLINE,
                     FG_PEAK_BW, FG_ACTIVE, FG_PERIOD,
                     CYCLE_TIME, label="CQF")

    print("\n── Circuit-Switched sweep ─────────────────────────────────────")
    cs_rows = sweep(arrival_rates, CircuitSwitchedController, cs_kw,
                    gpu_nodes, SIM_DURATION,
                    CG_DATA_KB, CG_REQ_BW,
                    FG_DATA_KB, FG_DEADLINE,
                    FG_PEAK_BW, FG_ACTIVE, FG_PERIOD,
                    CYCLE_TIME, label="Circuit-SW")

    # ── plot ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_comparison(arrival_rates, cqf_rows, cs_rows,
                    out_prefix="rdma_comparison")


if __name__ == "__main__":
    main()
