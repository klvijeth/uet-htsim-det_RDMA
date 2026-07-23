// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DETRDMA_CONTROLLER_H
#define DETRDMA_CONTROLLER_H

/*
 * CqfController -- packet-level port of the Cyclic Queuing and Forwarding
 * (CQF) central scheduler from detnet_simulation_legacy/CQF_RDMA.py
 * (CQFCentralController).
 *
 * This is a plain reactive object, not an EventSource: exactly like the
 * Python original, it only does work when a flow is admitted or released
 * (admit()/release()) -- there is no per-cycle global recomputation loop.
 * Per-cycle send gating happens inside each DetRdmaSrc's own re-arming
 * doNextEvent(), reading fields this controller mutates in place.
 *
 * Ported method-for-method from CQF_RDMA.py; see the .cpp for the mapping
 * back to the specific Python methods.
 *
 * Known, deliberate simplification vs. the Python model: the propagation/
 * processing-delay overhead term folded into coarse_bandwidth in
 * compute_link_bw_vectors (CQF_RDMA.py lines 68-90) is not modelled here --
 * it would require threading a Pipe* alongside every BaseQueue* on a path.
 * Choose cycle_time well above the topology's per-hop propagation delay
 * to stay in the regime that term exists to guard against.
 */

#include <map>
#include <vector>
#include "network.h"
#include "queue.h"
#include "route.h"

class DetRdmaSrc;

class CqfController {
public:
    CqfController(simtime_picosec cycle_time, mem_b packet_size);

    // Attempt to admit `flow` on one of `candidates` (as returned by
    // Topology::get_bidir_paths). On success, returns the index into
    // `candidates` of the chosen path (flow->_path is populated with that
    // path's BaseQueue hops, and, for coarse flows, flow->_cycle_packets /
    // flow->_assigned_bw are set), and the flow is reserved on every hop.
    // On failure (no path has enough headroom / fine-grained flow doesn't
    // fit its hyperperiod envelope), returns -1 and `flow` is left
    // untouched. Mirrors CQFCentralController.route_flow.
    int admit(DetRdmaSrc* flow, const std::vector<const Route*>& candidates);

    // Release a previously admitted flow and redistribute freed slack to
    // any remaining coarse flows on the touched links.
    // Mirrors CQFCentralController.release_bandwidth.
    void release(DetRdmaSrc* flow);

    simtime_picosec cycle_time() const { return _cycle_time; }
    mem_b packet_size() const { return _packet_size; }

private:
    // A hyperperiod envelope longer than this many slots is rejected
    // outright rather than allocated -- guards against the LCM of a
    // pathological set of FG periods blowing up (CQF_RDMA.py has no
    // equivalent guard; numpy/Python integers don't overflow the way a
    // naive C++ port would).
    static const size_t MAX_HYPERPERIOD_SLOTS = 1 << 20;

    // Bandwidth buffer margin (bps) withheld from FG admission, mirroring
    // CQFCentralController.buffer_a (CQF_RDMA.py line 25). Kept at 0 here
    // since Python's buffer_a=0.1 is expressed in Gbps of a Gbps-scale
    // model; re-tune via set_fg_buffer() if needed once real link speeds
    // are wired up.
    double _fg_buffer_bps;

    simtime_picosec _cycle_time;
    mem_b _packet_size;

    std::map<BaseQueue*, std::vector<DetRdmaSrc*> > _coarse_flows;
    std::map<BaseQueue*, std::vector<DetRdmaSrc*> > _fine_flows;

    static void extract_queues(const Route* route, std::vector<BaseQueue*>& out);
    static size_t lcm(size_t a, size_t b);

    // Bandwidth envelope (bps) over one hyperperiod of already-reserved
    // traffic on `q`. Mirrors compute_link_bw_vectors.
    std::vector<double> compute_link_bw_vector(BaseQueue* q) const;

    // Peak of compute_link_bw_vector. Mirrors compute_link_bandwidths.
    double compute_link_bandwidth(BaseQueue* q) const;

    // bitrate(q) - compute_link_bandwidth(q), floored at 0.
    // Mirrors calc_link_available_bandwidth.
    double calc_link_available_bandwidth(BaseQueue* q) const;

    // Does an FG flow's periodic profile fit alongside what's already
    // reserved on every hop of `path`? Mirrors check_fg_flow_fits.
    bool check_fg_flow_fits(const std::vector<BaseQueue*>& path, DetRdmaSrc* flow) const;

    // Pick the candidate route maximizing the minimum available bandwidth
    // across its queue hops. Scoped-down analogue of widest_path_dynamic:
    // the Python version searches an arbitrary graph; here we can only
    // choose among the ECMP candidates the topology already enumerated.
    int widest_path(const std::vector<const Route*>& candidates,
                     std::vector<BaseQueue*>& out_queues,
                     double& out_bottleneck) const;

    void reserve(DetRdmaSrc* flow, const std::vector<BaseQueue*>& path);

    // Recompute cycle_packets/assigned_bw for one coarse flow from current
    // slack across its path. Mirrors adjust_flow_bandwidth.
    void adjust_flow_bandwidth(DetRdmaSrc* cg_flow) const;
};

#endif
