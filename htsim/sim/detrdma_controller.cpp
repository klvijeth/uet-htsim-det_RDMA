// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "detrdma_controller.h"
#include "detrdma.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>

using namespace std;

/*
 * Method-for-method mapping back to detnet_simulation_legacy/CQF_RDMA.py's
 * CQFCentralController:
 *
 *   compute_link_bw_vector   <- compute_link_bw_vectors      (line 63)
 *   compute_link_bandwidth   <- compute_link_bandwidths       (line 119)
 *   calc_link_available_bandwidth <- calc_link_available_bandwidth (line 237)
 *   check_fg_flow_fits       <- check_fg_flow_fits            (line 295)
 *   widest_path              <- widest_path_dynamic           (line 245), scoped down
 *                                to the ECMP candidates the topology enumerates
 *                                rather than an arbitrary graph search
 *   reserve / release        <- reserve_bandwidth / release_bandwidth (127 / 145)
 *   adjust_flow_bandwidth    <- adjust_flow_bandwidth         (line 165)
 *   admit                    <- route_flow                    (line 327)
 */

CqfController::CqfController(simtime_picosec cycle_time, mem_b packet_size)
    : _fg_buffer_bps(0.0), _cycle_time(cycle_time), _packet_size(packet_size)
{
    assert(cycle_time > 0);
    assert(packet_size > 0);
}

void CqfController::extract_queues(const Route* route, vector<BaseQueue*>& out) {
    out.clear();
    // Whether the bandwidth-limited hops (queues, as opposed to Pipes,
    // which only add propagation delay) fall at odd or even indices
    // depends on queue_type: FatTreeTopology::get_bidir_paths conditionally
    // inserts an extra getRemoteEndpoint() queue per hop for
    // LOSSLESS_INPUT/LOSSLESS_INPUT_ECN (fat_tree_topology.cpp:1284-1287),
    // which firstfit.cpp's fixed odd-index walk doesn't need to handle
    // because it only ever walks already-connected routes for a specific
    // topology configuration. CqfController::admit(), by contrast, has to
    // inspect *raw* candidates straight out of get_bidir_paths(), before
    // add_endpoints() and regardless of queue_type -- so filter by actual
    // type instead of assuming a fixed stride.
    for (size_t i = 0; i < route->size(); i++) {
        BaseQueue* q = dynamic_cast<BaseQueue*>(route->at(i));
        if (q) {
            out.push_back(q);
        }
    }
}

size_t CqfController::lcm(size_t a, size_t b) {
    if (a == 0) return b;
    if (b == 0) return a;
    return (a / std::gcd(a, b)) * b;
}

vector<double> CqfController::compute_link_bw_vector(BaseQueue* q) const {
    double coarse_bandwidth = 0.0;
    auto cit = _coarse_flows.find(q);
    if (cit != _coarse_flows.end()) {
        for (DetRdmaSrc* f : cit->second) {
            coarse_bandwidth += f->_required_bw;
        }
    }

    auto fit = _fine_flows.find(q);
    if (fit == _fine_flows.end() || fit->second.empty()) {
        return vector<double>(1, coarse_bandwidth);
    }

    size_t T_L = 1;
    for (DetRdmaSrc* f : fit->second) {
        T_L = lcm(T_L, f->_period);
        if (T_L > MAX_HYPERPERIOD_SLOTS) {
            T_L = MAX_HYPERPERIOD_SLOTS;
            break;
        }
    }

    vector<double> aggregate(T_L, 0.0);
    for (DetRdmaSrc* f : fit->second) {
        if (f->_period == 0) {
            continue;
        }
        for (size_t i = 0; i < T_L; i++) {
            aggregate[i] += f->_profile[i % f->_period];
        }
    }
    for (size_t i = 0; i < T_L; i++) {
        aggregate[i] += coarse_bandwidth;
    }
    return aggregate;
}

double CqfController::compute_link_bandwidth(BaseQueue* q) const {
    vector<double> v = compute_link_bw_vector(q);
    return *std::max_element(v.begin(), v.end());
}

double CqfController::calc_link_available_bandwidth(BaseQueue* q) const {
    double used = compute_link_bandwidth(q);
    double capacity = (double)q->bitrate();
    double avail = capacity - used;
    return avail > 0.0 ? avail : 0.0;
}

bool CqfController::check_fg_flow_fits(const vector<BaseQueue*>& path, DetRdmaSrc* flow) const {
    for (BaseQueue* q : path) {
        double capacity = (double)q->bitrate();
        vector<double> current = compute_link_bw_vector(q);
        size_t current_TL = current.size();
        size_t new_TL = lcm(current_TL, flow->_period);

        if (new_TL > MAX_HYPERPERIOD_SLOTS) {
            // Can't safely evaluate a hyperperiod this large -- reject
            // conservatively rather than allocate unboundedly (CQF_RDMA.py
            // has no equivalent guard).
            return false;
        }

        for (size_t i = 0; i < new_TL; i++) {
            double prospective = current[i % current_TL] + flow->_profile[i % flow->_period];
            if (prospective > capacity - _fg_buffer_bps) {
                return false; // collision on this hop
            }
        }
    }
    return true;
}

int CqfController::widest_path(const vector<const Route*>& candidates,
                                vector<BaseQueue*>& out_queues,
                                double& out_bottleneck) const {
    int best_idx = -1;
    double best_bottleneck = -1.0;
    vector<BaseQueue*> best_queues;

    for (size_t i = 0; i < candidates.size(); i++) {
        vector<BaseQueue*> queues;
        extract_queues(candidates[i], queues);
        if (queues.empty()) {
            continue;
        }

        double bottleneck = numeric_limits<double>::infinity();
        for (BaseQueue* q : queues) {
            bottleneck = min(bottleneck, calc_link_available_bandwidth(q));
        }
        if (bottleneck > best_bottleneck) {
            best_bottleneck = bottleneck;
            best_idx = (int)i;
            best_queues = queues;
        }
    }

    if (best_idx >= 0) {
        out_queues = best_queues;
        out_bottleneck = best_bottleneck;
    }
    return best_idx;
}

void CqfController::reserve(DetRdmaSrc* flow, const vector<BaseQueue*>& path) {
    for (BaseQueue* q : path) {
        if (flow->_flow_type == DetRdmaFlowType::COARSE) {
            _coarse_flows[q].push_back(flow);
        } else {
            _fine_flows[q].push_back(flow);
        }
    }

    // Redistribute slack to every coarse flow sharing a link this flow just
    // touched (itself included, if coarse) -- mirrors reserve_bandwidth's
    // post-reservation loop (CQF_RDMA.py:139-142).
    for (BaseQueue* q : path) {
        auto it = _coarse_flows.find(q);
        if (it == _coarse_flows.end()) {
            continue;
        }
        vector<DetRdmaSrc*> flows_copy = it->second; // iterate over a copy, as Python's list(...) does
        for (DetRdmaSrc* cg : flows_copy) {
            adjust_flow_bandwidth(cg);
        }
    }
}

void CqfController::adjust_flow_bandwidth(DetRdmaSrc* cg_flow) const {
    double cycle_s = timeAsSec(_cycle_time);
    double packet_bits = (double)_packet_size * 8.0;
    uint64_t min_extra_packets = UINT64_MAX;

    for (BaseQueue* q : cg_flow->_path) {
        double total_required_bw = compute_link_bandwidth(q);
        double link_capacity = (double)q->bitrate();
        double slack_bandwidth = link_capacity - total_required_bw;
        if (slack_bandwidth < 0.0) {
            slack_bandwidth = 0.0;
        }

        uint64_t extra_packets_per_flow = 0;
        if (slack_bandwidth > 0.0) {
            double total_slack_packets_on_link = (cycle_s * slack_bandwidth) / packet_bits;
            auto it = _coarse_flows.find(q);
            size_t num_cg_flows = (it != _coarse_flows.end() && !it->second.empty())
                                       ? it->second.size()
                                       : 1;
            extra_packets_per_flow = (uint64_t)(total_slack_packets_on_link / (double)num_cg_flows);
        }

        if (extra_packets_per_flow < min_extra_packets) {
            min_extra_packets = extra_packets_per_flow;
        }
    }
    if (min_extra_packets == UINT64_MAX) {
        min_extra_packets = 0;
    }

    uint64_t required_packets = (uint64_t)((cg_flow->_required_bw * cycle_s) / packet_bits);

    cg_flow->_cycle_packets = required_packets + min_extra_packets;
    cg_flow->_assigned_bw = ((double)cg_flow->_cycle_packets * packet_bits) / cycle_s;
}

int CqfController::admit(DetRdmaSrc* flow, const vector<const Route*>& candidates) {
    vector<BaseQueue*> queues;
    double bottleneck = 0.0;
    int idx = widest_path(candidates, queues, bottleneck);

    if (idx < 0) {
        return -1; // no candidate path has any usable hop at all
    }

    if (flow->_flow_type == DetRdmaFlowType::COARSE) {
        if (flow->_required_bw > bottleneck) {
            return -1; // not enough headroom anywhere along the widest path
        }
    } else {
        if (!check_fg_flow_fits(queues, flow)) {
            return -1; // doesn't fit its hyperperiod envelope
        }
    }

    flow->_path = queues;
    flow->_admitted = true;
    reserve(flow, queues); // also runs adjust_flow_bandwidth for `flow` itself, if coarse

    return idx;
}

void CqfController::release(DetRdmaSrc* flow) {
    for (BaseQueue* q : flow->_path) {
        auto& v = (flow->_flow_type == DetRdmaFlowType::COARSE) ? _coarse_flows[q] : _fine_flows[q];
        v.erase(std::remove(v.begin(), v.end(), flow), v.end());
    }

    for (BaseQueue* q : flow->_path) {
        auto it = _coarse_flows.find(q);
        if (it == _coarse_flows.end()) {
            continue;
        }
        vector<DetRdmaSrc*> flows_copy = it->second;
        for (DetRdmaSrc* cg : flows_copy) {
            adjust_flow_bandwidth(cg);
        }
    }
}
