// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DETRDMA_H
#define DETRDMA_H

/*
 * DetRdmaSrc / DetRdmaSink -- the centrally-scheduled (CQF) RDMA transport.
 *
 * Structured like RoceSrc/RoceSink (see roce.h), but stripped of RoCE's own
 * congestion control: DetRdmaSrc does not self-pace off an RTT/window
 * estimate. Instead it is gated by CqfController -- a CqfController::admit()
 * call at flow start reserves this flow's share of every hop's bandwidth
 * for the life of the flow, and DetRdmaSrc's own re-arming doNextEvent()
 * (armed every controller->cycle_time(), the same self-rearm idiom RoceSrc
 * and Clock use) sends up to that reserved per-cycle packet budget and no
 * more.
 *
 * RDMAFlow's fields (network.py, legacy SimPy model) are folded directly
 * into DetRdmaSrc as members rather than kept in a separate flow object --
 * RoceSrc does the same with its own flow state, and it avoids a two-object
 * synchronization hazard between a "flow" and its "source".
 */

#include <string>
#include <vector>
#include "config.h"
#include "network.h"
#include "detrdmapacket.h"
#include "detrdma_controller.h"
#include "queue.h"
#include "eventlist.h"
#include "trigger.h"

class DetRdmaSink;

enum class DetRdmaFlowType { COARSE, FINE };

class DetRdmaSrc : public BaseQueue, public TriggerTarget {
    friend class DetRdmaSink;
    friend class CqfController;

public:
    DetRdmaSrc(CqfController& controller, EventList& eventlist);

    virtual void connect(Route* routeout, Route* routeback, DetRdmaSink& sink,
                          simtime_picosec startTime);

    void set_dst(uint32_t dst) { _dstaddr = dst; }
    void set_flowid(flowid_t flow_id) { _flow.set_flowid(flow_id); }
    void set_flowsize(uint64_t flow_size_in_bytes) { _flow_size = flow_size_in_bytes; }
    void set_end_trigger(Trigger& trigger) { _end_trigger = &trigger; }

    // Coarse-grained: persistent bandwidth-floor reservation.
    void make_coarse(double required_bw_bps) {
        _flow_type = DetRdmaFlowType::COARSE;
        _required_bw = required_bw_bps;
    }
    // Fine-grained: periodic bursty reservation. profile[i] is the
    // bandwidth (bps) this flow wants to send at during slot i of its
    // period; profile.size() == period (in cycles).
    void make_fine(const std::vector<double>& profile) {
        _flow_type = DetRdmaFlowType::FINE;
        _profile = profile;
        _period = profile.size();
    }

    void startflow();

    // Set once by CqfController::admit() -- true if admission succeeded.
    bool admitted() const { return _admitted; }

    virtual void activate() { startflow(); }

    virtual void doNextEvent();
    virtual void receivePacket(Packet& pkt);

    virtual mem_b queuesize() const { return 0; }
    virtual mem_b maxsize() const { return 0; }

    virtual const string& nodename() { return _nodename; }
    inline uint32_t flow_id() const { return _flow.flow_id(); }

    static uint32_t _global_node_count;

    PacketFlow _flow;

    // Bookkeeping the controller mutates directly (see CqfController).
    // Kept public to match how RoceSrc exposes its own send-side counters
    // for loggers -- CqfController is effectively the same kind of
    // "trusted external mutator" firstfit.cpp's FirstFit is for TcpSrc.
    DetRdmaFlowType _flow_type;
    double _required_bw;          // COARSE only (bps)
    std::vector<double> _profile; // FINE only (bps per slot)
    size_t _period;                // FINE only (slots)
    uint64_t _cycle_packets;      // COARSE only: packets this src may send per cycle
    double _assigned_bw;          // COARSE only: cycle_packets converted back to bps, for logging
    std::vector<BaseQueue*> _path; // filled in by CqfController::admit()

protected:
    CqfController& _controller;
    Trigger* _end_trigger;

    string _nodename;
    uint32_t _node_num;
    uint32_t _dstaddr;

    uint16_t _mss;
    uint64_t _flow_size;
    uint64_t _highest_sent; // seqno of next byte to send (bytes)
    bool _flow_started;
    bool _done;
    bool _admitted;
    size_t _cycle_index; // FINE only: which slot of the period we're in

    DetRdmaSink* _sink;
    const Route* _route;

    void send_packet();
    uint64_t budget_for_this_cycle() const;
};

class DetRdmaSink : public PacketSink, public DataReceiver {
    friend class DetRdmaSrc;

public:
    DetRdmaSink();

    virtual void receivePacket(Packet& pkt);

    uint64_t cumulative_ack() { return _cumulative_ack; }
    uint64_t total_received() const { return _cumulative_ack; }
    uint32_t drops() { return 0; }
    virtual const string& nodename() { return _nodename; }

    void set_src(uint32_t s) { _srcaddr = s; }

    DetRdmaSrc* _src;

protected:
    void connect(DetRdmaSrc& src, Route* route);

    const Route* _route;
    string _nodename;
    uint32_t _srcaddr;

    DetRdmaAck::seq_t _cumulative_ack;

    void send_ack(simtime_picosec ts);
};

#endif
