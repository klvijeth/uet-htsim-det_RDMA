// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include <iostream>
#include "detrdma.h"

using namespace std;

////////////////////////////////////////////////////////////////
//  DETRDMA SOURCE
////////////////////////////////////////////////////////////////

uint32_t DetRdmaSrc::_global_node_count = 0;

// The bitrate passed to BaseQueue() is unused by DetRdmaSrc (it never
// paces off BaseQueue::drainTime()/serviceCapacity() -- cyclic gating is
// driven entirely by CqfController's per-flow packet budget), so any
// nonzero placeholder is fine; BaseQueue divides by it at construction
// time and would fault on 0.
DetRdmaSrc::DetRdmaSrc(CqfController& controller, EventList& eventlist)
    : BaseQueue(speedFromMbps(100000.0), eventlist, NULL),
      _flow(nullptr),
      _controller(controller)
{
    _flow_type = DetRdmaFlowType::COARSE;
    _required_bw = 0.0;
    _period = 0;
    _cycle_packets = 0;
    _assigned_bw = 0.0;

    _end_trigger = NULL;
    _mss = (uint16_t)controller.packet_size();
    _dstaddr = UINT32_MAX;
    _flow_size = ((uint64_t)1) << 63;
    _highest_sent = 0;
    _flow_started = false;
    _done = false;
    _admitted = false;
    _cycle_index = 0;

    _sink = NULL;
    _route = NULL;

    _node_num = _global_node_count++;
    _nodename = "detRdmaSrc " + to_string(_node_num); // embedded space is load-bearing,
                                                       // see the token-layout comment on
                                                       // receivePacket()'s completion print
}

void DetRdmaSrc::connect(Route* routeout, Route* routeback, DetRdmaSink& sink,
                          simtime_picosec starttime) {
    assert(routeout);
    _route = routeout;
    _sink = &sink;
    _flow.set_id(get_id());
    _flow._name = _name;
    _sink->connect(*this, routeback);

    if (starttime != TRIGGER_START) {
        // Matches RoceSrc::connect's (slightly quirky but established)
        // convention of treating `starttime` as microseconds here.
        eventlist().sourceIsPending(*this, timeFromUs((double)starttime));
    } else {
        cout << "TRIGGER START " << _nodename << endl;
    }
}

void DetRdmaSrc::startflow() {
    _flow_started = true;
    _highest_sent = 0;
    _done = false;
    _cycle_index = 0;

    if (!_admitted) {
        // Admission runs in main_detrdma.cpp (CqfController::admit) before
        // connect() is called, because the chosen path has to be turned
        // into the Route objects connect() needs. Getting here without
        // having been admitted means the wiring code skipped that step.
        cout << "Flow " << _name << " " << get_id()
             << " cannot start: never admitted by CqfController" << endl;
        _done = true;
        return;
    }

    eventlist().sourceIsPendingRel(*this, 0);
}

uint64_t DetRdmaSrc::budget_for_this_cycle() const {
    if (_flow_type == DetRdmaFlowType::COARSE) {
        return _cycle_packets;
    }
    // FINE: read straight off this flow's own periodic profile -- unlike
    // coarse flows, fine-grained flows are not slack-adjusted at runtime
    // by the controller (mirrors CQF_RDMA.py: only coarse_flows go through
    // adjust_flow_bandwidth).
    if (_period == 0) {
        return 0;
    }
    double slot_bw_bps = _profile[_cycle_index % _period];
    double bits_per_cycle = slot_bw_bps * timeAsSec(_controller.cycle_time());
    double packet_bits = (double)_mss * 8.0;
    if (packet_bits <= 0.0) {
        return 0;
    }
    return (uint64_t)(bits_per_cycle / packet_bits);
}

void DetRdmaSrc::send_packet() {
    if (_flow_size && (_highest_sent * _mss >= _flow_size)) {
        // Already sent everything; waiting on the cumulative ack to mark
        // the flow done. Mirrors RoceSrc::send_packet's identical guard.
        return;
    }

    bool last_packet = false;
    if (_flow_size && (_highest_sent + 1) * _mss >= _flow_size) {
        last_packet = true;
    }

    DetRdmaPacket* p = DetRdmaPacket::newpkt(_flow, *_route, _highest_sent + 1, _mss,
                                              last_packet, _dstaddr);
    p->set_ts(eventlist().now());
    _highest_sent++;
    p->sendOn();
}

void DetRdmaSrc::doNextEvent() {
    if (!_flow_started) {
        startflow();
        return;
    }
    if (_done || !_admitted) {
        return;
    }

    uint64_t budget = budget_for_this_cycle();
    _cycle_index++;

    for (uint64_t i = 0; i < budget; i++) {
        if (_flow_size && _highest_sent * _mss >= _flow_size) {
            break;
        }
        send_packet();
    }

    if (!_done) {
        eventlist().sourceIsPendingRel(*this, _controller.cycle_time());
    }
}

void DetRdmaSrc::receivePacket(Packet& pkt) {
    if (_done) {
        pkt.free();
        return;
    }

    switch (pkt.type()) {
    case DETRDMAACK: {
        const DetRdmaAck& ack = (const DetRdmaAck&)pkt;
        DetRdmaAck::seq_t ackno = ack.ackno();

        if (_flow_size && ackno * _mss >= _flow_size) {
            // Token layout deliberately mirrors uec.cpp's completion
            // print, NOT roce.cpp's: validate.py extracts items[8] as the
            // FCT in microseconds, which only lines up because _nodename
            // ("detRdmaSrc <n>") contains an embedded space. See
            // detrdma_controller.h / the plan doc for the full rationale.
            cout << "Flow " << _name << " flowId " << flow_id() << " " << _nodename
                 << " finished at " << timeAsUs(eventlist().now())
                 << " total bytes " << (mem_b)ackno * _mss << endl;
            _done = true;
            _controller.release(this);
            if (_end_trigger) {
                _end_trigger->activate();
            }
        }
        pkt.free();
        return;
    }
    default:
        abort();
    }
}

////////////////////////////////////////////////////////////////
//  DETRDMA SINK
////////////////////////////////////////////////////////////////

DetRdmaSink::DetRdmaSink()
    : DataReceiver("detrdma_sink"), _cumulative_ack(0)
{
    _src = NULL;
    _route = NULL;
    _nodename = "detRdmaSink";
    _srcaddr = UINT32_MAX;
}

void DetRdmaSink::connect(DetRdmaSrc& src, Route* route) {
    _src = &src;
    _route = route;
    _cumulative_ack = 0;
}

void DetRdmaSink::receivePacket(Packet& pkt) {
    assert(pkt.type() == DETRDMADATA);

    DetRdmaPacket* p = (DetRdmaPacket*)&pkt;
    DetRdmaPacket::seq_t seqno = p->seqno();
    simtime_picosec ts = p->ts();

    // A flow admitted by CqfController holds a reservation covering its
    // full send rate on every hop, so it should never see loss or
    // reordering. A gap here means the admission-control math
    // under-reserved somewhere -- a controller bug worth crashing loudly
    // on during development, not a network condition to paper over with
    // NACK-style recovery (which DetRDMA deliberately doesn't implement;
    // see detrdmapacket.h).
    assert(seqno <= _cumulative_ack + 1);

    if (seqno == _cumulative_ack + 1) {
        _cumulative_ack = seqno;
        send_ack(ts);
    }

    p->free();
}

void DetRdmaSink::send_ack(simtime_picosec ts) {
    DetRdmaAck* ack = DetRdmaAck::newpkt(_src->_flow, *_route, _cumulative_ack, _srcaddr);
    ack->set_ts(ts);
    ack->sendOn();
}
