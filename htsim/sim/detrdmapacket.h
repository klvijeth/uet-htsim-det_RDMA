// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DETRDMAPACKET_H
#define DETRDMAPACKET_H

#include "network.h"

// DetRdmaPacket / DetRdmaAck are the data plane for the centrally-scheduled
// (CQF-style) DetRDMA transport. They follow the same PacketDB-backed
// newpkt()/free() reuse idiom as RocePacket/RoceAck (see rocepacket.h).
//
// There is deliberately no NACK/retransmit packet type: a flow that has
// been admitted by CqfController is guaranteed a reserved share of every
// hop's bandwidth for its cyclic slot, so it should never encounter queue
// contention. If a gap is ever observed at the sink, that indicates a bug
// in the admission-control accounting, not a real network condition -- see
// DetRdmaSink::receivePacket, which asserts rather than recovering.

class DetRdmaPacket : public Packet {
public:
    typedef uint64_t seq_t;

    inline static DetRdmaPacket* newpkt(PacketFlow& flow, const Route& route,
                                         seq_t seqno, int size,
                                         bool last_packet,
                                         uint32_t destination = UINT32_MAX) {
        DetRdmaPacket* p = _packetdb.allocPacket();
        p->set_route(flow, route, size, seqno + size - 1);
        p->_type = DETRDMADATA;
        p->_is_header = false;
        p->_seqno = seqno;
        p->_last_packet = last_packet;
        p->_path_len = route.size();
        p->_direction = NONE;
        p->set_dst(destination);
        return p;
    }

    void free() { _packetdb.freePacket(this); }
    virtual ~DetRdmaPacket() {}

    inline seq_t seqno() const { return _seqno; }
    inline bool last_packet() const { return _last_packet; }
    inline simtime_picosec ts() const { return _ts; }
    inline void set_ts(simtime_picosec ts) { _ts = ts; }
    virtual PktPriority priority() const { return Packet::PRIO_LO; }

    const static int ACKSIZE = 64;

protected:
    seq_t _seqno;
    simtime_picosec _ts;
    bool _last_packet;
    static PacketDB<DetRdmaPacket> _packetdb;
};

class DetRdmaAck : public Packet {
public:
    typedef DetRdmaPacket::seq_t seq_t;

    inline static DetRdmaAck* newpkt(PacketFlow& flow, const Route& route,
                                      seq_t ackno,
                                      uint32_t destination = UINT32_MAX) {
        DetRdmaAck* p = _packetdb.allocPacket();
        p->set_route(flow, route, DetRdmaPacket::ACKSIZE, ackno);
        p->_type = DETRDMAACK;
        p->_is_header = true;
        p->_ackno = ackno;
        p->_path_len = 0;
        p->_direction = NONE;
        p->set_dst(destination);
        return p;
    }

    void free() { _packetdb.freePacket(this); }
    virtual ~DetRdmaAck() {}

    inline seq_t ackno() const { return _ackno; }
    inline simtime_picosec ts() const { return _ts; }
    inline void set_ts(simtime_picosec ts) { _ts = ts; }
    virtual PktPriority priority() const { return Packet::PRIO_HI; }

protected:
    seq_t _ackno;
    simtime_picosec _ts;
    static PacketDB<DetRdmaAck> _packetdb;
};

#endif
