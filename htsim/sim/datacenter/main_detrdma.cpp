// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
//
// htsim_detrdma -- centrally-scheduled (CQF) RDMA transport, wired up the
// same way main_roce.cpp wires up RoCE, so the two binaries can be run
// against the same connection matrices/topologies for a like-for-like FCT
// comparison. See detrdma_controller.h for the port of
// detnet_simulation_legacy/CQF_RDMA.py's CQFCentralController this binary
// drives.
//
// Stage 1 scope (see the approved plan): every connection in the traffic
// matrix is admitted as a COARSE (persistent bandwidth-floor) flow at a
// uniform -cqf_cg_bw. Fine-grained (periodic/bursty) flow support exists in
// CqfController/DetRdmaSrc already, but isn't wired to a CLI input yet --
// that's the next stage once this minimal path is validated end-to-end.

#include "config.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <math.h>

#include "network.h"
#include "pipe.h"
#include "eventlist.h"
#include "logfile.h"
#include "clock.h"
#include "compositequeue.h"
#include "topology.h"
#include "connection_matrix.h"
#include "fat_tree_topology.h"
#include "fat_tree_switch.h"

#include "detrdma.h"
#include "detrdma_controller.h"

#include <list>
#include <map>

using namespace std;

EventList eventlist;

void exit_error(char* progr) {
    cout << "Usage " << progr << " [-nodes N]\n\t[-q queue_size]\n\t[-tm traffic_matrix_file]\n"
         << "\t[-topo topology_file]\n\t[-log log_level]\n\t[-seed random_seed]\n\t[-end end_time_in_usec]\n"
         << "\t[-mtu MTU]\n\t[-linkspeed Mbps]\n\t[-hop_latency x] per hop wire latency in us, default 1\n"
         << "\t[-switch_latency x] switching latency in us, default 0\n"
         << "\t[-cqf_cycle_time x] CQF cycle duration in us, default 10\n"
         << "\t[-cqf_cg_bw x] per-flow coarse-grained bandwidth floor in Mbps, default = -linkspeed\n"
         << endl;
    exit(1);
}

int main(int argc, char** argv) {
    mem_b queuesize = 15;
    linkspeed_bps linkspeed = speedFromMbps((double)100000);
    int packet_size = 4000;
    uint32_t no_of_nodes = 16;
    uint32_t tiers = 3;
    stringstream filename(ios_base::out);
    simtime_picosec hop_latency = timeFromUs((uint32_t)1);
    simtime_picosec switch_latency = timeFromUs((uint32_t)0);
    queue_type qt = COMPOSITE;
    queue_type snd_type = FAIR_PRIO;
    int seed = 13;
    int end_time = 1000; // in microseconds

    double cqf_cycle_time_us = 10.0;
    double cqf_cg_bw_mbps = -1.0; // -1 => default to linkspeed once parsed

    char* tm_file = NULL;
    char* topo_file = NULL;

    filename << "logout.dat";

    int i = 1;
    while (i < argc) {
        if (!strcmp(argv[i], "-o")) {
            filename.str(std::string());
            filename << argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-end")) {
            end_time = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-nodes")) {
            no_of_nodes = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-tiers")) {
            tiers = atoi(argv[i + 1]);
            assert(tiers == 2 || tiers == 3);
            i++;
        } else if (!strcmp(argv[i], "-tm")) {
            tm_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-topo")) {
            topo_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-q")) {
            queuesize = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-linkspeed")) {
            linkspeed = speedFromMbps(atof(argv[i + 1]));
            i++;
        } else if (!strcmp(argv[i], "-seed")) {
            seed = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mtu")) {
            packet_size = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-hop_latency")) {
            hop_latency = timeFromUs(atof(argv[i + 1]));
            i++;
        } else if (!strcmp(argv[i], "-switch_latency")) {
            switch_latency = timeFromUs(atof(argv[i + 1]));
            i++;
        } else if (!strcmp(argv[i], "-cqf_cycle_time")) {
            cqf_cycle_time_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-cqf_cg_bw")) {
            cqf_cg_bw_mbps = atof(argv[i + 1]);
            i++;
        } else {
            cout << "Unknown parameter " << argv[i] << endl;
            exit_error(argv[0]);
        }
        i++;
    }

    srand(seed);
    srandom(seed);

    if (cqf_cg_bw_mbps < 0) {
        cqf_cg_bw_mbps = (double)linkspeed / 1e6; // default: reserve full line rate
    }

    Packet::set_packet_size(packet_size);
    eventlist.setEndtime(timeFromUs((uint32_t)end_time));
    queuesize = memFromPkt(queuesize);

    cout << "Logging to " << filename.str() << endl;
    Logfile logfile(filename.str(), eventlist);
    logfile.setStartTime(timeFromSec(0));

    CqfController controller(timeFromUs(cqf_cycle_time_us), packet_size);
    double cg_bw_bps = cqf_cg_bw_mbps * 1e6;

    QueueLoggerFactory* qlf = 0;

    unique_ptr<FatTreeTopology> top;
    unique_ptr<FatTreeTopologyCfg> topo_cfg;
    if (topo_file) {
        topo_cfg = FatTreeTopologyCfg::load(topo_file, queuesize, qt, snd_type);
        if (topo_cfg->no_of_nodes() != no_of_nodes) {
            cerr << "Mismatch between connection matrix (" << no_of_nodes << " nodes) and topology ("
                 << topo_cfg->no_of_nodes() << " nodes)" << endl;
            exit(1);
        }
    } else {
        topo_cfg = make_unique<FatTreeTopologyCfg>(tiers, no_of_nodes, linkspeed, queuesize, hop_latency,
                                                     switch_latency, qt, snd_type);
    }
    top = make_unique<FatTreeTopology>(topo_cfg.get(), qlf, &eventlist, nullptr);

    vector<const Route*>*** net_paths;
    net_paths = new vector<const Route*>**[no_of_nodes];
    for (size_t s = 0; s < no_of_nodes; s++) {
        net_paths[s] = new vector<const Route*>*[no_of_nodes];
        for (size_t d = 0; d < no_of_nodes; d++) {
            net_paths[s][d] = NULL;
        }
    }

    ConnectionMatrix* conns = new ConnectionMatrix(no_of_nodes);
    if (tm_file) {
        cout << "Loading connection matrix from " << tm_file << endl;
        if (!conns->load(tm_file))
            exit(-1);
    } else {
        cout << "Loading connection matrix from standard input" << endl;
        conns->load(cin);
    }

    if (conns->N != no_of_nodes) {
        cout << "Connection matrix number of nodes is " << conns->N << " while I am using " << no_of_nodes
             << endl;
        exit(-1);
    }

    vector<connection*>* all_conns = conns->getAllConnections();
    vector<DetRdmaSrc*> detrdma_srcs;
    map<flowid_t, TriggerTarget*> flowmap;

    for (size_t c = 0; c < all_conns->size(); c++) {
        connection* crt = all_conns->at(c);
        int src = crt->src;
        int dest = crt->dst;

        if (!net_paths[src][dest]) {
            net_paths[src][dest] = top->get_bidir_paths(src, dest, false);
        }
        if (!net_paths[dest][src]) {
            net_paths[dest][src] = top->get_bidir_paths(dest, src, false);
        }

        cout << "Connection " << src << "->" << dest << " starting at " << timeAsUs(crt->start) << " size "
             << crt->size << endl;

        DetRdmaSrc* detrdmaSrc = new DetRdmaSrc(controller, eventlist);
        detrdmaSrc->set_dst(dest);
        detrdmaSrc->make_coarse(cg_bw_bps);
        if (crt->size > 0) {
            detrdmaSrc->set_flowsize(crt->size);
        }
        if (crt->flowid) {
            detrdmaSrc->set_flowid(crt->flowid);
            assert(flowmap.find(crt->flowid) == flowmap.end());
            flowmap[crt->flowid] = detrdmaSrc;
        }
        if (crt->send_done_trigger) {
            Trigger* trig = conns->getTrigger(crt->send_done_trigger, eventlist);
            detrdmaSrc->set_end_trigger(*trig);
        }

        DetRdmaSink* detrdmaSnk = new DetRdmaSink();
        detrdmaSnk->set_src(src);

        // Admission control (CqfController::admit) must run before we can
        // build the actual Route objects connect() needs, since it's the
        // one choosing which of net_paths[src][dest]'s candidates this flow
        // gets to use.
        int idx = controller.admit(detrdmaSrc, *net_paths[src][dest]);
        if (idx < 0) {
            cout << "Flow " << src << "->" << dest
                 << " REJECTED by CqfController (insufficient reservable bandwidth)" << endl;
            delete detrdmaSrc;
            delete detrdmaSnk;
            continue;
        }

        detrdmaSrc->setName("DetRdma_" + ntoa(src) + "_" + ntoa(dest));
        logfile.writeName(*detrdmaSrc);
        detrdmaSnk->setName("DetRdma_sink_" + ntoa(src) + "_" + ntoa(dest));
        logfile.writeName(*detrdmaSnk);

        ((HostQueue*)top->queues_ns_nlp[src][topo_cfg->HOST_POD_SWITCH(src)][0])->addHostSender(detrdmaSrc);

        // Route::add_endpoints is a no-op here (it only mutates _reverse,
        // which get_bidir_paths(..., false) never sets) -- the candidate
        // routes end at the last Pipe, so the destination endpoint has to
        // be appended explicitly or the final Pipe's sendOn() walks off
        // the end of the route.
        Route* routeout = new Route(*(net_paths[src][dest]->at(idx)));
        routeout->push_back(detrdmaSnk);

        Route* routein = new Route(*(net_paths[dest][src]->at(idx)));
        routein->push_back(detrdmaSrc);

        detrdmaSrc->connect(routeout, routein, *detrdmaSnk, crt->start);
        detrdma_srcs.push_back(detrdmaSrc);
    }

    Logged::dump_idmap();
    int pktsize = Packet::data_packet_size();
    logfile.write("# pktsize=" + ntoa(pktsize) + " bytes");
    logfile.write("# hostnicrate = " + ntoa(linkspeed / 1000000) + " Mbps");
    logfile.write("# cqf_cycle_time_us = " + ntoa(cqf_cycle_time_us));
    logfile.write("# cqf_cg_bw_mbps = " + ntoa(cqf_cg_bw_mbps));

    cout << "Starting simulation" << endl;
    while (eventlist.doNextEvent()) {
    }

    cout << "Done" << endl;
    cout << "Summary: Admitted " << detrdma_srcs.size() << " Total " << all_conns->size() << endl;

    return 0;
}
