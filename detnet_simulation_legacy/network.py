import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from collections import defaultdict
import copy
def create_network(link_df,router_dict,na_nodes):
    G = nx.Graph()
    # Adding nodes and edges with length and bandwidth in MBps
    for index, row in link_df.iterrows():
        source=row['a']
        dest=row['z']
        bw=row['speed']
        source_lat=router_dict[source] #getting latitude and longitude of source node
        dest_lat=router_dict[dest] #getting latitude and longitude of destination node
        distance= geodesic(source_lat, dest_lat).kilometers #getting distance between nodes
        if(source > dest):
            source, dest = dest, source
        if(source not in na_nodes and dest not in na_nodes):
            G.add_edge(source, dest, length=distance, bandwidth=bw, inverse_bw=1/bw)
            G.add_edge(dest, source, length=distance, bandwidth=bw, inverse_bw=1/bw) 
    # print("Number of edges in the network: ", len(G.edges()))
    return G 

#  Defining flow class each flow will have source, destination, number of packets, required bandwidth, deadline and type (time-triggered or event-triggered)


class flow:
    def __init__(self, flow_id, source, destination, num_packets, required_bw, flow_type, deadline):
        self.flow_id = copy.deepcopy(flow_id)
        self.source = copy.deepcopy(source)
        self.destination = copy.deepcopy(destination)
        self.num_packets = copy.deepcopy(num_packets)
        # self.path_split = defaultdict(float)
        self.path = []
        self.latency = 0
        self.accepted = False
        self.deadline = copy.deepcopy(deadline)
        self.start_time = 0
        self.end_time = 0 
        self.cycle_packets=0
        self.required_bw = copy.deepcopy(required_bw)
        self.type = copy.deepcopy(flow_type)
        self.num_paths=1
    
    def __str__(self):
        return "Flow id: %s, Source : %s, Dest: %s" % (self.flow_id, self.source, self.destination)


class RDMAFlow:
    """
    Flow for RDMA GPU cluster collective communications.
    type='fine'  : FG flow with deterministic periodic bandwidth profile.
    type='coarse': CG flow with a minimum bandwidth floor reservation.

    CG flows have two sub-modes:
      Fixed   — one-shot or persistent fixed-size transfer (total_bytes set, request_rate=None).
      Inference — random Poisson arrivals with log-normal payload sizes; bandwidth floor is
                  reserved permanently while requests arrive stochastically.
                  Set request_rate > 0 and request_size_mean_kb to enable.
    """
    def __init__(self, flow_id, source, destination, flow_type, total_bytes, deadline,
                 required_bw=0.0, profile=None, period=1,
                 request_rate=None, request_size_mean_kb=None, request_size_std_kb=None,
                 rng_seed=None):
        self.flow_id      = flow_id
        self.source       = source
        self.destination  = destination
        self.type         = flow_type    # "coarse" or "fine"
        self.total_bytes  = total_bytes  # KB — used only in fixed CG mode and FG mode
        self.deadline     = deadline     # per-request latency SLO (same units as cycle_time)

        # CG-specific
        self.required_bw   = required_bw  # floor bandwidth to reserve (Gbps)
        self.assigned_bw   = required_bw  # actual BW after slack sharing (set by controller)
        self.cycle_packets = 0            # packets per cycle (set by adjust_flow_bandwidth)

        # FG-specific
        self.profile = profile if profile is not None else []
        self.period  = period

        # Inference / random-burst mode (CG only)
        # request_rate: Poisson arrival rate (requests/second); None = fixed mode
        # request_size_mean_kb: mean payload per request (KB)
        # request_size_std_kb:  std dev of payload (KB); shapes the log-normal distribution
        self.request_rate          = request_rate
        self.request_size_mean_kb  = request_size_mean_kb
        self.request_size_std_kb   = request_size_std_kb if request_size_std_kb is not None \
                                     else (request_size_mean_kb * 0.5 if request_size_mean_kb else None)
        self.rng_seed              = rng_seed

        # Routing
        self.path = []

        # Per-flow metrics
        self.start_time               = 0.0
        self.completion_time          = None
        self.met_deadline             = None
        self.request_completion_times = []   # inference mode: one entry per request

    def __str__(self):
        return f"RDMAFlow(id={self.flow_id}, {self.source}->{self.destination}, type={self.type})"