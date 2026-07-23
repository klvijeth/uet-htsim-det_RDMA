#CQF code
import networkx as nx
from collections import defaultdict
import heapq
from tqdm import tqdm
import simpy


"""
Class: CQF controller
Implements the CQF Central Controller for bandwidth reservation and flow management.
"""
class CQFCentralController:
    # Initialization of the CQF controller
    def __init__(self, network, cycle_time, packet_size, max_paths=1, env=None):
        self.network      = network # NetworkX graph representing the network
        self.packet_size  = packet_size # Size of each packet in kilobytes
        self.cycle_time   = cycle_time # Duration of each CQF cycle in seconds
        self.max_paths    = max_paths # Maximum number of paths for multipath routing
        self.env          = env

        self.completed_flows = 0 # Counter for completed flows
        self.total_flows = 0  # Total number of flows
        self.progress_bar = None

        #flow tracking
        self.link_sensor_flows=defaultdict(list) #Sensor flows on each link
        self.link_data_flows=defaultdict(list) #Data flows on each link
        
        #metrics
        self.accepted_flows=0 # Counter for accepted flows
        self.rejected_flows=0 # Counter for rejected flows
        self.latency_list=[] # List to store latency of each accepted flow
        self.accepted_flows_data=0 # Counter for accepted data flows
        self.rejected_flows_data=0 # Counter for rejected data flows
        self.total_frer_flows=0 # Total number of FRER flows
        self.link_bws=defaultdict(dict) # Bandwidth utilization on each link at different timestamp
        self.network_utilization=defaultdict(float) # Network utilization metrics
        
        # Performance Metrics
        self.slo_met = 0
        self.slo_missed = 0
        self.total_bytes_delivered = 0


        self._reserve_lock = simpy.Resource(self.env, capacity=1)

    def set_progress_bar(self, total):
        self.progress_bar = tqdm(total=total, desc="Completed Flows")
    
    # normalize link representation (a,b) and (b,a) should be the same
    def normalize_link(self, u, v):
        return (min(u, v), max(u, v))
    
    # Compute the bandwidths used by sensor and data flows on a given link
    # def compute_link_bandwidths(self, link):
    #     sensor_bandwidth = sum(s.required_bw / s.num_paths for s in self.link_sensor_flows[link])
    #     data_bandwidth = sum((d.num_packets / d.num_paths) * self.packet_size / (d.deadline * 1000) for d in self.link_data_flows[link])
    #     return sensor_bandwidth, data_bandwidth
    

    # Compute the bandwidths used by sensor and data flows on a given link
    def compute_link_bandwidths(self, link):
        sensor_bandwidth = sum(s.required_bw for s in self.link_sensor_flows[link])
        data_bandwidth = sum((d.num_packets) * self.packet_size / (d.deadline * 1000) for d in self.link_data_flows[link])
        return sensor_bandwidth, data_bandwidth

    # Reserve bandwidth for a flow on its path
    def reserve_bandwidth(self, path, flow):
        if(flow.type=="sensor"):
            for i in range(len(path) - 1):
                link = self.normalize_link(path[i], path[i + 1])
                self.link_sensor_flows[link].append(flow) #sensor flow reservation
        else:
            for i in range(len(path) - 1):
                link = self.normalize_link(path[i], path[i + 1])
                self.link_data_flows[link].append(flow) #data flow reservation

    # Release bandwidth for a flow on its path after completion
    def release_bandwidth(self,path,flow):
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            if(flow.type=="sensor"):
                if(flow not in self.link_sensor_flows[link]):
                    print("Flow not found")
                    continue
                ind=self.link_sensor_flows[link].index(flow)
                self.link_sensor_flows[link].pop(ind)
            else:
                if(flow not in self.link_data_flows[link]):
                    print("Flow not found")
                    continue
                ind=self.link_data_flows[link].index(flow)
                self.link_data_flows[link].pop(ind)

    # Calculate number of packets that can be sent in the current cycle on a given path for a flow
    def send_window(self, path,flow):
        min_packets=100000000000
        propagation_speed = 200000  # speed of light in fiber (km/s)
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            sensor_bandwidth, min_data_bw = self.compute_link_bandwidths(link)
            avail_bw=self.network[link[0]][link[1]]['bandwidth']-(sensor_bandwidth+min_data_bw)
            distance=self.network[link[0]][link[1]]['length']
            packet_delay=(self.packet_size/(avail_bw*1000))
            propagation_delay = distance / propagation_speed
            link_packets=(self.cycle_time-propagation_delay)//packet_delay
            # link_packets=(self.cycle_time)//packet_delay
            flow_packets=int((((flow.num_packets / flow.num_paths)/flow.deadline)*self.cycle_time)+(link_packets/len(self.link_data_flows[link])))
            if(flow_packets<min_packets):
                min_packets=flow_packets

        return min_packets
    
    # Function to update bandwidth and network utilization metrics at each time step
    def set_bandwidth_utilization(self, time):
        total_utilization=0
        used_links=0
        for link in self.network.edges():
            if(link not in self.link_sensor_flows and link not in self.link_data_flows):
                continue
            sensor_bandwidth, min_data_bw = self.compute_link_bandwidths(link)
            link_utilization=(sensor_bandwidth+min_data_bw)/self.network[link[0]][link[1]]['bandwidth']
            self.link_bws[link][time]=sensor_bandwidth+min_data_bw
            total_utilization+=link_utilization
            used_links+=1
        if(used_links>0):
            self.network_utilization[time]=total_utilization/used_links
        else:
            self.network_utilization[time]=0

    # Start traffic episodes with sensor and data flows
    def new_traffic_source_cqf(self, env, cqf_episodes, frer_flows=0):
        # 1) Fire off all sensor flows at t=0
        for episode in cqf_episodes:
            sensor_flows, data_batches, intervals = episode
            for flow in sensor_flows:
                env.process(self.route_flow_cqf(env, flow))

            # 2) Schedule each data‐flow batch at the right interval
            for delay, batch in zip(intervals, data_batches):
                yield env.timeout(delay)
                if(frer_flows==0):
                    for flow in batch:
                        env.process(self.route_flow_cqf(env, flow))
                else:
                    i=0
                    frer_num=int(len(batch)*frer_flows)
                    frer_batch=batch[:frer_num]
                    non_frer_batch=batch[frer_num:]
                    # Process FRER flows first
                    for flow in frer_batch:
                        env.process(self.route_flow_cqf_frame_replication(env, flow))
                    # Process non-FRER flows
                    for flow in non_frer_batch:
                        env.process(self.route_flow_cqf(env, flow))

            yield env.timeout(200)  # Allow the environment to process events

    # Calculate the total bandwidth available on a path using (Total bandwidth - bandwidth required by current flows)
    def calc_path_bandwidth_cqf(self, path):
        min_packets=1000000000
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            sensor_bandwidth, min_data_bw = self.compute_link_bandwidths(link)
            avail_bw=self.network[link[0]][link[1]]['bandwidth']-sensor_bandwidth-min_data_bw
            distance=self.network[link[0]][link[1]]['length']
            packet_delay=(self.packet_size/(avail_bw*1000))
            propagation_speed = 200000  # speed of light in fiber (km/s)
            propagation_delay = distance / propagation_speed
            link_packets=(self.cycle_time-propagation_delay)//packet_delay
            # link_packets=(self.cycle_time)//packet_delay
            if(link_packets<min_packets):
                min_packets=link_packets
        min_bw=(min_packets*self.packet_size)/(self.cycle_time*1000)
        return min_bw

    def check_propagation_delay(self, path ):
        for i in range(len(path) - 1):
            link = self.normalize_link(path[i], path[i + 1])
            distance = self.network[link[0]][link[1]]['length']
            if distance >= 1200:
                # print("Propagation delay:", propagation_delay, "is greater than cycle time:", self.cycle_time)
                return False
        return True

    def calc_link_bandwidth_cqf(self, link):
        """
        Calculates the available bandwidth on a link (u, v) by checking the
        bandwidth of all flows using that link.
        """
        distance=self.network[link[0]][link[1]]['length']
        propagation_speed = 200000  # speed of light in fiber (km/s)
        propagation_delay = distance / propagation_speed
        if(propagation_delay>=self.cycle_time):
            # print("Propagation delay:", propagation_delay, "is greater than cycle time:", self.cycle_time)
            return -1
        sensor_bandwidth, min_data_bw = self.compute_link_bandwidths(link)
        avail_bw=self.network[link[0]][link[1]]['bandwidth']-sensor_bandwidth-min_data_bw
        packet_delay=(self.packet_size/(avail_bw*1000))
        link_packets=(self.cycle_time-propagation_delay)//packet_delay
        # link_packets=(self.cycle_time)//packet_delay
        avail_bw=(link_packets*self.packet_size)/(self.cycle_time*1000)
        return avail_bw


    def widest_path_dynamic(self, G, src, dst):
        """
        Finds a path from src to dst maximizing the minimum *dynamic* bandwidth
        along the path, by calling calc_bandwidth_path on each edge [u, v].
        Returns (path, bottleneck_capacity), or ([], 0) if no path exists.
        """
        # best possible min‑edge so far to reach each node
        bottleneck = {n: 0.0 for n in G.nodes()}
        bottleneck[src] = float('inf')
        parent     = {}
        seen       = set()
        # max‑heap ordered by current bottleneck
        heap = [(-bottleneck[src], src)]

        while heap:
            neg_cap, u = heapq.heappop(heap)
            cap_u = -neg_cap
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                break

            # explore neighbors dynamically
            for v in G.neighbors(u):
                # getting available bandwidth of the edge
                cap_uv = self.calc_link_bandwidth_cqf(tuple(sorted((u,v))))
                if cap_uv < 0:
                    continue
                path_cap = min(cap_u, cap_uv)
                if path_cap > bottleneck[v]:
                    bottleneck[v] = path_cap
                    parent[v]     = u
                    heapq.heappush(heap, (-path_cap, v))

        # if dst wasn’t reached
        if dst not in seen:
            return [], 0.0

        # reconstruct the path
        path = []
        cur  = dst
        while cur != src:
            path.append(cur)
            cur = parent[cur]
        path.append(src)
        path.reverse()

        return path, bottleneck[dst]

    def get_k_dynamic_wide_paths(self, flow, time, k=1):
        src, dst = flow.source, flow.destination
        # k        = self.max_paths
        per_req  = flow.required_bw

        # create a copy of the network to avoid modifying the original
        G = self.network.copy()
        selected = []
        path_bws=[]

        # bw_perc=(self.cycle_time-0.032)/self.cycle_time
        bw_perc=1
        for _ in range(k):
            path, bw = self.widest_path_dynamic(G, src, dst)
            if not path or ((bw_perc*bw) < per_req):
                break
            # if(self.check_propagation_delay(path) is False):
            #     print("Propagation delay exceeds cycle time for path:", path)
            #     break
            selected.append(path)
            path_bws.append(bw)
            self.reserve_bandwidth(path, flow)
            # *Remove* every edge on this path so it can’t be chosen again:
            G.remove_edges_from(zip(path, path[1:]))

        num_paths=len(selected)
        if(num_paths==0):
            # If no path is found with sufficient bandwidth
            return []
        # while((bw_perc*path_bws[-1])<(flow.required_bw/num_paths)):
        #     self.release_bandwidth(selected[-1], flow)
        #     selected.pop()
        #     path_bws.pop()
        #     num_paths-=1
        #     if(num_paths==0):
        #         break
        self.set_bandwidth_utilization(time)
        return selected

    # Define routing function for bandwidth reservation
    def route_flow_cqf(self, env, flow):
        # print("CQF Routing flow", flow.flow_id, "from", flow.source, "to", flow.destination, "with required bandwidth", flow.required_bw)
        start_time = env.now
        with self._reserve_lock.request() as req:
            yield req
            k_paths=self.get_k_dynamic_wide_paths(flow,int(env.now))
        
        if len(k_paths) == 0:
            self.rejected_flows+=1
            if self.progress_bar:
                self.progress_bar.update(1)
            return

        self.accepted_flows+=1
        flow.num_paths=1
        
        original_num_packets = flow.num_packets
        initial_deadline = flow.deadline
        
        if(flow.type=="sensor"):
            yield env.timeout(flow.deadline)
            
            for path in k_paths:
                self.release_bandwidth(path, flow)
            
            self.slo_met += 1 # Sensors generally always meet SLO if accepted
            self.completed_flows += 1
            if self.progress_bar:
                self.progress_bar.update(1)
        else:
            self.accepted_flows_data+=1
            max_path_length = max(len(p) for p in k_paths)
            yield env.timeout((max_path_length-2)*self.cycle_time)
            n=1
            packets={}
            while(flow.num_packets>=0):
                cycle_packets=0
                path_id=0
                if(n>=1):
                    for path in k_paths:
                        packets[path_id]=self.send_window(path,flow)
                        # self.set_bandwidth_utilization(int(env.now))
                        path_id+=1
                    n=0
                    path_id=0
                cycle_packets=sum(packets.values())
                
                if(cycle_packets<=0):
                    print("No packets sent in cycle", env.now, "Flow ID:", flow.flow_id)
                    break
                
                yield env.timeout(self.cycle_time)
                # if(cycle_packets>flow.num_packets):
                #     cycle_packets=flow.num_packets
                flow.num_packets -= cycle_packets
                flow.deadline-=self.cycle_time
                if(flow.deadline<=(-1)):
                    print("CQF Deadline exceeded************ by",abs(flow.deadline), "Number of left",flow.num_packets, "packets", int(cycle_packets), "initial deadline", initial_deadline)
                    break
                n+=self.cycle_time
            
            end_time = env.now
            # if(flow.deadline<=0 and abs(flow.deadline)>self.cycle_time):
            #     print("CQF Deadline exceeded************ by",abs(flow.deadline), "Number of packets sent",flow.num_packets, "packets", int(cycle_packets))
            # Release bandwidth after transmission
            for path in k_paths:
                self.release_bandwidth(path, flow)
            total_delay = round((end_time - start_time),5)
            self.latency_list.append(total_delay)
            
            self.total_bytes_delivered += (original_num_packets * self.packet_size * 1000) # Assuming packet_size is KB
            
            # Calculate SLO Attainment
            if total_delay <= initial_deadline:
                self.slo_met += 1
            else:
                self.slo_missed += 1
                
            self.completed_flows += 1
            if self.progress_bar:
                self.progress_bar.update(1)


    def route_flow_cqf_frame_replication(self, env, flow):
        # print("CQF Routing flow", flow.flow_id, "from", flow.source, "to", flow.destination, "with required bandwidth", flow.required_bw)
        start_time = env.now
        with self._reserve_lock.request() as req:
            yield req
            k_paths=self.get_k_dynamic_wide_paths(flow,int(env.now),2)
        
        if len(k_paths) == 0:
            self.rejected_flows+=1
            if self.progress_bar:
                self.progress_bar.update(1)
            return

        self.accepted_flows+=1
        flow.num_paths=1
        
        original_num_packets = flow.num_packets
        initial_deadline = flow.deadline

        if(len(k_paths)==2):
            self.total_frer_flows+=1
        if(flow.type=="sensor"):
            yield env.timeout(flow.deadline)
            
            for path in k_paths:
                self.release_bandwidth(path, flow)
            
            self.slo_met += 1
            self.completed_flows += 1
            if self.progress_bar:
                self.progress_bar.update(1)
        else:
            self.accepted_flows_data+=1
            max_path_length = max(len(p) for p in k_paths)
            yield env.timeout((max_path_length-2)*self.cycle_time)
            n=1
            packets={}
            while(flow.num_packets>=0):
                cycle_packets=0
                path_id=0
                if(n>=1):
                    for path in k_paths:
                        packets[path_id]=self.send_window(path,flow)
                        # self.set_bandwidth_utilization(int(env.now))
                        path_id+=1
                    n=0
                    path_id=0
                cycle_packets=max(packets.values())
                
                if(cycle_packets<=0):
                    print("CQF No packets sent in cycle", env.now, "Flow ID:", flow.flow_id)
                    break
                
                yield env.timeout(self.cycle_time)
                # if(cycle_packets>flow.num_packets):
                #     cycle_packets=flow.num_packets
                flow.num_packets -= cycle_packets
                flow.deadline-=self.cycle_time
                if(flow.deadline<=min(-1,(-1 * self.cycle_time))):
                    print("CQF Deadline exceeded************ by",abs(flow.deadline), "Number of left",flow.num_packets, "packets", int(cycle_packets), "initial deadline", initial_deadline)
                    break
                n+=self.cycle_time
            
            end_time = env.now
            # if(flow.deadline<=0 and abs(flow.deadline)>self.cycle_time):
            #     print("CQF Deadline exceeded************ by",abs(flow.deadline), "Number of packets sent",flow.num_packets, "packets", int(cycle_packets))
            # Release bandwidth after transmission
            for path in k_paths:
                self.release_bandwidth(path, flow)
            total_delay = round((end_time - start_time),5)
            self.latency_list.append(total_delay)
            
            self.total_bytes_delivered += (original_num_packets * self.packet_size * 1000)
            
            if total_delay <= initial_deadline:
                self.slo_met += 1
            else:
                self.slo_missed += 1
                
            self.completed_flows += 1
            if self.progress_bar:
                self.progress_bar.update(1)
                
    def get_performance_metrics(self, simulation_time):
        """Returns a dictionary of key performance metrics."""
        avg_latency = sum(self.latency_list) / len(self.latency_list) if self.latency_list else 0
        # Convert bytes to Gbps: (bytes * 8 bits / 1e9) / time
        throughput_gbps = (self.total_bytes_delivered * 8 / 1e9) / simulation_time if simulation_time > 0 else 0
        slo_attainment_rate = (self.slo_met / self.total_flows) * 100 if self.total_flows > 0 else 0
        
        return {
            "Total Flows": self.total_flows,
            "Accepted Flows": self.accepted_flows,
            "Rejected Flows": self.rejected_flows,
            "SLO Attainment (%)": slo_attainment_rate,
            "Average Latency (s)": avg_latency,
            "Throughput (Gbps)": throughput_gbps
        }