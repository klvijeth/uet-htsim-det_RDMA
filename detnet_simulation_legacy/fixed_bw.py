#oscars code
import networkx as nx
from collections import defaultdict
import heapq
from tqdm import tqdm
import simpy
# Define traffic source for bandwidth reservation

class FixedBwController:
    def __init__(self, network, packet_size, max_paths=1, batch_time=1,env=None):
        self.network = network
        self.packet_size=packet_size
        self.batch_time=batch_time
        self.max_paths=max_paths
        self.env          = env
        
        #flow tracking
        self.link_sensor_flows=defaultdict(list)
        self.link_data_flows=defaultdict(list)
        self.link_bws=defaultdict(dict)

        #metrics
        self.accepted_flows=0
        self.rejected_flows=0
        self.latency_list=[]
        self.accepted_flows_data=0
        self.rejected_flows_data=0
        self._reserve_lock = simpy.Resource(self.env, capacity=1)
        self.network_utilization=defaultdict(float)
    
    def set_progress_bar(self, total):
        self.progress_bar = tqdm(total=total, desc="Completed Flows")

    def compute_link_bandwidths(self, link):
        sensor_bandwidth = sum(s.required_bw / s.num_paths for s in self.link_sensor_flows[link])
        data_bandwidth = sum((d.num_packets / d.num_paths) * self.packet_size / (d.deadline * 1000) for d in self.link_data_flows[link])
        return sensor_bandwidth, data_bandwidth

    def reserve_bandwidth(self, path, flow):
        if(flow.type=="sensor"):
            for i in range(len(path) - 1):
                link = tuple(sorted((path[i], path[i + 1])))
                self.link_sensor_flows[link].append(flow)
        else:
            for i in range(len(path) - 1):
                link = tuple(sorted((path[i], path[i + 1])))
                self.link_data_flows[link].append(flow)
    
    def release_bandwidth(self,path,flow):
        for i in range(len(path) - 1):
            link = tuple(sorted((path[i], path[i + 1])))
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
    
    def set_utilized_bandwidth(self, time):
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


    def new_traffic_fixed_bw_reservation(self, env, fbw_episodes):
    # 1) Fire off all sensor flows at t=0
        for episode in fbw_episodes:
            sensor_flows, data_batches, intervals = episode
            for flow in sensor_flows:
                env.process(self.route_flow_fixed_bw_reservation(env, flow))

            # 2) Schedule each data‐flow batch at the right interval
            for delay, batch in zip(intervals, data_batches):
                yield env.timeout(delay)
                for flow in batch:
                    env.process(self.route_flow_fixed_bw_reservation(env, flow))

            yield env.timeout(200)  # Allow the environment to process events
            
    def calc_bandwidth_path(self, path):
        min_bw=100000000
        for i in range(len(path) - 1):
            link = tuple(sorted((path[i], path[i + 1])))
            total_bandwidth = self.network[link[0]][link[1]]['bandwidth']
            sensor_bandwidth=0
            for sensor in self.link_sensor_flows[link]:
                sensor_bandwidth+=(sensor.required_bw/sensor.num_paths)
            min_data_bw=0
            for data_flow in self.link_data_flows[link]:
                min_data_bw+=(data_flow.required_bw/data_flow.num_paths)
            avail_bw=total_bandwidth-sensor_bandwidth-min_data_bw
            if(avail_bw<min_bw):
                min_bw=avail_bw
        return min_bw

    def calc_bandwidth_link(self, link):
        total_bandwidth = self.network[link[0]][link[1]]['bandwidth']
        sensor_bandwidth=0
        for sensor in self.link_sensor_flows[link]:
            sensor_bandwidth+=(sensor.required_bw/sensor.num_paths)
        min_data_bw=0
        for data_flow in self.link_data_flows[link]:
            min_data_bw+=(data_flow.required_bw/data_flow.num_paths)
        avail_bw=total_bandwidth-sensor_bandwidth-min_data_bw
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
                cap_uv = self.calc_bandwidth_link(tuple(sorted((u,v))))
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

    def get_k_dynamic_wide_paths(self, flow,time):
        src, dst = flow.source, flow.destination
        k        = self.max_paths
        per_req  = flow.required_bw / k

        # create a copy of the network to avoid modifying the original
        G = self.network.copy()
        selected = []
        path_bws=[]
        for _ in range(k):
            path, bw = self.widest_path_dynamic(G, src, dst)
            if not path or bw < per_req:
                break
            selected.append(path)
            path_bws.append(bw)
            self.reserve_bandwidth(path, flow)
            # *Remove* every edge on this path so it can’t be chosen again:
            G.remove_edges_from(zip(path, path[1:]))

        num_paths=len(selected)
        if(num_paths==0):
            # If no path is found with sufficient bandwidth
            return []
        while(path_bws[-1]<(flow.required_bw/num_paths)):
            self.release_bandwidth(selected[-1], flow)
            selected.pop()
            path_bws.pop()
            num_paths-=1
            if(num_paths==0):
                break

        self.set_utilized_bandwidth(time)
        return selected


    # Define routing function for bandwidth reservation
    def route_flow_fixed_bw_reservation(self, env, flow):
        start_time = env.now
        with self._reserve_lock.request() as req:
            yield req
            k_paths = self.get_k_dynamic_wide_paths(flow,int(env.now))
        
        # If no path is found with sufficient bandwidth
        if(len(k_paths)==0):
            self.rejected_flows+=1
            return
        flow.num_paths=len(k_paths)
        self.accepted_flows += 1
        if(flow.type=="sensor"):
            yield env.timeout(flow.deadline)
            for path in k_paths:
                self.release_bandwidth(path, flow)
            return
        else:
            
            #Account for propagation delay
            # Propagation delay is calculated as the sum of the lengths of the links in the path divided by the speed of light in fiber (200,000 km/s)
            propogation_delay=0
            for path in k_paths:
                for i in range(len(path) - 1):
                    link = tuple(sorted((path[i], path[i + 1])))
                    propogation_delay+=self.network[link[0]][link[1]]['length']/200000
            yield env.timeout(propogation_delay)

            n=0
            while(flow.num_packets>0):
                batch_packets=0
                for path in k_paths:
                    # Calculate bandwidth only once per batch
                    bandwidth_avail = flow.required_bw/flow.num_paths
                    batch_packets+=int(( self.batch_time*bandwidth_avail * 1000)/self.packet_size)
                    # self.set_utilized_bandwidth(int(env.now))
                n+=1
                yield env.timeout(self.batch_time)
                flow.num_packets -= batch_packets
                flow.deadline -= self.batch_time
            end_time = env.now
            # Release bandwidth after transmission
            if(flow.deadline<=0 and abs(flow.deadline)>2):
                print("Fixed BW Deadline exceeded************ by",abs(flow.deadline))
            for path in k_paths:
                self.release_bandwidth(path, flow)
            total_delay = round(end_time - start_time, 5)
            total_delay = round((end_time - start_time),5)
            self.latency_list.append(total_delay)
            self.accepted_flows_data+=1
            if self.progress_bar:
                self.progress_bar.update(1)