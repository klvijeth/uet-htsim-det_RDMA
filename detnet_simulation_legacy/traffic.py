import random
import networkx as nx
from network import *

def get_random_connected_pair(graph, nodes):
    while True:
        src, dst = random.sample(nodes, 2)
        if nx.has_path(graph, src, dst):
            return src, dst

def generate_traffic_episode(network, num_sensor_flows, iterations, max_conc_flows, packet_size, load_factor=0.4):
    #Generating sensor load
    sensor_flows=[]
    intervals=[]
    flow_id=0
    for i in range(num_sensor_flows):
        nodes = list(network.nodes())
        # Choose two random unique nodes
        src, destination = get_random_connected_pair(network, nodes)
        duration = 500
        num_packets = 0
        required_bandwidth = random.randint(3000, 5000)
        sensor_flows.append((flow_id, src, destination, num_packets, required_bandwidth, "sensor", duration))
        flow_id += 1
    #Generating data load
    all_data_flows=[]
    # load_packets = (load_factor * 1000000000)/4
    load_packets = (load_factor * 100000000)
    for i in range(iterations):
        n=0
        conc_data_flows=[]
        for j in range(max_conc_flows):
            nodes = list(network.nodes())
            src, destination = get_random_connected_pair(network, nodes)
            # num_packets = random.randint(60000000,80000000)
            num_packets = random.randint(int(load_packets*0.9),int(load_packets*1.1))
            required_bandwidth = random.randint(7000 , 9000)
            deadline=(num_packets*packet_size)/(required_bandwidth*1000)
            conc_data_flows.append((flow_id, src,  destination, num_packets, required_bandwidth, "data", deadline))
            n+=1
            flow_id+=1
        all_data_flows.append(conc_data_flows)
        intervals.append(random.randint(9, 11))
    return sensor_flows, all_data_flows, intervals


def create_flows(traffic_episodes):
    cqf_episodes=[]
    oscars_episodes=[]
    fixedBW_episodes=[]
    for episode in traffic_episodes:
        all_sensor_flows_CQF=[]
        all_data_flows_CQF=[]
        all_sensor_flows_OSCARS=[]
        all_data_flows_OSCARS=[]
        all_sensor_flows_fixed_bw=[]
        all_data_flows_fixed_bw=[]
        sensor_flows, all_data_flows, intervals=episode
        for sensor_flow in sensor_flows:
            all_sensor_flows_CQF.append(flow(*sensor_flow))
            all_sensor_flows_OSCARS.append(flow(*sensor_flow))
            all_sensor_flows_fixed_bw.append(flow(*sensor_flow))

        for conc_flows in all_data_flows:
            conc_flows_CQF=[]
            conc_flows_OSCARS=[]
            conc_flows_fixed_bw=[]
            for data_flow in conc_flows:
                conc_flows_CQF.append(flow(*data_flow))
                conc_flows_OSCARS.append(flow(*data_flow))
                conc_flows_fixed_bw.append(flow(*data_flow))
            all_data_flows_CQF.append(conc_flows_CQF)
            all_data_flows_OSCARS.append(conc_flows_OSCARS)
            all_data_flows_fixed_bw.append(conc_flows_fixed_bw)
        cqf_episodes.append([all_sensor_flows_CQF,all_data_flows_CQF,intervals])
        oscars_episodes.append([all_sensor_flows_OSCARS,all_data_flows_OSCARS,intervals])
        fixedBW_episodes.append([all_sensor_flows_fixed_bw,all_data_flows_fixed_bw,intervals])
    return cqf_episodes, oscars_episodes, fixedBW_episodes