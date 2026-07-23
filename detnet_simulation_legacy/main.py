import pandas as pd
import simpy
import random
import matplotlib.pyplot as plt

from network import *
from cqf_sim import *
# from old_oscars_sim import *
from oscars_sim import *
from traffic import *

link_df = pd.read_csv('ESnet-BackBoneLinks.csv')
router_df = pd.read_csv('ESnet-core-routers.csv')
router_dict = {row['device']: {(row['latitude'], row['longitude'])} for _, row in router_df.iterrows()}

len(link_df[link_df['speed']>10000])
link_df=link_df[link_df['speed']>10000]


na_nodes=[]
network = create_network(link_df,router_dict,na_nodes) #Creating the network

closeness_dict = nx.closeness_centrality(network,distance='length') #Calculating closeness centrality for each node in the network

# 3. Find the node with the highest score
best_node = max(closeness_dict, key=closeness_dict.get)
highest_score = closeness_dict[best_node]

print(f"Optimal Central Controller: Node {best_node}")
print(f"Closeness Score: {highest_score:.4f}")

# #Defining the workload
# iterations=300
# max_conc_flows=30
# flow_id=0
# num_sensor_flows=15
# packet_size=9

# edge_centrality = nx.edge_betweenness_centrality(network)
# edge_centrality=dict(sorted(edge_centrality.items(), key=lambda item: item[1], reverse=True))
# sensor_flows, all_data_flows, intervals = generate_traffic_episode(network, num_sensor_flows, iterations, max_conc_flows, packet_size)

# import copy
# copy_data_flows=copy.deepcopy(all_data_flows)
# total_packets=0
# for iteration in copy_data_flows:
#     for data_flow in list(iteration):
#         total_packets+=data_flow[3]
#         dl=data_flow[-1]

# total_data=(total_packets*9)/(1000*1000*1000*1000)
# print("Total data in Pb",total_data)

# cycle_time=0.06
# k = 3  # Number of shortest paths to use for multipath routing


# cqf_controller = CQFCentralController(network, cycle_time, packet_size, k)
# OSCARS_controller = OscCentralController(network, packet_size, k,  0.05)


# env_OSCARS = simpy.Environment()
# env_cqf = simpy.Environment()

# all_sensor_flows_CQF=[]
# all_data_flows_CQF=[]
# all_sensor_flows_OSCARS=[]
# all_data_flows_OSCARS=[]


# for sensor_flow in sensor_flows:
#     all_sensor_flows_CQF.append(flow(*sensor_flow))
#     all_sensor_flows_OSCARS.append(flow(*sensor_flow))

# for conc_flows in all_data_flows:
#     conc_flows_CQF=[]
#     conc_flows_OSCARS=[]
#     for data_flow in conc_flows:
#         conc_flows_CQF.append(flow(*data_flow))
#         conc_flows_OSCARS.append(flow(*data_flow))
#     all_data_flows_CQF.append(conc_flows_CQF)
#     all_data_flows_OSCARS.append(conc_flows_OSCARS)

# env_cqf.process(new_traffic_source_cqf(env_cqf, intervals, cqf_controller, all_sensor_flows_CQF, all_data_flows_CQF))
# env_OSCARS.process(new_traffic_source_reservation(env_OSCARS, intervals, OSCARS_controller, all_sensor_flows_OSCARS, all_data_flows_OSCARS))

# # Run simulation
# env_cqf.run()
# env_OSCARS.run()
# print("*************************Displaying metrics for CQF*************************")
# print(f"Total Accepted flows={cqf_controller.accepted_flows}")
# print(f"Total rejected flows={cqf_controller.rejected_flows}")
# print(f"Average latency per flow = {round(sum(cqf_controller.latency_list)/cqf_controller.accepted_flows_data,5)}")
# print("Maximum latency is ",max(cqf_controller.latency_list))

# print("*************************Displaying metrics for OSCARS*************************")
# print(f"Total Accepted flows={OSCARS_controller.accepted_flows}")
# print(f"Total rejected flows={OSCARS_controller.rejected_flows}")
# print(f"Average latency per flow = {round(sum(OSCARS_controller.latency_list)/OSCARS_controller.accepted_flows_data,5)}")
# print("Maximum latency is ",max(OSCARS_controller.latency_list))


# avg_bw_util_list=[]
# edge_list=[]
# bw_centrality_list=[]
# max_bw_list=[]
# for edge, centrality in edge_centrality.items():
#     if(edge in cqf_controller.link_bws):
#         avg_bw=sum(cqf_controller.link_bws[edge].values())/len(cqf_controller.link_bws[edge].values())
#         edge_list.append(edge)
#         bw_centrality_list.append(centrality)
#         max_bw_list.append(cqf_controller.network[edge[0]][edge[1]]['bandwidth'])
#         avg_bw_util_list.append(round((avg_bw/cqf_controller.network[edge[0]][edge[1]]['bandwidth']),2))
# #         avg_bw_util_list.append(round((avg_bw),2))
#         # print(f"Edge {edge}: Betweenness Centrality: {centrality:.4f}", " Average bw=", round(avg_bw,4), "Total bw=", cqf_controller.network[edge[0]][edge[1]]['bandwidth'], "% utilization", round((avg_bw/cqf_controller.network[edge[0]][edge[1]]['bandwidth'])*100,2))
#         # print(f"Edge {edge}: Betweenness Centrality: {centrality:.4f}", "average % utilization", round((avg_bw/cqf_controller.network[edge[0]][edge[1]]['bandwidth'])*100,2))

# plt.figure(figsize=(10, 6))
# # plt.scatter(bw_centrality_list, avg_bw_util_list, color='blue', alpha=0.5)
# plt.scatter(max_bw_list, avg_bw_util_list, color='blue', alpha=0.5)

# # Annotate each point with the corresponding edge
# for i, edge in enumerate(edge_list):
#     # plt.annotate(f"{edge}", (bw_centrality_list[i], avg_bw_util_list[i]), fontsize=8, alpha=0.7)
#     plt.annotate(f"{edge}", (max_bw_list[i], avg_bw_util_list[i]), fontsize=8, alpha=0.7)

# plt.title('Average Bandwidth Utilization vs Maximum Available bandwidth')
# # plt.xlabel('Betweenness Centrality')
# plt.xlabel('Max Bandwidth')
# plt.ylabel('Average Bandwidth Utilization')
# plt.grid()
# plt.show()
# plt.savefig('avg_bw_util_vs_centrality.png')


# plt.figure(figsize=(10, 6))
# plt.scatter(bw_centrality_list, avg_bw_util_list, color='blue', alpha=0.5)

# # Annotate each point with the corresponding edge
# for i, edge in enumerate(edge_list):
#     plt.annotate(f"{edge}", (bw_centrality_list[i], avg_bw_util_list[i]), fontsize=8, alpha=0.7)

# plt.title('Average Bandwidth Utilization vs Betweenness Centrality')
# plt.xlabel('Betweenness Centrality')
# plt.ylabel('Average Bandwidth Utilization')
# plt.grid()
# plt.show()
# plt.savefig('avg_bw_util_vs_centrality.png')


# import statistics
# mean = statistics.mean(avg_bw_util_list)
# print(f"Mean: {mean}")

# # Calculate the variance
# variance = statistics.variance(avg_bw_util_list)
# print(f"Variance: {variance}")


# # Step 1: Calculate average normalized bandwidth for each link
# link_avg_bandwidths = {}

# for (src, dst), bw_dict in cqf_controller.link_bws.items():
#     bw_values = list(bw_dict.values())
#     total_bw = cqf_controller.network[src][dst]['bandwidth']
#     normalized_values = [bw / total_bw for bw in bw_values]
# #     avg_normalized_bw = np.mean(normalized_values)
#     avg_normalized_bw = sum(normalized_values)/len(normalized_values)
#     link_avg_bandwidths[(src, dst)] = (avg_normalized_bw, normalized_values, list(bw_dict.keys()))

# # Step 2: Select top 10 links by average
# top_10_links = sorted(link_avg_bandwidths.items(), key=lambda x: x[1][0], reverse=True)[:10]

# # Step 3: Plot
# plt.figure(figsize=(12, 8))

# for (src, dst), (avg_bw, normalized_values, times) in top_10_links:
#     label = f"{src} → {dst} (avg: {avg_bw:.2f})"
#     plt.plot(times, normalized_values, label=label)

# plt.xlabel('Date')
# plt.ylabel('Normalized Bandwidth Usage')
# plt.title('Top 10 Links by Average Normalized Bandwidth')
# plt.legend(loc='best')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()