import random
import networkx as nx

def get_random_connected_pair(graph, nodes):
    """
    Pick two distinct nodes by random sampling until there’s a path between them.
    """
    while True:
        src, dst = random.sample(nodes, 2)
        if nx.has_path(graph, src, dst):
            return src, dst

def generate_traffic_episode(
    network: nx.Graph,
    num_sensor_flows: int,
    iterations: int,
    max_conc_flows: int,
    packet_size: float,
    load_factor: float = 0.2 ,      # ρ ∈ [0,1], fraction of peak capacity to use
    mix: float = 0.2,              # α ∈ [0,1], fraction of load for sensors vs data
    sensor_bw_range=(3000, 5000),  # in Mbps
    data_bw_range=(3000, 5000),    # in Mbps
    sensor_flow_duration=900,      # mean seconds for sensor flows
    sensor_jitter=50,              # stddev for sensor duration
    interval_range=(7, 10)         # uniform seconds between data batches
):
    """
    Each episode has:
      - a fixed num_sensor_flows (scaled to C_sensor budget)
      - 'iterations' batches of data flows, each with
        dynamic_n = int(load_factor * max_conc_flows) flows (clamped [0, max_conc_flows])
    
    Returns:
      sensor_flows: List of tuples
        (flow_id, src, dst, num_packets, bandwidth_mbps, 'sensor', duration_s)
      all_data_flows: List (len=iterations) of lists of tuples
        (flow_id, src, dst, num_packets, bandwidth_mbps, 'data', deadline_s)
      intervals: List of inter-batch intervals in seconds
    """
    # Convert 100 Gbps → 100,000 Mbps
    network_capacity_mbps = 100_000
    C_sensor = load_factor * mix * network_capacity_mbps
    C_data   = load_factor * (1 - mix) * network_capacity_mbps

    nodes = list(network.nodes())
    flow_id = 0

    # 1) SENSOR FLOWS: fixed count, but scaled to fill C_sensor
    raw_sensor_rates = [random.uniform(*sensor_bw_range) for _ in range(num_sensor_flows)]
    total_raw_s = sum(raw_sensor_rates)
    scale_s = (C_sensor / total_raw_s) if total_raw_s > 0 else 0
    sensor_rates = [r * scale_s for r in raw_sensor_rates]

    sensor_flows = []
    for bw in sensor_rates:
        src, dst = get_random_connected_pair(network, nodes)
        duration_s = max(0.0, random.gauss(sensor_flow_duration, sensor_jitter))
        sensor_flows.append((
            flow_id,
            src,
            dst,
            0,         # num_packets not used
            bw,
            "sensor",
            duration_s
        ))
        flow_id += 1

    # 2) DATA FLOWS: dynamic count per batch, scaled to fill C_data
    all_data_flows = []
    intervals = []
    for _ in range(iterations):
        # determine how many data flows this batch
        n_data = min(max(0, int(load_factor * max_conc_flows)), max_conc_flows)

        # sample raw rates & scale to budget C_data
        raw_data_rates = [random.uniform(*data_bw_range) for _ in range(n_data)]
        total_raw_d = sum(raw_data_rates)
        scale_d = (C_data / total_raw_d) if total_raw_d > 0 else 0
        data_rates = [r * scale_d for r in raw_data_rates]

        batch = []
        for bw in data_rates:
            src, dst = get_random_connected_pair(network, nodes)
            num_packets = random.randint(50_000_000, 80_000_000)
            # convert bytes→megabits: *8/1e6
            total_megabits = num_packets * packet_size * 8 / 1e6
            deadline_s = total_megabits / bw

            batch.append((
                flow_id,
                src,
                dst,
                num_packets,
                bw,
                "data",
                deadline_s
            ))
            flow_id += 1

        all_data_flows.append(batch)
        intervals.append(random.uniform(*interval_range))

    return sensor_flows, all_data_flows, intervals
