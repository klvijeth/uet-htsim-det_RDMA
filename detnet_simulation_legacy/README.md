# Deterministic Networking Simulation

This repository contains a deterministic networking simulation framework built on SimPy and NetworkX. The framework simulates three routing paradigms:

1. Static Bandwidth Reservation.
2. Dynamic Bandwidth Reservation.
3. Cyclic Queuing and Forwarding.

It also implements multipath routing and compares the three paradigms based on latency and flow acceptance rate.

## Repository Structure

- **cqf_sim.py**: Contains the implementation of cyclic queuing and forwarding simulation.
- **fixed_bw.py**: Implements static bandwidth reservation simulation.
- **main.py**: Entry point for running simulations.
- **network.py**: Defines the network topology and flow functions.
- **traffic.py**: Handles traffic generation and management.
- **traffic2.py**: Alternative traffic generation module.
- **old_oscars_sim.py**: Legacy simulation script for OSCARS.
- **oscars_sim.py**: Updated Dynamic Bandwidth Reservation simulation script.
- **Cycle_time_experiments.ipynb**: Jupyter notebook for experiments related to cycle time.
- **multipath_experiments.ipynb**: Jupyter notebook for multipath routing experiments.
- **workload_experiments.ipynb**: Jupyter notebook for workload-based experiments.
- **Experiments.ipynb**: General-purpose notebook for running various experiments.
- **ESnet-BackBoneLinks.csv**: Dataset containing backbone link information for ESnet.
- **ESnet-core-routers.csv**: Dataset containing core router information for ESnet.
- **parameters.md**: Documentation for simulation parameters.
- **requirements.txt**: Lists the Python dependencies required to run the simulations.

## Installation

To set up the environment, ensure you have Python installed. Then, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Notebooks

For interactive experiments, open any of the provided Jupyter notebooks. For example:

```bash
jupyter notebook Cycle_time_experiments.ipynb
```

## Datasets

- **ESnet-BackBoneLinks.csv**: Contains backbone link data for ESnet.
- **ESnet-core-routers.csv**: Contains core router data for ESnet.

## Notes

- Ensure all dependencies are installed before running the scripts.
- Refer to `parameters.md` for detailed information on simulation parameters.
