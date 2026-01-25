# Mobility-Trajectories
This code implements the generative mobility model in the paper "Mobility Trajectories from Network-Driven Markov Dynamics" (https://arxiv.org/abs/2601.06020)

Please email Asif Shakeel: ashakeel@ucsd.edu with any questions


# Mobility Trajectories (H3) — Markov Network Pipeline

Implements the H3-based mobility generator from:

**Meyer & Shakeel (2026)**  
*Mobility Trajectories from Network-Driven Markov Dynamics*  
https://arxiv.org/abs/2601.06020  

---

## Pipeline Schematic

Polygon / Study Region
        ↓
H3 Discretization
        ↓
Undirected H3 Neighbor Graph
        ↓
graph_builder_h3.py
        ↓
Overlay Mobility Network
(hubs + corridors + feeders + metro links)
        ↓
pep_generator_h3.py
        ↓
Time-dependent Markov operators {P(b)}
        ↓
Synthetic mobility trajectories (PEP)
        ↓
Aggregated OD flows
        ↓
flow_verification_h3.py
(A_pep vs A_prod)
        ↓
timeElapsed_flows_h3.py
(single-step inflow / outflow / netflow)


code_h3/
├── graph_builder_h3.py
├── pep_generator_h3.py
├── flow_verification_h3.py
├── timeElapsed_flows_h3.py
│
├── data/
│   ├── regions/
│   ├── h3_nodes/
│   └── od_inputs/
│
├── outputs/
│   ├── overlays/
│   ├── pep_runs/
│   ├── verification/
│   └── timeElapsed_flows/
│
└── README.md

