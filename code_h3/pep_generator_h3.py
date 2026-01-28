#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pep_generator_h3.py
==================

Synthetic Periodic Mobility Generator on an H3 Overlay Graph
------------------------------------------------------------

This script constructs a *time-dependent, column-stochastic mobility operator*
P(b) on an H3 tessellation and uses it to simulate a synthetic population of
agents (PEPs: pseudo-empirical persons) moving through space and time.

The generator is built to operate on the *synthetic corridor overlay*
produced by `graph_builder_h3.py`.  The overlay encodes:

    • hub hierarchy
    • backbone corridors
    • feeders (primary / secondary)
    • metro hub–hub edges
    • potential / geometric direction fields

On top of this structural prior, this generator applies ** modular
behavioral fields (A–G)** that modulate transition probabilities in time.

The output is a fully periodic, schedule-aware, gravity-based mobility model
that produces:

    • P(b)   : per–time-bin Markov kernels
    • P̄     : daily mean kernel
    • Q     : daily operator (product of P(b))
    • X₀    : stationary (periodic fixed point) population
    • PEP   : synthetic agent trajectories
    • OD    : aggregated origin–destination flows
    • maps  : spatial diagnostics and animations


    ----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All results are written to an auto-generated run directory:

    outputs/runs/<region>/h3res-<R>/graph-<mode>/fdr-<fdr>/edge-<pot|geom>/ovr-<0|1>/
        <YYYYMMDDTHHMM_YYYYMMDDTHHMM>/m<T>/x-<init>/

Subfolders:

    npz/     numerical operators and kernels
    csvs/    synthetic OD flows and diagnostics
    maps/    interactive HTML maps
    plots/   time-series PNG diagnostics
    logs/    (reserved)

Core numerical outputs (npz/):

    nodes.npy        ordered list of H3 node IDs
    P_day0.npz       P(b) kernels for one full day
    Pbar.npy        daily mean kernel P̄
    Q.npy            daily operator Q = Π_b P(b)
    X0.npy           stationary (periodic fixed point) population

Synthetic mobility data (csvs/):

    pep_raw.csv
        One row per agent per time bin:
            PEP_ID, start_h3, end_h3, date, time,
            length_m, travel_time_min

    pep_od.csv
        Aggregated OD flows per time bin:
            start_h3, end_h3, date, time,
            trip_count,
            mean_length_m, median_length_m, std_length_m,
            mean_travel_time_min, median_travel_time_min, std_travel_time_min

    pep_population_by_tile.csv
        Initial vs final population per H3 cell:
            h3, initial_count, initial_share,
            final_count, final_share, delta_share,
            is_center, is_hub

    diag_temporal_center_netflow.csv
        Center inflow / outflow / netflow per time bin

    diag_mean_center_netflow_per_timebin.csv
        Daily mean center netflow by time of day


Maps (maps/):

    pep_population_map.html
        Choropleth of final (or initial) PEP population by H3 cell

    initial_population_map.html
        Static map of X₀ (stationary initial distribution)

    fixed_point_population_map.html
        Stationary population from Q (true periodic fixed point)

    mass_timeseries_map.html
        TimeSlider choropleth of mass evolution x(b)


Plots (plots/):

    center_flows_timeseries.png
        Center → periphery, periphery → center, and net flows vs time

----------------------------------------------------------------------
CONCEPTUAL MODEL
----------------------------------------------------------------------

Let P(b) be a column-stochastic matrix where:

    P[d, o] = Pr( destination = d | origin = o )  at time-bin b

The generator builds P(b) by composing:

    • Gravity kernel:
        w(o,d) ∝ (mass_d)^α · dist(o,d)^(-β)

    • Structural overlay:
        w(o,d) *= overlay_weight(o,d)

    • Time-dependent behavioral fields (A–G)

Then for each origin o:

    P[o,o] = p_stay(o, b)
    P[d,o] = (1 - p_stay(o,b)) · w(o,d) / Σ_d w(o,d)

This yields a *column-stochastic* operator suitable for mass evolution:

    x_{b+1} = P(b) · x_b


----------------------------------------------------------------------
MODULES (A–G)
----------------------------------------------------------------------

Each module is independently switchable and can operate in
absolute or multiplicative schedule modes.

A. Stay probability (origin-based)
    • center vs periphery baselines
    • daily schedules
    • optional override for hubs

B. Destination mass (destination-based)
    • center vs periphery baselines
    • daily schedules

C. Hub stay overrides
    • hub-specific stay modulation

D. Hub mass multipliers
    • amplifies or suppresses hub attraction

E. Feeder edge scaling
    • primary / secondary feeder boosts
    • center vs periphery differentiation

F. Hub outgoing metro scaling
    • time-varying hub-to-hub amplification

G. Directional bias
    • inward vs outward flow bias
    • driven by overlay potential / geometry


----------------------------------------------------------------------
PIPELINE
----------------------------------------------------------------------

1) Load corridor overlay manifest (H3-native)
2) Load nodes (lat/lon, hub, center)
3) Build one-day periodic P(b)
4) Compute:
       P̄  = mean_b P(b)
       Q  = Π_b P(b)
       X₀ = fixed point of Q
5) Simulate multinomial agent flows (PEPs)
6) Aggregate to OD and diagnostics
7) Generate maps and plots


----------------------------------------------------------------------
DESIGN PHILOSOPHY
----------------------------------------------------------------------

This is NOT a route choice model and NOT an activity-based model.

It is a ** mobility generator ** designed to:

    • encode spatial hierarchy
    • enforce periodic structure, not memory
    • allow time-varying directional asymmetry
    • remain analytically tractable (Markovian)

The generator is reproducible given the configuration
and random seed, and is intended as a *structural prior* for:

    • effective distance
    • accessibility
    • congestion proxies
    • corridor persistence
    • synthetic OD generation

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel (ashakeel@ucsd.edu)
"""


from __future__ import annotations

import os
import json
import math
from typing import Dict, Tuple, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import sys 
# h3 optional
try:
    import h3
except ImportError:
    h3 = None
    print("[WARN] h3 package not available; H3->lat/lon fallback will be disabled.")



def progress(msg):
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

# ---------------------------------------------------------------------
#    CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR   = "/Users/asif/Documents/nm24"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RUNS_DIR   = os.path.join(OUTPUT_DIR, "runs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

MODE = "generate"   # {"generate","validate"}

COUNTRY     = "USA" # "Mexico"
REGION_NAME = "Atlanta, Georgia" # "Mexico City, Mexico"



# Columns for raw / OD
START_COL = "start_h3"
END_COL   = "end_h3"
DATE_COL  = "date"
TIME_COL  = "time"



# Distance / gravity
ALPHA = 0.50
BETA  = 1.8

GRAPH_MODE =  "corridors"  # "corridors" | "grid"
H3_RESOLUTION = 6
NEIGHBOR_TOPOLOGY = "h3"
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "potential"
INCLUDE_BACKGROUND_EDGES = False
H3_OVERLAY_FLAG = 0
# -----------------------------------------------------
# CUSTOM CENTER NODES for A/B/D/F modules (overrides manifest)
# -----------------------------------------------------
CENTER_FIXED_IDS = ['8644c1a8fffffff', '8644c1aafffffff','8644c1a87ffffff'] #, '8644c1ab7ffffff',  '8644c1a97ffffff', '8644c1a9fffffff', 



# [
#     "864995b87ffffff", 
#     #"864995b8fffffff", 
#     "864995bafffffff", 
#     # "864995ba7ffffff", 
#     "864995bb7ffffff", 
#     #"864995b97ffffff", 
#     # "864995b9fffffff", 


# ]



SPAN_START   = "2025-06-01 00:00:00"
SPAN_END     = "2025-06-03 00:00:00"
TIME_RES_MIN = 30

SPAN_START_TS = pd.to_datetime(SPAN_START)
SPAN_END_TS   = pd.to_datetime(SPAN_END)
WINDOW_START_HH = 0
WINDOW_END_HH   = 24
BINS_PER_DAY = (WINDOW_END_HH - WINDOW_START_HH)*60 // TIME_RES_MIN


# -----------------------------------------------------
# Travel time attribution (Option 1)
# -----------------------------------------------------
SPEED_MIN_KMH = 10.0    # slow urban / congestion
SPEED_MAX_KMH = 90.0    # freeway upper bound



# Output controls
STORE_P_PER_BIN = True
STORE_PBAR      = True
STORE_Q         = True
STORE_X0        = True

# Initial population
INIT_X_MODE = "periodic_fixed_point"    #   "periodic_fixed_point"   # or "flat"

# ============================================================
# PER-MODULE CONSTANT OVERRIDES — independent switches
# ============================================================
FORCE_CONSTANT_A =  False # True #    # Stay probability
FORCE_CONSTANT_B =  False # True #   # Destination mass
FORCE_CONSTANT_C =  False # True #   # Hub pstay
FORCE_CONSTANT_D =  False # True #   # Hub mass multipliers
FORCE_CONSTANT_E =  False # True #   # Feeder scaling
FORCE_CONSTANT_F =  False # True #   # Metro outgoing scaling
FORCE_CONSTANT_G =  False # True #   # Directional bias

# ---------------------------------------------------------------------
# A. STAY PROBABILITIES
# ---------------------------------------------------------------------
ENABLE_PSTAY_SCHEDULE = True

P_STAY_BASE_MODE = (0.5, 0.5)  # (0.9, 0.1) # 

CENTER_STAY_SCHEDULE = [("06:00","11:00",0.9), ("15:00","21:00",0.5)] 
PERIPH_STAY_SCHEDULE = [("06:00","11:00",0.4), ("15:00","21:00",0.8)]

STAY_SCHEDULE_MODE = "absolute"

# ---------------------------------------------------------------------
# B. DESTINATION MASS
# ---------------------------------------------------------------------
ENABLE_MASS_SCHEDULE =  True #  False # 

MASS_BASE_MODE =   (1.0, 20.0) #  (1.0, 1.0)  # 

CENTER_MASS_SCHEDULE = [
    ("06:00","11:00", 20),
    ("15:00","20:00", 1.0),
]
PERIPH_MASS_SCHEDULE = [
    ("06:00","11:00", 1),
    ("15:00","20:00", 20),
]
MASS_SCHEDULE_MODE = "absolute"


# ---------------------------------------------------------------------
# C. HUB PSTAY
# ---------------------------------------------------------------------
ENABLE_HUB_PSTAY_SCALING = True

HUB_PSTAY_FIXED_CENTER_PERIPH =  (0.5, 0.5)

HUB_CENTER_PSTAY_SCHEDULE = [
    ("06:00","11:00", 0.1),
    ("15:00","23:00", 0.5),
]

HUB_PERIPH_PSTAY_SCHEDULE = [
    ("06:00","11:00", 0.1),
    ("15:00","20:00", 0.5),
]


HUB_PSTAY_SCHEDULE_MODE = "absolute"




# ---------------------------------------------------------------------
# D. HUB MASS MULTIPLIERS
# ---------------------------------------------------------------------
ENABLE_HUB_MASS_MULTIPLIERS = True


MASS_MULTIPLIER_HUB_CENTER_PERIPH = (1.0, 1.0)

# Used only if HUB_MASS_MODE == "scheduled":
# Each side gets its own schedule, independent of the other.
HUB_CENTER_MASS_SCHEDULE = [
    ("06:00","11:00", 1.0),
    ("15:00","23:00", 1.0),
]

HUB_PERIPH_MASS_SCHEDULE = [
    ("06:00","11:00",  10.0),
    ("15:00","23:00", 1.0),
]


HUB_MASS_SCHEDULE_MODE = "absolute"

# ---------------------------------------------------------------------
# E. FEEDER SCALING
# ---------------------------------------------------------------------
ENABLE_FEEDER_SCALING = True

FEEDER_SCALE_BASE_CENTER  = (1.0, 1.0)
FEEDER_SCALE_BASE_PERIPH  = (1.0, 1.0)

FEEDER_CENTER_PRIMARY_SCHEDULE = [
    ("06:00","11:00", 5.0),
    ("20:00","23:00", 1.0),
]

FEEDER_CENTER_SECONDARY_SCHEDULE = [
    ("06:00","11:00", 1.0),
    ("15:00","21:00", 1.0),
]

FEEDER_PERIPH_PRIMARY_SCHEDULE = [
    ("06:00","11:00", 5.0),
    ("20:00","23:00", 1.0),
]

FEEDER_PERIPH_SECONDARY_SCHEDULE = [
    ("06:00","11:00", 1.0),
    ("15:00","21:00", 1.0),
]

FEEDER_SCHEDULE_MODE = "absolute"
# ---------------------------------------------------------------------
# F. HUB OUTGOING METRO EDGE SCALING
# ---------------------------------------------------------------------
ENABLE_HUB_OUTGOING_METRO_SCALING = True

HUB_OUTGOING_METRO_FIXED = (1.0, 1.0)

HUB_OUTGOING_METRO_CENTER_SCHEDULE = [
    ("05:00","07:00", 5.0),
    ("21:00","23:00", 1.0),
]

HUB_OUTGOING_METRO_PERIPH_SCHEDULE = [
    ("05:00","07:00", 5.0),
   ("21:00","23:00", 1.0),
]


HUB_OUTGOING_METRO_SCHEDULE_MODE = "absolute"

# ---------------------------------------------------------------------
# G. DIRECTIONAL BIAS
# ---------------------------------------------------------------------
ENABLE_DIRECTIONAL_BIAS = True

DIRBIAS_BASE = (1.0, 1.0)


DIRBIAS_IN_MIN  = 1.0
DIRBIAS_IN_MAX  = 5.0
DIRBIAS_OUT_MIN = 1.0
DIRBIAS_OUT_MAX = 5.0
DIRBIAS_BASE = (DIRBIAS_IN_MIN, DIRBIAS_OUT_MIN)
DIRBIAS_NEUTRAL_BASE = 1.0


DIRBIAS_IN_SCHEDULE  = [("06:00","11:00",DIRBIAS_IN_MAX), ("15:00","20:00",DIRBIAS_IN_MIN)]
DIRBIAS_OUT_SCHEDULE = [ ("15:00","20:00",DIRBIAS_OUT_MAX), ("20:00","23:00",DIRBIAS_OUT_MIN)] 
DIRBIAS_NEUTRAL_SCHEDULE = [
    ("06:00","11:00", 1.0),
    ("15:00","20:00", 1.0),
]



DIRBIAS_SCHEDULE_MODE = "absolute"


WRITE_PEP_RAW        =  False # True #  
WRITE_PEP_OD         = True
WRITE_PEP_BIN_TOTALS = False # True
WRITE_PEP_TILE_POP   = True


# =========================
# Diagnostics: time series
# =========================
ENABLE_DIAG_TEMPORAL   = False # True   # trips per bin (date+time)
ENABLE_DIAG_MEAN_BIN   = True   # mean trips per time-of-day bin across days
ENABLE_DIAG_DAILY      = False # True   # total trips per day

# =========================
# Diagnostics / plots
# =========================
ENABLE_CENTER_OUTFLOW_PLOT = True  # center→non-center flow vs time (PNG in plots/)

# =========================
# Mass timeseries map (Option B)
# =========================
MAKE_MASS_MAP         = False      # keep False for now (we can add later cleanly)
MASS_MAP_MODE         = "count"    # "share" or "count"
MASS_MAP_USE_MANIFEST = False

# =========================
# PEP population map
# =========================
MAKE_PEP_MAP      = True
PEP_MAP_VALUE_COL = "initial_share"   # or "final_share"
PEP_MAP_MODE      = "count"           # "share" or "count"

TOTAL_PER_BIN_FIXED   = 120000        # you already wanted this
SEED                  = 12345         # or whatever you’re using

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def h3_to_latlon(hx: str) -> Tuple[float, float]:
    if h3 is None:
        return float("nan"), float("nan")
    try:
        lat, lon = h3.h3_to_geo(hx)
        return float(lat), float(lon)
    except Exception:
        return float("nan"), float("nan")

def h3_hex_edge_length_km(res: int) -> float:
    """
    Average H3 hexagon edge length (km) for a given resolution.
    Uses the h3 library’s built-in function.
    """
    if h3 is None:
        raise RuntimeError("h3 package not available")

    # h3-py v4+ API
    if hasattr(h3, "average_hexagon_edge_length"):
        return float(h3.average_hexagon_edge_length(res, unit="km"))

    # fallback for older h3-py variants (rare)
    if hasattr(h3, "hexagon_edge_length"):
        return float(h3.hexagon_edge_length(res, unit="km"))

    raise RuntimeError("Could not find an H3 edge-length function in this h3 package")


def h3_hex_diameter_km(res: int) -> float:
    """
    Approx diameter across opposite vertices (circumdiameter) of a regular hex.
    If e is edge length, circumradius R = e / tan(pi/6) = e / (1/sqrt(3)) = sqrt(3)*e
    Diameter = 2R = 2*sqrt(3)*e
    """
    e = h3_hex_edge_length_km(res)
    return float(2.0 * math.sqrt(3.0) * e)





def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(max(0,1-a)))
    return R*c

def build_distance_envelopes_h3(
    nodes_df: pd.DataFrame,
    res: int
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Build (min_km, max_km) distance envelopes for every ordered pair (i,j) of nodes.

    For O == D (stay moves):
        (0, D_hex)

    For O != D:
        ( max(0, d_center - D_hex),  d_center + D_hex )

    where D_hex is the hexagon diameter at resolution `res`.
    """

    node_list = nodes_df["h3"].astype(str).tolist()
    lat = nodes_df["lat"].to_numpy()
    lon = nodes_df["lon"].to_numpy()
    N = len(node_list)

    D_hex = h3_hex_diameter_km(res)

    env: Dict[Tuple[int,int], Tuple[float,float]] = {}

    for i in range(N):
        for j in range(N):
            if i == j:
                # internal movement within a single hex cell
                lo = 0.0
                hi = D_hex
            else:
                d = haversine_km(lat[i], lon[i], lat[j], lon[j])
                lo = max(0.0, d - D_hex)
                hi = d + D_hex
            env[(i, j)] = (float(lo), float(hi))

    return env

def minute_of_day_from_bin(b: int) -> int:
    return WINDOW_START_HH*60 + b*TIME_RES_MIN


def parse_hhmm(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh)*60 + int(mm)


def evaluate_ramp_schedule(minute: int,
                           schedule: Sequence[Tuple[str,str,float]],
                           default: float) -> float:
    """
    Ramp + hold schedule evaluator.

    Semantics:
        For entries (s,e,v):
            - Ramp from the current value at s to v across [s,e]
            - After reaching v at e, HOLD v until the next ramp block
        If the current time is before the first block → default
        If time sits in a gap → HOLD last arrived value

    A schedule like:
        [("06:00","11:00",0.9),
         ("15:00","21:00",0.5)]

    means:
        - START at default
        - 06:00 → begin ramp from default to 0.9 at 11:00
        - 11:00–15:00: hold 0.9
        - 15:00→ begin ramp from 0.9 to 0.5 at 21:00
        - 21:00–24:00: hold 0.5
    """

    if not schedule:
        return float(default)

    md = minute % 1440

    # parse and sort
    parsed = []
    for s,e,v in schedule:
        try:
            parsed.append((parse_hhmm(s), parse_hhmm(e), float(v)))
        except:
            pass
    if not parsed:
        return float(default)
    parsed.sort(key=lambda x: x[0])

    cur = float(default)

    for s,e,v in parsed:
        # before first ramp block
        if md < s:
            return cur

        # inside ramp block → interpolate
        if s <= md <= e:
            L = max(1, e-s)
            f = (md - s)/L
            return cur + f*(v - cur)

        # after ramp block → ramp has completed
        cur = v

    # after last block → hold last level
    return cur


def evaluate_module_schedule(
    bin_index: int,
    schedule: Sequence[Tuple[str,str,float]],
    base_value: float,
    mode: str,
    default_when_missing: float,
    force_constant: bool = False
) -> float:
    """
    Unified evaluator:

       if force_constant: return base_value

       absolute → final = schedule_value
       multiplier → final = base_value * schedule_value
    """

    # HARD OVERRIDE FOR THIS MODULE
    if force_constant:
        return float(base_value)

    # ---------- normal schedule evaluation ----------
    sched_val = evaluate_ramp_schedule(
        minute_of_day_from_bin(bin_index),
        schedule,
        default_when_missing
    )

    mode = (mode or "").lower()
    if mode == "absolute":
        return float(sched_val)
    else:
        return float(base_value) * float(sched_val)



# ---------------------------------------------------------------------
# CorridorOverlayH3
# ---------------------------------------------------------------------
class CorridorOverlayH3:

    def __init__(self):
        self.hub_attr: Dict[str, Dict[str,float]] = {}
        self.hub_ids: Set[str] = set()
        self.hub_pstay: Dict[str,float] = {}

        self.per_pair: Dict[Tuple[str,str], float] = {}
        self.dir_sign: Dict[Tuple[str,str], int] = {}
        self.feeder_label: Dict[Tuple[str,str], int] = {}
        self.overlay_flag: Dict[Tuple[str,str], int] = {}
        self.delta_geom_km: Dict[Tuple[str,str], float] = {}
        self.metro_pairs: Set[Tuple[str,str]] = set()

        self.manifest: Optional[dict] = None
        self.loaded = False
        self.enabled = False

    def is_hub(self, h3id: str) -> bool:
        return h3id in self.hub_attr

    def pstay_override(self, h3id: str) -> Optional[float]:
        return self.hub_attr.get(h3id,{}).get("pstay")

    # ------------------ Manifest loader ------------------
    def load(self, path: str):
        self.__init__()
        if not os.path.exists(path):
            print(f"[WARN] H3 manifest not found: {path}")
            return
        try:
            with open(path,"r",encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load manifest: {e}")
            return

        self.manifest = data

        # Hubs
        hubs_raw = data.get("hubs", {})
        if isinstance(hubs_raw, dict):
            if "h3" in hubs_raw and isinstance(hubs_raw["h3"], dict):
                hubs_iter = hubs_raw["h3"].items()
            else:
                hubs_iter = hubs_raw.items()
        elif isinstance(hubs_raw,list):
            hubs_iter = [(r.get("h3"),r) for r in hubs_raw if isinstance(r,dict)]
        else:
            hubs_iter = []

        for h3id, spec in hubs_iter:
            if h3id is None: 
                continue
            try:
                h3s = str(h3id)
                if isinstance(spec, dict):
                    attr = float(spec.get("attr",1.0))
                    pst  = spec.get("pstay")
                    if pst is not None:
                        pst = float(pst)
                else:
                    attr = float(spec)
                    pst  = None
                self.hub_attr[h3s] = {"attr":attr, "pstay":pst}
                self.hub_ids.add(h3s)
                if pst is not None:
                    self.hub_pstay[h3s] = pst
            except Exception:
                pass

        # Edges
        edges_raw = data.get("edges", {})
        if isinstance(edges_raw, dict):
            recs = []
            for v in edges_raw.values():
                if isinstance(v,list):
                    recs.extend(v)
        elif isinstance(edges_raw,list):
            recs = edges_raw
        else:
            recs = []

        weight_tmp = {}
        dir_tmp    = {}
        feeder_tmp = {}
        of_tmp     = {}
        geom_tmp   = {}
        metro_tmp  = {}

        for r in recs:
            if not isinstance(r,dict): 
                continue
            a = r.get("start_h3") or r.get("start")
            b = r.get("end_h3")   or r.get("end")
            if not a or not b: 
                continue
            a = str(a); b = str(b)

            w = r.get("w_norm") or r.get("weight") or r.get("mult") or 1.0
            try: w = float(w)
            except: w = 1.0
            if w > weight_tmp.get((a,b),0.0):
                weight_tmp[(a,b)] = w

            ds = r.get("delta_T") or r.get("delta") or r.get("deltaT")
            if ds is not None:
                try:
                    ds = int(ds)
                    ds = 1 if ds>0 else (-1 if ds<0 else 0)
                    dir_tmp[(a,b)] = ds
                    # if ds == 1:
                    #     if a == "8644c1a9fffffff":
                    #         print((a,b))
                except:
                    pass

            if "feeder_label" in r:
                try:
                    fl = int(r["feeder_label"])
                    if fl>0:
                        feeder_tmp[(a,b)] = max(fl, feeder_tmp.get((a,b),0))
                except: pass

            if "overlay_flag" in r:
                try:
                    ov = int(r["overlay_flag"])
                    of_tmp[(a,b)] = 1 if ov>0 else 0
                except: pass

            if "delta_geom_km" in r:
                try:
                    geom_tmp[(a,b)] = float(r["delta_geom_km"])
                except: pass

            if "is_metro" in r:
                try:
                    metro_tmp[(a,b)] = 1 if int(r["is_metro"])>0 else 0
                except: pass

        # symmetric weight
        undirected = {}
        for (a,b),w in weight_tmp.items():
            k = (a,b) if a<=b else (b,a)
            undirected[k] = max(w, undirected.get(k,0.0))

        for (a,b),w in undirected.items():
            self.per_pair[(a,b)] = w
            self.per_pair[(b,a)] = w

        # directional signs
        for (a,b),s in dir_tmp.items():
            self.dir_sign[(a,b)] = s
            if (b,a) not in dir_tmp:
                self.dir_sign[(b,a)] = -s

        # feeder
        self.feeder_label = dict(feeder_tmp)

        # overlay_flag
        self.overlay_flag = dict(of_tmp)

        # geom
        self.delta_geom_km = dict(geom_tmp)

        # metro edges (only if both ends are hubs)
        hubs = set(self.hub_attr.keys())
        metro_u = set()
        for (a,b),flag in metro_tmp.items():
            if flag>0 and a in hubs and b in hubs:
                key = (a,b) if a<=b else (b,a)
                metro_u.add(key)

        self.metro_pairs.clear()
        for (a,b) in metro_u:
            self.metro_pairs.add((a,b))
            self.metro_pairs.add((b,a))

        self.loaded = True
        self.enabled = bool(self.hub_attr or self.per_pair)
        print(f"[INFO] Overlay loaded: hubs={len(self.hub_attr)}, edges={len(self.per_pair)}")


# ---------------------------------------------------------------------
# Node load
# ---------------------------------------------------------------------
def load_nodes_from_h3_manifest(manifest: dict) -> pd.DataFrame:
    rows = []
    for h, spec in manifest.get("nodes", {}).items():
        if not isinstance(spec,dict):
            continue
        h3id = str(h)
        lat = spec.get("lat")
        lon = spec.get("lon")
        if (lat is None or lon is None) and h3 is not None:
            lat, lon = h3_to_latlon(h3id)
        if lat is None or lon is None:
            continue
        lat,lon = float(lat), float(lon)
        is_hub   = int(spec.get("is_hub",0))
        is_center= int(spec.get("is_center",0))
        rows.append((h3id,lat,lon,is_hub,is_center))
    df = pd.DataFrame(rows, columns=["h3","lat","lon","is_hub","is_center"])
    return df


# ---------------------------------------------------------------------
# Module A+C: stay_prob
# ---------------------------------------------------------------------
def stay_prob_for_origin(origin_h3: str,
                         is_center: bool,
                         bin_index: int,
                         overlay: CorridorOverlayH3) -> float:

    # Manifest override?
    pst_override = overlay.pstay_override(origin_h3)
    if pst_override is not None:
        return max(0.0, min(1.0, float(pst_override)))

    base0 = P_STAY_BASE_MODE[0] if is_center else P_STAY_BASE_MODE[1]
    sched = CENTER_STAY_SCHEDULE if is_center else PERIPH_STAY_SCHEDULE

    pst = base0
    if ENABLE_PSTAY_SCHEDULE:
        pst = evaluate_module_schedule(
            bin_index,
            sched,
            base0,
            STAY_SCHEDULE_MODE,
            default_when_missing=base0,
            force_constant=FORCE_CONSTANT_A,
        )


    # Hub overrides (Module C)
    if ENABLE_HUB_PSTAY_SCALING and overlay.is_hub(origin_h3):
        sched2 = HUB_CENTER_PSTAY_SCHEDULE if is_center else HUB_PERIPH_PSTAY_SCHEDULE
        pst = evaluate_module_schedule(
            bin_index,
            sched2,
            pst,
            HUB_PSTAY_SCHEDULE_MODE,
            default_when_missing=pst,
            force_constant=FORCE_CONSTANT_C,
        )


    return max(0.0, min(1.0, pst))


# ---------------------------------------------------------------------
# Module B+D: destination mass
# ---------------------------------------------------------------------
def mass_for_destination(dest_h3: str,
                         is_center: bool,
                         bin_index: int,
                         overlay: CorridorOverlayH3) -> float:

    base_val = MASS_BASE_MODE[0] if is_center else MASS_BASE_MODE[1]
    sched    = CENTER_MASS_SCHEDULE if is_center else PERIPH_MASS_SCHEDULE

    m = base_val
    if ENABLE_MASS_SCHEDULE:
        m = evaluate_module_schedule(
            bin_index,
            sched,
            base_val,
            MASS_SCHEDULE_MODE,
            default_when_missing=base_val,
            force_constant=FORCE_CONSTANT_B,
        )


    # Hub mass multiplier (Module D)
    if ENABLE_HUB_MASS_MULTIPLIERS and overlay.is_hub(dest_h3):
        base_center, base_periph = MASS_MULTIPLIER_HUB_CENTER_PERIPH
        base2  = base_center if is_center else base_periph
        sched2 = HUB_CENTER_MASS_SCHEDULE if is_center else HUB_PERIPH_MASS_SCHEDULE

        m *= evaluate_module_schedule(
            bin_index,
            sched2,
            base2,
            HUB_MASS_SCHEDULE_MODE,
            default_when_missing=base2,
            force_constant=FORCE_CONSTANT_D,
        )


    return float(m)



# ---------------------------------------------------------------------
# Module E: Feeder scaling
# ---------------------------------------------------------------------
def feeder_multiplier(origin_h3: str,
                      is_center: bool,
                      feeder_label: int,
                      bin_index: int) -> float:

    if not ENABLE_FEEDER_SCALING:
        return 1.0
    if feeder_label <= 0:
        return 1.0

    if is_center:
        base_pri, base_sec = FEEDER_SCALE_BASE_CENTER
        if feeder_label == 1:
            base = base_pri
            sched = FEEDER_CENTER_PRIMARY_SCHEDULE
        else:
            base = base_sec
            sched = FEEDER_CENTER_SECONDARY_SCHEDULE
    else:
        base_pri, base_sec = FEEDER_SCALE_BASE_PERIPH
        if feeder_label == 1:
            base = base_pri
            sched = FEEDER_PERIPH_PRIMARY_SCHEDULE
        else:
            base = base_sec
            sched = FEEDER_PERIPH_SECONDARY_SCHEDULE

    return evaluate_module_schedule(
        bin_index,
        sched,
        base,
        FEEDER_SCHEDULE_MODE,
        default_when_missing=base,
        force_constant=FORCE_CONSTANT_E,
    )



# ---------------------------------------------------------------------
# Module F: Metro-edge outgoing scaling
# ---------------------------------------------------------------------
def hub_outgoing_metro_multiplier(is_center_origin: bool,
                                  is_metro_edge: bool,
                                  bin_index: int) -> float:

    if not ENABLE_HUB_OUTGOING_METRO_SCALING:
        return 1.0
    if not is_metro_edge:
        return 1.0

    base_metro, base_non = HUB_OUTGOING_METRO_FIXED
    base  = base_metro
    sched = (HUB_OUTGOING_METRO_CENTER_SCHEDULE
             if is_center_origin else
             HUB_OUTGOING_METRO_PERIPH_SCHEDULE)

    return evaluate_module_schedule(
        bin_index,
        sched,
        base,
        HUB_OUTGOING_METRO_SCHEDULE_MODE,
        default_when_missing=base,
        force_constant=FORCE_CONSTANT_F,
    )




# ---------------------------------------------------------------------
# Module G: Directional bias
# ---------------------------------------------------------------------
def directional_bias_multiplier(dir_sign: Optional[int],
                                bin_index: int) -> float:

    if not ENABLE_DIRECTIONAL_BIAS:
        return 1.0

    try:
        ds = int(dir_sign) if dir_sign is not None else 0
    except:
        ds = 0
    if ds not in (-1, 0, 1):
        ds = 0

    in_base, out_base = DIRBIAS_BASE
    neutral_base      = DIRBIAS_NEUTRAL_BASE

    if ds > 0:
        base = in_base
        sched = DIRBIAS_IN_SCHEDULE
    elif ds < 0:
        base = out_base
        sched = DIRBIAS_OUT_SCHEDULE
    else:
        base = neutral_base
        sched = DIRBIAS_NEUTRAL_SCHEDULE

    return evaluate_module_schedule(
        bin_index,
        sched,
        base,
        DIRBIAS_SCHEDULE_MODE,
        default_when_missing=1.0,
        force_constant=FORCE_CONSTANT_G,
    )




# ---------------------------------------------------------------------
# BUILD P(b)
# ---------------------------------------------------------------------
def build_P_for_bin_h3(
    bin_index: int,
    nodes_df: pd.DataFrame,
    centers_set: Set[str],
    overlay: CorridorOverlayH3
) -> np.ndarray:

    node_list = nodes_df["h3"].astype(str).tolist()
    N = len(node_list)
    lat = nodes_df["lat"].to_numpy()
    lon = nodes_df["lon"].to_numpy()

    is_center = np.array([h in centers_set for h in node_list], dtype=bool)

    # ------------------------------------------------------------------
    # Precompute masses (destination-based)   → used in column weights
    # ------------------------------------------------------------------
    mass_vec = np.zeros(N, float)
    for j, h in enumerate(node_list):
        mass_vec[j] = mass_for_destination(
            h,
            is_center[j],
            bin_index,
            overlay
        )

    # ------------------------------------------------------------------
    # Precompute stay probabilities (origin-based)
    # ------------------------------------------------------------------
    pstay_vec = np.zeros(N, float)
    for o_idx, h in enumerate(node_list):
        pstay_vec[o_idx] = stay_prob_for_origin(
            h,
            is_center[o_idx],
            bin_index,
            overlay
        )

    # ------------------------------------------------------------------
    # Build COLUMN-stochastic P:  P[d, o] = Pr( origin=o → destination=d )
    # ------------------------------------------------------------------
    P = np.zeros((N, N), float)

    for o_idx, o in enumerate(node_list):

        pstay = pstay_vec[o_idx]
        P[o_idx, o_idx] = pstay               # stay at origin

        remain = 1.0 - pstay
        if remain <= 0.0:
            # Fully staying, entire column is zero except diagonal
            continue

        # Build UNNORMALIZED off-diagonal weights for this origin
        w = np.zeros(N, float)

        for d_idx, d in enumerate(node_list):

            if d_idx == o_idx:
                continue

            # ------------------- Gravity -------------------
            dist = haversine_km(lat[o_idx], lon[o_idx], lat[d_idx], lon[d_idx])
            if dist > 0:
                w_raw = (mass_vec[d_idx] ** ALPHA) * (dist ** (-BETA))
            else:
                w_raw = 0.0

            # --------------- Overlay edge weight ---------------
            w_raw *= overlay.per_pair.get((o, d), 0)

            # ---------------- Feeder scaling -------------------
            feeder_lbl = overlay.feeder_label.get((o, d), 0)

            w_raw *= feeder_multiplier(
                origin_h3=o,
                is_center=is_center[o_idx],
                feeder_label=feeder_lbl,
                bin_index=bin_index
            )

            # ---------------- Metro scaling ---------------------
            is_metro = (o, d) in overlay.metro_pairs
            w_raw *= hub_outgoing_metro_multiplier(
                is_center_origin=is_center[o_idx],
                is_metro_edge=is_metro,
                bin_index=bin_index
            )

            # ---------------- Directional bias -----------------
            ds = overlay.dir_sign.get((o, d), 0)
            w_raw *= directional_bias_multiplier(ds, bin_index)

            w[d_idx] = w_raw

        # ------------------- Normalize column -------------------
        off_sum = w.sum()
        if off_sum <= 0:
            # degenerate: force pure stay
            P[:, o_idx] = 0.0
            P[o_idx, o_idx] = 1.0
            continue

        # scale off-diagonals so they sum to remaining = 1 - pstay
        scale = remain / off_sum
        for d_idx in range(N):
            if d_idx != o_idx:
                P[d_idx, o_idx] = w[d_idx] * scale

    return P


# ---------------------------------------------------------------------
# DAILY MEAN + DAILY OPERATOR + PERIODIC FP
# ---------------------------------------------------------------------
def compute_daily_mean_P(P_list: List[np.ndarray]) -> np.ndarray:
    return sum(P_list) / len(P_list)


def compute_daily_operator_h3(P_list: List[np.ndarray]) -> np.ndarray:
    if not P_list:
        raise ValueError("Empty P_list")
    Q = np.eye(P_list[0].shape[0])
    for P in P_list:
        Q = P @ Q
    return Q


def compute_periodic_fixed_point_h3(
    Q: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20000
) -> np.ndarray:
    """
    Periodic fixed point for a column-stochastic daily operator Q.

    We treat x as a column vector of masses, and evolve:
        x_{k+1} = Q @ x_k
    looking for x such that Q @ x = x.
    """
    N = Q.shape[0]
    x = np.ones(N) / N  # start from flat

    for _ in range(max_iter):
        x_new = Q @ x      # NOTE: no transpose here
        s = x_new.sum()
        if s <= 0:
            raise RuntimeError("Degenerate Q (sum <= 0 in fixed-point iteration)")
        x_new /= s

        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new

        x = x_new

    print("[WARN] periodic FP not converged")
    return x


# ---------------------------------------------------------------------
# AUTO-RUN DIRECTORY
# ---------------------------------------------------------------------
def sanitize_region(r: str) -> str:
    return "".join(ch for ch in r if ch.isalnum())


def edge_tag() -> str:
    return "geom" if EDGE_DIRECTION_MODE=="geometric" else "pot"


def make_h3_run_directory() -> str:
    region_core = REGION_NAME.split(",")[0]
    tag = sanitize_region(region_core)

    span_tag = (
        SPAN_START_TS.strftime("%Y%m%dT%H%M") + "_" +
        SPAN_END_TS.strftime("%Y%m%dT%H%M")
    )

    if GRAPH_MODE == "corridors":
        run_dir = os.path.join(
            OUTPUT_DIR, "runs", tag,
            f"h3res-{H3_RESOLUTION}",
            f"graph-{GRAPH_MODE}",  
            f"fdr-{FEEDER_MODE}",
            f"edge-{edge_tag()}",
            f"ovr-{H3_OVERLAY_FLAG}",
            span_tag,
            f"m{TIME_RES_MIN}",
            f"x-{INIT_X_MODE}"
        )
    else:
        run_dir = os.path.join(
            OUTPUT_DIR, "runs", tag,
            f"h3res-{H3_RESOLUTION}",
             f"graph-{GRAPH_MODE}",           
            span_tag,
            f"m{TIME_RES_MIN}",
            f"x-{INIT_X_MODE}"
        )        

    for sub in ["npz","csvs","maps","logs"]:
        os.makedirs(os.path.join(run_dir,sub), exist_ok=True)

    print(f"[INFO] Run directory: {run_dir}")
    return run_dir

def write_temporal_diagnostics_h3(
    od_df: pd.DataFrame,
    centers_set: Set[str],
    run_dir: str,
):
    csv_dir = os.path.join(run_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    od = od_df.copy()
    od["trip_count"] = od["trip_count"].astype(float)
    od["start_h3"]   = od["start_h3"].astype(str)
    od["end_h3"]     = od["end_h3"].astype(str)

    centers_set_str = {str(c) for c in centers_set}

    od["is_center_origin"] = od["start_h3"].isin(centers_set_str)
    od["is_center_dest"]   = od["end_h3"].isin(centers_set_str)

    # -------------------------------------------------------
    # 1. Total trips per bin
    # -------------------------------------------------------
    total = (
        od.groupby(["date","time"],as_index=False)["trip_count"]
          .sum()
          .rename(columns={"trip_count":"total_trips"})
    )
    total.to_csv(os.path.join(csv_dir,"diag_temporal_total_trips.csv"),index=False)

    # -------------------------------------------------------
    # 2. Center outflow
    # -------------------------------------------------------
    out = (
        od.loc[od["is_center_origin"] & ~od["is_center_dest"]]
          .groupby(["date","time"],as_index=False)["trip_count"]
          .sum()
          .rename(columns={"trip_count":"center_outflow"})
    )

    # -------------------------------------------------------
    # 3. Center inflow
    # -------------------------------------------------------
    inflow = (
        od.loc[~od["is_center_origin"] & od["is_center_dest"]]
          .groupby(["date","time"],as_index=False)["trip_count"]
          .sum()
          .rename(columns={"trip_count":"center_inflow"})
    )

    # -------------------------------------------------------
    # 4. Merge into netflow
    # -------------------------------------------------------
    net = total.merge(out,how="left",on=["date","time"]) \
               .merge(inflow,how="left",on=["date","time"])

    net["center_outflow"] = net["center_outflow"].fillna(0)
    net["center_inflow"]  = net["center_inflow"].fillna(0)

    net["center_netflow"] = net["center_inflow"] - net["center_outflow"]

    # Write main diagnostic
    net.to_csv(os.path.join(csv_dir,"diag_temporal_center_netflow.csv"),index=False)

    # -------------------------------------------------------
    # 5. Mean netflow per time bin (THIS is the “daily average”)
    # -------------------------------------------------------
    mean_net = (
        net.groupby("time",as_index=False)["center_netflow"]
           .mean()
           .rename(columns={"center_netflow":"mean_center_netflow"})
    )

    mean_net.to_csv(os.path.join(csv_dir,"diag_mean_center_netflow_per_timebin.csv"),
                    index=False)



def simulate_pep_h3(
    P_list: List[np.ndarray],
    X0: np.ndarray,
    nodes_df: pd.DataFrame,
    centers_set: Set[str],
    overlay: CorridorOverlayH3,
    run_dir: str,
    total_per_bin: int = TOTAL_PER_BIN_FIXED,
    seed: int = SEED,
):
    """
    Multinomial PEP simulator:

      - Uses P(b) sequence (row-stochastic: row i → origin)
      - Initial population ~ X0
      - total_per_bin trips each active bin
      - Distance sampled from envelopes:
            O==D:         0–D_hex
            O!=D:         d_center ± D_hex
      - Writes:
            csvs/pep_raw.csv
            csvs/pep_od.csv
            csvs/pep_bin_totals.csv
            csvs/pep_population_by_tile.csv
    """

    # --- setup ---
    rng_moves   = np.random.default_rng(seed + 1)
    rng_lengths = np.random.default_rng(seed + 2)
    rng_speeds  = np.random.default_rng(seed + 3)

    nodes = nodes_df["h3"].astype(str).tolist()
    N = len(nodes)
    B = len(P_list)

    minutes_in_day = [
        WINDOW_START_HH * 60 + b * TIME_RES_MIN
        for b in range(B)
    ]

    # distance envelopes between all node pairs
    env = build_distance_envelopes_h3(nodes_df, H3_RESOLUTION)

    print("[DEBUG] H3 res:", H3_RESOLUTION)
    print("[DEBUG] edge_km:", h3_hex_edge_length_km(H3_RESOLUTION))
    print("[DEBUG] diameter_km:", h3_hex_diameter_km(H3_RESOLUTION))

    # agent pool & IDs
    POOL_SIZE = int(total_per_bin)
    agent_ids = np.array([f"P{(k+1):06d}" for k in range(POOL_SIZE)], dtype=object)

    # --- initial state from X0 ---
    if X0.shape[0] != N:
        raise ValueError(f"X0 length {X0.shape[0]} != number of nodes {N}")

    X0 = np.clip(X0, 0.0, None)
    s0 = X0.sum()
    if s0 <= 0:
        raise RuntimeError("X0 sums to zero")
    X0 = X0 / s0

    agent_state = rng_moves.choice(N, size=POOL_SIZE, p=X0, replace=True)
    initial_agent_state = agent_state.copy()

    out_rows: List[Tuple[str,str,str,int,str,float]] = []

    # --- time loop over [SPAN_START_TS, SPAN_END_TS) ---
    day = SPAN_START_TS.normalize()
    end_day = SPAN_END_TS.normalize()

    while day <= end_day:
        for b, minute in enumerate(minutes_in_day):
            tbin = day + pd.Timedelta(minutes=int(minute))

            # --- in-place progress ---
            progress(f"[PEP] {tbin}   bin {b+1}/{B}")

            if tbin < SPAN_START_TS or tbin >= SPAN_END_TS:
                continue

            P = P_list[b]
            if P.shape != (N, N):
                raise ValueError(f"P_list[{b}] has shape {P.shape}, expected ({N},{N})")

            # next-state buffer
            next_state = agent_state.copy()

            # multinomial step, grouped by origin
            for j in range(N):
                idx = np.where(agent_state == j)[0]
                if idx.size == 0:
                    continue

                # Use COLUMN j: P[:, j] = Pr(dest | origin = j)
                probs = P[:, j].astype(float)
                probs = np.clip(probs, 0.0, None)
                s = probs.sum()
                if s <= 0.0:
                    # degenerate: force self-loop
                    probs = np.zeros(N, dtype=float)
                    probs[j] = 1.0
                else:
                    probs /= s

                dest_choices = rng_moves.choice(N, size=idx.size, p=probs)
                next_state[idx] = dest_choices


            # record raw trips
            date_int = int(day.strftime("%Y%m%d"))
            time_str = tbin.strftime("%H:%M:%S") # tbin.strftime("%Y-%m-%d %H:%M:%S") 

            for a in range(POOL_SIZE):
                j = int(agent_state[a])
                i = int(next_state[a])

                lo_km, hi_km = env[(j, i)]
                if hi_km <= lo_km:
                    length_km = max(0.0, lo_km)
                else:
                    u = rng_lengths.random()
                    length_km = lo_km + (hi_km - lo_km) * float(u)

                length_m = float(length_km * 1000.0)

                # ---------------- travel time attribution ----------------
                # sample speed in km/h
                v_kmh = SPEED_MIN_KMH + (SPEED_MAX_KMH - SPEED_MIN_KMH) * rng_speeds.random()

                # travel time in minutes:  time = distance / speed
                travel_time_min = (length_km / v_kmh) * 60.0

                # cap to the time-bin duration
                travel_time_min = min(max(travel_time_min, 0.0), float(TIME_RES_MIN))


                out_rows.append((
                    agent_ids[a],
                    nodes[j],
                    nodes[i],
                    date_int,
                    time_str,
                    length_m,
                    float(travel_time_min),
                    # float(v_kmh),
                ))


            # advance state
            agent_state = next_state

        day = day + pd.Timedelta(days=1)

    # --- build dataframes / diagnostics ---
    csv_dir = os.path.join(run_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    cols = [
        "PEP_ID",
        START_COL,
        END_COL,
        DATE_COL,
        TIME_COL,
        "length_m",
        "travel_time_min",
        # "speed_kmh",
    ]



    pep_df = pd.DataFrame(out_rows, columns=cols)
    pep_df.sort_values(by=[DATE_COL, TIME_COL, START_COL, END_COL],
                       inplace=True, ignore_index=True)

    # RAW
    if WRITE_PEP_RAW:
        raw_path = os.path.join(csv_dir, "pep_raw.csv")
        pep_df.to_csv(raw_path, index=False)
        print(f"[OK] [WRITE] {raw_path}")


    # OD aggregate
    if WRITE_PEP_OD:

        g = pep_df.groupby([START_COL, END_COL, DATE_COL, TIME_COL], as_index=False)
        od = g.agg(
            trip_count=("length_m", "size"),

            mean_length_m=("length_m", "mean"),
            median_length_m=("length_m", "median"),
            std_length_m=("length_m", "std"),

            mean_travel_time_min=("travel_time_min", "mean"),
            median_travel_time_min=("travel_time_min", "median"),
            std_travel_time_min=("travel_time_min", "std"),
        )



        od.sort_values(by=[DATE_COL, TIME_COL, START_COL, END_COL],
                       inplace=True, ignore_index=True)
        od_path = os.path.join(csv_dir, "pep_od.csv")
        od.to_csv(od_path, index=False)
        print(f"[OK] [WRITE] {od_path}")

        # NEW: diagnostics
        write_temporal_diagnostics_h3(
            od_df   = od,           # or whatever you named the OD aggregate df
            centers_set = centers_set,
            run_dir = run_dir,
        )


        if WRITE_PEP_BIN_TOTALS:
            bt = od.groupby([DATE_COL, TIME_COL], as_index=False)["trip_count"].sum()
            bt_path = os.path.join(csv_dir, "pep_bin_totals.csv")
            bt.to_csv(bt_path, index=False)
            print(f"[OK] [WRITE] {bt_path}")

    # Per-tile population diagnostics (initial vs final)
    if WRITE_PEP_TILE_POP:
        def _counts_from_state(state: np.ndarray) -> np.ndarray:
            return np.bincount(state, minlength=N).astype(int)

        init_counts = _counts_from_state(initial_agent_state)
        final_counts = _counts_from_state(agent_state)

        def _share(counts: np.ndarray) -> np.ndarray:
            total = float(max(1, counts.sum()))
            return (counts.astype(float) / total) * 100.0

        is_center_mask = np.array([h in centers_set for h in nodes], dtype=bool)
        hubs_mask = np.array([overlay.is_hub(h) for h in nodes], dtype=bool)

        df_tiles = pd.DataFrame({
            "h3": nodes,
            "initial_count": init_counts,
            "initial_share": _share(init_counts),
            "final_count": final_counts,
            "final_share": _share(final_counts),
            "delta_share": _share(final_counts) - _share(init_counts),
            "is_center": is_center_mask,
            "is_hub": hubs_mask,
        }).sort_values(["is_center","final_count"],
                       ascending=[False, False], ignore_index=True)

        tiles_path = os.path.join(csv_dir, "pep_population_by_tile.csv")
        df_tiles.to_csv(tiles_path, index=False)
        print(f"[OK] [WRITE] {tiles_path}")

def plot_center_outflow_time_series(run_dir: str) -> None:
    """
    Plot center outflow / inflow / netflow vs time (one-day view) into
    run_dir/plots/center_flows_timeseries.png

    Expects diag_temporal_center_netflow.csv to exist.
    """
    csv_dir   = os.path.join(run_dir, "csvs")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    path = os.path.join(csv_dir, "diag_temporal_center_netflow.csv")
    if not os.path.exists(path):
        print(f"[WARN] Cannot plot center flows; missing {path}")
        return

    df = pd.read_csv(path)

    # If you ever simulate multiple days, we can either:
    #  - pick first date
    #  - or average by time
    # For now, you have one day → just sort by time.
    df = df.sort_values("time")

    times   = df["time"].astype(str)
    outflow = df["center_outflow"]
    inflow  = df["center_inflow"]
    netflow = df["center_netflow"]

    plt.figure(figsize=(10, 5))
    plt.plot(times, outflow, label="center → non-center (outflow)")
    plt.plot(times, inflow,  label="non-center → center (inflow)")
    plt.plot(times, netflow, label="net flow into center", linestyle="--")

    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Time of day")
    plt.ylabel("Trips per bin")
    plt.title("Center flows vs time")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(plots_dir, "center_flows_timeseries.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] wrote {out_png}")


import h3

def h3_polygon_geojson(h):
    """
    Returns a valid GeoJSON Feature for any H3 hexagon.
    Always lon/lat ordering.
    Never returns None.
    """

    try:
        # h3.cell_to_boundary returns list of (lat, lon)
        boundary = h3.cell_to_boundary(h)
        if not boundary:
            return None
    except Exception:
        return None

    # Convert to lon/lat order
    coords = [[lng, lat] for lat, lng in boundary]

    # Close polygon if not closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return {
        "type": "Feature",
        "properties": {
            "id": h
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        }
    }


def write_pep_population_html_map(
    nodes_df: pd.DataFrame,
    run_dir: str,
):
    """
    Folium HTML map for PEP population by tile.
    """

    import folium
    from folium import GeoJson

    maps_dir = os.path.join(run_dir, "maps")
    csv_dir  = os.path.join(run_dir, "csvs")
    os.makedirs(maps_dir, exist_ok=True)

    tiles_path = os.path.join(csv_dir, "pep_population_by_tile.csv")
    if not os.path.exists(tiles_path):
        print(f"[WARN] Cannot build HTML map: missing {tiles_path}")
        return

    df_tiles = pd.read_csv(tiles_path)

    df_nodes = nodes_df[["h3", "lat", "lon"]].copy()
    df_nodes["h3"] = df_nodes["h3"].astype(str)

    df = df_tiles.merge(df_nodes, on="h3", how="left")

    # pick column
    value_col = PEP_MAP_VALUE_COL
    if PEP_MAP_MODE == "count":
        src = "initial_count" if value_col.startswith("initial") else "final_count"
    else:
        src = value_col if value_col in df.columns else "final_share"

    df["value"] = df[src].astype(float)

    # base map
    m = folium.Map(zoom_start=11, tiles="cartodbpositron")
    m.fit_bounds([
        [df.lat.min(), df.lon.min()],
        [df.lat.max(), df.lon.max()],
    ])

    # color scale
    vmin, vmax = df["value"].min(), df["value"].max()
    colormap = folium.LinearColormap(
        ["#f7fbff", "#6baed6", "#08306b"],
        vmin=vmin, vmax=vmax,
        caption=f"PEP value ({value_col}, mode={PEP_MAP_MODE})",
    )
    colormap.add_to(m)

    # draw polygons
    for _, row in df.iterrows():
        h   = row["h3"]
        val = float(row["value"])
        is_center = int(row["is_center"])
        is_hub    = int(row["is_hub"])

        poly = h3_polygon_geojson(h)
        if poly is None:
            continue

        # correct closure
        def style_fn(feature, v=val, c=is_center, h=is_hub):
            style = {
                "fillColor": colormap(v),
                "color": "#000000",
                "weight": 1,
                "fillOpacity": 0.65,
            }
            if c:
                style["color"] = "#ff0000"
                style["weight"] = 2
            if h:
                style["color"] = "#00ff00"
                style["weight"] = 3
            return style

        tooltip = (
            f"<b>H3:</b> {h}<br>"
            f"<b>value:</b> {val:.4f}<br>"
            f"<b>is_center:</b> {is_center}<br>"
            f"<b>is_hub:</b> {is_hub}"
        )

        GeoJson(poly, style_function=style_fn, tooltip=tooltip).add_to(m)

    out = os.path.join(maps_dir, "pep_population_map.html")
    m.save(out)
    print(f"[OK] wrote {out}")



def compute_mass_timeseries(P_list: List[np.ndarray], X0: np.ndarray) -> List[np.ndarray]:
    """
    Computes mass X(b) for every time bin using the same dynamics
    as the simulator:

        x_{b+1} = P(b) @ x_b

    where P(b) is column-stochastic with P[d, o] = Pr(dest=d | origin=o).
    """
    N = len(X0)
    B = len(P_list)

    mass_ts: List[np.ndarray] = []
    x = X0.astype(float).copy()

    # Normalize once defensively
    s = x.sum()
    if s <= 0:
        raise RuntimeError("X0 sums to zero in compute_mass_timeseries")
    x /= s

    for b in range(B):
        # record current mass
        mass_ts.append(x.copy())

        # advance using column-stochastic dynamics
        P = P_list[b]
        if P.shape != (N, N):
            raise ValueError(f"P_list[{b}] has shape {P.shape}, expected ({N},{N})")

        x = P @ x  # NOTE: no transpose here
        s = x.sum()
        if s > 0:
            x /= s

    return mass_ts


def write_mass_timeseries_html_map(
    nodes_df: pd.DataFrame,
    mass_ts: List[np.ndarray],
    run_dir: str,
):
    """
    Working Folium TimeSliderChoropleth map for mass timeseries.
    """

    import folium
    from folium.plugins import TimeSliderChoropleth

    maps_dir = os.path.join(run_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    tiles = nodes_df["h3"].astype(str).tolist()

    # ---------------------------------------------------
    # Build polygons as full Feature objects with matching IDs
    # ---------------------------------------------------
    features = []
    for h in tiles:
        poly = h3_polygon_geojson(h)
        if poly is None:
            continue

        # Force correct structure
        feature = {
            "type": "Feature",
            "id": h,                       # slider matches on this
            "properties": {"id": h},       # required for Folium
            "geometry": poly["geometry"],  # already lon/lat correctly
        }
        features.append(feature)

    print("[DEBUG] total features:", len(features))

    # ---------------------------------------------------
    # Make the map
    # ---------------------------------------------------
    m = folium.Map(
        location=[nodes_df["lat"].mean(), nodes_df["lon"].mean()],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    m.fit_bounds([
        [nodes_df.lat.min(), nodes_df.lon.min()],
        [nodes_df.lat.max(), nodes_df.lon.max()],
    ])

    # ---------------------------------------------------
    # Global color scale
    # ---------------------------------------------------
    all_vals = np.concatenate(mass_ts)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    cmap = folium.LinearColormap(
        ["#ffffcc", "#41b6c4", "#0c2c84"],
        vmin=vmin,
        vmax=vmax,
    )

    # ---------------------------------------------------
    # Build styledict
    # keys = H3 ID
    # inner keys = timestamp in ms
    # ---------------------------------------------------
    styledict = {}
    BIN_MS = 30 * 60 * 1000   # 30 mins in ms

    for idx, h in enumerate(tiles):
        styledict[h] = {}
        for t, vec in enumerate(mass_ts):
            v = float(vec[idx])
            color = cmap(v) if not np.isnan(v) else "#00000000"
            timestamp = t * BIN_MS

            styledict[h][timestamp] = {
                "color": color,
                "fillColor": color,
                "fillOpacity": 0.75,
                "weight": 0.1,
            }

    # ---------------------------------------------------
    # Assemble FeatureCollection
    # ---------------------------------------------------
    fc = {
        "type": "FeatureCollection",
        "features": features,
    }

    # ---------------------------------------------------
    # Create slider
    # ---------------------------------------------------
    slider = TimeSliderChoropleth(
        fc,
        styledict=styledict,
    )
    slider.add_to(m)

    cmap.caption = "Mass per tile"
    cmap.add_to(m)

    out = os.path.join(maps_dir, "mass_timeseries_map.html")
    m.save(out)
    print(f"[OK] wrote {out}")

def moduleB_mass_no_overlay(dest_h3: str, is_center: bool) -> float:
    """
    Computes destination mass using only MASS_BASE_MODE,
    ignoring schedules and overlay.
    """
    return float(MASS_BASE_MODE[0] if is_center else MASS_BASE_MODE[1])


def write_moduleB_mass_map(nodes_df, centers_set, run_dir, bin_index=0):
    """
    Shows destination mass from Module B (mass_for_destination),
    but WITHOUT any schedule or overlay logic.
    Only MASS_BASE_MODE determines center/periphery weighting.
    """

    import folium
    from folium import GeoJson

    maps_dir = os.path.join(run_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    tiles = nodes_df["h3"].astype(str).tolist()
    is_center_mask = nodes_df["is_center"].astype(bool).tolist()

    # ------------------------------------------
    # Compute static B–mass for each tile
    # ------------------------------------------
    mass_vals = []
    for idx, h in enumerate(tiles):
        val = moduleB_mass_no_overlay(
            dest_h3=h,
            is_center=is_center_mask[idx]
        )
        mass_vals.append(float(val))

    df = nodes_df.copy()
    df["massB"] = mass_vals

    # ------------------------------------------
    # Build map
    # ------------------------------------------
    m = folium.Map(zoom_start=11, tiles="cartodbpositron")
    m.fit_bounds([
        [df.lat.min(), df.lon.min()],
        [df.lat.max(), df.lon.max()],
    ])

    vmin = df["massB"].min()
    vmax = df["massB"].max()

    colormap = folium.LinearColormap(
        ["#f7fbff", "#6baed6", "#08306b"],
        vmin=vmin, vmax=vmax,
        caption="Module-B Destination Mass"
    )
    colormap.add_to(m)

    # ------------------------------------------
    # Draw tiles
    # ------------------------------------------
    for _, row in df.iterrows():
        h = row["h3"]
        v = float(row["massB"])

        poly = h3_polygon_geojson(h)
        if poly is None:
            continue

        def style_fn(feature, vv=v):
            return {
                "fillColor": colormap(vv),
                "color": "#000000",
                "weight": 1,
                "fillOpacity": 0.65,
            }

        tooltip = (
            f"<b>H3:</b> {h}<br>"
            f"<b>mass_B:</b> {v:.2f}<br>"
            f"<b>is_center:</b> {int(row['is_center'])}"
        )

        GeoJson(poly, style_function=style_fn, tooltip=tooltip).add_to(m)

    out_path = os.path.join(maps_dir, "moduleB_mass_map.html")
    m.save(out_path)
    print(f"[OK] wrote {out_path}")



def write_initial_population_map(
    nodes_df: pd.DataFrame,
    X0: np.ndarray,
    run_dir: str,
):
    """
    Draws a static Folium map showing the initial population X0
    (length-N probability distribution used to seed the PEP simulator).

    X0[i] corresponds to nodes_df.iloc[i]["h3"].
    """

    import folium
    from folium import GeoJson

    maps_dir = os.path.join(run_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    if len(X0) != len(nodes_df):
        print("[ERROR] X0 length mismatch; cannot draw initial population map.")
        return
    
    df = nodes_df.copy()
    df["X0"] = X0.astype(float)*TOTAL_PER_BIN_FIXED

    # Build map
    m = folium.Map(zoom_start=11, tiles="cartodbpositron")
    m.fit_bounds([
        [df.lat.min(), df.lon.min()],
        [df.lat.max(), df.lon.max()],
    ])

    vmin = df["X0"].min()
    vmax = df["X0"].max()

    colormap = folium.LinearColormap(
        colors=["#f7fbff", "#6baed6", "#08306b"],
        vmin=vmin,
        vmax=vmax,
        caption="Initial Population X0 (share)",
    )
    colormap.add_to(m)

    # Draw polygons
    for _, row in df.iterrows():
        h = row["h3"]
        val = float(row["X0"])

        poly = h3_polygon_geojson(h)
        if poly is None:
            continue

        def style_fn(feature, vv=val):
            return {
                "fillColor": colormap(vv),
                "color": "#000000",
                "weight": 1,
                "fillOpacity": 0.7,
            }

        tooltip = (
            f"<b>H3:</b> {h}<br>"
            f"<b>X0:</b> {val:.6f}<br>"
            f"<b>is_center:</b> {int(row['is_center'])}<br>"
            f"<b>is_hub:</b> {int(row['is_hub'])}"
        )

        GeoJson(poly, style_function=style_fn, tooltip=tooltip).add_to(m)

    out_path = os.path.join(maps_dir, "initial_population_map.html")
    m.save(out_path)
    print(f"[OK] wrote {out_path}")

def write_fixed_point_population_map(nodes_df, overlay, Q, run_dir):
    """
    Visualizes the fixed point of column-stochastic Q using:
      • Node color = actual population (scaled by TOTAL_PER_BIN_FIXED)
      • Node size  = stationary probability
      • Hubs/centers overlayed as outlines ONLY (never altering node color)
      • Clean integer colorbar
    """
    import folium
    from folium import FeatureGroup
    import os
    import branca.colormap as cm

    # ---------------------------------------------------------
    # Bring global TOTAL_PER_BIN_FIXED
    # ---------------------------------------------------------
    # from __main__ import TOTAL_PER_BIN_FIXED
    # from __main__ import compute_periodic_fixed_point_h3

    maps_dir = os.path.join(run_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. FIXED POINT using your proper solver
    # ---------------------------------------------------------
    x_fp_prob = compute_periodic_fixed_point_h3(Q)            # stationary distribution
    x_fp_count = x_fp_prob * TOTAL_PER_BIN_FIXED              # actual population per bin

    df = nodes_df.copy()
    df["x_fp_prob"]  = x_fp_prob
    df["x_fp_count"] = x_fp_count

    # ---------------------------------------------------------
    # 2. Color scale (population)
    # ---------------------------------------------------------
    vmin = float(df["x_fp_count"].min())
    vmax = float(df["x_fp_count"].max())

    colormap = cm.LinearColormap(
        colors=["#d4f0ff", "#3186cc", "#08306b"],  # light → mid → dark
        vmin=vmin,
        vmax=vmax,
    )
    colormap.caption = "Fixed-Point Population (per cell)"
    colormap.format = lambda x: f"{int(x):,}"

    # ---------------------------------------------------------
    # 3. Base map + layers
    # ---------------------------------------------------------
    m = folium.Map(
        location=[df.lat.mean(), df.lon.mean()],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    layer_nodes   = FeatureGroup("Nodes (fixed point)", overlay=True)
    layer_centers = FeatureGroup("Centers", overlay=True)
    layer_hubs    = FeatureGroup("Hubs", overlay=True)
    layer_edges   = FeatureGroup("Edges (regular)", overlay=True)
    layer_metro   = FeatureGroup("Edges (metro)", overlay=True)


    # ---------------------------------------------------------
    # 4. Draw nodes FIRST (main visual layer)
    # ---------------------------------------------------------
    for _, r in df.iterrows():
        h   = r["h3"]
        lat = r["lat"]
        lon = r["lon"]
        p   = r["x_fp_prob"]
        c   = r["x_fp_count"]

        folium.CircleMarker(
            location=[lat, lon],
            radius=4 + 20 * p,
            color=colormap(c),
            fill=True,
            fill_color=colormap(c),
            fill_opacity=0.9,
            tooltip=(
                f"<b>H3:</b> {h}<br>"
                f"<b>Population:</b> {int(c):,}<br>"
                f"<b>Share:</b> {p*100:.4f}%"
            )
        ).add_to(layer_nodes)

    # ---------------------------------------------------------
    # 5. Draw centers + hubs as NON-INTRUSIVE thin outlines
    # ---------------------------------------------------------
    for _, r in df.iterrows():
        h = r["h3"]
        lat, lon = r["lat"], r["lon"]

        if r["is_center"] == 1:
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color="#ffcc33",
                weight=3,
                fill=False,   # <<< does NOT alter color
                opacity=1.0,
                tooltip=f"<b>Center</b><br>{h}"
            ).add_to(layer_centers)

        if overlay.is_hub(h):
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#33ff33",
                weight=3,
                fill=False,   # <<< does NOT alter color
                opacity=1.0,
                tooltip=f"<b>Hub</b><br>{h}"
            ).add_to(layer_hubs)

    # ---------------------------------------------------------
    # 6. Edges (unchanged)
    # ---------------------------------------------------------
    node_pos = {r["h3"]: (r["lat"], r["lon"]) for _, r in df.iterrows()}

    for (a, b), w in overlay.per_pair.items():
        if a not in node_pos or b not in node_pos:
            continue

        lat1, lon1 = node_pos[a]
        lat2, lon2 = node_pos[b]

        is_metro = (a, b) in overlay.metro_pairs
        layer    = layer_metro if is_metro else layer_edges

        folium.PolyLine(
            [[lat1, lon1], [lat2, lon2]],
            color="#ff0000" if is_metro else "#444444",
            weight=4 if is_metro else 1,
            opacity=0.5,
            tooltip=f"{a} → {b}<br>w={w:.4f}"
        ).add_to(layer)

    # ---------------------------------------------------------
    # 7. Final assembly
    # ---------------------------------------------------------
    layer_nodes.add_to(m)
    layer_centers.add_to(m)
    layer_hubs.add_to(m)
    layer_edges.add_to(m)
    layer_metro.add_to(m)

    colormap.add_to(m)


    folium.LayerControl(collapsed=False).add_to(m)

    style = """
    <style>
    .leaflet-control-colorbar {
        right: 200px !important;
    }
    .leaflet-control-layers {
        right: 200px !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(style))


    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.85);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 0 6px rgba(0,0,0,0.15);
    ">
        Periodic Fixed-Point Population Map
    </div>
    """

    m.get_root().html.add_child(folium.Element(title_html))


    out = os.path.join(maps_dir, "fixed_point_population_map.html")
    m.save(out)
    print(f"[OK] wrote {out}")


def generate_pep_h3(
    nodes_df: pd.DataFrame,
    centers_set: Set[str],
    overlay: CorridorOverlayH3,
    bins_per_day: int,
    run_dir: str
):
    """
    Fully consistent multinomial H3 PEP generator:
    - Builds P(b)
    - Stores P(b), P̄, Q, X₀
    - Simulates TOTAL_PER_BIN_FIXED agents per bin
    - Produces pep_raw.csv and pep_od.csv
    """

    print(f"[INFO] Starting H3 PEP generation ({bins_per_day} bins)")

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    N = len(nodes_df)
    node_list = nodes_df["h3"].astype(str).tolist()

    # ----------------------------------------------------
    # Save node ordering (authoritative basis)
    # ----------------------------------------------------
    nodes_path = os.path.join(run_dir, "npz", "nodes.npy")
    np.save(nodes_path, np.array(node_list, dtype=object))
    print(f"[OK] wrote {nodes_path}")

    # Preload lat/lon arrays
    # ----------------------------------------------------
    # 1. Compute all P(b)
    # ----------------------------------------------------
    # ----------------------------------------------------
    # 1. Compute daily-periodic P(b)
    # ----------------------------------------------------
    print("[INFO] Building P(b) kernels (daily periodic)…")

    # Build exactly one day's worth of P(b)
    P_day = []
    for b in range(bins_per_day):
        if b % 10 == 0:
            print(f"  building P_day[{b}/{bins_per_day}]")

        P = build_P_for_bin_h3(
            bin_index=b,
            nodes_df=nodes_df,
            centers_set=centers_set,
            overlay=overlay
        )
        P_day.append(P)

    # Expand across the whole span using modular indexing
    P_list = [P_day[b % bins_per_day] for b in range(bins_per_day)]


    print("[INFO] Writing Module-B mass map (bin 0)…")
    write_moduleB_mass_map(nodes_df, centers_set, run_dir, bin_index=0)


    # ----------------------------------------------------
    # 2. Save P(b)
    # ----------------------------------------------------
    if STORE_P_PER_BIN:
        npz_path = os.path.join(run_dir, "npz", "P_day0.npz")
        np.savez_compressed(
            npz_path,
            **{f"P_{b}": P_list[b] for b in range(bins_per_day)},
            nodes=np.array(node_list, dtype=object),
            bins_per_day=bins_per_day,
            time_res_min=TIME_RES_MIN
        )
        print(f"[OK] wrote {npz_path}")

    # ----------------------------------------------------
    # 3. Daily mean P̄
    # ----------------------------------------------------
    if STORE_PBAR:
        Pbar = compute_daily_mean_P(P_list)
        pbar_path = os.path.join(run_dir, "npz", "Pbar.npy")
        np.save(pbar_path, Pbar)
        print(f"[OK] wrote {pbar_path}")
    else:
        Pbar = None

    # store Q
    Q = compute_daily_operator_h3(P_list)
    if STORE_Q:
        qpath = os.path.join(run_dir,"npz","Q.npy")
        np.save(qpath, Q)
        print(f"[OK] wrote {qpath}")


    # NEW: write fixed point population map (true stationary distribution)
    # after computing Q
    write_fixed_point_population_map(
        nodes_df=nodes_df,
        overlay=overlay,
        Q=Q,
        run_dir=run_dir,
    )



    # store X0
    if STORE_X0:
        if INIT_X_MODE == "flat":
            X0 = np.ones(N)/N
            print("[INFO] X0 = flat")
        elif INIT_X_MODE == "periodic_fixed_point":
            print("[INFO] computing periodic fixed point of Q")
            X0 = compute_periodic_fixed_point_h3(Q)
        else:
            raise ValueError(INIT_X_MODE)

        x0path = os.path.join(run_dir,"npz","X0.npy")
        np.save(x0path, X0)
        print(f"[OK] wrote {x0path}")

        # NEW — draw initial population map
        write_initial_population_map(nodes_df, X0, run_dir)

        x0path = os.path.join(run_dir,"npz","X0.npy")
        np.save(x0path, X0)
        print(f"[OK] wrote {x0path}")
    else:
        # If not storing, still need X0 for simulation; default to flat
        X0 = np.ones(N)/N

    print("[DEBUG] X0 stats: min", X0.min(), "max", X0.max())
    print("[DEBUG] top 5 X0:", np.sort(X0)[-5:])


    # Compute time-series mass for debugging
    print("[INFO] Computing mass timeseries X(b)…")
    mass_ts = compute_mass_timeseries(P_list, X0)



    # Build interactive slider map
    write_mass_timeseries_html_map(
        nodes_df=nodes_df,
        mass_ts=mass_ts,
        run_dir=run_dir,
    )

    # --- NEW: simulate PEP raw trips using multinomial ---
    print("[INFO] Simulating multinomial PEP moves…")
    simulate_pep_h3(
        P_list      = P_list,
        X0          = X0,
        nodes_df    = nodes_df,
        centers_set = centers_set,
        overlay     = overlay,
        run_dir     = run_dir,
        total_per_bin = TOTAL_PER_BIN_FIXED,
        seed          = SEED,
    )

    print("[OK] PEP generation + simulation complete")


    # --- NEW: diagnostics / plots / maps that depend on simulated CSVs ---
    if ENABLE_CENTER_OUTFLOW_PLOT:
        plot_center_outflow_time_series(run_dir)

    if MAKE_PEP_MAP:
        write_pep_population_html_map(
            nodes_df=nodes_df,
            run_dir=run_dir
        )

    # if MAKE_MASS_MAP:
    #     write_mass_timeseries_map_h3(
    #         nodes_df=nodes_df,
    #         centers_set=centers_set,
    #         overlay=overlay,
    #         bins_per_day=bins_per_day,
    #         run_dir=run_dir,
    #     )

    print("[OK] PEP generation + simulation complete")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print(f"[INFO] MODE={MODE}")
    if GRAPH_MODE == "corridors":
        manifest_path = os.path.join(OUTPUT_DIR,"corridors_outputs_h3",f"corridors_manifest_{sanitize_region(REGION_NAME.split(',')[0])}_mode-h3_fdr-{FEEDER_MODE}_edge-{edge_tag()}_ovr-{H3_OVERLAY_FLAG}.json")
    else:
        manifest_path = os.path.join(OUTPUT_DIR,"corridors_outputs_h3",f"grid_manifest_{sanitize_region(REGION_NAME.split(',')[0])}_mode-h3.json")
    print(f"[INFO] Loading manifest: {manifest_path}")

    overlay = CorridorOverlayH3()
    overlay.load(manifest_path)
    if not overlay.loaded:
        print("[ERROR] Manifest not loaded.")
        return

    print("===== H3 FEEDER LABEL SUMMARY =====")
    print("Total edges in manifest:", len(overlay.per_pair))
    print("Feeder edges:", len(overlay.feeder_label))
    print("  primary:", sum(1 for x in overlay.feeder_label.values() if x == 1))
    print("  secondary:", sum(1 for x in overlay.feeder_label.values() if x == 2))
    print("Metro edges:", len(overlay.metro_pairs))
    print("===================================")



    nodes_df = load_nodes_from_h3_manifest(overlay.manifest)

    print("[DEBUG] nodes_df shape:", nodes_df.shape)
    print(nodes_df.head())
    print(nodes_df["lat"].isna().sum(), "lat NaN")
    print(nodes_df["lon"].isna().sum(), "lon NaN")



    # centers_set = set(nodes_df.loc[nodes_df["is_center"]!=0, "h3"].astype(str))

    # ==============================================================
    # OVERRIDE: centers come ONLY from CONFIG, never from manifest
    # ==============================================================

    print("[INFO] Overriding centers using CENTER_FIXED_IDS (manifest centers ignored).")

    nodes_df["is_center"] = nodes_df["h3"].astype(str).isin(CENTER_FIXED_IDS).astype(int)

    centers_set = set(CENTER_FIXED_IDS)

    print(f"[INFO] centers_set size: {len(centers_set)}")
    missing = centers_set - set(nodes_df["h3"].astype(str))
    if missing:
        print("[WARN] Some CENTER_FIXED_IDS were not found in nodes_df:", missing)
    else:
        print("[INFO] All CENTER_FIXED_IDS found in nodes_df.")


    if MODE == "validate":
        print("[INFO] Validate mode: stop after manifest load.")
        return

    run_dir = make_h3_run_directory()

    generate_pep_h3(
        nodes_df,
        centers_set,
        overlay,
        BINS_PER_DAY,
        run_dir
    )


if __name__ == "__main__":
    main()
