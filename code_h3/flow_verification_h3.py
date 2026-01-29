#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flow_verification_h3.py — End-to-end verification of H3 PEP vs. OD matrix products.

Author: Asif Shakeel
Email: ashakeel@ucsd.edu

Python 3.10+

This script validates the consistency between:
  • The multi-step transition matrix obtained by chaining OD matrices, and
  • The empirical end-to-end transition matrix reconstructed from PEP agents.

It computes both unweighted and population-weighted metrics:
  - L1 matrix distance
  - RMSE
  - Mean column L1
  - Mean column Jensen–Shannon divergence
  - Weighted L1
  - Weighted RMSE
  - Weighted JS

Inputs (from a completed generator run):
  • pep_od.csv   — time-binned OD matrices
  • pep_raw.csv  — agent trajectories

Outputs:
  • P_pep_end_to_end.csv
  • P_chain_from_OD.csv
  • verification_summary.csv
"""


import os
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG  (identical to generator)
# ==========================================================
BASE_DIR = "/Users/asif/Documents/nm24/outputs"
RUN_DIR  = os.path.join(BASE_DIR, "runs")
COUNTRY     = "USA"
REGION_NAME = "Atlanta, Georgia"

H3_RES = 6
NEIGHBOR_TOPOLOGY = "6"
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "pot"
INCLUDE_BACKGROUND_EDGES = False
INIT_X_MODE = "periodic_fixed_point"
GRAPH_MODE =  "corridors"   # "corridors" | "grid"

TIME_RES_MIN = 30

SPAN_START   = "2025-06-01 06:00:00"
SPAN_END     = "2025-06-01 09:00:00"
PEP_SPAN_START   = "2025-06-01 00:00:00"
PEP_SPAN_END     = "2025-06-30 00:00:00"

START_COL = "start_h3"
END_COL   = "end_h3"
DATE_COL  = "date"
TIME_COL  = "time"

EPS_SMOOTH = 1e-12

# Build tags (unchanged)
def _city_tag(): return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"
def _top_tag(): return f"h3res-{NEIGHBOR_TOPOLOGY}"
def _fdr_tag(): return f"fdr-{FEEDER_MODE}"
def _edge_tag(): return f"edge-{EDGE_DIRECTION_MODE}"
def _overlay_tag(): return "ovr-1" if INCLUDE_BACKGROUND_EDGES else "ovr-0"

city_tag = _city_tag()
top_tag  = _top_tag()
fdr_tag  = _fdr_tag()
edge_tag = _edge_tag()
overlay_tag = _overlay_tag()

PEP_SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
PEP_SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)
start_tag = PEP_SPAN_START_TS.strftime("%Y%m%dT%H%M")
end_tag   = PEP_SPAN_END_TS.strftime("%Y%m%dT%H%M")


if GRAPH_MODE == "corridors":
    RUN_NAME = (
        f"{city_tag}/h3res-{H3_RES}/graph-{GRAPH_MODE}/{fdr_tag}/{edge_tag}/{overlay_tag}/"
        f"{start_tag}_{end_tag}/m{TIME_RES_MIN}/x-{INIT_X_MODE}" 
    )
else:
    RUN_NAME = (
        f"{city_tag}/h3res-{H3_RES}/graph-{GRAPH_MODE}/"
        f"{start_tag}_{end_tag}/m{TIME_RES_MIN}/x-{INIT_X_MODE}" 
    )



DATA_DIR = os.path.join(RUN_DIR, RUN_NAME, "csvs")
OUTPUT_DIR = os.path.join(RUN_DIR, RUN_NAME, "flow_verification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OD_PATH  = os.path.join(DATA_DIR, "pep_od.csv")
RAW_PATH = os.path.join(DATA_DIR, "pep_raw.csv")

# ==========================================================
# Helpers (unchanged)
# ==========================================================
def build_time_grid(start, end, dt_min):
    out = []
    t = start
    dt = pd.Timedelta(minutes=dt_min)
    while t < end:
        out.append(t)
        t += dt
    return out

def load_M_stack(od_path, span_start, span_end, dt_min):
    od = pd.read_csv(od_path)
    od["_ts"] = pd.to_datetime(od[DATE_COL].astype(str) + " " + od[TIME_COL])

    ts_list = build_time_grid(span_start, span_end, dt_min)
    mask = (od["_ts"] >= span_start) & (od["_ts"] < span_end)
    od_span = od.loc[mask]

    starts = od_span[START_COL].astype(str).unique()
    ends   = od_span[END_COL].astype(str).unique()
    nodes  = sorted(set(starts) | set(ends))
    N = len(nodes)
    node_to_idx = {g:i for i,g in enumerate(nodes)}

    B = len(ts_list)
    M_stack = np.zeros((B, N, N))

    for b, ts in enumerate(ts_list):
        print(f"\r[load_M_stack] Processing time bin {b+1}/{B}", end="", flush=True)

        sub = od_span.loc[od_span["_ts"] == ts]
        M = np.zeros((N, N))
        for _, r in sub.iterrows():
            j = node_to_idx[r[START_COL]]
            i = node_to_idx[r[END_COL]]
            M[i, j] += r["trip_count"]

        for j in range(N):
            s = M[:, j].sum()
            if s > 0:
                M[:, j] /= s
            else:
                M[j, j] = 1.0

        M_stack[b] = M
    print()
    return M_stack, ts_list, nodes

def load_P_pep(raw_path, span_start, span_end, nodes):
    df = pd.read_csv(raw_path)
    df["_ts"] = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL])

    df_span = df.loc[
        (df["_ts"] >= span_start) & (df["_ts"] < span_end)
    ].sort_values(["PEP_ID","_ts"])

    node_to_idx = {g:i for i,g in enumerate(nodes)}
    N = len(nodes)

    F = np.zeros((N, N))
    x0 = np.zeros(N)
    unique_peps = df_span["PEP_ID"].nunique()

    for k, (pid, grp) in enumerate(df_span.groupby("PEP_ID")):
        print(f"\r[load_P_pep] Processed {k+1}/{unique_peps} PEPs", end="", flush=True)

        first = grp.iloc[0]
        last  = grp.iloc[-1]
        j = node_to_idx.get(first[START_COL], None)
        i = node_to_idx.get(last[END_COL], None)
        if j is None or i is None:
            continue
        F[i,j] += 1
        x0[j]  += 1

    P = np.zeros((N,N))
    for j in range(N):
        if x0[j] > 0:
            P[:,j] = F[:,j] / x0[j]
        else:
            P[j,j] = 1.0
    print()

    return P, x0

def multiply_chain(M_stack):
    P = np.eye(M_stack.shape[1])
    for k, M in enumerate(M_stack):
        print(f"\r[multiply_chain] Step {k+1}/{len(M_stack)}", end="", flush=True)
        P = M @ P
    print()
    return P

# ==========================================================
# METRICS (updated: includes weighted metrics)
# ==========================================================
def metrics(Pa, Pb, x0, eps=1e-12):
    D = Pa - Pb

    # original unweighted
    l1 = float(np.sum(np.abs(D)))
    rmse = float(np.sqrt(np.mean(D**2)))
    l1_cols = float(np.mean(np.sum(np.abs(D), axis=0)))

    # JS (unweighted)
    A = Pa + eps
    B = Pb + eps
    A /= A.sum(axis=0, keepdims=True)
    B /= B.sum(axis=0, keepdims=True)
    M = 0.5*(A + B)
    col_JS = (
        0.5*np.sum(A*np.log(A/M), axis=0) +
        0.5*np.sum(B*np.log(B/M), axis=0)
    )
    JS = float(np.mean(col_JS))

    # weighted metrics
    w = x0 + eps
    w = w / w.sum()

    col_L1  = np.sum(np.abs(D), axis=0)
    col_MSE = np.mean(D**2, axis=0)

    l1_w   = float(np.sum(w * col_L1))
    rmse_w = float(np.sqrt(np.sum(w * col_MSE)))
    js_w   = float(np.sum(w * col_JS))

    return (
        l1, rmse, l1_cols, JS,
        l1_w, rmse_w, js_w
    )

# ==========================================================
# MAIN
# ==========================================================
def main():
    span_start = pd.to_datetime(SPAN_START)
    span_end   = pd.to_datetime(SPAN_END)

    print("\n==============================================")
    print("H3 Mprod vs PEP — Weighted Transition Metrics")
    print("==============================================")
    print("DATA_DIR:", DATA_DIR)

    M_stack, ts_list, nodes = load_M_stack(
        OD_PATH, span_start, span_end, TIME_RES_MIN
    )
    print(f"Loaded OD matrices: B={len(ts_list)}, N={len(nodes)}")

    P_pep, x0 = load_P_pep(
        RAW_PATH, span_start, span_end, nodes
    )
    print("Loaded RAW end-to-end matrix.")

    P_chain = multiply_chain(M_stack)
    print("Built multi-step OD chain.")

    (
        l1, rmse, l1col, js,
        l1_w, rmse_w, js_w
    ) = metrics(P_pep, P_chain, x0, EPS_SMOOTH)

    print("\n=== SUMMARY ===")
    print(f"Bins={len(ts_list)}, N={len(nodes)}")
    print(f"L1_matrix     = {l1:.6f}")
    print(f"RMSE          = {rmse:.6f}")
    print(f"L1_col_mean   = {l1col:.6f}")
    print(f"JS_col_mean   = {js:.6f}")
    print(f"L1_weighted   = {l1_w:.6f}")
    print(f"RMSE_weighted = {rmse_w:.6f}")
    print(f"JS_weighted   = {js_w:.6f}")

    pd.DataFrame(P_pep,   index=nodes, columns=nodes).to_csv(
        os.path.join(OUTPUT_DIR, "P_pep_end_to_end.csv")
    )
    pd.DataFrame(P_chain, index=	nodes, columns=nodes).to_csv(
        os.path.join(OUTPUT_DIR, "P_chain	from_OD.csv")
    )

    print("\n[OK] wrote P_pep_end_to_end.csv and P_chain_from_OD.csv")

    # ==========================================================
    # WRITE VERIFICATION SUMMARY CSV
    # ==========================================================
    rows = [
        ("region", REGION_NAME),
        ("graph_mode", GRAPH_MODE),
        ("h3_resolution", H3_RES),
        ("time_res_min", TIME_RES_MIN),
        ("span_start", SPAN_START),
        ("span_end", SPAN_END),
        ("num_bins", len(ts_list)),
        ("num_nodes", len(nodes)),
        ("num_peps", int(x0.sum())),

        ("l1_matrix", l1),
        ("rmse", rmse),
        ("l1_col_mean", l1col),
        ("js_col_mean", js),

        ("l1_weighted", l1_w),
        ("rmse_weighted", rmse_w),
        ("js_weighted", js_w),
    ]

    summary_path = os.path.join(OUTPUT_DIR, "verification_summary.csv")

    df_out = pd.DataFrame(rows, columns=["field", "value"])
    df_out.to_csv(summary_path, index=False)

    print(f"[OK] wrote vertical verification summary to {summary_path}")


if __name__ == "__main__":
    main()
