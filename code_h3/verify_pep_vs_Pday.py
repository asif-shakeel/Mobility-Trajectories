#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG (identical to generator)
# ==========================================================
BASE_DIR = "/Users/asif/Documents/nm24/outputs"
RUN_DIR  = os.path.join(BASE_DIR, "runs")

REGION_NAME = "Atlanta, Georgia"
H3_RES = 6
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "pot"
INCLUDE_BACKGROUND_EDGES = False
INIT_X_MODE = "periodic_fixed_point"
GRAPH_MODE = "corridors"

TIME_RES_MIN = 30

SPAN_START = "2025-06-01 06:00:00"
SPAN_END   = "2025-06-01 09:00:00"

PEP_SPAN_START = "2025-06-01 00:00:00"
PEP_SPAN_END   = "2025-06-02 00:00:00"

START_COL = "start_h3"
END_COL   = "end_h3"
DATE_COL  = "date"
TIME_COL  = "time"

EPS_SMOOTH = 1e-12

# ----------------------------------------------------------
# Path helpers
# ----------------------------------------------------------
def _city_tag():
    return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"

def _overlay_tag():
    return "ovr-1" if INCLUDE_BACKGROUND_EDGES else "ovr-0"

city_tag = _city_tag()
overlay_tag = _overlay_tag()

PEP_SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
PEP_SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)

start_tag = PEP_SPAN_START_TS.strftime("%Y%m%dT%H%M")
end_tag   = PEP_SPAN_END_TS.strftime("%Y%m%dT%H%M")

RUN_NAME = (
    f"{city_tag}/h3res-{H3_RES}/graph-{GRAPH_MODE}/"
    f"fdr-{FEEDER_MODE}/edge-{EDGE_DIRECTION_MODE}/{overlay_tag}/"
    f"{start_tag}_{end_tag}/m{TIME_RES_MIN}/x-{INIT_X_MODE}"
)

DATA_DIR = os.path.join(RUN_DIR, RUN_NAME, "csvs")
NPZ_DIR  = os.path.join(RUN_DIR, RUN_NAME, "npz")
OUT_DIR  = os.path.join(RUN_DIR, RUN_NAME, "pep_vs_Pday_flow_verification")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH    = os.path.join(DATA_DIR, "pep_raw.csv")
P_DAY_PATH  = os.path.join(NPZ_DIR, "P_day0.npz")
NODES_PATH  = os.path.join(NPZ_DIR, "nodes.npy")

# ==========================================================
# Helpers
# ==========================================================
def minute_of_day(ts):
    return ts.hour * 60 + ts.minute

def bin_index_from_ts(ts, dt_min):
    return minute_of_day(ts) // dt_min

def load_P_pep(raw_path, span_start, span_end, nodes):
    df = pd.read_csv(raw_path)
    df["_ts"] = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL])

    df_span = df.loc[
        (df["_ts"] >= span_start) & (df["_ts"] < span_end)
    ].sort_values(["PEP_ID", "_ts"])

    node_to_idx = {g: i for i, g in enumerate(nodes)}
    N = len(nodes)

    F = np.zeros((N, N))
    x0 = np.zeros(N)

    for pid, grp in df_span.groupby("PEP_ID"):
        first = grp.iloc[0]
        last  = grp.iloc[-1]
        j = node_to_idx.get(first[START_COL])
        i = node_to_idx.get(last[END_COL])
        if j is None or i is None:
            continue
        F[i, j] += 1
        x0[j] += 1

    P = np.zeros((N, N))
    for j in range(N):
        if x0[j] > 0:
            P[:, j] = F[:, j] / x0[j]
        else:
            P[j, j] = 1.0

    return P, x0

def multiply_chain(M_list):
    P = np.eye(M_list[0].shape[0])
    for M in M_list:
        P = M @ P
    return P

def metrics(Pa, Pb, x0, eps=1e-12):
    D = Pa - Pb

    l1 = float(np.sum(np.abs(D)))
    rmse = float(np.sqrt(np.mean(D**2)))
    l1_cols = float(np.mean(np.sum(np.abs(D), axis=0)))

    A = Pa + eps
    B = Pb + eps
    A /= A.sum(axis=0, keepdims=True)
    B /= B.sum(axis=0, keepdims=True)
    M = 0.5 * (A + B)
    col_JS = (
        0.5 * np.sum(A * np.log(A / M), axis=0) +
        0.5 * np.sum(B * np.log(B / M), axis=0)
    )
    JS = float(np.mean(col_JS))

    w = x0 + eps
    w /= w.sum()

    col_L1 = np.sum(np.abs(D), axis=0)
    col_MSE = np.mean(D**2, axis=0)

    l1_w = float(np.sum(w * col_L1))
    rmse_w = float(np.sqrt(np.sum(w * col_MSE)))
    js_w = float(np.sum(w * col_JS))

    return l1, rmse, l1_cols, JS, l1_w, rmse_w, js_w

# ==========================================================
# MAIN
# ==========================================================
def main():
    span_start = pd.to_datetime(SPAN_START)
    span_end   = pd.to_datetime(SPAN_END)

    nodes = np.load(NODES_PATH, allow_pickle=True).tolist()
    npz = np.load(P_DAY_PATH, allow_pickle=True)
    bins_per_day = int(npz["bins_per_day"])
    P_day = [npz[f"P_{b}"] for b in range(bins_per_day)]

    bins = []
    t = span_start
    while t < span_end:
        bins.append(bin_index_from_ts(t, TIME_RES_MIN))
        t += pd.Timedelta(minutes=TIME_RES_MIN)
    bins = sorted(set(bins))

    P_pep, x0 = load_P_pep(RAW_PATH, span_start, span_end, nodes)
    P_chain = multiply_chain([P_day[b] for b in bins])

    results = metrics(P_pep, P_chain, x0, EPS_SMOOTH)

    print("\n=== METRICS ===")
    for name, val in zip(
        ["L1", "RMSE", "L1_col", "JS", "L1_w", "RMSE_w", "JS_w"],
        results
    ):
        print(f"{name}: {val:.6e}")

    (
        l1, rmse, l1col, js,
        l1_w, rmse_w, js_w
    ) = results

    rows = [
        ("region", REGION_NAME),
        ("graph_mode", GRAPH_MODE),
        ("h3_resolution", H3_RES),
        ("time_res_min", TIME_RES_MIN),
        ("span_start", SPAN_START),
        ("span_end", SPAN_END),
        ("num_bins", len(bins)),
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

    summary_path = os.path.join(OUT_DIR, "verification_summary.csv")
    pd.DataFrame(rows, columns=["field", "value"]).to_csv(
        summary_path, index=False
    )

if __name__ == "__main__":
    main()
