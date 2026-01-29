#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_builder_h3.py
-------------------

Synthetic corridor / feeder network generator on an H3 grid.

This script constructs a directed, weighted *synthetic mobility overlay*
on top of a base H3 adjacency graph for a given urban region.

The output is a network that mimics hierarchical urban mobility structure:
    • A central anchor (root)
    • Multiple concentric hub rings
    • Hub–center backbones
    • Primary (and optional secondary) feeders
    • Metro-style hub–hub links
    • Optional background grid edges

The result is a sparse, structured overlay graph that can be used as a
*transport substrate* for Markov mobility models, effective distance,
return-to-origin, corridor analysis, and other network-level mobility
measures.

----------------------------------------------------------------------
CONCEPTUAL PIPELINE
----------------------------------------------------------------------

1) Region & grid
   - Load region polygon from OSMnx.
   - Optional buffering.
   - Tile with H3 hexagons at H3_RESOLUTION.
   - Build base H3 adjacency.

2) Geometric anchor & center
   - Choose a geometric anchor (H3 or centroid).
   - Define the synthetic city center.

3) Hub placement (ring model)
   - Partition nodes into radial bands.
   - Sample hubs per ring with spacing constraints.

4) Backbone construction
   - For each hub, compute shortest path to center.
   - Paths define permanent backbone edges.

5) Feeder construction
   - Every non-hub node connects to nearest hub.
   - Optional secondary feeders (dual mode).

6) Metro tree
   - Connect hubs by a geometric spanning tree.

7) Overlay graph
   - Overlay = backbone + feeders + metro.
   - Optional background grid edges.

8) Direction assignment
   - Potential-field (hop distance from center), or
   - Geometric radial gradient.

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written to:

    outputs/corridors_outputs_h3/

with a filename tag:

    {city}_mode-h3_{fdr}_{edge}_{ovr}

The generator produces:

1) Network manifest (JSON)
    corridors_manifest_{city}_mode-h3_{fdr}_{edge}_{ovr}.json

   Contains:
     • node coordinates and hub labels
     • center anchor and potentials
     • directed edges with delta_T, feeder_label, overlay_flag
     • metro and backbone flags

2) Node table (CSV)
    corridor_potential_nodes_{city}_mode-h3_{fdr}_{edge}_{ovr}.csv

3) Edge table (CSV)
    corridor_potential_edges_{city}_mode-h3_{fdr}_{edge}_{ovr}.csv

4) Interactive maps (HTML + PNG)
    corridor_AM_map_{city}_mode-h3_{fdr}_{edge}_{ovr}.html
    corridor_PM_map_{city}_mode-h3_{fdr}_{edge}_{ovr}.html
    (PNG snapshots are auto-generated via Playwright)

5) Structural diagnostics
    max_hop_diameter_{city}_mode-h3_{fdr}_{edge}_{ovr}.csv
        – maximum hop distance in overlay graph

    corridor_feeder_diagnostics_{city}_mode-h3_{fdr}_{edge}_{ovr}.csv
    corridor_neighbor_distances_{city}_mode-h3_{fdr}_{edge}_{ovr}.csv
        – optional feeder and adjacency diagnostics

6) GRID mode outputs (if GRAPH_MODE="grid")

    grid_manifest_{city}_mode-h3.json
    grid_nodes_{city}_mode-h3.csv
    grid_edges_{city}_mode-h3.csv
    grid_map_{city}_mode-h3.html

----------------------------------------------------------------------
INTENDED USE
----------------------------------------------------------------------

This overlay is NOT a real road network.
It is a *synthetic structural prior* used to:

    • define Markov transition operators
    • compute effective distance
    • simulate time-elapsed mobility
    • study corridor persistence and accessibility
    • avoid individual-level trajectory assumptions

The generator is fully determined by its configuration
and random seed.

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


from __future__ import annotations
import os, math, json, random
from collections import defaultdict, deque
from typing import Optional

import h3
from shapely.ops import transform
from pyproj import CRS, Transformer

import folium
from folium import FeatureGroup
from folium.plugins.fullscreen import Fullscreen
# ============================================================
# CONFIG
# ============================================================

BASE_DIR   = "/Users/asif/Documents/nm24"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "outputs/corridors_outputs_h3")
os.makedirs(OUT_DIR, exist_ok=True)

REGION_NAME = "Atlanta, Georgia" #"Mexico City, Mexico"
REGION_POLY_PATH  = os.path.join(DATA_DIR, "region_boundary.geojson")
REGION_BUFFER_KM: Optional[float] = 20

H3_RESOLUTION = 6
SEED = 4012

GEOM_CENTER_MODE = "h3"   # or "centroid"
GEOM_CENTER_H3 = '8644c1aafffffff' # "864995b87ffffff"
HUB_CENTER_LAT = None
HUB_CENTER_LON = None
GEOM_TOL_KM = 0.5
FORCE_GEOM_CENTER_AS_HUB = False

GRAPH_MODE = "corridors"  # "corridors" | "grid"
FEEDER_MODE = "single"  # or "dual"
EDGE_DIRECTION_MODE = "potential"  # or "geometric"
# Whether to include background grid edges in outputs (in addition to corridor tree)
INCLUDE_BACKGROUND_EDGES = False


RING_RADII_KM = [0.0, 10, 20, 30, 40, 55]
HUBS_PER_RING = [3, 4, 5, 6, 7]
MIN_HUB_SEP_KM = [7, 7, 14, 14, 14]
ENFORCE_GLOBAL_HUB_SEPARATION = True
GLOBAL_MIN_HUB_SEP_KM = 7.0

HUB_RING_COLORS = [
    "#b30000","#ff7f0e","#1f77b4","#2ca02c","#9467bd",
]

NODE_RADIUS_HUB = 7
NODE_RADIUS_NON = 3
NODE_COLOR = "#555"
HUB_COLOR  = "#b30000"

kept_edges_undirected = set()
feeder_label = {}  # (u,v)->label
hub_backbone_flag = {}   # (u,v) → 1 if edge came from hub-backbone construction

# ============================================================
# UTILITIES
# ============================================================
# tags for filenames
def _city_tag() -> str:
    return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"

def _mode_tag() -> str:
    return "mode-h3"

def _feeder_tag() -> str:
    return f"fdr-{str(FEEDER_MODE).lower()}"

def _edge_dir_tag():
    if EDGE_DIRECTION_MODE == "potential":
        return "edge-pot"
    elif EDGE_DIRECTION_MODE == "geometric":
        return "edge-geom"
    else:
        return f"edge-{EDGE_DIRECTION_MODE}"

def _overlay_tag() -> str:
    return "ovr-1" if INCLUDE_BACKGROUND_EDGES else "ovr-0"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    a1 = math.radians(lat1)
    b1 = math.radians(lon1)
    a2 = math.radians(lat2)
    b2 = math.radians(lon2)
    dlat = a2 - a1
    dlon = b2 - b1
    h = (math.sin(dlat/2)**2 +
         math.cos(a1)*math.cos(a2)*math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(h))

def buffer_polygon_km(geom, buffer_km):
    if not buffer_km or buffer_km <= 0:
        return geom
    centroid = geom.centroid
    lon0, lat0 = centroid.x, centroid.y
    crs_wgs84 = CRS.from_epsg(4326)
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs(crs_wgs84, crs_local, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(crs_local, crs_wgs84, always_xy=True).transform
    geom_local = transform(to_local, geom)
    geom_buf = geom_local.buffer(buffer_km * 1000.0)
    return transform(to_wgs84, geom_buf)

import osmnx as ox

def load_region_polygon():
    """
    ALWAYS fetch region polygon fresh from OSMnx.
    REGION_NAME must be something Nominatim can resolve.
    """
    print(f"[INFO] Fetching region boundary from OSMnx: {REGION_NAME}")

    gdf = ox.geocode_to_gdf(REGION_NAME)
    poly = gdf.iloc[0].geometry

    # buffer if needed
    poly = buffer_polygon_km(poly, REGION_BUFFER_KM)

    return poly, poly.bounds


from shapely.geometry import Polygon

def h3_cell_polygon(h):
    boundary = h3.cell_to_boundary(h)  # list of (lat, lon)
    coords = [(lon, lat) for (lat, lon) in boundary]
    return Polygon(coords)


def build_h3_grid(poly, res):
    """
    Build H3 grid by keeping ALL H3 cells whose hex polygon intersects the region polygon.
    """
    outer = [(lat, lon) for (lon, lat) in poly.exterior.coords]

    # Get all candidate H3 cells (loose bounding)
    candidates = h3.polygon_to_cells(h3.LatLngPoly(outer), res=res)

    nodes = {}

    for c in candidates:
        cell_poly = h3_cell_polygon(c)

        # Keep the cell if its hex polygon intersects the region polygon
        if poly.intersects(cell_poly):
            lat, lon = h3.cell_to_latlng(c)
            nodes[c] = {"lat": lat, "lon": lon}

    print(f"[INFO] H3 grid: {len(nodes)} cells at resolution={res}")
    return nodes


def build_h3_adjacency(nodes):
    adj=defaultdict(list)
    S=set(nodes)
    for c in nodes:
        neigh=h3.grid_disk(c,1)
        for v in neigh:
            if v!=c and v in S:
                adj[c].append(v)
    return adj

def build_grid_edges(nodes, adj):
    """
    Pure grid topology, but EDGE SCHEMA MATCHES corridor mode.
    Every adjacency edge is an overlay edge in GRID mode.
    """
    edges = []

    for u in nodes:
        for v in adj.get(u, []):
            edges.append({
                "start_h3": u,
                "end_h3": v,
                "origin": u,
                "dest": v,

                "w_norm": 1.0,
                "delta_T": None,
                "delta_geom": 0,
                "delta_geom_km": 0.0,

                "am_pref": 0,
                "pm_pref": 0,

                "is_metro": 0,
                "feeder_label": 0,
                "overlay_flag": 1,        # ✅ GRID ⇒ ALL edges are overlay
                "hub_backbone": 0,
            })

    print(f"[INFO] Grid directed edges: {len(edges)}")
    return edges

# ============================================================
# CENTER SELECTION
# ============================================================

def choose_anchor_and_center(nodes):
    global HUB_CENTER_LAT, HUB_CENTER_LON

    lats=[nd["lat"] for nd in nodes.values()]
    lons=[nd["lon"] for nd in nodes.values()]
    cent_lat=sum(lats)/len(lats)
    cent_lon=sum(lons)/len(lons)

    if GEOM_CENTER_MODE=="h3":
        center_cell = GEOM_CENTER_H3
        anchor_lat,anchor_lon = h3.cell_to_latlng(center_cell)
    else:
        anchor_lat,anchor_lon = cent_lat,cent_lon
        best=None; center_cell=None
        for c,nd in nodes.items():
            d=haversine_m(anchor_lat,anchor_lon,nd["lat"],nd["lon"])
            if best is None or d<best:
                best=d; center_cell=c

    if HUB_CENTER_LAT is not None and HUB_CENTER_LON is not None:
        anchor_lat=float(HUB_CENTER_LAT)
        anchor_lon=float(HUB_CENTER_LON)

    HUB_CENTER_LAT=anchor_lat
    HUB_CENTER_LON=anchor_lon
    return anchor_lat,anchor_lon,center_cell

# ============================================================
# HUB PLACEMENT
# ============================================================

def place_hubs(nodes, center_lat, center_lon):
    """
    Hub placement that *only* selects hubs from existing tiles.
    If a ring band contains fewer available tiles than required,
    we pick as many as exist. Never create hubs outside the region.
    """

    # compute radial distances of all nodes
    radii_km = {
        c: haversine_m(center_lat, center_lon, nd["lat"], nd["lon"]) / 1000.0
        for c, nd in nodes.items()
    }

    hubs = []
    hub_ring_index = {}
    used = set()

    for k in range(len(RING_RADII_KM) - 1):
        r0, r1 = RING_RADII_KM[k], RING_RADII_KM[k+1]
        n_hubs = HUBS_PER_RING[k]
        sep_km = MIN_HUB_SEP_KM[k]

        # --- only pick tiles inside region & ring band ---
        candidates = [
            c for c in nodes
            if r0 <= radii_km[c] < r1
        ]

        # if empty, skip ring entirely
        if not candidates:
            print(f"[WARN] No tiles in ring {k} (r=[{r0},{r1}]) → skipping.")
            continue

        random.shuffle(candidates)

        chosen = []
        for c in candidates:
            if len(chosen) >= n_hubs:
                break

            lat_c, lon_c = nodes[c]["lat"], nodes[c]["lon"]

            # local spacing within ring
            ok = True
            for h in chosen:
                if haversine_m(lat_c, lon_c, nodes[h]["lat"], nodes[h]["lon"]) / 1000 < sep_km:
                    ok = False
                    break
            if not ok:
                continue

            # global spacing (optional)
            if ENFORCE_GLOBAL_HUB_SEPARATION:
                for h in used:
                    if haversine_m(lat_c, lon_c, nodes[h]["lat"], nodes[h]["lon"]) / 1000 < GLOBAL_MIN_HUB_SEP_KM:
                        ok = False
                        break
                if not ok:
                    continue

            chosen.append(c)

        # record
        for h in chosen:
            hubs.append(h)
            used.add(h)
            hub_ring_index[h] = k

    hubs = sorted(set(hubs))
    print(f"[INFO] Hubs placed (valid tiles only): {len(hubs)}")
    return hubs, hub_ring_index


# ============================================================
# BFS UTILITIES
# ============================================================

def bfs_shortest_path(adj,src,dst,forbidden=None):
    if forbidden is None: forbidden=set()
    dq=deque([src])
    parent={src:None}
    while dq:
        u=dq.popleft()
        if u==dst:
            path=[dst]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for v in adj[u]:
            if v in forbidden: continue
            if v not in parent:
                parent[v]=u
                dq.append(v)
    return None

def bfs_distance(adj,src,targets):
    visited={src:0}
    dq=deque([src])
    T=set(targets)
    while dq:
        u=dq.popleft()
        if u in T:
            return u,visited[u]
        for v in adj[u]:
            if v not in visited:
                visited[v]=visited[u]+1
                dq.append(v)
    return None,None

def bfs_farthest_with_parent(adj, src):
    visited = {src: 0}
    parent = {src: None}
    dq = deque([src])

    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in visited:
                visited[v] = visited[u] + 1
                parent[v] = u
                dq.append(v)

    far_node = max(visited, key=lambda k: visited[k])
    return far_node, visited[far_node], parent



def bfs_path(src, dst, forbidden, adj):
    dq = deque([src])
    parent = {src: None}
    while dq:
        u = dq.popleft()
        if u == dst:
            path=[dst]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for v in adj[u]:
            if v in forbidden: continue
            if v not in parent:
                parent[v]=u; dq.append(v)
    return None


def build_hub_backbone(nodes, hubs, center_cell, adj):
    """
    Backbone = shortest path from each hub directly to center.
    No hub-to-hub forcing. Paths may pass through other hubs naturally.
    """
    global kept_edges_undirected, hub_backbone_flag
    hub_backbone_flag = {}

    for h in hubs:
        if h == center_cell:
            continue

        p = bfs_path(h, center_cell, forbidden=set(), adj=adj)
        if not p:
            continue

        for u, v in zip(p[:-1], p[1:]):
            e = tuple(sorted((u, v)))
            kept_edges_undirected.add(e)
            hub_backbone_flag[(u, v)] = 1
            hub_backbone_flag[(v, u)] = 1


# ============================================================
# PRIMARY FEEDERS
# ============================================================

def build_primary_feeders(nodes, hubs, adj):
    global kept_edges_undirected, feeder_label
    hub_set=set(hubs)
    for c in nodes:
        if c in hub_set: continue
        nearest=None; nearest_hops=None
        for h in hubs:
            _,hops=bfs_distance(adj,c,{h})
            if hops is None: continue
            if nearest_hops is None or hops<nearest_hops:
                nearest_hops=hops; nearest=h
        if nearest is None: continue
        p=bfs_shortest_path(adj,c,nearest)
        if p:
            for u,v in zip(p[:-1],p[1:]):
                e=tuple(sorted((u,v)))
                kept_edges_undirected.add(e)
                feeder_label[(u,v)]=1


# ============================================================
# SECONDARY FEEDERS (DUAL)
# ============================================================

def build_secondary_feeders(nodes, hubs, center_cell, adj):
    global kept_edges_undirected, feeder_label
    hub_set=set(hubs)
    for c in nodes:
        if c in hub_set or c==center_cell: continue

        dist_list=[]
        for h in hubs:
            _,hops=bfs_distance(adj,c,{h})
            if hops is not None:
                dist_list.append((h,hops))
        dist_list.sort(key=lambda x:x[1])
        if len(dist_list)<2: continue

        h1,hops1=dist_list[0]
        h2,hops2=dist_list[1]

        forbidden={center_cell,h1}
        p=bfs_shortest_path(adj,c,h2,forbidden=forbidden)
        if not p: continue

        for u,v in zip(p[:-1],p[1:]):
            if feeder_label.get((u,v),0)>0 or feeder_label.get((v,u),0)>0:
                feeder_label[(u,v)]=3
            else:
                feeder_label[(u,v)]=2
            e=tuple(sorted((u,v)))
            kept_edges_undirected.add(e)
# ============================================================
# OVERLAY GRAPH + POTENTIALS
# ============================================================

def build_overlay_adj(corridor_edges, metro_edges):
    adj=defaultdict(list)
    for u,v in corridor_edges+metro_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def bfs_potential(adj, center):
    INF=10**9
    T={c:INF for c in adj}
    dq=deque([center])
    T[center]=0
    while dq:
        u=dq.popleft()
        for v in adj[u]:
            if T[v]>T[u]+1:
                T[v]=T[u]+1
                dq.append(v)
    return T

def compute_geom_dist(nodes,center_lat,center_lon):
    D={}
    for c,nd in nodes.items():
        D[c]=haversine_m(center_lat,center_lon,nd["lat"],nd["lon"])/1000.0
    return D



def build_edge_records(nodes, adj_base, corridor, metro, T_overlay, D_geom):
    metro_set = set(tuple(sorted(e)) for e in metro)
    corridor_set = set(tuple(sorted(e)) for e in corridor)

    all_undirected = kept_edges_undirected | metro_set | corridor_set

    edges = []
    tol = GEOM_TOL_KM

    for (u, v) in all_undirected:
        for (a, b) in [(u, v), (v, u)]:
            du = D_geom[a]
            dv = D_geom[b]
            ddk = du - dv

            if ddk > tol:
                dgeom = 1
            elif ddk < -tol:
                dgeom = -1
            else:
                dgeom = 0

            Tu = T_overlay.get(a, 10**9)
            Tv = T_overlay.get(b, 10**9)
            delta = None if (Tu >= 10**9 or Tv >= 10**9) else (Tu - Tv)

            delta_T = delta if EDGE_DIRECTION_MODE == "potential" else dgeom

            fl = feeder_label.get((a, b), 0)
            backbone_flag = hub_backbone_flag.get((a, b), 0)

            edges.append({
                "start_h3": a,
                "end_h3": b,
                "origin": a,
                "dest": b,
                "delta_T": delta_T,
                "delta_geom": dgeom,
                "delta_geom_km": ddk,
                "is_metro": int(tuple(sorted((a, b))) in metro_set),
                "overlay_flag": int(tuple(sorted((a, b))) in (corridor_set | metro_set)),
                "feeder_label": fl,
                "hub_backbone": backbone_flag,
            })

    print(f"[INFO] Directed edges: {len(edges)}")
    return edges



# ============================================================
# MANIFEST + CSV
# ============================================================

def write_outputs(nodes,hubs,hub_ring_index,center_cell,T_overlay,edges_dir,
                  center_lat,center_lon,manifest_path,nodes_csv,edges_csv):

    manifest_nodes={}
    for c,nd in nodes.items():
        manifest_nodes[c]={
            "lat":nd["lat"],
            "lon":nd["lon"],
            "is_hub":int(c in hubs),
            "is_center":int(c==center_cell),
            "T":None if T_overlay.get(c) is None else float(T_overlay[c]),
        }

    hubs_block={"h3":{h:{"attr":1.0} for h in hubs}}
    potentials_block={
        "T_hops":{c:(None if T_overlay.get(c) is None else float(T_overlay[c]))
                  for c in nodes},
        "centers":[center_cell],
    }

    def enc(e):
        return {
            "start_h3": e["start_h3"],
            "end_h3": e["end_h3"],
            "origin": e["origin"],
            "dest": e["dest"],

            "w_norm": 1.0,
            "delta_T": e["delta_T"],
            "delta_geom": e["delta_geom"],
            "delta_geom_km": e["delta_geom_km"],

            "am_pref": 0,
            "pm_pref": 0,

            "is_metro": e["is_metro"],
            "feeder_label": e["feeder_label"],
            "overlay_flag": e["overlay_flag"],
            "hub_backbone": e.get("hub_backbone", 0),
        }


    edges_AM=[enc(e) for e in edges_dir]
    edges_PM=[enc(e) for e in edges_dir]

    manifest={
        "seed":SEED,
        "region":REGION_NAME,
        "h3_resolution":H3_RESOLUTION,
        "edge_direction_mode":EDGE_DIRECTION_MODE,
        "geom_center_mode":GEOM_CENTER_MODE,
        "geom_center_h3":(GEOM_CENTER_H3 if GEOM_CENTER_MODE=="h3" else None),
        "geom_tol_km":GEOM_TOL_KM,
        "center_anchor":{"lat":center_lat,"lon":center_lon},
        "center_cell":center_cell,
        "hubs":hubs_block,
        "potentials":potentials_block,
        "nodes":manifest_nodes,
        "edges":{
            "AM":edges_AM,
            "PM":edges_PM,
            "preferred":{"AM":[],"PM":[]}
        }
    }

    with open(manifest_path,"w") as f:
        json.dump(manifest,f,indent=2)

    with open(nodes_csv,"w") as f:
        f.write("h3,lat,lon,T,is_hub,is_center,hub_ring\n")
        for c,nd in manifest_nodes.items():
            f.write(f"{c},{nd['lat']},{nd['lon']},"
                    f"{'' if nd['T'] is None else nd['T']},"
                    f"{nd['is_hub']},{nd['is_center']},"
                    f"{hub_ring_index.get(c,'')}\n")

    with open(edges_csv,"w") as f:
        f.write("start_h3,end_h3,w_norm,delta_T,delta_geom,delta_geom_km,am_pref,pm_pref,is_metro,feeder_label,overlay_flag\n")
        for e in edges_dir:
            f.write(f"{e["start_h3"]},{e["end_h3"]},1.0,{e['delta_T']},{e['delta_geom']},"
                    f"{e['delta_geom_km']},0,0,{e['is_metro']},"
                    f"{e['feeder_label']},{e['overlay_flag']}\n")
# ============================================================
# MAP BUILD
# ============================================================

def color_for_delta_T(delta_T):
    if delta_T is None:
        return "#aaaaaa"
    try:
        d=int(delta_T)
    except:
        return "#aaaaaa"
    if d>0: return "#d62728"
    if d<0: return "#1f77b4"
    return "#888888"


from playwright.sync_api import sync_playwright

def html_to_png(html_path, png_path, width=1200, height=1200):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.wait_for_timeout(2000)  # allow tiles to load
        page.screenshot(path=png_path)
        browser.close()

def h3_cell_style(feature):
    return {
        "fillColor": "#ffffff",
        "color": "#cccccc",
        "weight": 1,
        "fillOpacity": 0.15,
    }

def build_map(nodes, hubs, hub_ring_index, center_cell,
              edges_dir, metro_edges, corridor_edges, adj,
              prefix="h3_corridors", out_html=None):

    cells=list(nodes.keys())
    lat0=nodes[cells[0]]["lat"]
    lon0=nodes[cells[0]]["lon"]

    m=folium.Map(location=[lat0,lon0],zoom_start=10,
                 tiles="cartodbpositron",control_scale=True)
    Fullscreen().add_to(m)


    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.9);
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        Synthetic Mobility Network
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    g_cells = FeatureGroup("H3 cells", show=True)
    m.add_child(g_cells)

    g_regular=FeatureGroup("Corridor/background",show=True)



    g_metro=FeatureGroup("Metro edges",show=True)
    g_arrows=FeatureGroup("Feeder arrows",show=True)
    g_hubs=FeatureGroup("Hubs",show=True)
    g_nodes=FeatureGroup("Nodes",show=True)
    for g in (g_regular,g_metro,g_arrows,g_hubs,g_nodes):
        m.add_child(g)

    feeder_dir={(e["start_h3"],e["end_h3"]): e["feeder_label"] for e in edges_dir}

    from shapely.geometry import mapping

    for h, nd in nodes.items():
        poly = h3_cell_polygon(h)
        geojson = {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {}
                }


        folium.GeoJson(
            geojson,
            style_function=h3_cell_style,
            tooltip=h
        ).add_to(g_cells)


    # ============================================================
    # EDGE DRAWING WITH FULL LABELING (H3 ONLY)
    # ============================================================

    # unique undirected pairs
    unique_pairs = set(tuple(sorted((e["start_h3"], e["end_h3"]))) for e in edges_dir)

    for (u, v) in unique_pairs:
        latu, lonu = nodes[u]["lat"], nodes[u]["lon"]
        latv, lonv = nodes[v]["lat"], nodes[v]["lon"]

        # find directed edges
        e_uv = next((e for e in edges_dir if e["start_h3"] == u and e["end_h3"] == v), None)
        e_vu = next((e for e in edges_dir if e["start_h3"] == v and e["end_h3"] == u), None)

        # pick representative, preferring feeder direction
        if e_uv and e_uv["feeder_label"] > 0:
            sample = e_uv
        elif e_vu and e_vu["feeder_label"] > 0:
            sample = e_vu
        else:
            sample = e_uv if e_uv else e_vu

        # extract labels
        delta_T        = sample["delta_T"]
        delta_geom     = sample["delta_geom"]
        delta_geom_km  = sample["delta_geom_km"]
        feeder_label   = sample["feeder_label"]
        is_metro       = sample["is_metro"]
        overlay_flag   = sample["overlay_flag"]
        backbone_flag = sample.get("hub_backbone", 0)
        # color
        col = color_for_delta_T(delta_T)
        if feeder_label > 0:
            col = "#7e57c2"

        col = color_for_delta_T(delta_T)

        if backbone_flag == 1 and feeder_label == 0:
            col = "#00ff00"        # backbone (green)
        elif feeder_label > 0:
            col = "#7e57c2"        # feeder (purple)

        # tooltip
        tooltip_text = (
            f"{u} ↔ {v}"
            f"<br>ΔT: {delta_T}"
            f"<br>Δgeom: {delta_geom}"
            f"<br>Δgeom_km: {delta_geom_km:.3f}"
            f"<br>feeder_label: {feeder_label}"
            f"<br>is_metro: {is_metro}"
            f"<br>overlay_flag: {overlay_flag}"
           f"<br>hub_backbone: {backbone_flag}"
        )

        # draw edge
        folium.PolyLine(
            [(latu, lonu), (latv, lonv)],
            color=col,
            weight=4 if is_metro else 2 if overlay_flag else 1,
            opacity=0.9 if is_metro else (0.7 if overlay_flag else 0.4),
            tooltip=tooltip_text
        ).add_to(g_metro if is_metro else g_regular)

        # ============================================================
        # ARROWS (use the true feeder direction)
        # ============================================================

        fl_uv = feeder_dir.get((u, v), 0)
        fl_vu = feeder_dir.get((v, u), 0)

        if fl_uv > 0 and fl_vu == 0:
            direction = (u, v)
        elif fl_vu > 0 and fl_uv == 0:
            direction = (v, u)
        else:
            direction = None

        if direction and feeder_label > 0 and backbone_flag == 0:
            a, b = direction
            la, loa = nodes[a]["lat"], nodes[a]["lon"]
            lb, lob = nodes[b]["lat"], nodes[b]["lon"]
            dx = lb - la
            dy = lob - loa
            sz = 0.00045

            hx1 = lb - 0.2 * dx + sz * (-dy)
            hy1 = lob - 0.2 * dy + sz * (dx)
            hx2 = lb - 0.2 * dx - sz * (-dy)
            hy2 = lob - 0.2 * dy - sz * (dx)

            folium.PolyLine([(lb, lob), (hx1, hy1)],
                            color="#7e57c2",
                            weight=3, opacity=1).add_to(g_arrows)
            folium.PolyLine([(lb, lob), (hx2, hy2)],
                            color="#7e57c2",
                            weight=3, opacity=1).add_to(g_arrows)


        if direction and feeder_label > 0 and backbone_flag == 0:
            a,b=direction
            la,loa=nodes[a]["lat"],nodes[a]["lon"]
            lb,lob=nodes[b]["lat"],nodes[b]["lon"]
            dx=lb-la; dy=lob-loa
            sz=0.00045
            hx1=lb-0.2*dx+sz*(-dy)
            hy1=lob-0.2*dy+sz*(dx)
            hx2=lb-0.2*dx-sz*(-dy)
            hy2=lob-0.2*dy-sz*(dx)
            folium.PolyLine([(lb,lob),(hx1,hy1)],
                            color="#7e57c2",
                            weight=3,opacity=1).add_to(g_arrows)
            folium.PolyLine([(lb,lob),(hx2,hy2)],
                            color="#7e57c2",
                            weight=3,opacity=1).add_to(g_arrows)

    hubs_set=set(hubs)
    for h in hubs:
        nd=nodes[h]
        ring=hub_ring_index[h]
        col=HUB_RING_COLORS[ring] if ring<len(HUB_RING_COLORS) else HUB_COLOR
        folium.CircleMarker(
            [nd["lat"],nd["lon"]],
            radius=NODE_RADIUS_HUB,
            color=col,fill=True,fill_opacity=0.95,
            tooltip=f"Hub {h} ring {ring}"
        ).add_to(g_hubs)

    for c,nd in nodes.items():
        if c in hubs_set: continue
        if c==center_cell:
            col="#000"; rad=NODE_RADIUS_HUB
        else:
            col=NODE_COLOR; rad=NODE_RADIUS_NON
        folium.CircleMarker(
            [nd["lat"],nd["lon"]],
            radius=rad,color=col,fill=True,fill_opacity=0.85,
            tooltip=c
        ).add_to(g_nodes)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    html_to_png(out_html, out_html.replace(".html", ".png"))

def build_metro_tree(nodes, hubs, center_lat, center_lon):
    """
    Build the metro backbone by connecting each non-root hub
    to the nearest hub that is closer to the center (geometric distance).
    This reproduces the original metro-tree behavior.
    """
    if not hubs:
        return []

    # Geometric distance from center
    hub_dist = {
        h: haversine_m(center_lat, center_lon,
                       nodes[h]["lat"], nodes[h]["lon"])
        for h in hubs
    }

    # Sort hubs from closest to farthest from center
    ordered = sorted(hubs, key=lambda h: hub_dist[h])

    edges = set()

    # Every hub after root connects to closest "previous" hub
    for h in ordered[1:]:
        nearer = [x for x in ordered if hub_dist[x] < hub_dist[h]]
        if not nearer:
            continue

        # Choose the geometrically closest eligible hub
        best = None
        best_key = None
        for g in nearer:
            d = haversine_m(nodes[h]["lat"], nodes[h]["lon"],
                            nodes[g]["lat"], nodes[g]["lon"])
            key = (d, hub_dist[g])   # tie-break by distance to center
            if best_key is None or key < best_key:
                best_key = key
                best = g

        if best:
            edges.add(tuple(sorted((h, best))))

    return list(edges)


def run():
    random.seed(SEED)

    city_tag    = _city_tag()
    mode_tag    = _mode_tag()
    feeder_tag  = _feeder_tag()
    dir_tag     = _edge_dir_tag()
    overlay_tag = _overlay_tag()

    manifest_path = os.path.join(
        OUT_DIR,
        f"corridors_manifest_{city_tag}_{mode_tag}_{feeder_tag}_{dir_tag}_{overlay_tag}.json"
    )
    nodes_csv = os.path.join(
        OUT_DIR,
        f"corridor_potential_nodes_{city_tag}_{mode_tag}_{feeder_tag}_{dir_tag}_{overlay_tag}.csv"
    )
    edges_csv = os.path.join(
        OUT_DIR,
        f"corridor_potential_edges_{city_tag}_{mode_tag}_{feeder_tag}_{dir_tag}_{overlay_tag}.csv"
    )
    am_html = os.path.join(
        OUT_DIR,
        f"corridor_AM_map_{city_tag}_{mode_tag}_{feeder_tag}_{dir_tag}_{overlay_tag}.html"
    )
    # --- keep the rest of your run() exactly as-is ---

    random.seed(SEED)

    region_geom,_=load_region_polygon()
    nodes = build_h3_grid(region_geom, H3_RESOLUTION)
    adj   = build_h3_adjacency(nodes)

    # ==========================================================
    # GRID MODE (EARLY EXIT)
    # ==========================================================
    if GRAPH_MODE == "grid":
        print("[INFO] Running in GRID mode (no hubs, no corridors)")

        edges_dir = build_grid_edges(nodes, adj)

        # empty / neutral placeholders
        hubs = []
        hub_ring_index = {}
        center_cell = None
        T_overlay = {}



        # filenames (reuse same naming scheme, but tag mode)
        manifest_path = os.path.join(
            OUT_DIR, f"grid_manifest_{city_tag}_mode-h3.json"
        )
        nodes_csv = os.path.join(
            OUT_DIR, f"grid_nodes_{city_tag}_mode-h3.csv"
        )
        edges_csv = os.path.join(
            OUT_DIR, f"grid_edges_{city_tag}_mode-h3.csv"
        )
        map_html = os.path.join(
            OUT_DIR, f"grid_map_{city_tag}_mode-h3.html"
        )

        # write nodes
        with open(nodes_csv, "w") as f:
            f.write("h3,lat,lon\n")
            for c, nd in nodes.items():
                f.write(f"{c},{nd['lat']},{nd['lon']}\n")

        # write edges
        with open(edges_csv, "w") as f:
            f.write("start_h3,end_h3,w_norm\n")
            for e in edges_dir:
                f.write(f"{e['start_h3']},{e['end_h3']},1.0\n")

        # map (grid-only)
        build_map(
            nodes,
            hubs=[],
            hub_ring_index={},
            center_cell=None,
            edges_dir=edges_dir,
            metro_edges=[],
            corridor_edges=[],
            adj=adj,
            out_html=map_html
        )


        write_outputs(
            nodes=nodes,
            hubs=[],
            hub_ring_index={},
            center_cell=None,
            T_overlay={},
            edges_dir=edges_dir,
            center_lat=None,
            center_lon=None,
            manifest_path=manifest_path,
            nodes_csv=nodes_csv,
            edges_csv=edges_csv
        )

        print("[OK] GRID mode complete.")
        return

    # if not hubs:
    #     draw_hubs = False
    # else:
    #     draw_hubs = True


    center_lat,center_lon,center_cell=choose_anchor_and_center(nodes)

    if GEOM_CENTER_MODE=="h3" and center_cell not in nodes:
        lat,lon=h3.cell_to_latlng(center_cell)
        nodes[center_cell]={"lat":lat,"lon":lon}
        adj = build_h3_adjacency(nodes)

    hubs,hub_ring_index=place_hubs(nodes,center_lat,center_lon)


    global kept_edges_undirected, feeder_label
    kept_edges_undirected=set()
    feeder_label={}

    # ---------------------------------------------------------
    # ENSURE ALL CENTER CELLS ARE PAIRWISE CONNECTED (PERMANENT)
    # ---------------------------------------------------------
    # For now you have only one center; but this supports multiples.
    center_cells = [center_cell]   # or replace with a list if needed

    def bfs_shortest(u, v):
        dq = deque([u])
        parent = {u: None}
        while dq:
            x = dq.popleft()
            if x == v:
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            for y in adj[x]:
                if y not in parent:
                    parent[y] = x
                    dq.append(y)
        return None

    # pairwise-connect all center cells permanently
    for i in range(len(center_cells)):
        for j in range(i+1, len(center_cells)):
            c1 = center_cells[i]
            c2 = center_cells[j]
            path = bfs_shortest(c1, c2)
            if path:
                for a, b in zip(path[:-1], path[1:]):
                    kept_edges_undirected.add(tuple(sorted((a, b))))



    # build_hub_backbone(nodes, hubs, center_cell, center_lat, center_lon, adj, hub_ring_index)
    build_hub_backbone(nodes, hubs, center_cell, adj)

    build_primary_feeders(nodes,hubs,adj)
    if FEEDER_MODE=="dual":
        build_secondary_feeders(nodes,hubs,center_cell,adj)

    # Build metro edges between hubs
    metro_edges = build_metro_tree(nodes, hubs, center_lat, center_lon)

    # Corridor edges = kept feeder + backbone
    corridor_edges = list(kept_edges_undirected)
    
    overlay_adj = build_overlay_adj(corridor_edges,metro_edges)
    if center_cell not in overlay_adj:
        overlay_adj[center_cell]=[]
    T_overlay=bfs_potential(overlay_adj,center_cell)


    print("[INFO] Computing maximum hop distance in overlay graph…")

    max_hops = -1
    max_pair = None
    max_path = None

    for u in overlay_adj:
        v, dist, parent = bfs_farthest_with_parent(overlay_adj, u)
        if dist > max_hops:
            # reconstruct path u → v
            path = [v]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path = path[::-1]

            max_hops = dist
            max_pair = (u, v)
            max_path = path

    maxhop_csv = os.path.join(
        OUT_DIR,
        f"max_hop_diameter_{city_tag}_{mode_tag}_{feeder_tag}_{dir_tag}_{overlay_tag}.csv"
    )

    with open(maxhop_csv, "w") as f:
        f.write("node_u,node_v,hop_length,path\n")
        f.write(
            f"{max_pair[0]},{max_pair[1]},"
            f"{max_hops},\"{'->'.join(max_path)}\"\n"
        )

    print(f"[OK] Wrote max-hop diagnostic → {maxhop_csv}")




    D_geom_km=compute_geom_dist(nodes,center_lat,center_lon)
    edges_dir=build_edge_records(nodes,adj,corridor_edges,metro_edges,
                                 T_overlay,D_geom_km)


    write_outputs(nodes,hubs,hub_ring_index,center_cell,T_overlay,
                  edges_dir,center_lat,center_lon,
                  manifest_path,nodes_csv,edges_csv)



    build_map(nodes, hubs, hub_ring_index, center_cell,
            edges_dir, metro_edges, corridor_edges, adj,out_html=am_html)

    print("[OK] Done.")

if __name__=="__main__":
    run()
