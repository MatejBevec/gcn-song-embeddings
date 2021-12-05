import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import os, sys
from os import path
import random
import math

import networkx as nx
from networkx.algorithms import bipartite

from get_data import DatasetCollector

LEAF_RATIO = 0.9 # remove LEAF_RATIO of allr degree 1 nodes to prune the graph
NORMALIZE_DEGREE = True # normalize visit counts with respect to degree (nerf popular nodes)
K = 3 # show K nearest neighbours when crawling graph
RANDOM_SEED = 420


def get_random_track(dc):
    rnd = random.randrange(0, len(dc.graph["tracks"]))
    rnd_id = dc.graph["tracks"][rnd]
    return rnd_id

def short_str(str, n):
    return str if len(str) < n else (str[0:n-3] + "...")
    

def count_walks_from(g, src_track_id, n, alpha, norm_degree):
    # do n*2 (always step track->col->track) steps of random walks with restarts
    # from track_id and count visits to neighbors
    # = Personalized PageRank

    visits = {}
    track_id = src_track_id

    degree = g.degree[src_track_id]
    num_steps = int(n * math.pow(degree, 1/1.5))

    for i in range(0, num_steps):
        rnd1 = random.randrange(0, len(g[track_id]))
        col_id = list(g[track_id])[rnd1]
        rnd2 = random.randrange(0, len(g[col_id]))
        track_id = list(g[col_id])[rnd2]

        if track_id not in visits:
            visits[track_id] = 0
        visits[track_id] += 1

        if random.random() < alpha:
            track_id = src_track_id

    if norm_degree:
        for node in visits:
            visits[node] /= math.log(g.degree[node]+1)

    del visits[src_track_id]

    return dict(sorted(visits.items(), reverse=True, key=lambda a: a[1]))

def count_walks_norm_total_n(g, src_track_id, n, alpha):
    # do n*2 (always step track->col->track) steps of random walks with restarts
    # from track_id and count visits to neighbors
    # = Personalized PageRank

    visits = {}
    track_id = src_track_id

    degree = g.degree[src_track_id]
    num_steps = int(n * math.pow(degree, 1/1.5))

    for i in range(0, num_steps):
        rnd1 = random.randrange(0, len(g[track_id]))
        col_id = list(g[track_id])[rnd1]
        rnd2 = random.randrange(0, len(g[col_id]))
        track_id = list(g[col_id])[rnd2]

        if track_id not in visits:
            visits[track_id] = 0
        visits[track_id] += 1

        if random.random() < alpha:
            track_id = src_track_id

    sum = 0
    for node in visits:
        sum += visits[node]

    del visits[src_track_id]

    for node in visits:
        visits[node] /= (num_steps)

    return dict(sorted(visits.items(), reverse=True, key=lambda a: a[1]))   

def count_walks_torch(g, track_ids, n, alpha):
    
    n_hops = n
    n_nodes = len(track_ids)
    trace = torch.zeros(n_nodes, n_hops, dtype=torch.int64)

    id_list = sorted(list(g.nodes()))
    index_map = {id_list[i]: i for i in range(len(id_list))}

    for i in range(0, n_nodes):
        #item = nodeset[i]
        item = track_ids[i]
        for j in range(0, n_hops):
            neighbors = list(g[item])
            rnd = torch.randint(len(neighbors), ()).item()
            collection = neighbors[rnd]
            neighbors = list(g[collection])
            rnd = torch.randint(len(neighbors), ()).item()
            item = neighbors[rnd]
            
            ind = index_map[item]
            trace[i, j] = ind

            if torch.rand(()) < alpha:
                item = track_ids[i]

    
    n_all_nodes = len(g)
    visit_counts = torch.zeros(n_nodes, n_all_nodes, dtype=torch.float64) \
        .scatter_add_(1, trace, torch.ones_like(trace, dtype=torch.float64))
    visit_prob = visit_counts / visit_counts.sum(1, keepdim=True) # or just /n_hops
    for i in range(0, n_nodes):
        visit_prob[i, index_map[track_ids[i]] ] = 0

    topk_weights, topk_nodes = visit_prob.topk(3, 1)

    nodes_row = topk_nodes[0,:]
    weights_row = topk_weights[0,:]
    visit_dict = {id_list[nodes_row[i]]: weights_row[i] for i in range(nodes_row.shape[0])}
    
    return visit_dict


def print_nn(dc, g, nn, k):
    nn_ids = list(nn)
    print("\u001b[36m Nearest neighbors:")
    for i in range(0, k):
        track = dc.tracks[nn_ids[i]]
        print(f"{i}. [{nn[nn_ids[i]]:.2f}] {short_str(track['name'], 30)} - {track['artist']} ({g.degree[nn_ids[i]]})")
    print("\033[0m")

def show_info(dc, g):

    #collections = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
    #tracks = set(g) - collections

    n = len(g)
    print("# nodes: ", n)
    num_edges = len(g.edges)
    print("# edges: ", num_edges)
    print("# tracks in tracks.json: ", len(dc.tracks))
    print("# tracks in graph.json: ", len(dc.graph["tracks"]))
    print("# cols in collections.json: ", len(dc.collections))
    print("# cols in graph.json: ", len(dc.graph["collections"]))

    num_components = nx.number_connected_components(g)
    component_sizes = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    print("# components: ",  num_components)
    print("Component sizes: ", component_sizes)

    g_degrees = sorted([d for n, d in g.degree()], reverse=True)
    print("Mean degree: ", np.mean(g_degrees))
    print("Median degree: ", np.median(g_degrees))
    
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(*np.unique(g_degrees, return_counts=True))
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(component_sizes, bins=200)
    ax2.set_title("Component sizes")
    ax2.set_xlabel("Size")
    ax2.set_ylabel("# of Components")
    plt.show()

def print_node_info(dc, id):
    if id in dc.tracks:
        info = dc.tracks[id]
        print("\033[0;33m", info["name"], "\033[0m")
        print(info["artist"])
    elif id in dc.collections:
        info = dc.collections[id]
        print("\033[0;33m", info["name"], "\033[0m")
        print(info["type"])
    else:
        print("woot")

def crawl(dc, g):

    print("# nodes: ", len(g))
    print("# edges: ", len(g.edges))
    degs = g_degrees = sorted([d for n, d in g.degree()], reverse=True)
    print("Mean degree: ", np.mean(degs))

    id = get_random_track(dc)
    
    while True:
        print("\n\033[0;31m---------------------\033[0m")
        print_node_info(dc, id)
        print("\033[0;31m---------------------\033[0m\n")

        if id in dc.tracks:
            # find most similar (track) nodes with Personalized PageRank
            nn = count_walks_from(g, id, 2000, 0.85, NORMALIZE_DEGREE)
            print_nn(dc, g, nn, K)

            # TEMP: try different random walk algos
            print("Without degree normalization: ")
            nn = count_walks_from(g, id, 2000, 0.85, False)
            print_nn(dc, g, nn, K)
            #print("Divide by total hops: ")
            #nn = count_walks_norm_total_n(g, id, 2000, 0.85)
            #print_nn(dc, g, nn, K)
            print("Pytorch implementation: ")
            nn = count_walks_torch(g, [id], 2000, 0.85)
            print_nn(dc, g, nn, K)

        print("Links:")
        neighbors = list(g[id])
        for i,ne in enumerate(neighbors):
            print(i+1, "."),
            print_node_info(dc, ne)
            print()
        ans = input("\nPick a link or enter 'r' for a random track:")
        if ans == "r":
            id = get_random_track(dc)
        else:
            id = neighbors[int(ans)-1]

def show_sample(dc, g):
    # loops threw a few examples in the dataset  
    random.seed(RANDOM_SEED)
    for i in range(0, 5):
        id = get_random_track(dc)
        nn = count_walks_from(g, id, 2000, 0.85, NORMALIZE_DEGREE)
        print_node_info(dc, id)
        print_nn(dc, g, nn, K)
        #input()

def filter_dataset_with_graph(dc, g):
    for t in list(dc.tracks):
        if t not in g:
            del dc.tracks[t]
    for c in list(dc.collections):
        if c not in g:
            del dc.collections[c]
    
    dc.graph["tracks"] = [t for t in dc.graph["tracks"] if t in g]
    dc.graph["collections"] = [c for c in dc.graph["collections"] if c in g]

    kept_e = [e for e in dc.graph["edges"] if (e["from"], e["to"]) in g.edges ]
    dc.graph["edges"] = kept_e

def make_mini_dataset(dc, g, save_dir):
    # sample a small subset of the graph and save to save_dir
    min_deg = 4 #8 # remove nodes with smaller degree
    max_deg = 120 #80 # remove tracks with larger degree

    print(f"{len(g)} nodes, cutting min, max degree...")
    for node in list(g.nodes):
        if node in dc.tracks and g.degree[node] > max_deg:
            g.remove_node(node)
    for node in list(g.nodes):
        if g.degree[node] < min_deg:
            g.remove_node(node)
    print(f"{len(g)} nodes left.")

    filter_dataset_with_graph(dc, g)
    dc.save_dataset_as(save_dir)

def to_nx_graph(dc, remove_leafs=True, giant_only=True):
    g = nx.Graph()
    g.add_nodes_from(dc.graph["collections"], bipartite=0)
    g.add_nodes_from(dc.graph["tracks"], bipartite=1) 
    edge_tuples = [ (e["from"], e["to"]) for e in dc.graph["edges"] ] 
    g.add_edges_from( edge_tuples )

    # remove degree 1 nodes (leafs)
    if remove_leafs:
        for node in list(g.nodes):
            if g.degree[node] <= 1:
                if random.random() < LEAF_RATIO:
                    g.remove_node(node)

    # keep only the giant component
    if giant_only:
        giant = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])
        g = nx.Graph(giant)
        filter_dataset_with_graph(dc, g)

    return g

if __name__ == "__main__":

    if len(sys.argv) < 2 or (sys.argv[1] not in ["info", "crawl", "sample", "mini", "positives"]):
        print("Unrecognized command.")
        exit()
    mode = sys.argv[1]
    top_comp = int(sys.argv[2]) if len(sys.argv) > 2 else None #use only largest top_comp components  

    dc = DatasetCollector("../pinsage_code/dataset_mini", 10) 
    g = to_nx_graph(dc, remove_leafs=True, giant_only=True)

    try:
        if mode == "info":
            show_info(dc, g)
        if mode == "crawl":
            crawl(dc, g)
        if mode == "sample":
            show_sample(dc, g)
        if mode == "mini":
            make_mini_dataset(dc, g, sys.argv[2])
    except KeyboardInterrupt:
        print("Exiting...")

    #dc.save_dataset()

