import os
from os import path
import json
import time
from networkx.classes.function import all_neighbors

import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh

from spotify_graph import SpotifyGraph

DEF_T_PRECOMP = 100
DEF_HOPS = 500
DEF_ALPHA = 0.85

# bipartide graph in dgl (2 options):
# all nodes are same type, pass in number of item nodes (currently using)
# heterogenous graph

def get_embeddings(h, nodeset, d):
    return h[nodeset, :d]

def put_embeddings(h, nodeset, nodeset_new_h):
    #n_nodes = nodeset.shape[0] # nodeset is just the batch, new_h has all relevant nodes!
    #n_features = h.shape[1]
    #new_h = h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), nodeset_new_h)
    new_h = h.clone().detach()
    pad_cols = new_h.shape[1] - nodeset_new_h.shape[1]
    pad_rows = nodeset.shape[0]
    new_h[nodeset, :] = torch.cat([nodeset_new_h, torch.zeros(pad_rows, pad_cols)], 1)
    # print(nodeset_new_h.shape)
    # print(new_h.shape)
    # print(nodeset_new_h[0,:])
    # print()
    return new_h

def do_random_walks(g, nodeset, n_hops, alpha):

    n_nodes = nodeset.shape[0]
    trace = torch.zeros(n_nodes, n_hops, dtype=torch.int64)

    for i in range(0, n_nodes):
        item = nodeset[i]
        for j in range(0, n_hops):
            neighbors = g.successors(item)
            rnd = torch.randint(len(neighbors), ())
            collection = neighbors[rnd]
            neighbors = g.successors(collection)
            rnd = torch.randint(len(neighbors), ())
            item = neighbors[rnd]
            
            trace[i, j] = item

            if torch.rand(()) < alpha:
                item = nodeset[i]

    return trace

def sample_neighborhood_topt_early_stop(g, n_items, nodeset, n_hops, alpha, T, np, nv):
    # terminate when at least "np" pins have been visited "nv" or more times

    n_nodes = nodeset.shape[0]
    visit_counts = torch.zeros((n_nodes, n_items), dtype=torch.int64)

    for i in range(0, n_nodes):
        item = nodeset[i]
        term_counter = 0
        for j in range(0, n_hops):
            neighbors = g.successors(item)
            rnd = torch.randint(len(neighbors), ())
            collection = neighbors[rnd]
            neighbors = g.successors(collection)
            rnd = torch.randint(len(neighbors), ())
            item = neighbors[rnd]

            visit_counts[i, item] += 1

            if visit_counts[i, item] == nv:
                term_counter += 1
                if term_counter == np:
                    break

            if torch.rand(()) < alpha:
                item = nodeset[i]

    visit_counts = visit_counts.to(torch.float64)
    visit_prob = visit_counts / visit_counts.sum(1, keepdim=True)
    visit_prob[range(0, n_nodes), nodeset] = 0
    return visit_prob.topk(T, 1)

def sample_neighborhood(g, n_items, nodeset, n_hops, alpha):

    n_nodes = nodeset.shape[0]
    n_all_nodes = g.number_of_nodes()

    trace = do_random_walks(g, nodeset, n_hops, alpha)
    visit_counts = torch.zeros(n_nodes, n_all_nodes, dtype=torch.float64) \
        .scatter_add_(1, trace, torch.ones_like(trace, dtype=torch.float64))
    visit_prob = visit_counts / visit_counts.sum(1, keepdim=True) # or just /n_hops
    visit_prob[range(0, n_nodes), nodeset] = 0

    return visit_prob

def sample_neighborhood_topt(g, n_items, nodeset, n_hops, alpha, T):
    # called "N(u)" in paper

    visit_prob = sample_neighborhood(g, n_items, nodeset, n_hops, alpha)
    return visit_prob.topk(T, 1)

def precompute_neighborhoods_topt(g, n_items, n_hops, alpha, T, path):

    if os.path.isfile(path): 
        nodes, weights = torch.load(path)
        if weights.shape[0] == n_items and weights.shape[1] == T:
            return (nodes, weights)

    batch_size = 256
    all_nb_nodes = torch.zeros(n_items, T, dtype=torch.int64)
    all_nb_weights = torch.zeros(n_items, T, dtype=torch.float64)

    t0 = time.time()
    for i in range(0, n_items, batch_size):
        batch = torch.tensor(range(i, min(i+batch_size, n_items)))
        nb_weights, nb_nodes = sample_neighborhood_topt(g, n_items, batch, n_hops, alpha, T)
        all_nb_nodes[batch, :] = nb_nodes
        all_nb_weights[batch, :] = nb_weights
        print(f"{i + batch_size}/{n_items} done.")
        print(f"{time.time() - t0}s elapsed.")
    
    torch.save( (all_nb_weights, all_nb_nodes), path )
    return (all_nb_weights, all_nb_nodes)


def sample_hard_negatives(g, n_items, visit_prob, hn_per_query, min_rank, max_rank):

    range = visit_prob.topk(max_rank, 1)[1][:, min_rank:]
    sample = torch.randint(0, range.shape[1], (hn_per_query,))
    return range[:, sample]

def relevant_nodes_per_layer(g, n_items, nodeset, n_layers, n_hops, alpha, T):
    # called "S" in paper

    S = []
    cur_nodeset = nodeset
    for i in reversed(range(0, n_layers)):
        nb_weights, nb_nodes = sample_neighborhood_topt(g, n_items, cur_nodeset, n_hops, alpha, T)
        #nb_weights, nb_nodes = sample_neighborhood_topt_early_stop(
        #   g, n_items, cur_nodeset, n_hops, alpha, T, T, 20)
        S.insert(0, (cur_nodeset, nb_weights, nb_nodes))
        cur_nodeset = torch.cat([nb_nodes.flatten(), cur_nodeset]).unique()

    return S

def relevant_nodes_per_layer_precomp(nodeset, n_layers, T , nbhds):

    all_nb_weights, all_nb_nodes = nbhds
    S = []
    cur_nodeset = nodeset
    for i in reversed(range(0, n_layers)):
        nb_weights, nb_nodes = all_nb_weights[cur_nodeset, :T], all_nb_nodes[cur_nodeset, :T]
        S.insert(0, (cur_nodeset, nb_weights, nb_nodes))
        cur_nodeset = torch.cat([nb_nodes.flatten(), cur_nodeset]).unique()
    
    return S
    

class ConvLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ConvLayer, self).__init__()

        self.in_dim = in_dim            # node features dim
        self.out_dim = out_dim          # output after every conv layer (not final emb.)
        self.hidden_dim = hidden_dim    # neighbor aggregate dim

        self.Q = nn.Linear(in_dim, hidden_dim) # NN layer for every neighbor pre-aggregation (m)
        torch.nn.init.xavier_uniform_(self.Q.weight)
        self.Q.bias.data.fill_(0.01)

        self.W = nn.Linear(in_dim + hidden_dim, out_dim) # NN layer post-aggregation (d)
        torch.nn.init.xavier_uniform_(self.W.weight)
        self.W.bias.data.fill_(0.01)

    def forward(self, h, nodeset, nb_nodes, nb_weights):
        
        n_nodes, T = nb_nodes.shape

        # possible problem: repeat nodes in neighbor_h
        
        nodeset_h = get_embeddings(h, nodeset, self.in_dim)
        neighbor_h = get_embeddings(h, nb_nodes.flatten(), self.in_dim)
        neighbor_h = neighbor_h.view(n_nodes, T, self.in_dim) # back to T neighbors per node

        # transform neighbor features through NN and aggregate
        neighbor_h = nn.functional.relu( self.Q(neighbor_h) )
        agg = (nb_weights[:, :, None] * neighbor_h) .sum(1) / nb_weights.sum(1, keepdim=True)
        # (maybe put this division by weight sum in sample_neighborhood_topt?)

        # should be [n_nodes * hidden_features]

        # concatenate node features with aggreagate, run through 2nd NN, normalize output vector
        concat = torch.cat([nodeset_h, agg], 1).float()
        new_h = nn.functional.relu( self.W(concat) )
        new_h_norm = new_h / new_h.norm(dim=1, keepdim=True)

        return new_h_norm


class PinSageModel(nn.Module):

    def __init__(self, g, n_items, n_layers, dimensions, n_hops, alpha, T, nbhds):
        super(PinSageModel, self).__init__()

        self.g = g
        self.n_items = n_items
        self.T = T
        self.n_hops = n_hops
        self.alpha = alpha
        self.nbhds = nbhds

        self.n_layers = n_layers
        self.in_dim = dimensions[0]
        self.hidden_dim = dimensions[1]
        self.out_dim = dimensions[2]
        self.in_dim_per_layer = [self.in_dim] + [self.out_dim for l in range(n_layers-1)]
        
        self.conv_layers = nn.ModuleList()
        for i in range(0, self.n_layers):
            self.conv_layers.append( 
                ConvLayer(self.in_dim_per_layer[i], self.out_dim, self.hidden_dim))

        self.G1 = nn.Linear(self.out_dim, self.out_dim)
        torch.nn.init.xavier_uniform_(self.G1.weight)
        self.G1.bias.data.fill_(0.01)

        self.G2 = nn.Linear(self.out_dim, self.out_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.G2.weight)

    def forward(self, initial_h, nodeset):
        
        t1 = time.time()
        # relevant_nodes = relevant_nodes_per_layer(
        #     self.g, self.n_items, nodeset, self.n_layers, self.n_hops, self.alpha, self.T)
        relevant_nodes = relevant_nodes_per_layer_precomp(nodeset, self.n_layers, self.T, self.nbhds)
        h = initial_h
        t2 = time.time()
        #print("t:", t2-t1)

        # print(h[nodeset,:][0,:64])

        for i, (nodeset, nb_weights, nb_nodes) in enumerate(relevant_nodes):
            nodeset_new_h = self.conv_layers[i](h, nodeset, nb_nodes, nb_weights)
            h = put_embeddings(h, nodeset, nodeset_new_h)

        nodeset_new_h = self.G2( nn.functional.relu( self.G1(nodeset_new_h) ) )
        h = put_embeddings(h, nodeset, nodeset_new_h)

        t3 = time.time()
        #print("t:", t3-t2)

        return get_embeddings(h, nodeset, self.out_dim) # -> very "functional", keep this in mind
            



if __name__ == "__main__":

    print("hell")

    dataset = SpotifyGraph("./dataset_micro", "./dataset_micro/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()

    batch_size = 128
    sample = torch.randint(0, len(track_ids), (batch_size,))
    nodeset = sample # dgl ids are just indices!, nodeset = batch of nodes

    # visit_prob = sample_neighborhood(g, nodeset, 1000, 0.85)
    # range = sample_hard_negatives(g, visit_prob, 6, 50, 100)
    # print(range) # .values .indices

    initial_h = features
    #print(initial_h[0:5, 0:5])

    n_items = initial_h.shape[0] # number of nodes of type item (not collection)
    n_layers = 2
    in_dim = initial_h.shape[1]
    dimensions = (in_dim, 1024, 512)
    n_hops = 500
    alpha = 0.85
    T = 3

    nbhds = precompute_neighborhoods_topt(g, n_items, n_hops, alpha, DEF_T_PRECOMP, "./neighborhoods_micro.pt")
    pinsage_conv = PinSageModel(g, n_items, n_layers, dimensions, n_hops, alpha, T, nbhds)

    nodeflow = relevant_nodes_per_layer_precomp(nodeset, 2, T, nbhds)

    nodes, w, nb = nodeflow[0]
    track_info = dataset.tracks
    for i in range(0, 10):
        q_id = track_ids[nodes[i]]
        first_id = track_ids[nb[i,0]]
        second_id = track_ids[nb[i,1]]
        print(track_info[q_id]["name"] + " - " + track_info[q_id]["artist"])
        print(track_info[first_id]["name"] + " - " + track_info[first_id]["artist"])
        print(track_info[second_id]["name"] + " - " + track_info[second_id]["artist"])
        print()
