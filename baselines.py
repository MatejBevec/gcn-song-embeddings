import os
from os import path
import time
import math
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import scipy as sp
import dgl
import torch
import torch.nn as nn

from snore import SNoRe
from torch.nn.modules.distance import CosineSimilarity
import generate_node_features as cb



class PredictionModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, g, ids, train_set, test_set, features):
        pass

    @abstractmethod
    def knn(self, nodeset, k):
        pass

class EmbeddingModel(PredictionModel):
    @abstractmethod
    def embed(self, nodeset):
        pass


def knn_from_emb_torch(emb, q, k, sim_func):
    # this needs terabytes of memory lol
    q_tensor = q.unsqueeze(1).repeat(1, emb.shape[0], 1)
    emb_tensor = emb.repeat(q.shape[0], 1, 1)
    knn_tensor = sim_func(q_tensor, emb_tensor)
    return knn_tensor

def knn_from_emb(emb, q, k, sim_func):
    w_list = []
    n_list = []
    #print(q[0:10])
    #print(emb[0:10, 0:20])
    for i,node in enumerate(q):
        if i % 100 == 0:
            print(f"{i}/{q.shape[0]}")
        sim = sim_func(emb[node, :].unsqueeze(0), emb)
        #print(sim[0:10])
        knn_w, knn_n = sim.topk(k, 0, largest=True)
        #print(knn_n[:10])
        # now hope garbage collection gets rid of sim
        w_list.append(knn_w)
        n_list.append(knn_n)

    return torch.stack(w_list, 0), torch.stack(n_list, 0)

def knn_from_emb_batches(emb, q, k, sim_func):
    #doesnt work rn

    sim = torch.nn.CosineSimilarity(dim=2)
    b_mem = 1e9 # make batches appr. 1GB in size to fit in memory
    q_size = q.shape[0]
    b_size = int( b_mem / (emb.shape[1]*8) )
    b_size = 32 # larger batches don't seem to give any speedup
    print("b_size:", b_size)
    w_list = []
    n_list = []

    #q.to('cuda')
    #emb.to('cuda')
    #sim.to('cuda')

    for i in range(0, q_size, b_size):
        t1 = time.time()
        q_batch = emb[q[i:i+b_size]]
        print("i=", i)
        col_list = []
        for j in range(0, emb.shape[0], b_size):
            emb_batch = emb[j:j+b_size, :]
            #print(emb_batch.shape)
            #print("j=",j)
            q_tensor = q_batch.unsqueeze(1).repeat(1, emb_batch.shape[0], 1)
            emb_tensor = emb_batch.repeat(q_batch.shape[0], 1, 1)
            knn_tensor = sim_func(q_tensor, emb_tensor)
            col_list.append(knn_tensor)
        
        w, nodes = torch.cat(col_list, dim=1).topk(k, dim=1)
        w_list.append(w)
        n_list.append(nodes)

        print("Elapsed: ", time.time() - t1)

    return torch.cat(w_list, dim=0), torch.cat(n_list, dim=0)


class PersPageRank(PredictionModel):

    def __init__(self):
        self.n_hops = 1000
        self.alpha = 0.85

    def visit_prob(self, g, nodeset, n_hops, alpha):

        n_nodes = nodeset.shape[0]
        trace = torch.zeros(n_nodes, n_hops, dtype=torch.int64)

        for i in range(0, n_nodes):
            if i % 100 == 0:
                print(f"{i}/{n_nodes}")
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

        n_nodes = nodeset.shape[0]
        n_all_nodes = g.number_of_nodes()
        visit_counts = torch.zeros(n_nodes, n_all_nodes, dtype=torch.float64) \
            .scatter_add_(1, trace, torch.ones_like(trace, dtype=torch.float64))
        visit_prob = visit_counts / visit_counts.sum(1, keepdim=True) # or just /n_hops
        visit_prob[range(0, n_nodes), nodeset] = 0

        return visit_prob

    def train(self, g, ids, train_set, test_set, features):
        self.g = g
        # or we could precompute here

    def knn(self, nodeset, k):
        visit_prob = self.visit_prob(self.g, nodeset, self.n_hops, self.alpha)
        return visit_prob.topk(k, 1)


class Node2Vec(EmbeddingModel):

    def __init__(self):
        pass

    def train(self, g, ids, train_set, test_set, features):
        pass

    def embed(self, nodeset):
        pass

    def knn(self, nodeset, k):
        pass

class Snore(EmbeddingModel):

    def __init__(self):
        self.model = SNoRe(dimension=256, fixed_dimension=True)
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)

    def train(self, g, ids, train_set, test_set, features):
        adj = g.adj(scipy_fmt="csr")
        sparse_emb = self.model.embed(adj)
        print(sparse_emb.toarray().size)
        self.embedding = torch.from_numpy(sparse_emb.toarray())[:len(ids), :]
        print(self.embedding.shape)
        #print(self.embedding[0:10,0:20])

    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)


# TEMP - LOAD COMPUTED EMBEDDING
class EmbLoader(EmbeddingModel):

    def __init__(self, load_dir):
        #self.load_dir = os.path.join(run_dir, "emb")
        self.load_dir = load_dir
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)

    def train(self, g, ids, train_set, test_set, features):
        emb_list = []
        for i,track_id in enumerate(ids):
            print(i)
            load_path = os.path.join(self.load_dir, track_id + ".pt")
            emb_list.append(torch.load(load_path))

        self.embedding = torch.stack(emb_list, dim=0)

    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)



if __name__ == "__main__":

    emb = torch.tensor([
        [1,2,3],
        [1,2,2],
        [1,1,0],
        [6,7,3],
        [1,9,0],
        [1,1,1]
    ], dtype=torch.float64)

    q = torch.tensor([0,1])
    
    knn = knn_from_emb(emb, q, 3, torch.nn.CosineSimilarity(dim=1))
    print(knn[0])
    print(knn[1])

    #mem = n_list.element_size() * n_list.nelement()
    #print(f"{mem} bytes")