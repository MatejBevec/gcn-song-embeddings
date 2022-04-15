import os
from os import path
import time
import math
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix
import dgl
import torch
import torch.nn as nn
import networkx as nx
from networkx.algorithms import bipartite

from snore import SNoRe
from node2vec import Node2Vec as N2V
import fastnode2vec as fn2v
from torch.nn.modules.distance import CosineSimilarity
from surprise import Dataset as SpDataset
from surprise import Reader as SpReader
from surprise import NormalPredictor, SVD
import implicit
from tqdm import tqdm

from lib.gnns.GNNs_unsupervised import GNN

from spotify_graph import SpotifyGraph



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


def cosine_sim_mat(batch, eps=1e-10):
    # given a batch of vectors (n*d), return n*n matrix of pairwise similaries
    n, d = batch.shape
    dot_prod = torch.mm( batch, batch.transpose(1, 0) )
    lengths = torch.norm(batch, dim=1)
    lengths_mat = lengths.unsqueeze(1).repeat((1, n))
    lengths_mul = lengths_mat * lengths_mat.transpose(1, 0)

    cosine_sim = dot_prod / lengths_mul
    return cosine_sim

def cosine_sim_ab(a, b, eps=1e-16):
    dot_prod = torch.mm(a, b.transpose(1, 0))
    a_lengths = torch.norm(a, dim=1)
    b_lengths = torch.norm(b, dim=1)
    lengths_mat = torch.mm(a_lengths.unsqueeze(1), b_lengths.unsqueeze(0))
    lengths_mat = lengths_mat + eps

    cosine_sim = dot_prod / lengths_mat
    return cosine_sim



def knn_from_emb_slow(emb, q, k, sim_func):
    w_list = []
    n_list = []
    for i,node in enumerate(q):
        sim = sim_func(emb[node, :].unsqueeze(0), emb)
        knn_w, knn_n = sim.topk(k+1, 0, largest=True)
        # now hope garbage collection gets rid of sim
        w_list.append(knn_w[1:])
        n_list.append(knn_n[1:])

    return torch.stack(w_list, 0), torch.stack(n_list, 0)

def knn_from_emb(emb, q, k, sim_func):
    w_list = []
    n_list = []
    b_size = 128
    for i in range(0, len(q), b_size):
        nodes = q[i:min(i+b_size, len(q))]
        q_batch = emb[nodes, :]
        cosine_sim = cosine_sim_ab(q_batch, emb)
        knn_w, knn_n = cosine_sim.topk(k+1, dim=1, largest=True)
        w_list.append(knn_w[:, 1:])
        n_list.append(knn_n[:, 1:])
    
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

class SimpleSimilarity(PredictionModel):

    def __init__(self):
        self.func = None #a networkx similarity function

    def train(self, g, ids, train_set, test_set, features):
        #g_hom = dgl.to_homogeneous(g)
        #self.g = dgl.to_networkx(g_hom).to_undirected()
        # bodge
        adj = g.adj(scipy_fmt="csr")
        self.g = nx.from_scipy_sparse_matrix(adj)
        self.n = len(ids)

    def knn(self, nodeset, k):
        knn_list = []
        for q in nodeset:
            pairs = [(q.item(), n2) for n2 in range(0, self.n)]
            scores = self.func(self.g, pairs)
            knn_to_q = torch.tensor( [sc[2] for sc in scores] )
            knn_list.append(knn_to_q)
        return torch.stack(knn_list, dim=0).topk(k, dim=1)

class JaccardIndex(SimpleSimilarity):
    def __init__(self):
        self.func = nx.preferential_attachment

class AdamicAdar(SimpleSimilarity):
    def __init__(self):
        self.func = nx.adamic_adar_index

class Preferential(SimpleSimilarity):
    def __init__(self):
        self.func = nx.preferential_attachment


# class Node2Vec(EmbeddingModel):

#     def __init__(self, projected=True):
#         self.model = None
#         self.embedding = None
#         self.sim_func = nn.CosineSimilarity(dim=1)
#         self.projected = projected

#     def train(self, g, ids, train_set, test_set, features):
#         all_nodes = np.arange(0, len(ids))
#         if self.projected:
#             adj = project_bipartite_graph(all_nodes, dgl_g=g)
#         else:
#             adj = g.adj(scipy_fmt="csr")
#         self.g = nx.from_scipy_sparse_matrix(adj)
#         #self.g = dgl.to_networkx(g)
#         self.model = N2V(self.g, dimensions=64, walk_length=20, num_walks=200, workers=4)
#         # do one of the above scramble the ids?
#         self.wv = self.model.fit(window=10, min_count=1, batch_words=4).wv
#         vec_list = []
#         for i in range(0, len(ids)):
#             vec_list.append(torch.tensor(self.wv.get_vector(i)))
#         self.embedding = torch.stack(vec_list, dim=0)
#         print(self.embedding)

#     def embed(self, nodeset):
#         return self.embedding[nodeset, :]

#     def knn(self, nodeset, k):
#         return knn_from_emb(self.embedding, nodeset, k, self.sim_func)

class FastNode2Vec(EmbeddingModel):

    def __init__(self, projected=True):
        self.model = None
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)
        self.projected = projected

    def train(self, g, ids, train_set, test_set, features):
        all_nodes = np.arange(0, len(ids))
        if self.projected:
            adj = project_bipartite_graph(all_nodes, dgl_g=g)
        else:
            adj = g.adj(scipy_fmt="csr")
        nx_g = nx.from_scipy_sparse_matrix(adj)
        edges = [(e[0], e[1], e[2]["weight"]) for e in nx_g.edges(data=True)]
        self.g = fn2v.Graph(edges, directed=False, weighted=self.projected)
        self.model = fn2v.Node2Vec(self.g, dim=128, walk_length=20, context=10, p=2.0, q=0.5, workers=4)
        self.model.train(epochs=10)
        self.wv = self.model.wv
        vec_list = []
        for i in range(0, len(ids)):
            vector = self.wv.get_vector(i)
            vec_list.append(torch.tensor(vector))
        self.embedding = torch.stack(vec_list, dim=0)

    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)

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
        print(f"Loading embeddings from {self.load_dir} ...")
        emb_list = []

        pbar = tqdm(total=len(ids), desc="Loading embeddings")
        bsize = 1000
        for i, track_id in enumerate(ids):
            if i % bsize == 0:
                pbar.update(bsize)
                #print(f"{i}/{len(ids)}")
            load_path = os.path.join(self.load_dir, track_id + ".pt")
            emb_list.append(torch.load(load_path))

        self.embedding = torch.stack(emb_list, dim=0)
        pbar.close()

    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)

class Random(PredictionModel):

    def __init__(self):
        pass

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

    def knn(self, nodeset, k):
        #nodes_list = torch.randint(0, self.n-1, size=(nodeset.shape[0], k))
        nodes_list = [torch.randperm(self.n)[0:k] for i in range(0, nodeset.shape[0])]
        nodes = torch.stack(nodes_list, dim=0)
        print(nodes)
        weights = torch.ones_like(nodes)
        print(weights)
        return weights, nodes

# CF METHODS

def to_col_track_matrix(g, ids):
    n_tracks = len(ids)
    n_cols = len(g) - n_tracks

    #mat = np.zeros((n_cols, n_tracks))
    mat = lil_matrix((n_cols, n_tracks), dtype=np.int32)
    for col in range(n_tracks, n_tracks+n_cols):
        neighbors = list( set(g.successors(col)) | set(g.predecessors(col)) )
        neighbors = sorted(neighbors)
        mat[col-n_tracks, neighbors] = 1
    print("matrix built")
    mat = mat.tocsr(copy=True)
    return mat

def to_track_track_matrix(ids, positives):
    n = len(ids)
    #mat = np.zeros((n, n))
    print("building matrix")
    pos_tuples = [(positives[i, 0], positives[i, 1]) for i in range(positives.shape[0])]
    pos_tuples.sort(key=lambda x: x[1])
    mat = lil_matrix((n, n), dtype=np.int32)
    for (a, b) in pos_tuples:
        if mat[a, b]:
            mat[a, b] += 1
        else:
            mat[a, b] = 1
    print("matrix built")
    mat = mat.tocsr(copy=True)
    return mat

def to_ratings_df(mat):
    ratings = []
    for col in range(mat.shape[0]):
        for tr in range(mat.shape[1]):
            if mat[col, tr] > 0:
                ratings.append({"userID": col, "itemID": tr, "rating": mat[col, tr]})
    #df = pd.DataFrame.from_dict(ratings, orient="index")
    df = pd.DataFrame(ratings)
    print("df built")
    return df
            
# def to_surprise_trainset(df):
#     reader = SpReader(rating_scale=(0, 5))
#     data = SpDataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
#     trainset = data.build_full_trainset()
#     print("trainset built")
#     return trainset

# def new_surprise_testset(ids, sample):
#     n = len(ids)
#     mat = np.zeros((n, n))
#     mat[sample, :] = 1
#     df = to_ratings_df(mat)
#     trainset = to_surprise_trainset(df)
#     return trainset.build_testset()

# def surprise_pred_to_knn(predictions, sample, n_items, k):
#     sample = sample.numpy()
#     rating_mat = np.zeros((len(sample), n_items))
#     global_to_batch_map = {sample[i]: i for i in range(len(sample))}
#     for uid, iid, true_r, est, _ in predictions:
#         rating_mat[global_to_batch_map[uid], iid] = est
    
#     topn_scores = np.flip(np.sort(rating_mat, axis=1), axis=1).copy()[:, :k]
#     topn_nodes = np.flip(np.argsort(rating_mat, axis=1), axis=1).copy()[:, :k]
#     return topn_scores, topn_nodes


def project_bipartite_graph(bottom_nodes, dgl_g=None, nx_g=None, adj_mat=None):
    assert dgl_g or nx_g or adj_mat

    print("Projecting bipartite graph...")

    if dgl_g:
        adj_mat = dgl_g.adj(scipy_fmt="csr")
    elif nx_g:
        print(nx_g.nodes())
        adj_mat = nx.adjacency_matrix(nx_g)

    nx_graph = nx.from_scipy_sparse_matrix(adj_mat)
    proj_graph = bipartite.weighted_projected_graph(nx_graph, bottom_nodes)
    proj_adj_mat = nx.to_scipy_sparse_matrix(proj_graph, nodelist=None, format="csr")
    
    return proj_adj_mat


# class TrackTrackSVD(PredictionModel):

#     def __init__(self):
#         pass

#     def train(self, g, ids, train_set, test_set, features):
#         self.ids = ids
#         self.n = len(ids)
        
#         ratings_df = to_ratings_df(to_track_track_matrix(ids, train_set))
#         print(ratings_df.sort_values("rating", ascending=False))
#         trainset = to_surprise_trainset(ratings_df)
#         self.algo = SVD()
#         self.algo.fit(trainset)

#     def knn(self, nodeset, k):
        
#         testset = new_surprise_testset(self.ids, nodeset)
#         predictions = self.algo.test(testset)
#         topn_scores, topn_nodes = surprise_pred_to_knn(predictions, nodeset, len(self.ids), k)
#         return torch.from_numpy(topn_scores), torch.from_numpy(topn_nodes)


class TrackTrackCF(PredictionModel):

    def __init__(self, algo="als", factors=128):
        #os.system("export OPENBLAS_NUM_THREADS=1")
        #os.system("export MKL_NUM_THREADS=1")
        self.algo = algo
        self.factors = factors

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

        ttmat = to_track_track_matrix(ids, train_set)
        
        if self.algo == "als":
            self.model = implicit.cpu.als.AlternatingLeastSquares(factors=self.factors)
        elif self.algo == "lmf":
            self.model = implicit.cpu.lmf.LogisticMatrixFactorization(factors=self.factors)
        else:
            self.model = implicit.cpu.bpr.BayesianPersonalizedRanking(factors=self.factors)

        self.model.fit(ttmat)

    def knn(self, nodeset, k):
        
        topn_nodes, topn_scores = self.model.similar_items(nodeset, N=k+1)
        #print(topn_scores)
        #print(topn_nodes)
        return torch.from_numpy(topn_scores)[:, 1:], torch.from_numpy(topn_nodes)[:, 1:]

class ColTrackCF(PredictionModel):

    def __init__(self, algo="als", factors=128):
        self.algo = algo
        self.factors = factors

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

        ctmat = to_col_track_matrix(g, ids)

        if self.algo == "als":
            self.model = implicit.cpu.als.AlternatingLeastSquares(factors=self.factors)
        elif self.algo == "lmf":
            self.model = implicit.cpu.lmf.LogisticMatrixFactorization(factors=self.factors)
        else:
            self.model = implicit.cpu.bpr.BayesianPersonalizedRanking(factors=self.factors)

        self.model.fit(ctmat)

    def knn(self, nodeset, k):

        topn_nodes, topn_scores = self.model.similar_items(nodeset, N=k+1)
        return torch.from_numpy(topn_scores)[:, 1:], torch.from_numpy(topn_nodes)[:, 1:]


class GraphSAGE(EmbeddingModel):

    def __init__(self, projected=True):
        self.projected = projected

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

        all_nodes = np.arange(0, len(ids))
        self.features = features.clone().detach().numpy()

        if self.projected:
            adj = project_bipartite_graph(all_nodes, dgl_g=g)
        else:
            adj = g.adj(scipy_fmt="csr")

        self.model = GNN(self.proj_adj_mat, features=self.features, supervised=False, model="graphsage", device="cuda")
        self.model.fit()
        self.embedding = self.mode.generate_embeddings()

    def embed(self, nodeset):
        #nn.CosineSimilarity(dim=1)
        return torch.from_numpy(self.embedding[nodeset, :])

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, nn.CosineSimilarity(dim=1))


if __name__ == "__main__":

    # emb = torch.tensor([
    #     [1,2,3],
    #     [1,2,2],
    #     [1,1,0],
    #     [6,7,3],
    #     [1,9,0],
    #     [1,1,1]
    # ], dtype=torch.float64)

    # q = torch.tensor([0,1])
    # knn = knn_from_emb(emb, q, 3, torch.nn.CosineSimilarity(dim=1))

    dataset = SpotifyGraph("dataset_micro", None)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    # pos = dataset.load_positives("dataset_small/positives_lfm_large.json")

    adj_mat = project_bipartite_graph(np.arange(0, len(track_ids)), dgl_g=g)
    nx_g = nx.from_scipy_sparse_matrix(adj_mat)

    edges = [(e[0], e[1], e[2]["weight"]) for e in nx_g.edges(data=True)]
    for e in edges:
        print(e)

    # sample = torch.randperm(len(track_ids))[:100]

    # model = TrackTrackALS()
    # model.train(g, track_ids, pos, pos, features)

    # knn = model.knn(sample, 3)
    # print(knn)

    # emb = torch.tensor([
    #     [1,5,3],
    #     [4,4,5],
    #     [-1,9,0],
    #     [-1, 2, 2],
    #     [-2, 0, 1],
    #     [1, 5, 6],
    #     [5,8,1],
    #     [9,8,7]
    # ]).float()

    # q = [0, 2, 4]

    # knn_w, knn_n = knn_from_emb(emb, q, 3, None)
    # print(knn_w, knn_n)

    # knn_w, knn_n = knn_from_emb_slow(emb, q, 3, nn.CosineSimilarity(dim=1))
    # print(knn_w, knn_n)    