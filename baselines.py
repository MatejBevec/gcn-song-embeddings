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

#from snore import SNoRe
#from node2vec import Node2Vec as N2V
import fastnode2vec as fn2v
from torch.nn.modules.distance import CosineSimilarity
# from surprise import Dataset as SpDataset
# from surprise import Reader as SpReader
# from surprise import NormalPredictor, SVD
import implicit
from tqdm import tqdm

from lib.gnns.GNNs_unsupervised import GNN

from spotify_graph import SpotifyGraph
from pinsage_training import PinSage, save_embeddings


class PredictionModel(ABC):
    """Base recommender class."""

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
    """And embedding-based recommender."""
    
    @abstractmethod
    def embed(self, nodeset):
        pass


# --- HELPER METHODS ---

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
    """Nearest graph neighbors via PPR aka. random walks with restarts."""

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
    """Node similarity measures from network analysis."""

    def __init__(self, projected=True):
        self.func = None #a networkx similarity function
        self.projected = projected

    def train(self, g, ids, train_set, test_set, features):
        all_nodes = np.arange(0, len(ids))
        if self.projected:
            adj = project_bipartite_graph(all_nodes, dgl_g=g)
        else:
            adj = g.adj(scipy_fmt="csr")
        self.g = nx.from_scipy_sparse_array(adj)
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
    def __init__(self, projected=True):
        self.func = nx.preferential_attachment
        self.projected = projected

class AdamicAdar(SimpleSimilarity):
    def __init__(self, projected=True):
        self.func = nx.adamic_adar_index
        self.projected = projected

class Preferential(SimpleSimilarity):
    def __init__(self, projected=True):
        self.func = nx.preferential_attachment
        self.projected = projected


class JaccardFast(PredictionModel):

    def __init__(self):
        pass

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

        ctmat = to_col_track_matrix(g, ids)

        self.intersect_sizes = ctmat.transpose() * ctmat
        self.nbh_sizes = self.intersect_sizes.diagonal()
        n = ctmat.shape[1]


    def knn(self, nodeset, k):
        intersect_tensor = torch.from_numpy(self.intersect_sizes[nodeset, :].toarray())
        nbh_tensor = torch.from_numpy(self.nbh_sizes)
        nbh_nodeset_tensor = nbh_tensor[nodeset]
        a = nbh_nodeset_tensor.repeat((nbh_tensor.shape[0], 1)).transpose(1,0)
        b = nbh_tensor.unsqueeze(0).transpose(1, 0).repeat((1, nbh_nodeset_tensor.shape[0])).transpose(1,0)
        union_tensor = a + b - intersect_tensor
        scores = intersect_tensor / (union_tensor + 1e-10)

        topk_scores, topk_nodes = torch.topk(scores, k)
        return topk_scores[:, 1:], topk_nodes[:, 1:]


class FastNode2Vec(EmbeddingModel):
    """Node2vec random walk based node embedding."""

    def __init__(self, projected=True):
        self.model = None
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)
        self.projected = projected

    def train(self, g, ids, train_set, test_set, features):
        print("TRAINING NODE2VEC")
        all_nodes = np.arange(0, len(ids))
        if self.projected:
            adj = project_bipartite_graph(all_nodes, dgl_g=g)
        else:
            adj = g.adj(scipy_fmt="csr")
        nx_g = nx.from_scipy_sparse_array(adj)
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


# class Snore(EmbeddingModel):
#     """Symbolic (explainable) random walk based node embedding."""

#     def __init__(self):
#         self.model = SNoRe(dimension=256, fixed_dimension=True)
#         self.embedding = None
#         self.sim_func = nn.CosineSimilarity(dim=1)

#     def train(self, g, ids, train_set, test_set, features):
#         adj = g.adj(scipy_fmt="csr")
#         sparse_emb = self.model.embed(adj)
#         print(sparse_emb.toarray().size)
#         self.embedding = torch.from_numpy(sparse_emb.toarray())[:len(ids), :]
#         print(self.embedding.shape)
#         #print(self.embedding[0:10,0:20])

#     def embed(self, nodeset):
#         return self.embedding[nodeset, :]

#     def knn(self, nodeset, k):
#         return knn_from_emb(self.embedding, nodeset, k, self.sim_func)


def _load_embeddings(ids, load_dir):
    emb_list = []
    pbar = tqdm(total=len(ids), desc="Loading embeddings")
    bsize = 1000
    for i, track_id in enumerate(ids):
        if i % bsize == 0:
            pbar.update(bsize)
            #print(f"{i}/{len(ids)}")
        load_path = os.path.join(load_dir, track_id + ".pt")
        emb_list.append(torch.load(load_path))

    embedding = torch.stack(emb_list, dim=0)
    pbar.close()
    return embedding


class EmbLoader(EmbeddingModel):
    """Load precomputed embeddings as a recommender method."""

    def __init__(self, load_dir):
        #self.load_dir = os.path.join(run_dir, "emb")
        self.load_dir = load_dir
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)

    def train(self, g, ids, train_set, test_set, features):
        print(f"Loading embeddings from {self.load_dir} ...")
        
        # emb_list = []
        # pbar = tqdm(total=len(ids), desc="Loading embeddings")
        # bsize = 1000
        # for i, track_id in enumerate(ids):
        #     if i % bsize == 0:
        #         pbar.update(bsize)
        #         #print(f"{i}/{len(ids)}")
        #     load_path = os.path.join(self.load_dir, track_id + ".pt")
        #     emb_list.append(torch.load(load_path))

        # self.embedding = torch.stack(emb_list, dim=0)
        # pbar.close()

        self.embedding = _load_embeddings(ids, self.load_dir)

    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)


class PinSageWrapper(EmbeddingModel):
    """A wrapper for training PinSage with the same interface as baseline methods."""

    def __init__(self, train_params=None, run_name=None, log=True):
        self.embedding = None
        self.sim_func = nn.CosineSimilarity(dim=1)
        self.train_params = train_params if train_params else {}
        self.run_name = run_name if run_name else time.strftime("%X %x")
        self.log = log

    def train(self, g, ids, train_set, test_set, features):
        print("Training PinSage with parameters:")
        print(self.train_params)

        self.trainer = PinSage(g, len(ids), features, train_set, log=self.log, load_save=False)

        for param in self.train_params:
            exec(f"self.trainer.{param} = {self.train_params[param]}")
        self.trainer.run_name = self.run_name
        self.trainer.train()
        print("Embedding...")

        # HACK: save embeddings
        track_ids = ids
        n = len(track_ids)
        bsize = 256
        emb_dir = os.path.join("temp_runs", self.run_name, "emb")
        if not os.path.isdir(emb_dir):
            os.makedirs(emb_dir)
        for i in range(0, n, bsize):
            iids = torch.arange(i, min(i+bsize, n))
            emb = self.trainer.embed(iids)
            for id in iids:
                str_id = track_ids[id]
                save_path = os.path.join(emb_dir, str_id + ".pt")
                if os.path.isfile(save_path):
                    continue
                torch.save(emb[id-i, :].clone().detach(), save_path)

        self.embedding = _load_embeddings(ids, emb_dir)

        
    def embed(self, nodeset):
        return self.embedding[nodeset, :]

    def knn(self, nodeset, k):
        return knn_from_emb(self.embedding, nodeset, k, self.sim_func)


class Random(PredictionModel):
    """Gives random recommendations."""

    def __init__(self):
        pass

    def train(self, g, ids, train_set, test_set, features):
        self.ids = ids
        self.n = len(ids)

    def knn(self, nodeset, k):
        #nodes_list = torch.randint(0, self.n-1, size=(nodeset.shape[0], k))
        nodes_list = [torch.randperm(self.n)[0:k] for i in range(0, nodeset.shape[0])]
        nodes = torch.stack(nodes_list, dim=0)
        #print(nodes)
        weights = torch.ones_like(nodes)
        #print(weights)
        return weights, nodes


# --- MORE HELPER METHODS ---

def to_col_track_matrix(g, ids):
    n_tracks = len(ids)
    n_cols = len(g) - n_tracks

    #mat = np.zeros((n_cols, n_tracks))
    mat = lil_matrix((n_cols, n_tracks), dtype=np.int32)
    for col in range(n_tracks, n_tracks+n_cols):
        neighbors = list( set(g.successors(col)) | set(g.predecessors(col)) )
        neighbors = sorted(neighbors)
        mat[col-n_tracks, neighbors] = 1
    mat = mat.tocsr(copy=True)
    return mat

def to_track_track_matrix(ids, positives):
    n = len(ids)
    pos_tuples = [(positives[i, 0], positives[i, 1]) for i in range(positives.shape[0])]
    pos_tuples.sort(key=lambda x: x[1])
    mat = lil_matrix((n, n), dtype=np.int32)
    for (a, b) in pos_tuples:
        if mat[a, b]:
            mat[a, b] += 1
        else:
            mat[a, b] = 1    
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

def project_bipartite_graph(bottom_nodes, dgl_g=None, nx_g=None, adj_mat=None):
    assert dgl_g or nx_g or adj_mat

    print("Projecting bipartite graph...")

    if dgl_g:
        adj_mat = dgl_g.adj(scipy_fmt="csr")
    elif nx_g:
        print(nx_g.nodes())
        adj_mat = nx.adjacency_matrix(nx_g)

    nx_graph = nx.from_scipy_sparse_array(adj_mat)
    proj_graph = bipartite.weighted_projected_graph(nx_graph, bottom_nodes)
    proj_adj_mat = nx.to_scipy_sparse_array(proj_graph, nodelist=None, format="csr")
    
    return proj_adj_mat



class TrackTrackCF(PredictionModel):
    """Matrix factorization of the track-track co-occurence matrix."""

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
    """Matrix factorization of the playlist-track membership matrix."""

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
    """Unsupervised GraphSAGE GCN node embedding."""

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

    dataset = SpotifyGraph("dataset_micro", None)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    #pos = dataset.load_positives("dataset_small/positives_lfm_large.json")

    jaccard = JaccardFast()
    jaccard.train(g, track_ids, None, None, None)
    print(jaccard.knn([1,4,5,6,7,8,12,333], 3))

