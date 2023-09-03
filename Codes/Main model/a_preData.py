import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm

class PREPROCESS:
    def __init__(self, data, label):
        self.data = data
        self.data = self.data.sort_index()
        # del_threshold = 0.06
        # self.del_threshold_ubiq = self.data.shape[1] * (1-del_threshold)
        # self.del_threshold_rare = self.data.shape[1] * del_threshold
        # ## filter genes whose expression value > 0 in at least 94% cells
        # self.data = self.data[np.sum(self.data > 0, axis=1) < self.del_threshold_ubiq]   
        # ## filter genes whose expression value > 2 in less than 6% cells
        # self.data = self.data[np.sum(self.data > 2, axis=1) > self.del_threshold_rare]
        self.X = preprocessing.scale(np.array(self.data), axis = 1)
        self.gene = np.array(self.data.index.tolist())
        self.label = label
        
    def buildNetwork(self):
        self.__loadData()
        self.A = self.computeAdj()
        self.D, self.dist = self.getLaplacian(self.A)
        self.L = self.laplacianNormalize(self.A, self.D)
        
    def __loadData(self):
        self.network = pd.read_table('data/HomoSapiens_binary_hq.txt')    # pd.read_table('HomoSapiens_binary_hq.txt')
        self.network = self.network.loc[:, ['Gene_A', 'Gene_B']]
        self.__getNetworkPartition()
        
    def __getNetworkPartition(self):
        network_ind = []
        for i in range(self.network.shape[0]):
            if self.network.iloc[i, 0] in self.gene and self.network.iloc[i, 1] in self.gene:
                network_ind.append(i)

        self.network = self.network.iloc[network_ind, :]
        self.network = self.network[self.network['Gene_A'] != self.network['Gene_B']]
        
    def computeAdj(self):
        adj = np.zeros((len(self.gene), len(self.gene)))
        for i in range(self.network.shape[0]):
            gene1 = self.network.iloc[i, 0]
            gene2 = self.network.iloc[i, 1]
            gene1_ind = np.where(gene1 == self.gene)[0][0]
            gene2_ind = np.where(gene2 == self.gene)[0][0]
            adj[gene1_ind, gene2_ind] = 1

        adj += adj.T - np.diag(np.diag(adj))
        adj[np.where(adj == 2)] = 1
        return adj
        
    def getLaplacian(self, adjacency_matrix):
        rows, cols = np.where((adjacency_matrix == 1) | (adjacency_matrix == 2))
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        all_rows = range(0, adjacency_matrix.shape[0])
        # classnames, labels = np.unique(label, return_inverse=True)
        # node_color = get_color(labels)
        for n in all_rows:
            G.add_node(n)
        G.add_edges_from(edges)
        dist = nx.all_pairs_shortest_path_length(G)
        dist = list(dist)
        m = len(dist)
        degree_matrix = np.zeros((m, m))
        distance_matrix = np.zeros((m, m))
        for i in range(m):
            values = np.array(list(dist[i][1].values()))
            degree_matrix[i, i] = sum(values == 1)
            values = np.array(list(dist[i][1].values()))
            position = zip([i]*len(dist[i][1].keys()), list(dist[i][1].keys()))
            position = np.array(list(position))
            distance_matrix[position[:, 0], position[:, 1]] = list(dist[i][1].values())
            
        distance_matrix[np.where(distance_matrix == 0)] = -1
        distance_matrix = distance_matrix + np.diag(-np.diag(distance_matrix))
        return degree_matrix, distance_matrix
    
    def laplacianNormalize(self, A, D):
        L = np.identity(D.shape[0]) - np.dot(np.dot(np.linalg.inv(D ** 0.5 + np.diag(np.array(np.zeros(D.shape[0]) + 0.001))), A), np.linalg.inv(D ** 0.5 + np.diag(np.array(np.zeros(D.shape[0]) + 0.001))))
        return L
        