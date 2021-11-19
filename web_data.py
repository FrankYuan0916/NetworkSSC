import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp  # 稀疏矩阵
import networkx as nx


def split_str(data):
    return data.split()

def pre_data(dataset="cornell", path="C:\\Users\\THINK\\Desktop\\RA_Yu Tianwei\\SSC 5\\"):
    
    # idx_features_labels = path+dataset.content = ./data/cora/cora.content
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    # idx_features_labels = idx_features_labels
    # 压缩稀疏矩阵，变成一个feature的matrix
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    onehot_labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0])
    # j: label  i: index，给每个
    idx_map = {j: i for i, j in enumerate(idx)}

    # 返回连接两个node的edge
    cite = pd.read_csv('{}{}.cites'.format(path, dataset), header=None)
    edges_origin = np.array(list(map(split_str, cite.iloc[:,0])))
    # 变成一维数组
    flat = edges_origin.flatten()
    # 将原先的label替换为index
    edges = np.array(list(map(idx_map.get, flat)), dtype=np.int32).reshape(edges_origin.shape)
    # 生成邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
    # 转化为对称形式
    adj += adj.T - sp.diags(adj.diagonal())

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    # y_train, y_val, y_test, train_mask, val_mask, test_mask = split_data(onehot_labels)

    return adj, features, idx_features_labels[:, -1] #, y_train, y_val, y_test, train_mask, val_mask, test_mask


def encode_onehot(labels):
    unique = set(labels)
    seq = enumerate(unique)
    uni_dict = {id: np.identity(len(unique))[i,:] for i, id in seq}
    # 根据labels得到对应的onehot encode
    onehot_labels = np.array(list(map(uni_dict.get, labels)), dtype=np.int32)
    # print(onehot_labels)
    return onehot_labels

def get_laplacian(adjacency_matrix, label, draw_plot = True):
    rows, cols = np.where((adjacency_matrix == 1) | (adjacency_matrix == 2))  # 2是因为有双向引用的（？？
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    # classnames, labels = np.unique(label, return_inverse=True)
    # node_color = get_color(labels)
    for n in all_rows:
        G.add_node(n) 
    G.add_edges_from(edges)
    # if draw_plot == True:      
    #     plt.figure(figsize=((15,15)))
    #     nx.draw(G, with_labels=False,node_color=node_color)
    #     plt.show()
    dist = nx.all_pairs_shortest_path_length(G)
    dist = list(dist)
    m = len(dist)
    degree_matrix = np.zeros((m, m))
    distance_matrix = np.zeros((m, m))
    for i in range(m):
        values = np.array(list(dist[i][1].values()))
        degree_matrix[i, i] = sum(values == 1)
        key = list(dist[i][1].keys())
        values = np.array(list(dist[i][1].values()))
        position = zip([i]*len(dist[i][1].keys()), list(dist[i][1].keys()))
        position = np.array(list(position))
        distance_matrix[position[:,0], position[:,1]] = list(dist[i][1].values())
    laplacian_matrix = degree_matrix - adjacency_matrix
        
    distance_matrix[np.where(distance_matrix == 0)] = 1000
    distance_matrix = distance_matrix + np.diag(-np.diag(distance_matrix))
    return adjacency_matrix, degree_matrix, laplacian_matrix, distance_matrix