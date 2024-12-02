from operator import index

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import argparse
import tensorflow._api.v2.compat.v1 as tf
from train import GATETrainer

'''
相似度矩阵 matrix 进行二值化，所有大于等于 threshold 的值设置为 1，小于 threshold 的值设置为 0。
'''
def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy

'''
根据输入的网络（邻接矩阵）和节点特征生成图的邻接矩阵和特征矩阵。
'''
def single_generate_graph_adj_and_feature(network, feature):
    """

    :param network: 网络（邻接矩阵）
    :param feature: 特征矩阵
    :return: adj：图的邻接矩阵（稀疏矩阵格式）。 features：节点的特征矩阵
    """
    features = sp.csr_matrix(feature).tolil().todense()

    graph = nx.from_numpy_array(network)
    adj = nx.adjacency_matrix(graph)
    adj = sp.coo_matrix(adj)


    return adj, features

'''
使用 GATE 模型提取图节点的嵌入特征
'''
def get_gate_feature(adj, features, epochs, l):
    args = parse_args(epochs=epochs,l=l)
    feature_dim = features.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_graph_data(adj)
    gate_trainer = GATETrainer(args)
    gate_trainer(G, features, S, R)
    embeddings, attention = gate_trainer.infer(G, features, S, R)
    tf.reset_default_graph()
    return embeddings

'''
准备图数据，将邻接矩阵转化为适合 GATE 模型训练的格式
'''
def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data = adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col


def parse_args(epochs,l):
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run gate.")

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')
    #默认128，64
    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[256,128],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()

if __name__ == '__main__':
    df_disease = pd.read_excel(r'C:\HJH\circPreTest\pythonProject\feature\circRNA_disease_features_128.xlsx', index_col=0)
    df_sim = pd.read_excel(r'C:\HJH\circPreTest\pythonProject\feature\circRNA_disease_cell_similarity.xlsx',header=0,index_col=0)

    feature = df_disease.values
    similarity = df_sim.values
    # 确保 similarity 是数值类型
    similarity = pd.to_numeric(similarity.flatten(), errors='coerce').reshape(df_sim.shape)

    #二值化
    network = sim_thresholding(similarity,0.8)
    adj, features = single_generate_graph_adj_and_feature(network, feature)
    embeddings = get_gate_feature(adj, features,100, 1)
    print(embeddings.shape)

    # 指定要保存的CSV文件的路径
    file_path = r'C:\HJH\circPreTest\pythonProject\feature\gate_feature_circRNAAnddisease_0.8_128_0.01.csv'

    np.savetxt(file_path, embeddings, delimiter=',',)
