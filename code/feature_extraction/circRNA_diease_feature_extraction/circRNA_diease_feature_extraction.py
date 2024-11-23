import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    df_circRNA_dis = pd.read_excel('../../../dataset/circ_disease/circRNA_Disease_Matrix_noHead.xlsx', header=None)
    circRNA_dis_matrix = df_circRNA_dis.values
    print(circRNA_dis_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加circRNA节点
    for i in range(1668):
        G.add_node(f"circRNA_{i}")

    # 添加疾病节点
    for i in range(271):
        G.add_node(f"disease_{i}")

    # 遍历矩阵，添加边
    for circRNA_index in range(1668):
        for disease_index in range(271):
            interaction = circRNA_dis_matrix[circRNA_index, disease_index]
            if interaction != 0:
                G.add_edge(f"miRNA_{circRNA_index}", f"disease_{disease_index}")

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=8)

    # 训练模型
    model = node2vec.fit()

    # 提取 circRNA 特征向量
    circRNA_features = {}
    for circRNA_node in range(1668):
        circRNA_node_str = f"circRNA_{circRNA_node}"
        circRNA_features[circRNA_node_str] = model.wv[circRNA_node_str]

    df = pd.DataFrame.from_dict(circRNA_features, orient='index')
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名可以根据需要自定义
    print(df)
    df.to_excel('../../../feature/miRNA_disease_feature_128.xlsx')