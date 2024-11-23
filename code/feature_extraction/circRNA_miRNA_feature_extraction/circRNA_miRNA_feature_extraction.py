import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    df_circRNA_miRNA = pd.read_excel('../../../dataset/circ_mi/circRNA_miRNA_Matrix_noHead.xlsx', header=None)
    circRNA_miRNA_matrix = df_circRNA_miRNA.values
    print(circRNA_miRNA_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加circRNA节点
    for i in range(703):
        G.add_node(f"circRNA_{i}")

    # 添加miRNA节点
    for i in range(1859):
        G.add_node(f"miRNA{i}")

    # 遍历矩阵，添加边
    for circRNA_index in range(703):
        for miRNA_index in range(1859):
            interaction = circRNA_miRNA_matrix[circRNA_index, miRNA_index]
            if interaction != 0:
                G.add_edge(f"miRNA_{circRNA_index}", f"miRNA_{miRNA_index}")

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=8)

    # 训练模型
    model = node2vec.fit()

    # 提取 circRNA 特征向量
    circRNA_features = {}
    for circRNA_node in range(703):
        circRNA_node_str = f"circRNA_{circRNA_node}"
        circRNA_features[circRNA_node_str] = model.wv[circRNA_node_str]

    df = pd.DataFrame.from_dict(circRNA_features, orient='index')
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名可以根据需要自定义
    print(df)
    df.to_excel('../../../feature/miRNA_miRNA_feature_128.xlsx')