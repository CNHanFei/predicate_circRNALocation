import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':

    # 读取 Excel 文件
    df_circRNA_miRNA = pd.read_excel('../../../dataset/circ_disease/circRNA_Disease_Matrix.xlsx', header=None)

    # 提取矩阵和节点名字
    circRNA_miRNA_matrix = df_circRNA_miRNA.iloc[1:, 1:].values  # 关联矩阵
    circRNA_names = df_circRNA_miRNA.iloc[1:, 0].values  # circRNA 名字
    disease_names= df_circRNA_miRNA.iloc[0, 1:].values  # disease 名字
    print(f"矩阵大小: {circRNA_miRNA_matrix.shape}")
    print(f"circRNA 数量: {len(circRNA_names)}, disease 数量: {len(disease_names)}")
    print(circRNA_names)

    # 创建图
    G = nx.Graph()

    # 添加节点
    for circRNA_name in circRNA_names:
        G.add_node(circRNA_name)  # circRNA 节点
    for disease_name in disease_names:
        G.add_node(disease_name)  # disease 节点

    # 添加边
    for i, circRNA_name in enumerate(circRNA_names):
        for j, disease_name in enumerate(disease_names):
            if circRNA_miRNA_matrix[i, j] == 1:  # 如果有关联，添加边
                G.add_edge(circRNA_name, disease_name)

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=8)

    # 训练模型
    model = node2vec.fit()

    # 提取 circRNA 特征向量
    circRNA_features = {}
    for circRNA_name in circRNA_names:
        circRNA_features[circRNA_name] = model.wv[circRNA_name]  # 提取 circRNA 的特征向量

    # 转换为 DataFrame
    df = pd.DataFrame.from_dict(circRNA_features, orient='index')
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 添加特征列名
    df.index.name = 'circRNA_Name'  # 设置索引名为 circRNA 名字
    df.reset_index(inplace=True)  # 将索引变为第一列

    # 输出到 Excel
    output_path = '../../../feature/circRNA_disease_features_128.xlsx'
    df.to_excel(output_path, index=False)
    print(f"特征向量保存至 {output_path}")