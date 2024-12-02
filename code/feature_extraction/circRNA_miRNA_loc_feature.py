import csv
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


if __name__ == '__main__':
    df_circRNA_miRNA = pd.read_excel(r'C:\HJH\circPreTest\pythonProject\dataset\circ_mi\circRNA_miRNA_Matrix.xlsx',header=0,index_col=0)
    circRNA_miRNA_matrix = df_circRNA_miRNA.values

    df_miRNA_loc = pd.read_excel(r'C:\HJH\circPreTest\pythonProject\dataset\circ_mi\miRNA_loc.xlsx', header=0,index_col=0)
    miRNA_loc = df_miRNA_loc.values

    print("circRNA_miRNA_matrix shape:", circRNA_miRNA_matrix.shape)
    print("mRNA_loc shape:", miRNA_loc.shape)

    circRNA_miRNA_ratio_vector = []

    for i in range(circRNA_miRNA_matrix.shape[0]):
        # 初始化 temp 列表，7 个位置的计数值，初始为0
        temp = [0, 0, 0, 0, 0, 0, 0]
        for j in range(circRNA_miRNA_matrix.shape[1]):
            if circRNA_miRNA_matrix[i][j] == 1:
                #找对应j行的mRNA的定位情况
                for k in range(7):
                    if miRNA_loc[j][k] == 1:
                        temp[k] += 1
        #处理完了一个miRNA
        sum_number = sum(temp)
        if sum_number != 0:
            for k in range(len(temp)):
                temp[k] = temp[k] / sum_number
        circRNA_miRNA_ratio_vector.append(temp)

    print(circRNA_miRNA_ratio_vector)
    # 指定要写入的CSV文件名
    csv_file = r"C:\HJH\circPreTest\pythonProject\feature\circRNA_miRNA_loc_feature.csv"
    # 将结果保存为 Excel 文件
    output_file = r"C:\HJH\circPreTest\pythonProject\feature\circRNA_miRNA_loc_feature.xlsx"

    # 将结果转换为 pandas DataFrame，列名为位置的标签（7个位置），行名为 circRNA 的名字
    df_result = pd.DataFrame(circRNA_miRNA_ratio_vector, columns=["Cytoplasm", "Exomere", "Extracellular exosome",
                                                                  "Extracellular vesicle", "Microvesicle",
                                                                  "Mitochondrion", "Supermere"],
                             index=df_circRNA_miRNA.index)

    # 保存为 Excel 文件，index=False 表示不保存行名
    df_result.to_excel(output_file, index=True)

    # 打印保存完成的信息
    print(f"处理完成，结果已保存到 {output_file}")
