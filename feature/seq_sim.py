import pandas as pd
from Bio import pairwise2
import math
import numpy as np

# 读取 Excel 文件
df = pd.read_excel('../dataset/circ_seq/seq1_unique_cells_and_sequences.xlsx')

# 提取细胞名字和序列
cells = df.iloc[:, 0].tolist()  # 细胞名字
sequences = df.iloc[:, 1].tolist()  # 细胞对应的序列


# 定义计算相似性的函数
def smith_waterman_similarity(seq1, seq2):
    # 计算 Smith-Waterman 比对得分
    alignments = pairwise2.align.localxx(seq1, seq2)
    best_alignment = alignments[0]  # 获取最佳对比结果
    sp_mi_mj = best_alignment[2]  # 局部比对得分

    # 计算归一化的相似性得分
    sp_mi_mi = pairwise2.align.localxx(seq1, seq1)[0][2]  # miRNA mi 本身的比对得分
    sp_mj_mj = pairwise2.align.localxx(seq2, seq2)[0][2]  # miRNA mj 本身的比对得分

    # 计算相似性得分
    similarity_score = sp_mi_mj / math.sqrt(sp_mi_mi * sp_mj_mj)
    return similarity_score


# 初始化相似性矩阵
num_cells = len(cells)
print(num_cells)
similarity_matrix = np.zeros((num_cells, num_cells))

# 计算相似性矩阵
for i in range(num_cells):
    for j in range(i, num_cells):
        similarity = smith_waterman_similarity(sequences[i], sequences[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # 因为矩阵是对称的

print(similarity_matrix)

# 将相似性矩阵保存为 DataFrame 并写入 Excel
similarity_df = pd.DataFrame(similarity_matrix, index=cells, columns=cells)
similarity_df.to_excel('cell_similarity_matrix.xlsx', index=True)

print("相似性矩阵已保存为 'cell_similarity_matrix.xlsx'")
