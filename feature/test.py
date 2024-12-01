from Bio import Align
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# 读取 Excel 文件
df = pd.read_excel('circ_for_seqSim.xlsx')  # 修改为实际文件路径

# 提取细胞名字和序列
cells = df.iloc[:, 0].tolist()  # 细胞名字
sequences = df.iloc[:, 1].tolist()  # 细胞对应的序列

# 创建 PairwiseAligner 对象
aligner = Align.PairwiseAligner()

# 限制每个序列的最大长度，以提高性能（如果适用）
# max_length = 200  # 可以根据实际情况调整
# sequences = [seq[:max_length] for seq in sequences]  # 截取序列的前100个碱基

# 定义计算相似性的函数
def smith_waterman_similarity(seq1, seq2):
    # 使用 PairwiseAligner 计算局部对齐
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]  # 获取最佳对比结果
    sp_mi_mj = best_alignment.score  # 局部比对得分

    # 计算归一化的相似性得分
    sp_mi_mi = aligner.align(seq1, seq1)[0].score  # 细胞 mi 本身的比对得分
    sp_mj_mj = aligner.align(seq2, seq2)[0].score  # 细胞 mj 本身的比对得分

    # 计算相似性得分
    similarity_score = sp_mi_mj / math.sqrt(sp_mi_mi * sp_mj_mj)
    return similarity_score


# 定义并行计算的函数
def compute_similarity_for_pair(pair):
    i, j = pair
    similarity = smith_waterman_similarity(sequences[i], sequences[j])
    return (i, j, similarity)


# 计算并存储相似度矩阵的上三角部分（并行计算）
def calculate_and_save_upper_triangle_similarity_matrix():
    # 初始化相似性矩阵
    num_cells = len(cells)

    # 输出文件路径
    output_file = '_disease_cell_similarity.xlsx'

    # 创建一个空的 DataFrame 来存储上三角部分的相似度
    similarity_matrix = np.zeros((num_cells, num_cells))

    # 生成要计算的细胞对（i, j），只包含上三角部分
    pairs = [(i, j) for i in range(num_cells) for j in range(i + 1, num_cells)]

    # 使用 Pool 进行并行计算
    with Pool(processes=16) as pool:
        # 使用进度条
        with tqdm(total=len(pairs), desc="计算相似度", unit="对") as pbar:
            results = []
            for result in pool.imap(compute_similarity_for_pair, pairs):
                results.append(result)
                pbar.update(1)

    # 填充相似度矩阵
    for i, j, similarity in results:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # 因为矩阵是对称的

    # 将相似性矩阵转换为 DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=cells, columns=cells)

    # 将相似性矩阵保存到 Excel 文件
    similarity_df.to_excel(output_file, index=True)

    print(f"相似性矩阵已保存到 {output_file}")


if __name__ == '__main__':
    calculate_and_save_upper_triangle_similarity_matrix()
