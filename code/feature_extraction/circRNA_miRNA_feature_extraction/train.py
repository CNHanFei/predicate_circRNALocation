import tensorflow._api.v2.compat.v1 as tf #因为 GATE 使用的是 TensorFlow 1.x 版本的 API
from gate import GATE
import scipy.sparse as sp
tf.disable_eager_execution()

def conver_sparse_tf2np(input):
    # 函数将TensorFlow 稀疏矩阵转换为 SciPy 稀疏矩阵格式
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

'''
假设一个图，包含 100 个节点，节点特征是 128 维向量，图的邻接矩阵表示节点之间的连接关系。
可以将这些数据传递给 GATETrainer 类来训练 GATE 模型，并得到每个节点的嵌入向量
'''
class GATETrainer():

    def __init__(self, args):
        #初始化参数
        self.args = args
        self.build_placeholders()
        gate = GATE(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config) #创建一个新的 TensorFlow 会话，用来执行计算图
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()]) #初始化全局和局部变量


    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)  #使用 Adam 优化器来最小化损失函数
        gradients, variables = zip(*optimizer.compute_gradients(loss))  #计算损失函数相对于各个变量的梯度
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping) #通过梯度裁剪防止梯度爆炸
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))    #应用梯度更新模型参数


    def __call__(self, A, X, S, R):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R)


    def run_epoch(self, epoch, A, X, S, R):

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R}) #将输入数据传递给 TensorFlow 计算图

        return loss

    #推理函数，用于在训练完成后获取模型的嵌入表示 H 和图注意力层 C 的计算结果
    def infer(self, A, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})   #执行计算图，得到最终的节点表示 H 和图注意力结果 C


        return H, conver_sparse_tf2np(C)    #将图注意力结果从 TensorFlow 的稀疏格式转换为 NumPy 格式，以便后续处理




