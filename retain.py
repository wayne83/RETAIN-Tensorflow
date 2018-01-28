#coding:utf-8
import tensorflow as tf
import numpy as np

class Retain(object):

    def __init__(self,inputDimSize,embDimSize,alphaHiddenDimSize,betaHiddenDimSize,outputDimSize,keep_prob=0.8,
                 L2=1e-8,opt=tf.train.AdadeltaOptimizer(learning_rate=10),init_scale=0.01):
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.outputDimSize = outputDimSize
        self.L2 = L2
        self.opt = opt
        self.init_scale = init_scale
        self.keep_prob = keep_prob
        self.opt = opt

        self.tparams = {}

        # 第一层网络的降维，可载入之前的模型初始化
        self.tparams["W_emb"] = tf.Variable(tf.random_normal([inputDimSize,embDimSize],stddev=self.init_scale,dtype=tf.float32),name="W_emb")
        self.tparams["b_emb"] = tf.Variable(tf.random_normal([embDimSize,],stddev=self.init_scale,dtype=tf.float32),name="b_emb")

        # 初始化RNNα
        self.tparams["W_gru_a"] = tf.Variable(
            tf.random_normal([self.embDimSize,3*self.alphaHiddenDimSize],stddev=self.init_scale,dtype=tf.float32),name="W_gru_a")
        self.tparams["U_gru_a"] = tf.Variable(
            tf.random_normal([alphaHiddenDimSize,3*alphaHiddenDimSize],stddev=self.init_scale,dtype=tf.float32),name="U_gru_a")
        self.tparams["b_gru_a"] = tf.Variable(tf.random_normal([3*alphaHiddenDimSize,],stddev=self.init_scale,dtype=tf.float32),name="b_gru_a")

        #初始化RNNβ
        self.tparams["W_gru_b"] = tf.Variable(
            tf.random_normal([self.embDimSize, 3 * self.betaHiddenDimSize],stddev=self.init_scale, dtype=tf.float32),name="W_gru_b")
        self.tparams["U_gru_b"] = tf.Variable(
            tf.random_normal([betaHiddenDimSize, 3 * betaHiddenDimSize], stddev=self.init_scale, dtype=tf.float32),name="U_gru_b")
        self.tparams["b_gru_b"] = tf.Variable(
            tf.random_normal([3 * betaHiddenDimSize,], stddev=self.init_scale, dtype=tf.float32),name="b_gru_b")

        # 初始化RNNα与RNNβ后的网络结构
        self.tparams["w_alpha"] = tf.Variable(tf.random_normal([alphaHiddenDimSize,1],stddev=self.init_scale,dtype=tf.float32),name="w_alpha")
        self.tparams["b_alpha"] = tf.Variable(tf.random_normal([1,],stddev=self.init_scale,dtype=tf.float32),name="b_alpha")
        self.tparams["w_beta"] = tf.Variable(tf.random_normal([betaHiddenDimSize,embDimSize],stddev=self.init_scale,dtype=tf.float32),name="w_beta")
        self.tparams["b_beta"] = tf.Variable(tf.random_normal([embDimSize,],stddev=self.init_scale,dtype=tf.float32),name="b_beta")

        # 最后一层输出网络
        self.tparams["w_output"] = tf.Variable(tf.random_normal([self.embDimSize,self.outputDimSize],stddev=self.init_scale,dtype=tf.float32),name="w_output")
        self.tparams["b_output"] = tf.Variable(tf.random_normal([self.outputDimSize,],stddev=self.init_scale,dtype=tf.float32),name="b_output")

        #初始化输入
        self.x = tf.placeholder(tf.float32,[None,None,self.inputDimSize],name="x_input")   # x是用于训练的3维向量，maxlen × n_samples × inputDimSize
        self.y = tf.placeholder(tf.float32,[None,None,self.outputDimSize],name="y_label")  # y与x相差一天，用于计算x预测的误差用于反向传播
        self.length = tf.placeholder(tf.int32,[None])   # 输入batch所有病人的visit数目
        self.counts = tf.placeholder(tf.int32,[None],name="counts")
        self.first_h_a = tf.placeholder(tf.float32, [None, self.alphaHiddenDimSize])
        self.first_h_b = tf.placeholder(tf.float32, [None, self.betaHiddenDimSize])
        self.first_ch = tf.placeholder(tf.float32,  [None, self.embDimSize])

        self.loss = self.build_model()

        self.opt = self.opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _slice(self,x,n,dim):
        return x[:,n*dim:(n+1)*dim]

    # RNN的GRU门，传入的emb是按照时间顺序倒序
    # 返回len × n_samples × hiddenDimSize维度的3维向量
    def gru_layer(self,emb,name,hiddenDimSize):

        # 按照时间倒序得到gi~g1
        def stepFn(h,wx):
            uh = tf.matmul(h, self.tparams["U_gru_" + name])
            r = tf.nn.sigmoid(self._slice(wx, 0, hiddenDimSize) + self._slice(uh, 0, hiddenDimSize))
            z = tf.nn.sigmoid(self._slice(wx, 1, hiddenDimSize) + self._slice(uh, 1, hiddenDimSize))
            h_tilde = tf.nn.tanh(self._slice(wx, 2, hiddenDimSize) + r * self._slice(uh, 2, hiddenDimSize))
            h_new = z * h + ((1. - z) * h_tilde)
            return h_new

        Wx = tf.einsum("ijk,kl -> ijl" , emb, self.tparams["W_gru_" + name]) + self.tparams["b_gru_" + name]
        # 用于RNN训练的循环
        if name == "a":
            results = tf.scan(fn=stepFn,elems=Wx,initializer=self.first_h_a)
        else:
            results = tf.scan(fn=stepFn, elems=Wx, initializer=self.first_h_b)

        return results

    # 两个attention的处理，其中att_timesteps是目前为止的步数
    # 返回的是一个3维向量，维度为(n_timesteps × n_samples × embDimSize)
    def attentionStep(self,x,att_timesteps):
        reverse_emb_t = self.emb[:att_timesteps][::-1]
        reverse_h_a = self.gru_layer(reverse_emb_t, "a", self.alphaHiddenDimSize)[::-1] * 0.5
        reverse_h_b = self.gru_layer(reverse_emb_t, "b", self.betaHiddenDimSize)[::-1] * 0.5

        preAlpha = tf.einsum("ijk,kl -> ijl",reverse_h_a, self.tparams["w_alpha"]) + self.tparams["b_alpha"]
        preAlpha = tf.squeeze(preAlpha,[2,2])
        alpha = tf.transpose(tf.nn.softmax(tf.transpose(preAlpha)))
        beta = tf.tanh( tf.einsum("ijk,kl -> ijl", reverse_h_b, self.tparams["w_beta"]) + self.tparams["b_beta"])

        c_t = tf.reduce_mean( ( tf.expand_dims(alpha ,2) * beta * self.emb[:att_timesteps]) , axis=0 )
        return c_t

    # 构建训练网络
    def build_model(self):

        # 第一层网络，进行输入元素的降维处理
        self.emb = tf.einsum("ijk,kl -> ijl",self.x,self.tparams["W_emb"]) + self.tparams["b_emb"]
        if self.keep_prob < 1:
            self.emb = tf.nn.dropout(self.emb,self.keep_prob)

        #counts = tf.range(n_timesteps) + 1
        self.c_t = tf.scan(fn=self.attentionStep,elems=self.counts,initializer=self.first_ch)

        if self.keep_prob < 1.0:
            self.c_t = tf.nn.dropout(self.c_t,self.keep_prob)

        temp_y = tf.einsum("ijk,kl -> ijl",self.c_t,self.tparams["w_output"]) + self.tparams["b_output"]
        self.y_hat = tf.nn.softmax( temp_y )

        # 将y_hat维度变为 patients × maxlen × outputDimSize
        # self.y_hat = tf.transpose(self.y_hat,[1,0,2])

        loss = tf.reduce_sum( -( self.y * tf.log(self.y_hat + self.L2 )  + (1.- self.y) * tf.log( 1.-self.y_hat + self.L2 ) ) )

        return loss

    # 对原始的seqs(patients × visits × medical code)进行处理，得到便于RNN训练的结构（maxlen × n_samples × inputDimSize）
    # 返回的length表示这个seqs中所有病人对应的visit的数目
    def padTrainMatrix(self,seqs):
        lengths = np.array( [ len(seq) for seq in seqs ] ).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen,n_samples,self.inputDimSize]).astype(np.float32)
        for idx,seq in enumerate(seqs):
            for xvec,subseq in zip(x[:,idx,:],seq):
                for tuple in subseq:
                    xvec[tuple[0]] = tuple[1]
        return x,lengths


    # 开始训练
    def startTrain(self,x = None,y = None,lengths = None):

        counts = np.arange(x.shape[0]) + 1
        first_h_a = np.zeros([x.shape[1],self.alphaHiddenDimSize])
        first_h_b = np.zeros([x.shape[1],self.betaHiddenDimSize])
        first_ch = np.zeros([x.shape[1],self.embDimSize])
        loss,y_hat,y,opt = self.sess.run( (self.loss,self.y_hat,self.y,self.opt),feed_dict={self.x:x,self.y:y,self.counts:counts,self.first_ch:first_ch,
                                                                                            self.first_h_a:first_h_a,self.first_h_b:first_h_b,self.length:lengths} )
        return loss

    # 根据输入的x与y获取结果和loss
    def get_reslut(self,x = None,y = None,lengths = None):
        counts = np.arange(x.shape[0]) + 1
        first_h_a = np.zeros([x.shape[1], self.alphaHiddenDimSize])
        first_h_b = np.zeros([x.shape[1], self.betaHiddenDimSize])
        first_ch = np.zeros([x.shape[1], self.embDimSize])

        loss,y_hat = self.sess.run((self.loss,self.y_hat),feed_dict={self.x:x,self.y:y,self.counts:counts,self.first_ch:first_ch,
                                                                     self.first_h_a:first_h_a,self.first_h_b:first_h_b,self.length:lengths})
        return loss,y_hat