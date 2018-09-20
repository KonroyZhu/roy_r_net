import json
import math

import tensorflow as tf


class RNet:
    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(
            tf.truncated_normal(shape=[dim_in, dim_out], name=name, stddev=stddev / math.sqrt(float(dim_in))))

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal(shape=[dim]), name=name)

    def mat_weigth_mul(self, mat, weight):
        # mat*weight => [batch_size,n,m] * [m,p]=[batch_size,p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])  # 检查矩阵是否可以相乘
        mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])  # [batch_size*n,m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size*n,p]
        return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])  # [batch_size,n,p]

    def DropoutWrappedLSTMCell(self, hidden_size, in_keep_prob, name=None):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, name=name)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def __init__(self):
        self.options = json.load(open('model/config.json', 'r'))['rnet']['train']  # 加载配置文件中的配置项
        opts = self.options

        # weights
        self.W_uQ = self.random_weight(dim_in=2 * opts['hidden_size'], dim_out=opts['hidden_size'])
        self.W_uP = self.random_weight(dim_in=2 * opts['hidden_size'], dim_out=opts['hidden_size'])

        self.W_vP = self.random_weight(dim_in=opts['hidden_size'], dim_out=opts['hidden_size'])
        self.W_QP = self.random_weight(dim_in=4 * opts["hidden_size"], dim_out=4 * opts["hidden_size"])  # GATE权重

        self.W_smP1 = self.random_weight(dim_in=opts["hidden_size"], dim_out=opts["hidden_size"])
        self.W_smP2 = self.random_weight(dim_in=opts["hidden_size"], dim_out=opts["hidden_size"])
        self.W_g_SM = self.random_weight(dim_in=2 * opts['hidden_size'], dim_out=2 * opts['hidden_size'])

        self.W_ruQ = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=2 * opts["hidden_size"])

        self.W_VrQ = self.random_weight(dim_in=opts["q_length"], dim_out=opts['hidden_size'])  # it's a parameter
        self.W_VQ = self.random_weight(dim_in=opts["hidden_size"], dim_out=2 * opts["hidden_size"])

        self.W_hP = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts['hidden_size'])
        self.W_ha = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts['hidden_size'])

        # biases 用于attention训练出权重，给每个time step赋予权重
        self.B_v_QP = self.random_bias(dim=opts["hidden_size"])  # 公式4 vT attention for Question-Passage match
        self.B_v_SM = self.random_bias(dim=opts["hidden_size"])  # 公式8 vT 为生成attention权重准备的向量
        self.B_v_rQ = self.random_bias(dim=2 * opts['hidden_size'])  # 公式11 vT
        self.B_v_ap = self.random_bias(dim=opts['hidden_size'])  # 公式 9 vT

        # QP_match
        with tf.variable_scope("QP_match"):
            self.QPmatch_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],
                                                            in_keep_prob=opts["in_keep_prob"])
            self.QPmatch_state = self.QPmatch_cell.zero_state(batch_size=opts["batch_size"],  # QP需要提取中间状态所以保留
                                                              dtype=tf.float32)  # RNN单元的初始状态

        # Ans Pointer
        with tf.variable_scope("Ans_Ptr"):
            self.AnsPtr_cell = self.DropoutWrappedLSTMCell(hidden_size=2 * opts["hidden_size"],
                                                           in_keep_prob=opts["in_keep_prob"])

    def build_model(self):
        opts = self.options

        # placeholder
        # -输入
        paragraph = tf.placeholder(shape=[opts['batch_size'], opts['p_length'], opts['word_emb_dim']], dtype=tf.float32)
        question = tf.placeholder(shape=[opts['batch_size'], opts['q_length'], opts['word_emb_dim']], dtype=tf.float32)
        # -输出
        answer_si = tf.placeholder(shape=[opts['batch_size'], opts['p_length']], dtype=tf.float32)
        answer_ei = tf.placeholder(shape=[opts['batch_size'], opts['p_length']], dtype=tf.float32)

        print("Layer1: Question and Paragraph Encoding Layer")
        # 由于计算能力不足，此步骤省略字符级别的embedding
        eQcQ = question
        ePcP = paragraph
        # tf.unstack: 在axis指定的轴心上减少一个维度
        # ( 因为训练rnn的时候，需要用q/p_length作为 time step 所以要作unstack转换)
        unstacked_eQcQ = tf.unstack(value=eQcQ, axis=1)  # [ batch,q_length,dim] => q_length 个 [batch,dim]
        unstacked_ePcP = tf.unstack(value=ePcP, axis=1)  # [ batch,q_length,dim] => p_length 个 [batch,dim]

        with tf.variable_scope("encoding"):
            stacked_enc_fw_cells = [self.DropoutWrappedLSTMCell(
                hidden_size=opts["hidden_size"],
                in_keep_prob=opts["in_keep_prob"]) for _ in range(2)]  # 包含两个forward LSTM 单元的列表
            stacked_enc_bw_cells = [self.DropoutWrappedLSTMCell(
                hidden_size=opts["hidden_size"],
                in_keep_prob=opts['in_keep_prob']) for _ in range(2)]  # 包含两个backward LSTM 单元的列表

            q_enc_outputs, q_enc_final_fw, q_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_eQcQ, dtype=tf.float32, scope="context_encoding"
            )  # 整个序列（q_length 个 [b, 2*h])||fw的最后向量( b, 2*h)||bw的最后向量( b, 2*h)

            p_enc_output, p_enc_fianl_fw, p_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_ePcP, dtype=tf.float32, scope="context_encoding"
            )  # 整个序列（p_length 个 [b, 2*h])||fw的最后向量( b, 2*h)||bw的最后向量( b, 2*h)

            # tf.stack 在axis指定的轴心上增加一个维度（将列表合并）
            u_Q = tf.stack(q_enc_outputs, axis=1)  # q_lenth个 [batch,2*hidden] => [batch,q_length,2*hidden]
            u_P = tf.stack(p_enc_output, axis=1)  # p_lenth个 [batch,2*hidden] => [batch,p_length,2*hidden]
        u_Q = tf.nn.dropout(u_Q, opts["in_keep_prob"])  # (b,q,2*h)
        u_P = tf.nn.dropout(u_P, opts["in_keep_prob"])  # (b,q,2*h)
        print(u_Q)
        print(u_P)

        print("Layer2: Question-Passage Matching")
        v_P = []  # RNN( v_Pt-1,ct)
        # 使用attention要对RNN序列输出的整个序列作处理
        for t in range(opts['p_length']):
            # 利用question向量和paragraph中每个词语的关系，构造对于u_P的attention
            # arg1: u_Q * weight
            W_u_Q = self.mat_weigth_mul(u_Q, self.W_uQ)  # (b,q,2*h ) * (2*h,h)=>(b,q,h)
            # arg2: paragraph * weight
            tiled_u_tP = tf.concat(
                [tf.reshape(u_P[:, t, :], shape=[opts["batch_size"], 1, -1])] * opts['q_length'],  # 注意力与u_Q(b,q,2h)结合
                axis=1)  # 由于每步只取u_P在词语t时的张量（b,1,2*h）,需要将其广播（复制）成（b,q,2*h)才能正常运算
            W_u_P = self.mat_weigth_mul(tiled_u_tP, self.W_uP)  # (b,q,2*h ) * (2*h,h)=>(b,q,h)

            # arg3: vP[t-1] * weight
            if t == 0:
                tanh = tf.nn.tanh(W_u_P + W_u_Q)  # v_P的上一个为空 (b,q,h)
            else:
                tiled_v_tP_qp = tf.concat(
                    [tf.reshape(v_P[t - 1], shape=[opts["batch_size"], 1, -1])] * opts["q_length"],  # 注意力与u_Q(b,q,2h)结合
                    axis=1)  # 由于每一步的v_P[t-1]形状为（b,1,2*h),需要将其广播（复制）为（b,q,2*h)才能正常参与计算
                W_v_tP_qp = self.mat_weigth_mul(tiled_v_tP_qp, self.W_vP)
                tanh = tf.nn.tanh(W_u_Q + W_u_P + W_v_tP_qp)  # 3 个（b,q,h)相加，还是(b,q,h)

            # ATTENTION
            # s_t 为attention给每个time step（q）的权重（由输入tanh与B_v_OP的乘积得到）
            s_t = tf.squeeze(  # squeeze 用于去除张量中形状为零的维度   ### 此处的B_v_QP就是公式4中的vT
                self.mat_weigth_mul(tanh, tf.reshape(self.B_v_QP, [-1, 1]))  # (b,q,h)*(h,1) => (b,q,1) 用于attention
            )  # squeeze之后形状为1的最后一个维度被去除，变为(b,q)
            a_t = tf.nn.softmax(s_t, axis=1)  # 将attention权重s_t (b,q) 在q方向上进行softmax
            tiled_a_t = tf.concat(
                [tf.reshape(a_t, shape=[opts["batch_size"], -1, 1])] * 2 * opts["hidden_size"], axis=2
            )  # 前面用B_v_QP将向量变成了(b,q,1)现在广播(复制）成(b,q,2*h)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q),
                                axis=1)  # sum((b,q,2*h) x (b,q,2*h)) => (b,2*h) 加上attention的u_Q

            # GATE
            u_tP_c_t = tf.expand_dims(  # u_P 与 c_t 的拼接
                tf.concat(
                    [tf.squeeze(u_P[:, t, :]), c_t], axis=1  # u_tP(b,1,h) =squeeze=> (b,2*h) | concat(axis=1)=> (b,4*h)
                ), axis=1  # expanded => (b,1,4*h)
            )
            g_t = tf.sigmoid(self.mat_weigth_mul(u_tP_c_t, self.W_QP))  # (b,1,4*h)*(4*h,4*h) => (b,1,4*h) gate权重
            u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t))  # (b,1,4*h) x (b,1,4*h) => (b,1,4*h) 乘上gate权重
            # squeeze 后 剩下（b,4*h)

            # QP_match
            with tf.variable_scope("QP_match"):
                output, self.QPmatch_state = self.QPmatch_cell(u_tP_c_t_star,  # (b,4*h)
                                                               self.QPmatch_state)  # 吧输入 和上一步的状态一起作为输入
                v_P.append(output)
        v_P = tf.stack(v_P, axis=1)  # (b,p,h) 经过RNN后 形状末端的 4*h又变回h（由QPmatch_cell的隐层决定 ）
        v_P = tf.nn.dropout(v_P, opts["in_keep_prob"])
        print('v_P:', v_P)

        print("Layer3: Self-Matching Attention Layer")

        SM_star = []
        for t in range(opts["p_length"]):
            # 让v_P和自身的各部分计算attention（此处相当于QP Attention中的u_Q）
            # arg1: v_P * weight
            W_p1_v_P = self.mat_weigth_mul(v_P, self.W_smP1)  # v_P (b,q,h) * (h,h) => (b,q,h)

            # arg2: v_Pt * weight
            tiled_v_tP_sm = tf.concat(
                [tf.reshape(v_P[:, t, :], shape=[opts["batch_size"], 1, -1])] * opts["p_length"],  # p:自注意v_P(b,p,h)
                axis=1)  # 广播（复制）形为（b,1,h)的v_P[t]成（b,p,h)
            W_p2_v_P = self.mat_weigth_mul(tiled_v_tP_sm, self.W_smP2)  # (b,q,h)*(h,h) => (b,q,h)

            tanh = tf.tanh(W_p1_v_P + W_p2_v_P)  # (b,q,h)
            s_t = tf.squeeze(  # B_v_SM 就是公式 8 中的 vT
                self.mat_weigth_mul(tanh, tf.reshape(self.B_v_SM, shape=[-1, 1])))  # (b,q,1) =squeeze=> (b,q)
            a_t = tf.nn.softmax(s_t, axis=1)  # (b,q)
            tiled_a_t = tf.concat(
                [tf.reshape(a_t, shape=[opts['batch_size'], -1, 1])] * opts['hidden_size'],
                axis=2  # (b,q) => (b,q,1) => (b,q,h) 作为attention的权重
            )
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, v_P), axis=1)  # (b,q,h) x (b,q,h) => (b,q,h) =sum=> (b,h)

            # gate
            v_tP_c_t = tf.expand_dims(
                tf.concat(
                    [tf.squeeze(v_P[:, t, :]), c_t],  # （b,1,h) =squeeze=>(b,h)
                    axis=1  # (b,h) (b,h) =conc=> (b,2h)
                ), axis=1  # (b,2h) =expand=> (b,1,2h)
            )
            g_t = tf.sigmoid(self.mat_weigth_mul(v_tP_c_t, self.W_g_SM))  # (b,1,2h) * (2h,2h) =>(b,1,2h)
            v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))  # (b,1,2h) x (b,1,2h) => (b,1,2h) 与gate权重相乘
            SM_star.append(v_tP_c_t_star)
        SM_star = tf.stack(SM_star, axis=1)  # 先于神经网络单元的处理 (b,p,2h)

        unstacked_SM_star = tf.unstack(SM_star, axis=1)  # 由于RNN要顺着p（每一time step）循环，SM_star要分割了才进入双向神经网络
        with tf.variable_scope("Self_match"):
            SM_fw_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["in_keep_prob"])
            SM_bw_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["in_keep_prob"])
            SM_output, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
                SM_fw_cell, SM_bw_cell, unstacked_SM_star, dtype=tf.float32
            )
            h_P = tf.stack(SM_output, axis=1)  # (b,p,2h) fw+bw
        h_P = tf.nn.dropout(h_P, opts['in_keep_prob'])
        print("h_P:", h_P)

        print("Layer4:Output Layer")
        # u_Q * weight
        W_ruQ_uQ = self.mat_weigth_mul(u_Q, self.W_ruQ)  # (b,q,2h)*(2h,2h) => (b,q,2h)
        W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_VQ)  # (q,h)*(h,2h) => (q,2h)`W_VrQ是一个参数？？？
        W_vQ_V_rQ1 = tf.stack([W_vQ_V_rQ] * opts["batch_size"], axis=0)  # (b,q,2h) TODO: 形状有出入

        tanh = tf.tanh(W_ruQ_uQ + W_vQ_V_rQ1)  # (b,q,2h)
        s_t = tf.squeeze(
            self.mat_weigth_mul(tanh, tf.reshape(self.B_v_rQ, shape=[-1, 1]))  # Reshape实则是矩阵转置
        )  # (b,q,2h)*(2h,1) => (b,q,1) =squeeze=> (b,q)
        a_t = tf.nn.softmax(s_t, axis=1)  # (b,q)在q方向上
        tiled_a_t = tf.concat(
            [tf.reshape(a_t, shape=[opts["batch_size"], -1, 1])] * 2 * opts["hidden_size"],
            # (b,q) =>(b,q,1) => (b,q,2h)
            axis=2)
        r_Q = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), axis=1)  # (b,q,2h) x (b,q,2h) => (b,q,2h) =sum=> (b,2h)
        r_Q = tf.nn.dropout(r_Q, opts["in_keep_prob"])
        print("r_Q", r_Q)

        h_a = None  # r_Q是h_a的初始状态
        p = [None for _ in range(2)]
        for t in range(2):  # 需要预测出两个p：p1是开始位置，p2是结束位置
            W_hP_h_P = self.mat_weigth_mul(h_P, self.W_hP)  # (b,p,2h)*(2h,h) => (b,p,h)

            if t == 0:
                h_t1a = r_Q  # TODO:预测p1？
            else:
                h_t1a = h_a  # TODO: 预测p2？
            print(t, "h_t1a", h_t1a)  # (b,2h)
            tiled_h_t1a = tf.concat(
                [tf.reshape(h_t1a, shape=[opts['batch_size'], 1, -1])] * opts['p_length'],
                axis=1
            )  # (b,p,2h)
            print("tiled_h_t1a", tiled_h_t1a)
            W_ha_h_t1a = self.mat_weigth_mul(tiled_h_t1a, self.W_ha)  # (b,p,2h)*(2h,h)=>(b,p,h)

            tanh = tf.nn.tanh(W_ha_h_t1a + W_hP_h_P)  # (b,p,h)
            s_t = tf.squeeze(
                self.mat_weigth_mul(tanh, tf.reshape(self.B_v_ap, [-1, 1]))
            )  # (b,p)
            a_t = tf.nn.softmax(s_t, axis=1)  # 指针网络省去了attention机制的最后一行公式 直接利用softmax结果

            p[t] = a_t  # (b,p) 最终通过argmax找出最大的idx作为输出

            tiled_a_t = tf.concat(
                [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['hidden_size'], 2)  # 广播(b,p,2h)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, h_P), 1)  # (b,p,2h) x (b,p,2h) =sum=> (b,2h)

            if t == 0:
                AnsPtr_state = self.AnsPtr_cell.zero_state(opts['batch_size'], dtype=tf.float32)
                h_a, _ = self.AnsPtr_cell(c_t, (AnsPtr_state, r_Q))
                h_a = h_a[1]
                print("h_a:", h_a)

        print(p)
        p1 = p[0]  # (b,p)
        p2 = p[1]  # (b,p)

        # 正确标签
        answer_si_idx = tf.cast(tf.argmax(answer_si, 1), tf.int32)  # (b,p) =argmax=>(b,)
        answer_ei_idx = tf.cast(tf.argmax(answer_ei, 1), tf.int32)  # (b,p) =argmax=>(b,)

        batch_idx = tf.reshape(tf.range(0, opts['batch_size']), [-1, 1])  # (b,1)
        answer_si_re = tf.reshape(answer_si_idx, [-1, 1])  # (b,1)
        batch_idx_si = tf.concat([batch_idx, answer_si_re], 1)
        print("batch_idx_si",batch_idx_si)
        answer_ei_re = tf.reshape(answer_ei_idx, [-1, 1])  # (b,1)
        batch_idx_ei = tf.concat([batch_idx, answer_ei_re], 1)
