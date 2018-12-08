from __future__ import print_function
# import os, time, cPickle
# cPickle.x中移除，直接是内置的包：pickle
import os
import time
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim 是tensorflow中 一个轻量级的库，用于定义、训练、评估复杂模型。
from random import shuffle
# random伪随机数生成器
# shuffle 打乱给定序列
# sklearn 和 scikit-learn是什么呢
import sklearn.preprocessing
# 导入时，在包，右键make dictionary as Source Root
from base_model import BaseModel, BaseModelParams, BaseDataIter
import utils
from flip_gradient import flip_gradient


class DataIter(BaseDataIter):  # 定义数据迭代类

    # 初始化--定义训练集和测试集的大小；通过base_modal.py
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        # 读取图片特征、文本向量、标签，用于训练和测试。（训练集与测试集相同）
        with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
            self.train_img_feats = pickle.load(f, encoding='bytes')
        with open('./data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
            self.train_txt_vecs = pickle.load(f, encoding='bytes')
        with open('./data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
            self.train_labels = pickle.load(f, encoding='bytes')
        with open('./data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
            self.test_img_feats = pickle.load(f, encoding='bytes')
        with open('./data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
            self.test_txt_vecs = pickle.load(f, encoding='bytes')
        with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as f:
            self.test_labels = pickle.load(f, encoding='bytes')

        self.num_train_batch = len(self.train_img_feats) / self.batch_size
        self.num_test_batch = len(self.test_img_feats) / self.batch_size

    # 训练数据
    def train_data(self):
        # print(self.num_train_batch)
        self.num_train_batch = int(self.num_train_batch)
        # print(self.num_train_batch)
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i*self.batch_size: (i+1)*self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i*self.batch_size: (i+1)*self.batch_size]
            batch_labels = self.train_labels[i*self.batch_size: (i+1)*self.batch_size]
            # yield出现即，定义了一个生成器函数
            yield batch_img_feats, batch_txt_vecs, batch_labels, i

    # 测试数据
    def test_data(self):
        # print(self.num_test_batch)
        self.num_test_batch = int(self.num_test_batch)
        # print(self.num_test_batch)
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i*self.batch_size: (i+1)*self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i*self.batch_size: (i+1)*self.batch_size]
            batch_labels = self.test_labels[i*self.batch_size: (i+1)*self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i


class ModelParams(BaseModelParams):  # 模型参数类
    def __init__(self):
        BaseModelParams.__init__(self)

        # 初始化轮次为200，批大小63，图像特征维度4096，文本向量维度5000
        self.epoch = 200
        self.batch_size = 64
        self.visual_feat_dim = 4096
        # self.word_vec_dim = 300
        self.word_vec_dim = 5000
        # 初始化
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        # 定义检索结果为50，语义_嵌入_维度40
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'wikipedia_dataset'
        self.model_name = 'adv_semantic_zsl'
        self.model_dir = 'adv_semantic_zsl_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        # 定义checkpoint（检查点）
        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    # 更新 checkpoint_dir,sample_dir,log_dir,dataset_dir
    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.visual_feats = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.word_vecs = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size,10])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size,1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.visual_feats)
        self.emb_w = self.label_embed(self.word_vecs)
        # self.corr_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        # self.corr_loss = tf.reduce_mean(self.corr_loss)

        # dissimilar loss
        emb_v_ = tf.reduce_sum(self.emb_v, axis=1, keepdims=True)
        emb_w_ = tf.reduce_sum(self.emb_w, axis=1, keepdims=True)
        distance_map = tf.matmul(emb_v_, tf.ones([1, self.model_params.batch_size])) - tf.matmul(self.emb_v,
                                                                                                 tf.transpose(
                                                                                                   self.emb_w)) + \
                       tf.matmul(tf.ones([self.model_params.batch_size, 1]), tf.transpose(emb_w_))
        mask_initial = tf.to_float(tf.matmul(self.y_single, tf.ones([1, self.model_params.batch_size],
                                                                    dtype=tf.int32)) - tf.matmul(tf.ones([
            self.model_params.batch_size, 1], dtype=tf.int32), tf.transpose(self.y_single)))
        mask = tf.to_float(tf.not_equal(mask_initial, tf.zeros_like(mask_initial)))
        masked_dissimilar_loss = tf.multiply(distance_map, mask)
        self.dissimilar_loss = tf.reduce_mean(tf.maximum(0., 0.1*tf.ones_like(mask)-masked_dissimilar_loss))
        # self.similar_loss = tf.reduce_mean(tf.abs(distance_map-masked_dissimilar_loss))
        self.similar_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        self.similar_loss = tf.reduce_mean(self.similar_loss)
        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits_v) + \
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 100*self.label_loss + self.similar_loss + 0.02*self.dissimilar_loss
        # 域分类器
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)
        # 初始化all_emb_v/w
        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                                   tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                                   tf.ones([self.model_params.batch_size, 1])], 1)
        # 判别 损失
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.emb_v_class, labels=all_emb_w)\
                                 + \
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)

        self.t_vars = tf.trainable_variables()
        # 图像特征变量
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        # 文本特征变量
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        # 域分类器变量
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        # 标签分类器变量
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]

    # 图像特征——嵌入
    def visual_feature_embed(self, X, is_training=True, reuse=False):
        # 全连接层，激活函数为none,reuse指定是否共享层或者和变量
        # 三层全连接层
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            # slim.fully_connected 第一个参数是输入。 第二个参数是num_outputs-输出个数。scope名字。
            # 计算 tanh
            net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    # 文本特征——嵌入（同图像特征——嵌入）
    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, self.model_params.semantic_emb_dim, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net

    # 标签分类器（模态内）
    def label_classifier(self, X, reuse=False):
        # 一层全连接层，激活函数为none
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net

    # 域分类器（模态间）-判别器
    def domain_classifier(self, E, l, is_training=True, reuse=False):
        # 三层全连接层
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            # 逆梯度
            E = flip_gradient(E, l)
            # print(self.model_params.semantic_emb_dim/2)
            # print(int(self.model_params.semantic_emb_dim/2))
            temp1 = int(self.model_params.semantic_emb_dim/2)
            temp2 = int(self.model_params.semantic_emb_dim/4)
            # print(temp1)
            # print(temp2)
            # net = slim.fully_connected(E, self.model_params.semantic_emb_dim/2, scope='dc_fc_0')
            # net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='dc_fc_1')
            # temp1,temp2替换上面的
            net = slim.fully_connected(E, temp1, scope='dc_fc_0')
            net = slim.fully_connected(net, temp2, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    # 训练
    def train(self, sess):
        # self.check_dirs()

        # 总的损失=嵌入损失+判别损失
        total_loss = self.emb_loss + self.domain_class_loss
        # Adam优化
        # learning_rate 学习率，beta1
        total_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_total,
            beta1=0.5).minimize(total_loss)
        # 训练-嵌入损失-生成器
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars+self.vf_vars)
        # 训练-判别损失--判别器
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        # 初始化所有变量
        tf.initialize_all_variables().run()
        # 保存训练器的模型的变量
        self.saver = tf.train.Saver()

        start_time = time.time()
        # ???
        map_avg_ti = []
        map_avg_it = []
        # 对抗损失
        adv_loss = []
        # 嵌入损失
        emb_loss = []
        # 遍历
        for epoch in range(self.model_params.epoch):
            # 不是第一轮次
            if epoch % 1 == 0:
                adv = 0
                emb = 0
                # 测试-图像特征转换
                test_img_feats_trans = []
                # 测试-文本向量转换
                test_txt_vecs_trans = []
                # 测试-标签（输出）
                test_labels = []
                p = float(epoch) / self.model_params.epoch
                # l没用到啊
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # 遍历-训练数据
                for feats, vecs, labels, i in self.data_iter.train_data():
                    labels_ = labels - np.ones_like(labels)
                    # 标签二值化
                    label_binarizer = sklearn.preprocessing.LabelBinarizer()
                    # fit相当于去重
                    label_binarizer.fit(range(max(labels_)+1))
                    # 将多标签转换为2值标签
                    b = label_binarizer.transform(labels_)
                    # feed_dict给使用placeholder创建出来的tensor赋值
                    #  np.transpose转置
                    emb_loss_val, domain_loss_val = sess.run([self.emb_loss, self.domain_class_loss],
                                                         feed_dict={self.visual_feats: feats,
                                                                    self.word_vecs: vecs,
                                                                    self.y: b,
                                                                    self.y_single: np.transpose([labels]),
                                                                    self.l: l})
                    emb += emb_loss_val
                    adv += domain_loss_val
                # 25是？？？
                emb = float(emb)/float(25)
                adv = float(adv)/float(25)
                adv_loss.append(adv)
                emb_loss.append(emb)

                # 遍历-测试数据
                for feats, vecs, labels, i in self.data_iter.test_data():           
                    feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
                    vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
                    test_labels += labels
                    for ii in range(len(feats)):
                        test_img_feats_trans.append(feats_trans[ii])
                        test_txt_vecs_trans.append(vecs_trans[ii])
                test_img_feats_trans = np.asarray(test_img_feats_trans)
                test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)

                k = self.model_params.top_k
                # 平均精确度 avg_precision
                avg_precs = []
                for i in range(len(test_txt_vecs_trans)):
                    query_label = test_labels[i]
                    # distances and sort by distances
                    wv = test_txt_vecs_trans[i]
                    diffs = test_img_feats_trans - wv
                    # np.linalg.norm计算范数，此处，2范数
                    dists = np.linalg.norm(diffs, axis=1)
                    # 升序排列返回原序列的下标
                    sorted_idx = np.argsort(dists)
                    # for each k do top-k
                    # precision
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0 : topk]
                        if query_label != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if retrieved_label == query_label:
                                hits += 1
                        # 计算精确度
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    # 计算平均精确度
                    avg_precs.append(np.average(precs))
                # 文本-平均精确度
                map_avg_ti.append(np.mean(avg_precs))

                # 平均精确度
                avg_precs_ = []
                # 图像标签作为查询标签，计算平均精确度
                for i in range(len(test_img_feats_trans)):
                    query_img_feat = test_img_feats_trans[i]
                    # 真实标签
                    ground_truth_label = test_labels[i]
                    # calculate distance and sort
                    diffs = test_txt_vecs_trans - query_img_feat
                    dists = np.linalg.norm(diffs, axis=1)
                    sorted_idx = np.argsort(dists)
                    # for each k in top-k
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0 : topk]
                        if ground_truth_label != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if ground_truth_label == retrieved_label:
                                hits += 1
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    avg_precs_.append(np.average(precs))
                # 图像-平均精确度
                map_avg_it.append(np.mean(avg_precs_))

            # 最后一轮次
            if epoch == (self.model_params.epoch-1):
                # 将结果写入文件
                with open('./data/wikipedia_dataset/map_avg_ti.pkl', 'wb') as f:
                    pickle.dump(map_avg_ti, f, pickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/map_avg_it.pkl', 'wb') as f:
                    pickle.dump(map_avg_it, f, pickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/adv_loss.pkl', 'wb') as f:
                    pickle.dump(adv_loss, f, pickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/emb_loss.pkl', 'wb') as f:
                    pickle.dump(emb_loss, f, pickle.HIGHEST_PROTOCOL)
                print('dump map_avg')

            # 第一轮次
            if epoch == 0:
                test_img_feats_trans = []
                test_txt_vecs_trans = []
                for feats, vecs, labels, i in self.data_iter.train_data():
                    feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
                    vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
                    for ii in range(len(feats)):
                        test_img_feats_trans.append(feats_trans[ii])
                        test_txt_vecs_trans.append(vecs_trans[ii])
                test_img_feats_trans = np.asarray(test_img_feats_trans)
                test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
                # 将 测试的图像与文本特征 存入一个测试特征中
                test_feats_trans = np.concatenate((test_img_feats_trans[0:1000], test_txt_vecs_trans[-1000:]))
                # 并写入文件--（特征转换前？）
                with open('./data/wikipedia_dataset/before_feats_transformed.pkl', 'wb') as f:
                    pickle.dump(test_feats_trans, f, pickle.HIGHEST_PROTOCOL)
                # 这个文件不存在啊
                # with open('./data/wiki_shallow/before_feats_transformed.pkl', 'wb') as f:
                # pickle.dump(test_feats_trans, f, pickle.HIGHEST_PROTOCOL)
                print('dump before')

            p = float(epoch) / self.model_params.epoch
            # l没用到啊
            l = 2. / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, idx in self.data_iter.train_data():
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(max(batch_labels_)+1))
                b = label_binarizer.transform(batch_labels_)
                #_, label_loss_val, dissimilar_loss_val, similar_loss_val = sess.run([total_train_op, self.label_loss, self.dissimilar_loss, self.similar_loss], feed_dict={self.visual_feats: batch_feat, self.word_vecs: batch_vec, self.y: b, self.y_single: np.transpose([batch_labels]),self.l: l})
                sess.run([emb_train_op, domain_train_op],
                         feed_dict={
                             self.visual_feats: batch_feat,
                             self.word_vecs: batch_vec,
                             self.y: b,
                             self.y_single: np.transpose([batch_labels_]),
                             self.l: l})
                label_loss_val, similar_loss_val, emb_loss_val, domain_loss_val, dissimilar_loss_val= sess.run([self.label_loss, self.similar_loss, self.emb_loss, self.domain_class_loss, self.dissimilar_loss],
                                                         feed_dict={self.visual_feats: batch_feat,
                                                                    self.word_vecs: batch_vec,
                                                                    self.y: b,
                                                                    self.y_single: np.transpose([batch_labels]),
                                                                    self.l: l})
                print('Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: '
                      '%.8f, domain_loss: %.8f, label_loss: %.8f, similar_loss: %.8f, '
                      'dissimilar_loss: %.8f'
                      % (epoch, idx, self.data_iter.num_train_batch, time.time() - start_time,
                         emb_loss_val,
                         domain_loss_val, label_loss_val, similar_loss_val, dissimilar_loss_val))
            # if epoch == (self.model_params.epoch - 1):
            #    self.emb_v_eval, self.emb_w_eval = sess.run([self.emb_v, self.emb_w],
            #    self.emb_v_eval, self.emb_w_eval = sess.run([self.emb_v, self.emb_w],
            #             feed_dict={
            #                 self.visual_feats: batch_feat,
            #                 self.word_vecs: batch_vec,
            #                 self.y: b,
            #                 self.y_single: np.transpose([batch_labels]),
            #                 self.l: l})
            #    with open('./data/wikipedia_dataset/train_img_emb.pkl', 'wb') as f:
            #        pickle.dump(self.emb_v_eval, f, pickle.HIGHEST_PROTOCOL)
            #    with open('./data/wikipedia_dataset/train_txt_emb.pkl', 'wb') as f:
            #        pickle.dump(self.emb_w_eval, f, pickle.HIGHEST_PROTOCOL)

    # 评估-随机-排序
    def eval_random_rank(self):
        start = time.time()
        # 读取测试标签
        with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as fpkl:
           test_labels = pickle.load(fpkl)
        # with open('./data/wiki_shallow/L_te.pkl', 'rb') as fpkl:
        #     test_labels = pickle.load(fpkl)
        k = self.model_params.top_k
        avg_precs = []
        for i in range(len(test_labels)):
            query_label = test_labels[i]

            # distances and sort by distances
            sorted_idx = range(len(test_labels))
            # 打乱
            shuffle(sorted_idx)

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0: topk]
                if query_label != test_labels[top_k[-1]]:
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if query_label != retrieved_label:
                        hits += 1
                # 准确率=击中/总数
                precs.append(float(hits) / float(topk))
            # 平均准确率
            avg_precs.append(np.sum(precs) / float(k))
        # 平均准确率
        mean_avg_prec = np.mean(avg_precs)
        # mAP
        print('[Eval - random] mAP: %f in %4.4fs' % (mean_avg_prec, (time.time() - start)))

    # 评估
    def eval(self, sess):
        start = time.time()

        test_img_feats_trans = []
        test_txt_vecs_trans = []
        test_labels = []
        # 迭代测试数据--获取测试图像和文本转换特征
        for feats, vecs, labels, i in self.data_iter.test_data():
            feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
            vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
            test_labels += labels
            for ii in range(len(feats)):
                test_img_feats_trans.append(feats_trans[ii])
                test_txt_vecs_trans.append(vecs_trans[ii])
        # 连接为测试特征转换矩阵
        test_img_feats_trans = np.asarray(test_img_feats_trans)
        test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
        test_feats_trans = np.concatenate((test_img_feats_trans[0:1000], test_txt_vecs_trans[-1000:]))
        # 写入测试-特征-转换矩阵
        with open('./data/wikipedia_dataset/test_feats_transformed.pkl', 'wb') as f:
           pickle.dump(test_feats_trans, f, pickle.HIGHEST_PROTOCOL)
        # ？？？
        # with open('./data/wiki_shallow/test_feats_transformed.pkl', 'wb') as f:
        #     pickle.dump(test_feats_trans, f, pickle.HIGHEST_PROTOCOL)
        # 输出特征转换时间
        print('[Eval] transformed test features in %4.4f' % (time.time() - start))
        # k = self.model_params.top_k

        # t2i
        avg_precs = []
        # 所有-精确度
        all_precs = []
        # ???.定义不同的结果集维度，来显示精确率？
        # all_k = [5,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
        all_k =[50]
        for k in all_k:
            for i in range(len(test_txt_vecs_trans)):
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_txt_vecs_trans[i]
                diffs = test_img_feats_trans - wv
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(query_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(retrieved_label) == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        # 输出：评估-txt2img的mAP，以及运行时间
        print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
        # 定义t2i的精确度
        t2i = all_precs[0]
        # ？？？
        # 写入文件
        with open('./data/wikipedia_dataset/txt2img_all_precision.pkl', 'wb') as f:
           pickle.dump(all_precs, f, pickle.HIGHEST_PROTOCOL)
        # with open('./data/wiki_shallow/txt2img_all_precision.pkl', 'wb') as f:
        #     pickle.dump(all_precs, f, pickle.HIGHEST_PROTOCOL)

        # 同上：i2t
        avg_precs = []
        all_precs = []
        # all_k = [5,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
        all_k = [50]
        for k in all_k:
            for i in range(len(test_img_feats_trans)):
                query_img_feat = test_img_feats_trans[i]
                ground_truth_label = test_labels[i]

                # calculate distance and sort
                diffs = test_txt_vecs_trans - query_img_feat
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k in top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(ground_truth_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(ground_truth_label) == retrieved_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        # 输出：评估-img2txt的mAP，时间
        print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

        # 写入文件
        with open('./data/wikipedia_dataset/text_words_map.pkl', 'wb') as f:
           pickle.dump(all_precs, f, pickle.HIGHEST_PROTOCOL)
        # with open('./data/wiki_shallow/text_words_map.pkl', 'wb') as f:
        #     pickle.dump(all_precs, f, pickle.HIGHEST_PROTOCOL)

        # # ？？？utils.is_text_relevant
        # # 第二种评估
        # # with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
        # #    txt_words = pickle.load(f)
        # # with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
        # #    img_words = pickle.load(f)
        # # with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
        # #    test_txt_names = pickle.load(f)
        # # with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
        # #    test_img_names = pickle.load(f)
        # with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
        #     txt_words = pickle.load(f)
        # with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
        #     img_words = pickle.load(f, encoding='utf8')
        #     print(img_words)
        # with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
        #     test_txt_names = pickle.load(f)
        # with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
        #     test_img_names = pickle.load(f)

        # # t2i
        # # Precision-scope for text query
        # scope = 100
        # retrieval_results = []
        # precisions = np.zeros(scope)
        # for i in range(len(test_txt_vecs_trans)):
        #     # print(i)
        #     # print(test_img_names[i])
        #     # print(img_words[test_img_names[i]])
        #     query_words = img_words[test_img_names[i]]
        #
        #     # distances and sort by distances
        #     wv = test_txt_vecs_trans[i]
        #     diffs = test_img_feats_trans - wv
        #     dists = np.linalg.norm(diffs, axis=1)
        #     sorted_idx = np.argsort(dists)
        #
        #     hits = np.zeros(scope)
        #     p = np.zeros(scope)
        #     for k in range(scope):
        #         retrieved = img_words[test_img_names[sorted_idx[k]]]
        #         if utils.is_text_relevant(query_words, retrieved, None):
        #             hits[k] = 1.0
        #     for k in range(scope):
        #         p[k] = np.sum(hits[0:k]) / float(k + 1)
        #     precisions += p
        #
        #     if i in sorted_idx[0:5] and np.sum(hits[0:5]) >= 4:
        #         result = {
        #             'query': test_txt_names[i],
        #             'retrieval': [test_img_names[hh] for hh in sorted_idx[0:5]]
        #         }
        #         retrieval_results.append(result)
        #
        # # 将检索结果写入文件
        # with open('./data/wikipedia_dataset/txt2img-retrievals.pkl', 'wb') as f:
        #     pickle.dump(retrieval_results, f, pickle.HIGHEST_PROTOCOL)
        # # 打印时间
        # print('[Eval - txt2img] finished precision-scope in %4.4fs' % (time.time() - start))
        #
        # # i2t
        # # Precision-scope for image query
        # retrieval_results = []
        # precisions = np.zeros(scope)
        # for i in range(len(test_img_feats_trans)):
        #     query_img_feat = test_img_feats_trans[i]
        #     query_img_name = test_img_names[i]
        #     ground_truth_words = img_words[query_img_name]
        #
        #     # calculate distance and sort
        #     diffs = test_txt_vecs_trans - query_img_feat
        #     dists = np.linalg.norm(diffs, axis=1)
        #     sorted_idx = np.argsort(dists)
        #
        #     hits = np.zeros(scope)
        #     p = np.zeros(scope)
        #     for k in range(scope):
        #         retrieved = img_words[test_img_names[sorted_idx[k]]]
        #         if utils.is_text_relevant(retrieved, ground_truth_words, None):
        #             hits[k] = 1.0
        #     for k in range(scope):
        #         p[k] = np.sum(hits[0:k]) / float(k + 1)
        #     precisions += p
        #
        #     if i in sorted_idx[0:5] and np.sum(hits[0:5]) >= 4 :
        #         result = {
        #             'query': test_img_names[i],
        #             'retrieval': [test_txt_names[hh] for hh in sorted_idx[0:5]]
        #         }
        #         retrieval_results.append(result)
        #
        # with open('./data/wikipedia_dataset/img2txt-retrievals.pkl', 'wb') as f:
        #     pickle.dump(retrieval_results, f, pickle.HIGHEST_PROTOCOL)
        # print('[Eval - img2txt] finished precision-scope in %4.4fs' % (time.time() - start))
