# -*- coding:utf-8

# 查看各个文件的维度
# import pickle
#
# with open('../data/nuswide/img_train_id_feats.pkl', 'rb') as f:
#     train_img_feats = pickle.load(f, encoding='bytes')
# with open('../data/nuswide/train_id_bow.pkl', 'rb') as f:
#     train_txt_vecs = pickle.load(f, encoding='bytes')
#
# with open('../data/nuswide/train_id_label_map.pkl', 'rb') as f:
#     train_labels = pickle.load(f, encoding='bytes')
#
# with open('../data/nuswide/img_test_id_feats.pkl', 'rb') as f:
#     test_img_feats = pickle.load(f, encoding='bytes')
#     print(len(test_img_feats))
# with open('../data/nuswide/test_id_bow.pkl', 'rb') as f:
#     test_txt_vecs = pickle.load(f, encoding='bytes')
#     print(len(test_txt_vecs))
# with open('../data/nuswide/test_id_label_map.pkl', 'rb') as f:
#     test_labels = pickle.load(f, encoding='bytes')
#     print(len(test_labels))
# with open('../data/nuswide/train_ids.pkl', 'rb') as f:
#     train_ids = pickle.load(f, encoding='bytes')
#     print(len(train_ids))
# with open('../data/nuswide/test_ids.pkl', 'rb') as f:
#     test_ids = pickle.load(f, encoding='bytes')
#     print(len(test_ids))
# with open('../data/nuswide/train_id_label_single.pkl', 'rb') as f:
#     train_labels_single = pickle.load(f, encoding='bytes')
#     print(len(train_labels_single))
# with open('../data/nuswide/test_id_label_single.pkl', 'rb') as f:
#     test_labels_single = pickle.load(f, encoding='bytes')
#     print(len(test_labels_single))
#

# # 测试reduce_sum
# import tensorflow as tf
# # 定义一个矩阵（2*3）.
# # Tensor一维是向量；二维是矩阵
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# print(x)
# y = tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
# print(y)

# a = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
# b = [1, 2, 3]
# print(type(a))
# print(type(a[1]))
# print(a-b)

# with open("../data/resulta.txt", "a") as file:
#     file.write("sss\n")
# with open("../data/resulta.txt", "a") as file:
#     file.write("sssss\n")

import numpy as np
x = np.array([[1, 2, 3], [9, 8, 7], [6, 5, 4]])
y = np.array([1, 1, 1])
z = x - y
print(z)

