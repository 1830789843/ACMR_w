import tensorflow as tf
import numpy as np
import utils
import pickle
import sklearn.preprocessing


x = tf.constant([[1, 2, 3], [4, 5, 6]])
y = tf.transpose(x)
print(y)

# Equivalently
y = tf.transpose(x, perm=[1, 0])
print(y)

# output:Tensor("transpose:0", shape=(3, 2), dtype=int32)
# [[1, 4]
#  [2, 5]
#  [3, 6]]

i=1
j=1 -\
  2
print(j)
# j = -1

a = 3
print(a ** 2)

p = 0.1
out = np.exp(-10. * p)
out = 2. / (1. + np.exp(-10. * p)) - 1
print(out)

a = tf.constant([10, 20])
a = tf.placeholder(float, [None, 10])
a = tf.placeholder(float, shape=(None, 10))
print("111111111111")
print(a)
# sess = tf.Session()
# sess.run(a)

x = np.array([3, 1, 2])
y = np.argsort(x)
z = y[0:5]
u = z[-1]
print(x)
print(y)
print(z)
print(u)

x = range(12)  # 类似于定义一个数组
print(x)
for i in x:
  print(i)

# print(utils.is_text_relevant(23, 32, None))

# help(utils)

# with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
#     train_img_feats = pickle.load(f, encoding='bytes')
# print(train_img_feats)
with open('E:/pythonProject/ACMR_demo-master/data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
    tt = pickle.load(f, encoding='bytes')
# print(tt)
print(len(tt))
print(tt[0])
print(len(tt[0]))
with open('E:/pythonProject/ACMR_demo-master/data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
    tt = pickle.load(f, encoding='bytes')
# print(tt)
print(len(tt))
print(tt[0])
print(len(tt[0]))
with open('E:/pythonProject/ACMR_demo-master/data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
    tt = pickle.load(f, encoding='bytes')
print(tt)
print(len(tt))
print(tt[0])
# print(len(tt[0]))

t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
print(tf.shape(t1))
print(tf.concat([t1, t2], -1))

batch_labels_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(batch_labels_)+1))
b = label_binarizer.transform(batch_labels_)
print(batch_labels_)
print(max(batch_labels_))
print(range(max(batch_labels_)))
print(b)
print(np.transpose(b))
adj_mat = np.dot(b, np.transpose(b))
print(adj_mat)


x1 = np.arange(9.0).reshape((3, 3))
print(x1)
x2 = np.arange(3.0)
print(x2)
x = np.multiply(x1, x2)
print(x)
x = np.dot(x1, x2)
print(x)





