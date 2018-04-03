# coding = utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

dic = unpickle('data\\data_batch_1')
data_train = dic[b'data']
labels_train = np.array(dic[b'labels'])

dic = unpickle('data\\data_batch_2')
data_train = np.concatenate((data_train, dic[b'data']))
labels_train = np.concatenate((labels_train, np.array(dic[b'labels'])))

dic = unpickle('data\\data_batch_3')
data_train = np.concatenate((data_train, dic[b'data']))
labels_train = np.concatenate((labels_train, np.array(dic[b'labels'])))

dic = unpickle('data\\data_batch_4')
data_train = np.concatenate((data_train, dic[b'data']))
labels_train = np.concatenate((labels_train, np.array(dic[b'labels'])))

dic = unpickle('data\\data_batch_5')
data_train = np.concatenate((data_train, dic[b'data']))
labels_train = np.concatenate((labels_train, np.array(dic[b'labels'])))

dic = unpickle("data\\test_batch")
data_test = dic[b'data'].reshape(10000, 3, 32, 32)
labels_test = np.array(dic[b'labels'])

data_train = data_train.reshape(50000, 3, 32, 32)

temp = np.zeros([32, 32, 3])
temp1 = np.zeros([50000, 32, 32, 3])

for i in range(len(data_train)):
    for j in range(32):
        for k in range(32):
            temp[j][k][0] = data_train[i][0][j][k]
            temp[j][k][1] = data_train[i][1][j][k]
            temp[j][k][2] = data_train[i][2][j][k]
    temp1[i] = temp

data_train = temp1



# print('Training data shape: ', data_train.shape)
# print('Training labels shape: ', labels_train.shape)
# print('Test data shape: ', data_test.shape)
# print('Test labels shape: ', labels_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for i, cls in enumerate(classes):
    idxs = np.flatnonzero(labels_train == i)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for j, idx in enumerate(idxs):
        plt_idx = j * num_classes + i + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(data_train[idx].astype('uint8'))
        plt.axis('off')
        if j == 0:
            plt.title(cls)
plt.show()

