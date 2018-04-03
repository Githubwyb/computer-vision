# coding = utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import data, exposure
from skimage.feature import local_binary_pattern
from sklearn import neighbors
from sklearn import svm

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
data_test = dic[b'data']
labels_test = np.array(dic[b'labels'])

try:
    data_train = np.load("data\\data_sum.npy")
    print("File 'data_sum.npy' is here!")
except FileNotFoundError:
    print("No such file named 'data_sum.npy'!")
    data_train = data_train.reshape(50000, 3, 32, 32)
    temp = np.zeros([50000, 32, 32, 3])

    for i in range(len(data_train)):
        for j in range(32):
            for k in range(32):
                temp[i][j][k][0] = data_train[i][0][j][k]
                temp[i][j][k][1] = data_train[i][1][j][k]
                temp[i][j][k][2] = data_train[i][2][j][k]
    data_train = temp
    np.save("data\\data_sum.npy", data_train)

try:
    data_test = np.load("data\\data_test.npy")
    print("File 'data_test.npy' is here!")
except FileNotFoundError:
    print("No such file named 'data_test.npy'!")
    data_test = data_test.reshape(10000, 3, 32, 32)
    temp = np.zeros([10000, 32, 32, 3])

    for i in range(len(data_test)):
        for j in range(32):
            for k in range(32):
                temp[i][j][k][0] = data_test[i][0][j][k]
                temp[i][j][k][1] = data_test[i][1][j][k]
                temp[i][j][k][2] = data_test[i][2][j][k]
    data_test = temp
    np.save("data\\data_test.npy", data_test)

# print('Training data shape: ', data_train.shape)
# print('Training labels shape: ', labels_train.shape)
# print('Test data shape: ', data_test.shape)
# print('Test labels shape: ', labels_test.shape)

# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for i, cls in enumerate(classes):
#     idxs = np.flatnonzero(labels_train == i)
#     idxs = np.random.choice(idxs, samples_per_class, replace = False)
#     for j, idx in enumerate(idxs):
#         plt_idx = j * num_classes + i + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(data_train[idx].astype('uint8'))
#         plt.axis('off')
#         if j == 0:
#             plt.title(cls)
# plt.show()

num_training = 10000
mask = list(range(num_training))
data_train = data_train[mask]
labels_train = labels_train[mask]

num_test = 1000
mask = list(range(num_test))
data_test = data_test[mask]
labels_test = labels_test[mask]

data_train_hog = np.array([hog(rgb2gray(image), orientations = 8, pixels_per_cell = (16, 16), cells_per_block = (1, 1)) for image in data_train])
data_test_hog = np.array([hog(rgb2gray(image), orientations = 8, pixels_per_cell = (16, 16), cells_per_block = (1, 1)) for image in data_test])

data_train_lbp = np.reshape(np.array([local_binary_pattern(rgb2gray(image), P = 1, R = 2) for image in data_train]), (data_train.shape[0], -1))
data_test_lbp = np.reshape(np.array([local_binary_pattern(rgb2gray(image), P = 1, R = 2) for image in data_test]), (data_test.shape[0], -1))

data_train_row = np.reshape(data_train, (data_train.shape[0], -1))
data_test_row = np.reshape(data_test, (data_test.shape[0], -1))
# print(data_train_row.shape, data_test_row.shape)

# n_neightbors = 5
# knn = neighbors.KNeighborsClassifier(n_neightbors, weights = 'uniform')
# knn.fit([data_train_row, data_train_lbp], labels_train)
# print('KNN where n = 5 and input are row data', knn.score([data_test_row, data_train_row], labels_test))

# n_neightbors = 5
# knn_hog = neighbors.KNeighborsClassifier(n_neightbors, weights = 'uniform')
# knn_hog.fit(data_train_hog, labels_train)
# print('KNN where n = 5 and input are hog data', knn_hog.score(data_test_hog, labels_test))

# n_neightbors = 5
# knn_lbp = neighbors.KNeighborsClassifier(n_neightbors, weights = 'uniform')
# knn_lbp.fit(data_train_lbp, labels_train)
# print('KNN where n = 5 and input are lbp data', knn_lbp.score(data_test_lbp, labels_test))

C = 1.0
X_train = data_train_row
Y_train = labels_train
X_test = data_test_row
Y_test = labels_test
titles = ('SVC with linear kernel',
    'LinearSVC(linear kernel)',
    'SVC with RBF kernel',
    'SVC with polynomial(degree 3) kernel')
models = (svm.SVC(kernel = 'linear', C = C),
    svm.LinearSVC(C = C),
    svm.SVC(kernel = 'rbf', gamma = 0.7, C = C),
    svm.SVC(kernel = 'poly', degree = 3, C = C))
models = (clf.fit(X_train, Y_train) for clf in models)
for svmclf, title in zip(models, titles):
    print(title, ': ', svmclf.score(X_test, Y_test))


