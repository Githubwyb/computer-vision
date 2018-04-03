# coding = utf-8

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    import numpy as np
    dic = unpickle('..\\doc\\data_batch_1')
    data = np.array(dic[b'data'])
    labels = np.array(dic[b'labels'])

    dic = unpickle('..\\doc\\data_batch_2')
    data = np.concatenate((data, np.array(dic[b'data'])))
    labels = np.concatenate((labels, np.array(dic[b'labels'])))

    dic = unpickle('..\\doc\\data_batch_3')
    data = np.concatenate((data, np.array(dic[b'data'])))
    labels = np.concatenate((labels, np.array(dic[b'labels'])))

    dic = unpickle('..\\doc\\data_batch_4')
    data = np.concatenate((data, np.array(dic[b'data'])))
    labels = np.concatenate((labels, np.array(dic[b'labels'])))

    dic = unpickle('..\\doc\\data_batch_5')
    data = np.concatenate((data, np.array(dic[b'data'])))
    labels = np.concatenate((labels, np.array(dic[b'labels'])))

    dic_test = unpickle("..\\doc\\test_batch")
    data_test = np.array(dic[b'data'])
    labels_test = np.array(dic[b'labels'])

    from sklearn import linear_model
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(data, labels)
    score = lr.score(data_test, labels_test)
    print("Logistic regression is", score)

    # from sklearn import svm #support vector Machine
    # model=svm.SVC(kernel='rbf', C=1, gamma=0.1)
    # model.fit(data,labels)
    # score = model.score(data_test, labels_test)
    # print('Accuracy for rbf SVM is ',score)
