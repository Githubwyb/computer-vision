# coding = utf-8
import time
import matplotlib.pyplot as plt
import os
import os.path
import numpy as np

start = time.time()

DATA_ROOT = 'data/'

imgData = []
imgName = []
temp = []

folders = os.listdir(DATA_ROOT + 'images')
for folder in folders:
    files = os.listdir(DATA_ROOT + 'images/' + folder)
    for file in files:
        if file.find('.jpeg') != -1:
            imgName.append(int(file[: -5]))
            imgData.append(plt.imread(DATA_ROOT + 'images/' + folder + '/' + file))

with open(DATA_ROOT + 'labels/training_label') as f:
    file_lines = f.readlines()

with open(DATA_ROOT + 'labels/words') as f:
    labelName = f.readlines()

labels_train = []
data_train = []

for j in range(len(file_lines)):
    lines = file_lines[j].split(' ')
    temp = []
    for i in range(1, len(lines)):
        if lines[i].isdigit():
            temp.append(int(lines[i]))
    labels_train.append(temp)
    index = imgName.index(int(lines[0]))
    data_train.append(imgData[index])
