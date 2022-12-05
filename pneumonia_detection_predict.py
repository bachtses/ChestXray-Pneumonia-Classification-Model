import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.metrics import AUC
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
import cv2
import os.path
import sys
import time
from numpy import array
from numpy import argmax
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

test_folder = "images/test/"
print("test folder: ", test_folder)

img_width = 200
img_height = 200
img_channels = 3
n_classes = 2  # [pneumonia, normal]

X = []
y = []
X = np.array(X)
y = np.array(y)

model = load_model('pneumonia_detection_model.h5')

for directory in os.listdir(test_folder):
    print("directory: ", directory)
    for item in os.listdir(os.path.join(test_folder, directory)):
        print("item: ", item)
        img = cv2.imread(os.path.join(os.path.join(test_folder, directory), item))
        img = cv2.resize(img, (img_width, img_height))
        # img = np.mean(img, axis=2)  # convert to 1-dim grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        # print(img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        if directory == 'NORMAL':
            real_label = [0, 1]
        elif directory == 'PNEUMONIA':
            real_label = [1, 0]

        # img = img/255
        Xtest = img
        Xtest = Xtest.reshape(-1, img_width, img_height, img_channels)
        # print("test image shape", img.shape)

        # demonstrate prediction

        yhat = model.predict(Xtest, verbose=0)

        print("Real label: ", real_label)
        print("AI model's prediction: ", yhat)
        print("\n")





