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

train_folder = "images/train/"
print("train folder: ", train_folder)

img_width = 200
img_height = 200
img_channels = 3
batch = 16
epochs = 5

n_classes = 2  # [pneumonia, normal]

filenames_list = []
X = []
Y = []
#  Data Read
for directory in os.listdir(train_folder):
    print("directory: ", directory)
    for item in os.listdir(os.path.join(train_folder, directory)):
        # print("item: ", item)
        img = cv2.imread(os.path.join(os.path.join(train_folder, directory), item))
        img = cv2.resize(img, (img_width, img_height))
        # img = np.mean(img, axis=2)  # convert to 1-dim grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        # print(img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        if directory == 'NORMAL':
            label = [0, 1]
        elif directory == 'PNEUMONIA':
            label = [1, 0]

        # img = img/255
        X.append(img)
        Y.append(label)

print("X shape: ", np.shape(X))
print("Y shape: ", np.shape(Y))

print("Number of samples in", train_folder, ":", len(X))
print("\n")

X = np.array(X)
Y = np.array(Y)


start_time = time.time()

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(activation='relu', units=128))
model.add(Dense(activation='relu', units=64))
model.add(Dense(activation='sigmoid', units=2))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
history = model.fit(X, Y, validation_split=0.2, batch_size=batch, epochs=epochs, verbose=1, callbacks=[checkpoint])


early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
callbacks_list = [early, learning_rate_reduction]

model.save('my_model.h5')
pd.DataFrame(model.history.history).plot()


elapsed_time = time.time() - start_time
elapsed_time = elapsed_time/60
elapsed_time = str(round(elapsed_time, 2))
print("Training duration : ", elapsed_time, "minutes")
