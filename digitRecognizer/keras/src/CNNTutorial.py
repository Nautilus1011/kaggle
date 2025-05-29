# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("/home/jinysd/workspace/repo/kaggle/digitRecognizer/data"))


output_path = "/home/jinysd/workspace/repo/kaggle/digitRecognizer/keras/output"


# read train 
train = pd.read_csv("/home/jinysd/workspace/repo/kaggle/digitRecognizer/data/train.csv")
print(train.shape)
train.head()

# read test 
test= pd.read_csv("/home/jinysd/workspace/repo/kaggle/digitRecognizer/data/test.csv")
print(test.shape)
test.head()

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"], axis = 1)

# visualize number of digits classes
plt.figure(figsize=(15, 7))
g = sns.countplot(x = Y_train, palette="icefire")
plt.title("Number of digit classes")
save_path = os.path.join(output_path, "digit_class_distribution.png")
plt.savefig(save_path)
Y_train.value_counts()

# plot some samples
img = X_train.iloc[0].values
img = img.reshape((28, 28))
plt.figure()
plt.imshow(img, cmap = 'gray')
plt.title(train.iloc[0,0])
plt.axis("off")
save_path = os.path.join(output_path, "sample_digit(1)")
plt.savefig(save_path)

# plot some samples
img = X_train.iloc[3].values    # iloc[]はPandasのDataFrameの行を取得, valuesをつけるとnumpy配列になる
img = img.reshape((28, 28))
plt.figure()
plt.imshow(img, cmap = 'gray')
plt.title(train.iloc[3,0])
plt.axis("off")
save_path = os.path.join(output_path, "sample_digit(3)")
plt.savefig(save_path)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("X_train shape:", X_train.shape)
print("test shape:", test.shape)

# Reshape   -1:データのサンプル数(-1は自動で計算を指す) 28 x 28の2D画像に変換, 1:チャネル数 カラーなら3
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("X_train shape:", X_train.shape)
print("test shape:", test.shape)

# Label Encoding  ラベルをワンホットエンコーディング 3:[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

# examples 9
plt.figure()
plt.imshow(X_train[2][:,:,0], cmap = 'gray')
plt.title(train.iloc[3,0])
plt.axis("off")
save_path = os.path.join(output_path, "sample_digit(9)")
plt.savefig(save_path)



# model
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', 
                 activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# Define the optimizer
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 10
batch_size = 250

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# Plot the loss and accuracy curves for training and validation
plt.figure()
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
save_path = os.path.join(output_path, "Test_Loss")
plt.savefig(save_path)


# confusion matrix
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
save_path = os.path.join(output_path, "Confusion_Matrix")
plt.savefig(save_path)