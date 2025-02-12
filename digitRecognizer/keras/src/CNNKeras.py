import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import os

data_path = "/home/jinysd/workspace/repo/kaggle/digitRecognizer/data"
output_path = "/home/jinysd/workspace/repo/kaggle/digitRecognizer/keras/output"

# read train and test
train = pd.read_csv(os.path.join(data_path, "train.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))

# put labels into y_train variable
Y_train = train["label"]    # 正解ラベル
# Drop "label" column
X_train = train.drop(labels = ["label"], axis=1)    # 謎１　labelsとは

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape 
X_train = X_train.values.reshape(-1,28,28,1)
test= test.values.reshape(-1,28,28,1)

# Label Encoding
Y_train = to_categorical(Y_train, num_classes=10)   # これをする理由わからない

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)






#model
model = Sequential()
# 畳み込み層　
model.add(Conv2D(filters=8, kernel_size=(5,5), padding="Same",
                 activation="relu", input_shape=(28,28,1)))
# フィルターの数８であるから出力の特徴マップは(28,28,8)
# padding="same"であるから入力と出力画像のサイズは同じ
# relu関数であるからマイナスの数値は0となる

# プーリング層
model.add(MaxPool2D(pool_size=(2,2)))
# 最大プーリング(2x2)を適用することで特徴マップを縮小
# 入力(28,28,8) → 出力(14,14,8)

# ドロップアウト
model.add(Dropout(0.25))
# 過学習を防ぐため無作為に25%のニューロンを消す

# 平坦化
model.add(Flatten())
# 全結合層に入力するために3D特徴マップ(14,14,8)を1Dベクトル(1568)に変換

# 全結合層 
model.add(Dense(256, activation="relu")) 
# 256ニューロンの層
# 1568のニューロンと256のニューロン一つ一つがすべてと繋がっている

# ドロップアウト
model.add(Dropout(0.5))
# 過学習を防ぐため無作為に50%のニューロンを消す

# 2つ目の全結合層
model.add(Dense(10, activation="softmax"))
# 分類問題であるから出力層でもある
# ソフトマックス関数で確率として出力される
# ニューロン数は分類したい物の数, 今回は10




# Defin the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)    # 謎
# オプティマイザの定義
# Adam:SGD(確率的勾配降下法)を改良したアルゴリズム
# 学習率を適応的に調整するのが特徴

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics = ["accuracy"])
# モデルのコンパイル
# 損失関数：クロスエントロピー, 多クラス分類に適した関数, 
# metrics まだ理解できていない


epochs = 10
batch_size = 250

# データ拡張
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



# confusion matrix
# Predict the values from the validation dataset
Y_pred = model.predict(test)
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
save_path = os.path.join(output_path, "Confusion_Matrix_test")
plt.savefig(save_path)