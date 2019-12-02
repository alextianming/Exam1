# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:24:12 2019
#https://www.kesci.com/home/project/5c00abbc954d6e001068d37b/code
#最普通的CNN来做图像的分类，同时也会对训练后的模型进行评估与分析解读
@author: wfq
"""
# In[] 导入包
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Activation,Flatten,BatchNormalization

import pickle

# In[] 导入数据

def unpickle(file):
      with open(file, 'rb') as fo:
              dict = pickle.load(fo, encoding='bytes')
      return dict

# In[] 训练集文件
train_files = ['data_batch_'+str(i) for i in range(1,6)]

train_data_list,train_label_list = [],[]
for f in train_files:
    
    fpath = 'E:/PythonEx/DataInputDeal/Data/' + f
    batch_dict = unpickle(fpath)
    
    batch_data = batch_dict[b'data']
    batch_labels = batch_dict[b'labels']
    train_data_list.append(batch_data)
    train_label_list.append(batch_labels)

X_train = np.concatenate(train_data_list, axis = 0)
y_train = np.concatenate(train_label_list, axis = 0)
# In[] 测试集文件
test_batch = unpickle('E:/PythonEx/DataInputDeal/Data/test_batch')

X_test = np.array(test_batch[b'data'])
y_test = np.array(test_batch[b'labels']) # list type
# In[] 
label_names_batch = unpickle('E:/PythonEx/DataInputDeal/Data/batches.meta')
label_names = label_names_batch[b'label_names']
label_names = [l.decode("utf-8") for l in label_names]

print('训练集特征：',X_train.shape,'，训练集label',y_train.shape)
print('测试集特征：',X_test.shape,'，测试集label', y_test.shape)
print("类别名字：",label_names)
print(32*32)
# In[] 数据的预 展示 
num_classes = 10
X_train = X_train.reshape(-1,3,32,32)
X_test = X_test.reshape(-1,3,32,32)

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = X_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(label_names[i])
    plt.imshow(im)
plt.show()

# In[] 数据 预 处理
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 归一化
X_train /= 255
X_test /= 255

# 将class vectors转变成binary class metrics
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# In[] *****构建 模型 *****************
#Keras有两种后台可以实现，一种是TensorFlow backend,一种是Theano backend。
print('Backend:',keras.backend.image_dim_ordering())
# 查看当前数据集图像的维度顺序
X_train.shape[1:]
# Method 1: Switch backend
#keras.backend.set_image_dim_ordering('th')
#print('Backend:',keras.backend.image_dim_ordering())
# Method 2: reshape
# X_train = X_train.reshape(-1,32,32,3)
# X_train.shape

def base_model(opt):
    model = Sequential()
    
    # 32个卷积核(feature maps),步长为1，特征图的大小不会改变（周边补充空白），
    model.add(Conv2D(32,(3,3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # channel是在前面 (theano后台)
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    # 64个卷积核
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    model.add(Flatten())   # Flatten layer
    model.add(Dense(512))  # fully connected layer with 512 units
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes)) # Fully connected output layer with 10 units
    model.add(Activation('softmax')) # softmax activation function
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']) # 要优化的是准确率
    return model

# In[] 模型训练
# 初始化 RMSprop 优化器
opt1 = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# 初始化 Adam 优化器
# opt2 = keras.optimizers.Adam(lr=0.0001)

# 用RMSProp训练模型
cnn = base_model(opt1)
cnn.summary() # 打印网络结构及其内部参数

# In[] 模型训练的误差图
# 进行100轮批次为32的训练,默认训练过程中会使用正则化防止过拟合            
history = cnn.fit(X_train, y_train, 
                    epochs = 100, batch_size = 32, 
                    validation_data=(X_test,y_test), 
                    shuffle=True)
 
def plot_loss_and_accuracy(history):
    # Plots for training and testing process: loss and accuracy
 
    plt.figure(0)
    plt.plot(history.history['acc'],'r')
    plt.plot(history.history['val_acc'],'g')
    plt.xticks(np.arange(0, 101, 20))
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])
     
     
    plt.figure(1)
    plt.plot(history.history['loss'],'r')
    plt.plot(history.history['val_loss'],'g')
    plt.xticks(np.arange(0, 101, 20))
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
     
    plt.show()
    
plot_loss_and_accuracy(history)
# In[] 模型测试     ***&&&********
# 对样本进行，默认不使用正则化
score = cnn.evaluate(X_test,y_test)
print("损失值为{0:.2f},准确率为{1:.2%}".format(score[0],score[1]))

# In[] 模型的评估 混淆矩阵
from sklearn.metrics import confusion_matrix
Y_pred = cnn.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
 
# for ix in range(10):
#     print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, 
            annot=True,
            annot_kws={"size": 12})# font size
plt.show()