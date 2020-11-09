import os
import numpy as np
import cv2
import random as ran
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import models, optimizers, regularizers
from keras.callbacks import LearningRateScheduler

weight_decay = 5e-4
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
optimizer = 'SGD'
epoch_num = 20

def VGG16():
    model = models.Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Flatten())  # 2*2*512
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


if __name__ == '__main__':
    
    # label = {
    #     0:"airplane",
    #     1:"automobile",
    #     2:"bird",
    #     3:"cat",
    #     4:"deer",
    #     5:"dog",
    #     6:"frog",
    #     7:"horse",
    #     8:"ship",
    #     9:"truck"
    # }
    label = {
        '0':"airplane",
        '1':"automobile",
        '2':"bird",
        '3':"cat",
        '4':"deer",
        '5':"dog",
        '6':"frog",
        '7':"horse",
        '8':"ship",
        '9':"truck"
    }
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # # d = np.array(y_train.copy(),dtype='int8')
    # # print(d)
    # for i in range(10):
    #     r = ran.randint(0,9)
    #     cv2.imshow(label[d],x_train[r])
    #     print(label[y_train[r]])
    #     print(y_train[r])
        

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # print(y_train)

    # batch_size = 32
    # learning_rate = 0.001
    # optimizer = 'SGD'
    print("hyperparameters:")
    print("batch size:",batch_size)
    print("learning rate:",learning_rate)
    print("optimizer:",optimizer)

    # get model
    model = VGG16()

    # show model
    model.summary()

    # train
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    change_lr = LearningRateScheduler(scheduler)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch_num,callbacks=[change_lr] ,validation_data=(x_test,y_test))