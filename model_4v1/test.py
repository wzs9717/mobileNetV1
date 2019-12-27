from __future__ import print_function
import keras
import sys
sys.setrecursionlimit(10000)

#import densenet
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img

from PIL import Image
def pic_transfer(X):
#    resize X from 32 by 32 to 224by224
    number_X=2000
    X_resize=np.zeros((number_X,224,224,3))
    for i in range(number_X):
        pic_tem=array_to_img(X[i,:,:,:])
        pic_tem2=pic_tem.resize((224, 224),Image.BILINEAR)
        X_resize[i,:,:,:]=img_to_array(pic_tem2)
    return X_resize
    
(trainX, trainY), (testX, testY) = cifar100.load_data()
trainY = trainY.reshape(trainY.shape[0])
testY = testY.reshape(testY.shape[0])
print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
trainX = trainX.astype('float32')
testX = testX.astype('float32')

    
#image=np.ones((100,32,32,3))
testX2=pic_transfer(testX)
#ex1=testX[1,:,:,:]
#ex2=array_to_img(ex1)
#ex2=ex2.resize((224, 224),Image.BILINEAR)
#ex3=img_to_array(ex2)
#plt.imshow(ex2)
#plt.show()
#print(testX2.shape)
testXimg=array_to_img(testX[1,:,:,:])
testXimg2=array_to_img(testX2[1,:,:,:])
plt.imshow(testXimg)
plt.show()
plt.imshow(testXimg2)
plt.show()





