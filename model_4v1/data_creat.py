from __future__ import print_function
import keras
import sys
sys.setrecursionlimit(10000)

#import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential
from PIL import Image
import tensorflow as tf
import cv2
import scipy.misc


def pic_creat(X,file_dic,file_name):
    number_X=X.shape[0]
    for i in range(number_X):
        pic_tem=array_to_img(X[i,:,:,:])
        pic_tem2=pic_tem.resize((224, 224),Image.BILINEAR)
        X_resize=img_to_array(pic_tem2)
        scipy.misc.imsave(file_dic+"/"+str(i)+".jpg",X_resize)
    	# scipy.misc.imsave(file_dic+"/"+str(i)+".jpg",X_resize)
        # pic_tem=array_to_img(X[i,:,:,:])






(trainX, trainY), (testX, testY) = cifar100.load_data()
trainY = trainY.reshape(trainY.shape[0])
testY = testY.reshape(testY.shape[0])
print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
trainX = trainX.astype('float32')
testX = testX.astype('float32')

pic_creat(testX,"./data/test","test")
pic_creat(trainX,"./data/train","train")
# pic_creat(valX,"./data/val","val")


# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


# train=unpickle('train')
# print(train.keys())

# l=train[b'fine_labels']
# print(l[0])
# def append_ext(fn):
#     return fn+".jpg"

# traindf=pd.DataFrame({'id':np.arange(trainY.shape[0]),'label':trainY},index=np.arange(trainY.shape[0]))
# valdf=pd.DataFrame({'id':np.arange(valY.shape[0]),'label':valY},index=np.arange(valY.shape[0]))
# testdf=pd.DataFrame({'id':np.arange(testY.shape[0]),'label':testY},index=np.arange(testY.shape[0]))

# traindf["id"]=traindf["id"].apply(append_ext)
# valdf["id"]=valdf["id"].apply(append_ext)
# testdf["id"]=testdf["id"].apply(append_ext)

# train_generator = datagen.flow_from_dataframe（dataframe = traindf，
#                                       directory =“./data/ train /”，
#                                       x_col =“id”，
#                                       y_col =“label”，
#                                       subset =“training”，
#                                       batch_size = 32，
#                                       seed = 42，
#                                       shuffle = True，
#                                       class_mode =“sparse”，
#                                       target_size =（224,224））
# valid_generator = datagen.flow_from_dataframe（dataframe = traindf，
#                                       directory =“./data/ train /”，
#                                       x_col =“id”，
#                                       y_col =“label”，
#                                       subset =“validation”，
#                                       batch_size = 32，
#                                       seed = 42，
#                                       shuffle = True，
#                                       class_mode =“sparse”，
#                                       target_size =（224,224））

# testdatagen = ImageDataGenerator（rescale = 1. / 255。） 

# test_generator = test_datagen.flow_from_dataframe（dataframe = testdf，
#                                      directory =“./data/ test /”，
#                                      x_col =“id”，
#                                      y_col = None，
#                                      batch_size = 32，
#                                      seed = 42，
#                                      shuffle = False，
#                                      class_mode = None，
#                                      target_size =（32,32））
