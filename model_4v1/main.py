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
import gc

def pic_transfer(X,number_batch):
#    resize X from 32 by 32 to 224by224
    number_X=number_batch
    X_resize=np.zeros((number_X,224,224,3))
    for i in range(number_X): 	
        pic_tem=array_to_img(X[i,:,:,:])
        pic_tem2=pic_tem.resize((224, 224),Image.BILINEAR)
        X_resize[i,:,:,:]=img_to_array(pic_tem2)
    return X_resize
    
    
    
batch_size = 64
nb_classes = 100
nb_epoch = 12
classes=100
img_rows, img_cols = 224, 224
img_channels = 3

img_dim = (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 12
bottleneck = False
reduction = 0.0
dropout_rate = 0.0 # 0.0 for data augmentation

#--------------------------------------------------------------------------------------------
alpha=1.0
#pre_model = Sequential()

base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=img_dim, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=100)
shape = (1, 1, int(1024 * 1.0))
x = base_model.output
x = keras.layers.Flatten(name='flatten')(x)
x = keras.layers.Dense(classes, activation='softmax',
                         use_bias=True, name='Logits')(x)
inputs = base_model.input
model = keras.models.Model(inputs, x)

for layer in model.layers[:8]:
       layer.trainable = False
       
print("Model created")
#--------------------------------------------------------------------------------------------
model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar100.load_data()
trainY = trainY.reshape(trainY.shape[0])
testY = testY.reshape(testY.shape[0])
print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX=pic_transfer(trainX,10000)
testX=pic_transfer(testX,500)
trainY=trainY[0:10000]
testY=testY[0:500]

#del trainX, trainYtestX, testY
#gc.collect()

trainX /= 255.
testX /= 255.

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)
#print(Y_train.shape)
#print(Y_test.shape)
generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                              )
#generator_test=ImageDataGenerator(rotation_range=0,
#                               width_shift_range=0,
#                               height_shift_range=0,
#                               rescale=7)

# trainX=tf.keras.backend.resize_images(
    # trainX,
    # height_factor=7,
    # width_factor=7,
    # data_format="channels_last",
    # interpolation='nearest'
# )
generator.fit(trainX, seed=0)
#generator.fit(testX, seed=0)

#transform_parameters['zx']=7

#testX=apply_transform(testX, zx=7,zy=7)

# Load model
# model.load_weights("weights/DenseNet-BC-100-12-CIFAR100.h5")
# print("Model loaded.")

lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
model_checkpoint= ModelCheckpoint("weights/MobilenetV2-CIFAR100.h5", monitor="val_acc", save_best_only=True,
                                  save_weights_only=True)

callbacks=[lr_reducer, early_stopper, model_checkpoint]


model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
                   callbacks=callbacks,
				   validation_data=(testX, Y_test),
                   nb_val_samples=testX.shape[0], verbose=1)
yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
