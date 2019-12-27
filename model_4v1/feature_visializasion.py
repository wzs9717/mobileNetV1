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
from keras.models import Sequential,load_model
from PIL import Image
import tensorflow as tf
import gc
import pandas as pd
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
import os
from keras.utils import multi_gpu_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
 
def main():

    batch_size = 128
    nb_classes = 100
    nb_epoch = 6000
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
    alpha=1.0
    # base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=img_dim, alpha=alpha, include_top=False, weights="imagenet", input_tensor=None, pooling=None, classes=100)
    # shape = (1, 1, int(1280 * alpha))
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 1280))(x)
    # x = Dropout(0.3, name='Dropout')(x)
    # x = Conv2D(nb_classes, (1, 1), padding='same')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Activation('softmax', name='softmax')(x)
    # output = Reshape((nb_classes,))(x)
    # inputs = base_model.input
    # model = keras.models.Model(inputs, x)
    # optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    # model.load_weights("weights/MobilenetV2-CIFAR100.h5")
    # print("Model loaded.")
    # model.save('weights/model_feature_extraction.h5')

    model=load_model('weights/model_feature_extraction.h5')
    model.summary()
    # model = load_model("weights/MobilenetV2-CIFAR100.h5") #replaced by your model name
    # Get all our test images.
    image='/home/kleong013/Documents/intern document/wang zhisheng/detection/MobileNet-master (2)/model_3/data/test/0.jpg'
    images=cv2.imread('/home/kleong013/Documents/intern document/wang zhisheng/detection/MobileNet-master (2)/model_3/data/test/0.jpg')
    cv2.imshow("Image", images)
    cv2.waitKey(0)
    # Turn the image into an array.
    image_arr = images
    image_arr = np.expand_dims(image_arr, axis=0)
 
# 设置可视化的层
    layer_1 = K.function([model.layers[0].input], [model.layers[2].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [112, 112]
        plt.subplot(4, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    # conv layer: 299

    # layer_name='block_1_depthwise'
    # intermediate_layer_model = Model(input=model.input,
    #                              output=model.get_layer(layer_name).output)

    layer_1 = K.function([model.layers[0].input], [model.layers[3].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [show_img.shape[1],show_img.shape[2]]
        plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

    layer_1 = K.function([model.layers[0].input], [model.get_layer('expanded_conv_project_BN').output])
    f1 = layer_1([image_arr])[0]
    for _ in range(16):
        show_img = f1[:, :, :, _]
        show_img.shape = [112, 112]
        plt.subplot(2, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

    layer_1 = K.function([model.layers[0].input], [model.get_layer('block_1_project_BN').output])
    f1 = layer_1([image_arr])[0]
    for _ in range(24):
        show_img = f1[:, :, :, _]
        show_img.shape = [56, 56]
        plt.subplot(3, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()


    layer_1 = K.function([model.layers[0].input], [model.get_layer('block_3_project_BN').output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [28, 28]
        plt.subplot(4, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

    layer_1 = K.function([model.layers[0].input], [model.get_layer('block_7_project_BN').output])
    f1 = layer_1([image_arr])[0]
    for _ in range(64):
        show_img = f1[:, :, :, _]
        show_img.shape = [14, 14]
        plt.subplot(8, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()


    layer_1 = K.function([model.layers[0].input], [model.get_layer('block_15_project_BN').output])
    f1 = layer_1([image_arr])[0]
    for _ in range(40):
        show_img = f1[:, :, :, _]
        show_img.shape = [7, 7]
        plt.subplot(5, 8, _ + 1)
        # plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

    print('This is the end !')
 
if __name__ == '__main__':
    main()

