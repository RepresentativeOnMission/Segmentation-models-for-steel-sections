import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
import pandas as pd

from PIL import Image as Img
import cv2
from tensorflow.keras.utils import Sequence


def identity_block(x, filters, kernel_size, block, stage):
    base_conv_name = 'res' + str(stage) + block + '_branch'
    base_bn_name = 'bn' + str(stage) + block + '_branch'

    f1, f2, f3 = filters

    shortcut_x = x

    x = layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=1, padding='valid', name=base_conv_name + '2a')(x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=f2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                      name=base_conv_name + '2b')(x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=1, padding='valid', name=base_conv_name + '2c')(x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2c')(x)

    x = layers.Add()([x, shortcut_x])
    x = layers.Activation('relu')(x)

    return x


def convolutional_block(input_x, filters, kernel_size, strides, block, stage):
    base_conv_name = 'res' + str(stage) + block + '_branch'
    base_bn_name = 'bn' + str(stage) + block + '_branch'

    f1, f2, f3 = filters

    x = layers.Conv2D(f1, (1, 1), padding='valid', strides=strides, name=base_conv_name + 'a')(input_x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f2, (kernel_size, kernel_size), padding='same', strides=1, name=base_conv_name + '2b')(x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f3, (1, 1), padding='valid', strides=1, name=base_conv_name + '2c')(x)
    x = layers.BatchNormalization(axis=3, name=base_bn_name + '2c')(x)

    shortcut_x = layers.Conv2D(f3, (1, 1), padding='valid', strides=strides, name=base_conv_name + '1')(input_x)
    shortcut_x = layers.BatchNormalization(axis=3, name=base_bn_name + 'shortcut')(shortcut_x)

    x = layers.Add()([x, shortcut_x])
    x = layers.Activation('relu')(x)
    return x

# Es: convolutional_block(x, (64,64,256), 3, 2, 'a', 1)

def ResNet50_pretrained(input_shape, classes, retrain=False, include_top=True):
    X_input = layers.Input(input_shape)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(X_input)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, name='conv1')(x)
    x = layers.BatchNormalization(axis=3, name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)
    pool1 = x
    x = convolutional_block(x, [64, 64, 256], kernel_size=3, strides=1, block='a', stage=1)
    x = identity_block(x, [64, 64, 256], kernel_size=3, block='b', stage=1)
    x = identity_block(x, [64, 64, 256], kernel_size=3, block='c', stage=1)
    pool2 = x
    x = convolutional_block(x, [128, 128, 512], kernel_size=3, strides=2, block='a', stage=2)
    x = identity_block(x, [128, 128, 512], kernel_size=3, block='b', stage=2)
    x = identity_block(x, [128, 128, 512], kernel_size=3, block='c', stage=2)
    x = identity_block(x, [128, 128, 512], kernel_size=3, block='d', stage=2)
    pool3 = x
    x = convolutional_block(x, [256, 256, 1024], kernel_size=3, strides=2, block='a', stage=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3, block='b', stage=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3, block='c', stage=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3, block='d', stage=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3, block='e', stage=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3, block='f', stage=3)
    pool4 = x
    x = convolutional_block(x, [512, 512, 2048], kernel_size=3, strides=2, block='a', stage=4)
    x = identity_block(x, [512, 512, 2048], kernel_size=3, block='b', stage=4)
    x = identity_block(x, [512, 512, 2048], kernel_size=3, block='c', stage=4)
    pool5 = x
    if(include_top):
        # x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    model = models.Model(inputs=X_input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='softmax', metrics=['accuracy'])
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    return model, pool1, pool2, pool3, pool4, pool5