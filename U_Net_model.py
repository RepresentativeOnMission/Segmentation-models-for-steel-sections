import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
from PIL import Image
import cv2
import albumentations as A
import os
from tensorflow.keras.utils import Sequence
import sys
import argparse
import json
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models as sm


class U_Net_vanilla(tf.keras.Model):

    def __init__(self, n_classes, shape):
        super(U_Net_vanilla, self).__init__()
        self.n_classes = n_classes
        self.shape = shape

    def encoder_downconv_block(self, x, filters):
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_encoder_' + str(filters) + '_1')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_encoder_' + str(filters) + '_1')(x)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_encoder_' + str(filters) + '_2')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_encoder_' + str(filters) + '_2')(x)
        x_skip = x
        x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='maxpool_encoder_' + str(filters))(x)
        return x, x_skip

    def decoder_upconv_block(self, x, x_skip, filters):
        x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_1')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[0]) + '_1')(x)
        x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_2')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[0]) + '_2')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear',
                                         name='upsample_decoder_' + str(filters[0]))(x)
        x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(2, 2), strides=1, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   bias_initializer='zeros', name='conv_' + str(filters[1]) + '_upsample')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[1]) + '_3')(x)
        x = tf.keras.layers.Concatenate()([x, x_skip])
        return x

    def final_layers(self, x, filters, n_classes):
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters) + '_1')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters) + '_2')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=n_classes, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='final_layer_' + str(n_classes))(x)
        return x

    def call(self, inputs, training=True):
        x, x_skip_1 = self.encoder_downconv_block(inputs, 64)
        x, x_skip_2 = self.encoder_downconv_block(x, 128)
        x, x_skip_3 = self.encoder_downconv_block(x, 256)
        x, x_skip_4 = self.encoder_downconv_block(x, 512)
        x = self.decoder_upconv_block(x, x_skip_4, [1024, 512])
        x = self.decoder_upconv_block(x, x_skip_3, [512, 256])
        x = self.decoder_upconv_block(x, x_skip_2, [256, 128])
        x = self.decoder_upconv_block(x, x_skip_1, [128, 64])
        x = self.final_layers(x, 64, self.n_classes)
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=self.shape)
        return models.Model(inputs=[x], outputs=self.call(x))

def U_Net_resnet50(n_classes,  input_shape=(224,224,3), retrain=False,):
    #i need this line of code because, with retrain=True, i mean: don't freeze the encoder weights, while he means freeze them
    encoder_freeze = not retrain
    return sm.Unet('resnet50',encoder_weights='imagenet', classes=n_classes, encoder_freeze=encoder_freeze, activation='linear', input_shape=input_shape)

def U_Net_resnet34(n_classes, input_shape=(224,224,3), retrain=False):
    #i need this line of code because, with retrain=True, i mean: don't freeze the encoder weights, while he means freeze them
    encoder_freeze = not retrain
    return sm.Unet('resnet34',encoder_weights='imagenet', classes=n_classes, encoder_freeze=encoder_freeze, activation='linear', input_shape=input_shape)
    