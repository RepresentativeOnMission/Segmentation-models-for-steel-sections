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

def PSPNet_resnet50(n_classes, input_shape, retrain=False):
    encoder_freeze = not retrain
    return sm.PSPNet('resnet50', input_shape=input_shape ,encoder_weights='imagenet', classes=n_classes, encoder_freeze=encoder_freeze, activation='linear')

class psp_unet():
    
    def __init__(self, backbone, n_classes, retrain=False, conv_filters=256):
        self.backbone = backbone
        self.n_classes = n_classes
        self.retrain = retrain
        self.conv_filters = conv_filters
        
    def spatial_block(self, x, up_size, conv_filters):
        
        x = layers.MaxPool2D(pool_size=(up_size, up_size), strides=None, padding='same', data_format=None,name='pyramid_pooling_max_pool_'+str(up_size))(x)
        x = layers.Conv2D(filters=self.conv_filters, kernel_size=(1,1), strides=1, name='pyramid_pooling_conv1x1_'+str(up_size), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(up_size, interpolation='bilinear', name='pyramid_pooling_upsampling_'+str(up_size))(x)
        return x
    
    def decoder_upconv_block(self, x, x_skip, filters, dilation=False, concatenate=True):
        
        if(dilation):
            x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, dilation_rate=2, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_1')(x)
        else:
            x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                              bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_1')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[0]) + '_1')(x)
        
        
        if(dilation):
            x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, dilation_rate=2, padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                              bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_2')(x)
        else:
            x = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                          bias_initializer='zeros', name='conv_decoder_' + str(filters[0]) + '_2')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[0]) + '_2')(x)
        
        
        x = tf.keras.layers.UpSampling2D(size=2, data_format=None, interpolation='bilinear',
                                         name='upsample_decoder_' + str(filters[0]))(x)
        x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=1, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   bias_initializer='zeros', name='conv_' + str(filters[1]) + '_upsample')(x)
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_decoder_' + str(filters[1]) + '_3')(x)
        if(concatenate):
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

    def get_model(self):

        for layer in self.backbone.layers:
            layer.trainable = self.retrain
        
        if(self.backbone.name=='short_resnet50'):
            
            pool_4 = self.backbone.get_layer('conv4_block6_out')
            pool_3 = self.backbone.get_layer('conv3_block4_out')
            pool_2 = self.backbone.get_layer('conv2_block3_out')
            pool_1 = self.backbone.get_layer('conv1_relu')

        inputs = self.backbone.inputs
        x = pool_4.output
        
        #build pyramid pooling, using pool_3 as filter map
        x1 = self.spatial_block(x, up_size=1, conv_filters=self.conv_filters)
        x2 = self.spatial_block(x, up_size=2, conv_filters=self.conv_filters)
        x3 = self.spatial_block(x, up_size=3, conv_filters=self.conv_filters)
        x6 = self.spatial_block(x, up_size=6, conv_filters=self.conv_filters)
        x = layers.Concatenate(axis=-1, name='psp_concat')([x, x1, x2, x3, x6])
        #build 2 atrous convolutions, with 2 and 2 as dilation
        x = self.decoder_upconv_block(x, pool_3.output, [1024,512], dilation=True)
        #build UNet decoder part
        x = self.decoder_upconv_block(x, pool_2.output, [512,256])
        x = self.decoder_upconv_block(x, pool_1.output, [256,64])
        x = self.decoder_upconv_block(x, None, [64,32], concatenate=False)
        x = self.final_layers(x, 32, self.n_classes)
        
        return models.Model(inputs=inputs, outputs=x, name='psp_unet')