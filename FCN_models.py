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
import abc

#arguments:
#
# * backbone: this is a functional (or sequential) model of an encoder, currently this class supports only ResNet50
# * n_classes: this is the number of classes, the last layer will have this as quantity of filters
# * filters: contains the number of filters of each block. The number of filters for the last block is n_classes
# * block_types: contains the types of blocks. You can use transposed convolution or Upsampling + Convolution
# * retrain: Allow the weights of the encoder to be modified by backpropagation during training
# * strides: This is the multiplier, for each block, which multiply height and width of the image. (2,2,8) = 2*2*8
# * apply_postprocess_conv: this is an optional convolutional layer with strides=1, filters=n_classes and kernel_size=3, that will be positioned as the final layer
#


#this class summarize all classes below, which are deprecated. If you want to understand a specific model, however, i suggest to watch the deprecated models.
class FCN_8():
    
    def __init__(self,
                 backbone,
                 n_classes,
                 filters = [128,64], 
                 block_types = ['conv_transposed','conv_transposed','conv_transposed'],
                 strides = [2, 2, 8],
                 retrain=False,
                 model_name = 'FCN_8',
                 apply_postprocess_conv=False):
        self.backbone = backbone
        self.filters = filters
        self.block_types = block_types
        self.retrain = retrain
        self.strides = strides
        self.n_classes = n_classes
        self.model_name = model_name
        self.apply_postprocess_conv = apply_postprocess_conv

    def conv_transposed_block(self, x, filters, stride, skip_conn=None, i=-1, is_last_layer=False):
        

        if not is_last_layer:
            x = layers.Conv2DTranspose(filters=filters, kernel_size=(2 * stride, 2 * stride), strides=stride,
                                       name='transposed_conv_block_' + str(i) + '_layer_1', padding='SAME')(x)
            x = layers.Add()([x, skip_conn])
        else:
            x = layers.Conv2DTranspose(filters=filters, kernel_size=(2 * stride, 2 * stride), strides=stride,
                                       name='transposed_conv_block_' + str(i) + '_layer_1', padding='SAME')(x)

        return x

    def upsampling_conv_block(self, x, filters, upsample_diluation, skip_conn=None, i=-1, is_last_layer=False):

        x = layers.UpSampling2D(size=(upsample_diluation, upsample_diluation), data_format=None,
                                interpolation='bilinear', name='upsample_block_' + str(i) + '_layer_1')(x)
        if not is_last_layer:
            x = layers.Conv2D(filters=filters, kernel_size=(2 * upsample_diluation, 2 * upsample_diluation), strides=1,
                              name='conv_block_' + str(i) + '_layer_2', padding='SAME')(x)
            x = layers.Add()([x, skip_conn])
        else:
            x = layers.Conv2D(filters=filters, kernel_size=(2 * upsample_diluation, 2 * upsample_diluation), strides=1,
                              name='conv_block_' + str(i) + '_layer_2', padding='SAME')(x)
        return x

    def get_model(self):
            
        
        
        for layer in self.backbone.layers:
            layer.trainable = self.retrain

            
        skip_connections = []
        if(self.backbone.name=='resnet50'):
            
            pool_5 = self.backbone.get_layer('conv5_block3_out')
            pool_4 = self.backbone.get_layer('conv4_block6_out')
            skip_connections.append(pool_4)
            pool_3 = self.backbone.get_layer('conv3_block4_out')
            skip_connections.append(pool_3)
            pool_2 = self.backbone.get_layer('conv2_block3_out')
            skip_connections.append(pool_2)
            pool_1 = self.backbone.get_layer('conv1_relu')
            skip_connections.append(pool_1)

        inputs = self.backbone.inputs
        x = pool_5.output
        
        #layers with skip connections
        for i in range(0, len(self.block_types)-1):
            #calculate current skip connection
            skip_conn = layers.Conv2D(filters=self.filters[i], kernel_size=(1, 1), strides=1, name='pool_'+str(i)+'_1', padding='SAME')(skip_connections.pop(0).output)
            if self.block_types[i] == 'conv_transposed':
                x = self.conv_transposed_block(x, self.filters[i], self.strides[i], skip_conn, i, is_last_layer=False)
            elif self.block_types[i] == 'upsampling_conv':
                x = self.upsampling_conv_block(x, self.filters[i], self.strides[i], skip_conn, i, is_last_layer=False)
        
        #last layer is an upsample + conv layer, that doesn't use skip connections
        if self.block_types[-1] == 'conv_transposed':
            x = self.conv_transposed_block(x, self.n_classes, self.strides[-1], is_last_layer=True)
        elif self.block_types[-1] == 'upsampling_conv':
            x = self.upsampling_conv_block(x, self.n_classes, self.strides[-1], is_last_layer=True)
        
        #postprocess conv (optional)
        if(self.apply_postprocess_conv):
            x = layers.Conv2D(filters=self.n_classes, kernel_size=(3, 3), strides=1, name='postprocess_conv', padding='SAME')(x)
        
        return models.Model(inputs = inputs, outputs = x, name=self.model_name)


# FCN-32
def FCN_32(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs
    x = pool_5.output
    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(64, 64), strides=32, name='transposedConv1')(x)
    return models.Model(inputs=inputs, outputs=x, name='FCN_32')


# FCN-16
def FCN_16(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs
    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, name='pool_5_upsample2x')(pool_5.output)
    pool_4_conv = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, name='pool_4_conv')(pool_4.output)
    x = layers.Add()([x, pool_4_conv])
    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(32, 32), strides=16, name='transposedConv1')(x)
    return models.Model(inputs=inputs, outputs=x, name='FCN_16')



#---------------------------------------------------------DEPRECATED------------------------------------------------------------------------
# FCN-8
def FCN_8_original(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs

    pool_5_up = layers.Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, name='pool_5_upsample2x',
                                       padding='SAME')(pool_5.output)

    pool_4_n = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, name='pool_4_1', padding='SAME')(pool_4.output)

    x1 = layers.Add()([pool_5_up, pool_4_n])

    x1_up = layers.Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, name='x1_upsample2x',
                                   padding='SAME')(x1)

    pool_3_n = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, name='pool_3_1', padding='SAME')(pool_3.output)

    x = layers.Add()([x1_up, pool_3_n])

    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(16, 16), strides=8, name='x2_upsample2x',
                               padding='SAME')(x)

    return models.Model(inputs=inputs, outputs=x, name='FCN_8_original')

# FCN-8
def FCN_8_original_filters_mod(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs

    pool_5_up = layers.Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=2, name='pool_5_upsample2x',
                                       padding='same')(pool_5.output)

    pool_4_n = layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, name='pool_4_1', padding='same')(pool_4.output)

    x1 = layers.Add()([pool_5_up, pool_4_n])

    x1_up = layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=2, name='x1_upsample2x',
                                   padding='same')(x1)

    pool_3_n = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, name='pool_3_1', padding='same')(pool_3.output)

    x = layers.Add()([x1_up, pool_3_n])

    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(16, 16), strides=8, name='x2_upsample2x',
                               padding='same')(x)

    return models.Model(inputs=inputs, outputs=x, name='FCN_8_original')

# FCN-8
def FCN_8_original_end_upsample(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs

    pool_5_up = layers.Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=2, name='pool_5_upsample2x',
                                       padding='same')(pool_5.output)

    pool_4_n = layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, name='pool_4_1', padding='same')(pool_4.output)

    x1 = layers.Add()([pool_5_up, pool_4_n])

    x1_up = layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=2, name='x1_upsample2x',
                                   padding='same')(x1)

    pool_3_n = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, name='pool_3_1', padding='same')(pool_3.output)

    x = layers.Add()([x1_up, pool_3_n])

    x = layers.UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear', name='upsample_last_conv')(x)
    x = layers.Conv2D(filters=n_classes, kernel_size=(16, 16), name='last_conv', padding='same')(x)

    return models.Model(inputs=inputs, outputs=x, name='FCN_8_end_upsample')


# FCN-8
def FCN_8_deconv_deep(backbone, n_classes, retrain=False):

    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs

    pool_5_up = layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=2, name='pool_5_upsample2x',padding='same')(pool_5.output)
    pool_4_conv = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, name='pool_4_conv')(pool_4.output)
    x1 = layers.Add()([pool_5_up, pool_4_conv])

    x1_up = layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=2, name='x1_upsample2x',padding='same')(x1)
    pool_3_conv = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, name='pool_3_conv')(pool_3.output)
    x = layers.Add()([x1_up, pool_3_conv])

    x = layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=2, name='x2_upsample2x',padding='same')(x)
    pool_2_conv = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, name='pool_2_conv')(pool_2.output)
    x = layers.Add()([x, pool_2_conv])

    x = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=2, name='x3_upsample2x',padding='same')(x)
    pool_1_conv = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, name='pool_1_conv')(pool_1.output)
    x = layers.Add()([x, pool_1_conv])

    x = layers.Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, name='transposed_1',padding='same')(x)

    return models.Model(inputs=inputs, outputs=x, name='FCN_8_deconv_deep')


# FCN-8
def FCN_8_Upsampling(backbone, n_classes, retrain=False):
    
    for layer in backbone.layers:
        layer.trainable = retrain

    if(backbone.name=='resnet50'):
        pool_1 = backbone.get_layer('conv1_relu')
        pool_2 = backbone.get_layer('conv2_block3_out')
        pool_3 = backbone.get_layer('conv3_block4_out')
        pool_4 = backbone.get_layer('conv4_block6_out')
        pool_5 = backbone.get_layer('conv5_block3_out')

    inputs = backbone.inputs

    pool_5_up = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear', name='pool5_upsample_2x')(pool_5.output)
    pool_5_up = layers.Conv2D(filters=n_classes, kernel_size=(4, 4), name='pool5_conv_2x',padding='same')(pool_5_up)

    pool_4_n = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, name='pool_4_1')(pool_4.output)
    x1 = layers.Add()([pool_5_up, pool_4_n])

    x1_up = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear', name='x1_upsample_2x')(x1)
    x1_up = layers.Conv2D(filters=n_classes, kernel_size=(4, 4), name='x1_conv_2x',padding='same')(x1_up)
    pool_3_n = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, name='pool_3_1')(pool_3.output)

    x = layers.Add()([x1_up, pool_3_n])

    x = layers.UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear', name='upsample_last_conv')(x)
    x = layers.Conv2D(filters=n_classes, kernel_size=(16, 16), name='last_conv',padding='same')(x)

    return models.Model(inputs=inputs, outputs=x, name='FCN_8_Upsampling')

#---------------------------------------------------------DEPRECATED------------------------------------------------------------------------
