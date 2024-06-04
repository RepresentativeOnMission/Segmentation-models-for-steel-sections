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
import tensorflow.keras.backend as K

def softmax_cross_entropy_loss(y_true, y_pred):
    #y_true = tf.cast(y_true, dtype="float16")#tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
def weighted_cross_entropy_loss(y_true, y_pred):
    #y_true = tf.cast(y_true, dtype="float16")# tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) * tf.reduce_sum(y_true * [1.,1.8,1.5,1.], axis=-1)
    return loss

def dice_loss(y_true, y_pred):
    inputs = K.flatten(tf.nn.softmax(y_pred))
    targets = K.flatten(y_true)
    intersection = K.sum(targets*inputs)
    dice_loss = 1 - (2*intersection + 1e-6) / (K.sum(targets) + K.sum(inputs) + 1e-6)
    
def weighted_cross_entropy_dice_loss(y_true, y_pred):
    #y_true = tf.cast(y_true, dtype="float16")# tf.float32)
    inputs = K.flatten(tf.nn.softmax(y_pred))
    targets = K.flatten(y_true)
    w_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) * tf.reduce_sum(y_true * [1.,1.8,1.5,1.], axis=-1)
    intersection = K.sum(targets*inputs)
    dice_loss = 1 - (2*intersection + 1e-6) / (K.sum(targets) + K.sum(inputs) + 1e-6)
    return 0.7*w_cce+ 0.3*dice_loss

   