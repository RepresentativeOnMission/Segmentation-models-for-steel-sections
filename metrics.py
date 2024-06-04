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


#this class performs the IoU for a batch of size 1 x h x w x channels, it doesn't work if you input data that doesn't come from a generator with batch_size=1
def IoU_per_class_single(n_th_class, n_classes):
    def IoU(y_true, y_pred):
        #de one-hot-encoded input and mask
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        #calculate confusion matrix
        res_y_true = tf.reshape(y_true[0], [-1])
        res_y_pred = tf.reshape(y_pred[0], [-1])
        conf_matrix = tf.math.confusion_matrix(labels=res_y_true, predictions=res_y_pred, num_classes=n_classes, dtype=tf.dtypes.float32)
        #calculate the IoU per class
        true_positives = tf.linalg.diag_part(conf_matrix)
        #sum_over_row = FN + TP
        #sum_over_col = FP + TP
        sum_over_row = tf.reduce_sum(conf_matrix, axis=0)
        sum_over_col = tf.reduce_sum(conf_matrix, axis=1)
        epsilon = 0.0001
        #TP / (FN + TP + FP + TP - TP) = TP / (FN + FP + TP)
        return (true_positives / (sum_over_row + sum_over_col - true_positives + epsilon))[n_th_class]

    IoU.__name__ = 'IoU_class_' + str(n_th_class) + '_single'
    return IoU


#this class performs the IoU for a batch of size batch_size x h x w x channels, it doesn't work if you input data that doesn't come from a generator with batch_size=1
#this class performs IoU for a SINGLE BATCH, it is not CUMULATIVE for all batches in the generator
def IoU_mean_single(n_classes):
    def IoU(y_true, y_pred):
        #de one-hot-encoded input and mask
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        #calculate confusion matrix, by summing confusion matrices from every element of the batch
        res_y_true = tf.reshape(y_true[0], [-1])
        res_y_pred = tf.reshape(y_pred[0], [-1])
        conf_matrix = tf.math.confusion_matrix(labels=res_y_true, predictions=res_y_pred, num_classes=n_classes, dtype=tf.dtypes.float32)
        for i in range(1, tf.shape(y_true)[0]):
            res_y_true = tf.reshape(y_true[i], [-1])
            res_y_pred = tf.reshape(y_pred[i], [-1])
            conf_matrix = conf_matrix + tf.math.confusion_matrix(labels=res_y_true, predictions=res_y_pred, num_classes=n_classes, dtype=tf.dtypes.float32)
        #calculate the IoU per class
        true_positives = tf.linalg.diag_part(conf_matrix)
        #sum_over_row = FN + TP
        #sum_over_col = FP + TP
        sum_over_row = tf.reduce_sum(conf_matrix, axis=0)
        sum_over_col = tf.reduce_sum(conf_matrix, axis=1)
        epsilon = 0.0001
        #TP / (FN + TP + FP + TP - TP) = TP / (FN + FP + TP)
        a = tf.math.divide_no_nan(true_positives, (sum_over_row + sum_over_col - true_positives + epsilon))
        return tf.math.reduce_mean(a)

    IoU.__name__ = 'mean_IoU_single'
    return IoU



#makes cumulative mean IoU, given a generator
class Mean_IoU_cumulative(tf.keras.metrics.MeanIoU):
    def __init__(self, n_classes, name='IoU_mean_cumulative', dtype=None):
        super().__init__(n_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
    
#makes cumulative IoU_per_class given a generator
class IoU_per_class_cumulative(tf.keras.metrics.MeanIoU):
    
    def __init__(self, n_th_class, n_classes, name=None, dtype=None):
        super().__init__(n_classes, name)
        self.n_th_class = n_th_class

    def result(self):
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        true_positives = tf.linalg.diag_part(self.total_cm)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row[self.n_th_class] + sum_over_col[self.n_th_class] \
            - true_positives[self.n_th_class]

        iou = tf.math.divide_no_nan(true_positives[self.n_th_class], denominator)

        return iou

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
