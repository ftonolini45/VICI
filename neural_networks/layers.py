'''
Functions to implement layers
'''

import numpy as np
import collections
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import linalg as tfl
from tools import vae_utils

def tf_conv_layer(x_in,W,b,st,nonlinearity):
    '''
    Convolutional Layer
    INPUTS:
    x_in - input in the format [batch,H,W,channels]
    W - linear weights, in the format [H,W,channels_in,channels_out]
    b - additive weights, in the format [H,W,channels_in,channels_out]
    st - strieds (single number for symmetric strides)
    nonlinearity - the non-linearity to be used, e.g. tf.nn.sigmoid
    OUTPUTS:
    y_out - output
    '''
    
    x = x_in
    st_inv = tf.divide(1.0,st)
    szf0 = tf.shape(W)[0]
    szf1 = tf.shape(W)[1]
    nh_out = tf.shape(W)[3]
    height = tf.cast(tf.shape(x)[1],tf.float32)
    width = tf.cast(tf.shape(x)[2],tf.float32)
    if st<1.0:
        B = tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(tf.ones([tf.shape(x)[0],tf.cast(height*st_inv,tf.int32),tf.cast(width*st_inv,tf.int32),tf.shape(x)[3]]),b, strides=[1,1,1,1],padding='SAME')
        hidden1_pre = tf.add(tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(tf.image.resize(x,[tf.cast(height*st_inv,tf.int32),tf.cast(width*st_inv,tf.int32)]),W, strides=[1,1,1,1],padding='SAME'),0.0*B)
        y_out = nonlinearity(hidden1_pre)
    else:
        B = tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(tf.ones([tf.shape(x)[0],tf.cast(height,tf.int32),tf.cast(width,tf.int32),tf.shape(x)[3]]),b, strides=[1,1,1,1],padding='SAME')
        hidden1_pre = tf.add(tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME'),0.0*B)
        y_out = nonlinearity(hidden1_pre)
        y_out = tf.image.resize(y_out,[tf.cast(height*st_inv,tf.int32),tf.cast(width*st_inv,tf.int32)])
    
    return y_out