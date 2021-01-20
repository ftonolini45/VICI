'''
Functions to implement layers
'''

import tensorflow as tf
import numpy as np
from neural_networks import NN_utils as vae_utils

def reshape_2D(x,im_sz=None):
    
    if im_sz:
        n_ch = tf.cast(tf.divide(tf.shape(x)[1],im_sz[0]*im_sz[1]),tf.float32)
    else:
        im_sz = tf.cast(tf.sqrt(tf.shape(x)[1]),tf.float32)
        n_ch = 1
        
    x_new = tf.reshape(x,[tf.shape(x)[0],im_sz[0],im_sz[1],n_ch])
    
    return x_new

def flatten_from_2D(X):
    
    x = tf.reshape(X,[tf.shape(X)[0],tf.shape(X)[1]*tf.shape(X)[2]*tf.shape(X)[3]])
    
    return x

def downsample(x,d=2):
    
    height = tf.cast(tf.shape(x)[1],tf.float32)
    width = tf.cast(tf.shape(x)[2],tf.float32)
    y = tf.image.resize(x,[tf.cast(height/d,tf.int32),tf.cast(width/d,tf.int32)])
    
    return y

def upsample(x,d=2):
    
    height = tf.cast(tf.shape(x)[1],tf.float32)
    width = tf.cast(tf.shape(x)[2],tf.float32)
    y = tf.image.resize(x,[tf.cast(height*d,tf.int32),tf.cast(width*d,tf.int32)])
    
    return y

def xavier_init(dims):
    
    tf.random.set_seed(np.random.randint(low=0, high=1000))
    
    N_in = dims[0]*dims[1]*dims[2]
    N_out = dims[3]
    N_avg = (N_in + N_out)/2
    var = tf.cast(1/N_avg,tf.float32)
    w0 = var*tf.random.uniform(dims)
    
    return w0

def tf_fc_layer(x_in,W,b,nonlinearity):
    
    y_out = tf.add(tf.matmul(x_in, W), b)
    if nonlinearity:
        y_out = nonlinearity(y_out)
    
    return y_out

def tf_fc_weights_W(n_h_in,n_h_out,name=None):
    
    w = tf.Variable(vae_utils.xavier_init(n_h_in,n_h_out), dtype=tf.float32)

    return w

def tf_fc_weights_b(n_h_out,name=None):

    b = tf.Variable(tf.zeros(n_h_out, dtype=tf.float32))

    return b

def tf_conv2D_layer(x_in,W,st,nonlinearity):
    '''
    Convolutional Layer
    INPUTS:
    x_in - input in the format [batch,H,W,channels]
    W - linear weights, in the format [H,W,channels_in,channels_out]
    st - strieds for convolutional layer st>1.0 will lead to a down-sampling, while st<1.0 will lead to an up-sampling
    nonlinearity - the non-linearity to be used, e.g. tf.nn.sigmoid
    OUTPUTS:
    y_out - output
    '''
    
    st_inv = tf.divide(1.0,st)
    szf0 = tf.shape(W)[0]
    szf1 = tf.shape(W)[1]
    nh_out = tf.shape(W)[3]
    
    if st<1.0:
        hidden1_pre = tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(upsample(x_in,st_inv),W, strides=[1,1,1,1],padding='SAME')
        y_out = hidden1_pre
        if nonlinearity:
            y_out = nonlinearity(y_out)
    else:
        hidden1_pre = tf.cast((1/(szf0*szf1*nh_out)),tf.float32)*tf.nn.conv2d(x_in,W, strides=[1,1,1,1],padding='SAME')
        if nonlinearity:
            y_out = nonlinearity(hidden1_pre)
        y_out = downsample(hidden1_pre,d=st)
    
    return y_out

def tf_conv2D_weights(szf,ch_in,ch_out,name=None):
    
    W = tf.Variable(xavier_init([szf,szf,ch_in,ch_out]), dtype=tf.float32, name=name)
    
    return W

