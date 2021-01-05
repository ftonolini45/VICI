import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

def reshape_and_extract(x,im_sz):
    
    xsz = tf.cast(tf.divide(tf.shape(x)[1],2),tf.int32)
    A = x[:,xsz:]
    x = x[:,:xsz]
    
    l = tf.expand_dims(x[:,-1],1)
    x = x[:,:-1]
    
    Al = tf.expand_dims(A[:,-1],1)
    A = A[:,:-1]
    
    x_new = tf.reshape(x,[tf.shape(x)[0],im_sz[0],im_sz[1],1])
    A_new = tf.reshape(A,[tf.shape(x)[0],im_sz[0],im_sz[1],1])
    x_new = tf.concat([x_new,A_new],3)
    
    l = tf.concat([l,Al],1)
    
    return x_new, l

def flatten(X):
    
    x = tf.reshape(X,[tf.shape(X)[0],tf.shape(X)[1]*tf.shape(X)[2]*tf.shape(X)[3]])
    
    return x

def reshape_to_images(x,im_sz):
    
    n_ch = tf.cast(tf.divide(tf.shape(x)[1],im_sz[0]*im_sz[1]),tf.int32)
    X = tf.reshape(x,[tf.shape(x)[0],im_sz[0],im_sz[1],n_ch])
    
    return X

def compute_size(im_sz,strides):
    
    im_sz = np.asarray(im_sz)
    l_sz = np.zeros((np.shape(strides)[0], np.shape(im_sz)[0]))
    
    fac = 1.0
    for i in range(np.shape(strides)[0]):
        
        fac = fac/strides[i]
        l_sz[i,:] = im_sz.astype(float)*fac
        
    l_sz = np.rint(l_sz).astype(int)
    
    
    return l_sz

def xavier_init(fan_in, fan_out, constant = 1):

    low = -constant * tfm.sqrt(6.0 / (tf.cast(fan_in,dtype=tf.float32) + tf.cast(fan_out,dtype=tf.float32)))
    high = constant * tfm.sqrt(6.0 / (tf.cast(fan_in,dtype=tf.float32) + tf.cast(fan_out,dtype=tf.float32)))
#    low = -constant * np.sqrt(6.0 / (np.float32(fan_in) + np.float32(fan_out)))
#    high = constant * np.sqrt(6.0 / (np.float32(fan_in) + np.float32(fan_out)))
#    low = -constant * np.sqrt(6.0 / (fan_in +fan_out))
#    high = constant * np.sqrt(6.0 / (fan_in +fan_out))
    return tf.random.uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)
