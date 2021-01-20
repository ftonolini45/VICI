'''
Functions to implement layers
'''

import tensorflow as tf
from neural_networks import NN_utils
import numpy as np

def reshape_1D(x,sig_sz=None):
    '''
    Reshape data array from [number_of_samples,dimensions] to [number_of_samples,signal_length,channels]
    INPUTS:
        x - input data array in the format [number_of_samples,dimensions].
    OPTIONAL INPUTS:
        sig_sz - length of the 1D signals. By default it is assumed that there is one channel, so sig_sz=dimensions.
    OUTPUTS:
        x_new - reshaped array of data in format [number_of_samples,signal_length,channels].
    '''
    
    if sig_sz:
        n_ch = tf.cast(tf.divide(tf.shape(x)[1],sig_sz),tf.float32)
    else:
        sig_sz = tf.shape(x)[1]
        n_ch = 1
        
    x_new = tf.reshape(x,[tf.shape(x)[0],sig_sz,n_ch])
    
    return x_new

def flatten_from_1D(X):
    '''
    Reshape data array from [number_of_samples,signal_length,channels] to [number_of_samples,dimensions]
    INPUTS:
        X - input data array in the format [number_of_samples,signal_length,channels].
    OUTPUTS:
        x - reshaped array of data in format [number_of_samples,dimensions].
    '''
    
    x = tf.reshape(X,[tf.shape(X)[0],tf.shape(X)[1]*tf.shape(X)[2]])
    
    return x

def reshape_2D(x,im_sz=None):
    '''
    Reshape data array from [number_of_samples,dimensions] to [number_of_samples,height,width,channels]
    INPUTS:
        x - input data array in the format [number_of_samples,dimensions].
    OPTIONAL INPUTS:
        im_sz - [height,width] of the images. By default it is assumed that there is one channel and images are square, so sig_sz=[sqrt(dimensions),sqrt(dimensions)].
    OUTPUTS:
        x_new - reshaped array of data in format [number_of_samples,height,width,channels].
    '''
    
    if im_sz:
        n_ch = tf.cast(tf.divide(tf.shape(x)[1],im_sz[0]*im_sz[1]),tf.float32)
    else:
        im_sz = [tf.cast(tf.sqrt(tf.shape(x)[1]),tf.float32),tf.cast(tf.sqrt(tf.shape(x)[1]),tf.float32)]
        n_ch = 1
        
    x_new = tf.reshape(x,[tf.shape(x)[0],im_sz[0],im_sz[1],tf.cast(n_ch,tf.int32)])
    
    return x_new

def flatten_from_2D(X):
    '''
    Reshape data array from [number_of_samples,height,width,channels] to [number_of_samples,dimensions]
    INPUTS:
        X - input data array in the format [number_of_samples,height,width,channels].
    OUTPUTS:
        x - reshaped array of data in format [number_of_samples,dimensions].
    '''
    
    x = tf.reshape(X,[tf.shape(X)[0],tf.shape(X)[1]*tf.shape(X)[2]*tf.shape(X)[3]])
    
    return x

def downsample(x,d=2):
    '''
    Down-sample 2D images by d 
    INPUTS:
        x - input images in the format [number_of_samples,height,width,channels]
    OPTIONAL INPUTS:
        d - down-sampling ratio
    OUTPUTS:
        y - down-sampled images in the format [number_of_samples,height,width,channels]
    '''
    
    height = tf.cast(tf.shape(x)[1],tf.float32)
    width = tf.cast(tf.shape(x)[2],tf.float32)
    y = tf.image.resize(x,[tf.cast(height/d,tf.int32),tf.cast(width/d,tf.int32)])
    
    return y

def upsample(x,d=2):
    '''
    Up-sample 2D images by d 
    INPUTS:
        x - input images in the format [number_of_samples,height,width,channels]
    OPTIONAL INPUTS:
        d - up-sampling ratio
    OUTPUTS:
        y - up-sampled images in the format [number_of_samples,height,width,channels]
    '''
    
    height = tf.cast(tf.shape(x)[1],tf.float32)
    width = tf.cast(tf.shape(x)[2],tf.float32)
    y = tf.image.resize(x,[tf.cast(height*d,tf.int32),tf.cast(width*d,tf.int32)])
    
    return y

def downsample_1D(x,d=2):
    '''
    Down-sample 1D signals by d 
    INPUTS:
        x - input signals in the format [number_of_samples,length,channels]
    OPTIONAL INPUTS:
        d - down-sampling ratio
    OUTPUTS:
        y - down-sampled signals in the format [number_of_samples,length,channels]
    '''
    
    x_ext = tf.expand_dims(x,axis=2)
    length = tf.cast(tf.shape(x)[1],tf.float32)
    y_ext = tf.image.resize(x_ext,[tf.cast(length/d,tf.int32),1])
    y = tf.squeeze(y_ext,axis=2)
    
    return y

def upsample_1D(x,d=2):
    '''
    Up-sample 1D signals by d 
    INPUTS:
        x - input signals in the format [number_of_samples,length,channels]
    OPTIONAL INPUTS:
        d - up-sampling ratio
    OUTPUTS:
        y - up-sampled signals in the format [number_of_samples,length,channels]
    '''
    
    x_ext = tf.expand_dims(x,axis=2)
    length = tf.cast(tf.shape(x)[1],tf.float32)
    y_ext = tf.image.resize(x_ext,[tf.cast(length*d,tf.int32),1])
    y = tf.squeeze(y_ext,axis=2)
    
    return y

def xavier_init(dims):
    '''
    Xavier variable initialisation for convolutional filters
    INPUTS:
        dims - dimensions of the variable
    OUTPUTS:
        w0 - Xavier initialised variable
    '''
    N_in = 1
    for i in range(np.shape(dims)[0]-1):
        N_in = N_in*dims[i]
        
    N_out = dims[-1]
    N_avg = (N_in + N_out)/2
    var = tf.cast(1/N_avg,tf.float32)
    w0 = var*tf.random.uniform(dims)
    
    return w0

def tf_fc_layer(x_in,W,b,nonlinearity):
    '''
    run fully connected layer
    INPUTS:
        x_in - input in the format [n_samples, dimensions]
        W - linear weights
        b - addidive weights
        nonlinearity - non-linearity to be used, e.g. tf.nn.relu
    OUTPUTS:
        y_out - output of the layer in the format [n_samples, dimensions]
    '''
    
    y_out = tf.add(tf.matmul(x_in, W), b)
    if nonlinearity:
        y_out = nonlinearity(y_out)
    
    return y_out

def tf_fc_weights_W(n_h_in,n_h_out,name=None):
    '''
    initialise linear weights for fully connected layer
    INPUTS:
        n_h_in - number dimensions input
        n_h_out - number dimensions out
    OPTIONAL INPUT:
        name - name to assign the weights
    OUTPUTS:
        W - initialised linear weights in the format [n_h_in,n_h_out]
    '''
    
    w = tf.Variable(NN_utils.xavier_init(n_h_in,n_h_out), dtype=tf.float32)

    return w

def tf_fc_weights_b(n_h_out,name=None):
    '''
    initialise additive weights for fully connected layer
    INPUTS:
        n_h_out - number dimensions out
    OPTIONAL INPUT:
        name - name to assign the weights
    OUTPUTS:
        b - initialised additive weights in the format [n_h_out]
    '''
    
    b = tf.Variable(tf.zeros(n_h_out, dtype=tf.float32))

    return b

def tf_conv1D_layer(x_in,W,st,nonlinearity):
    '''
    Run 1D convolutional Layer
    INPUTS:
        x_in - input in the format [batch,length,channels]
        W - convolutional weights, in the format [length,channels_in,channels_out]
        st - strieds for convolutional layer st>1.0 will lead to a down-sampling, while st<1.0 will lead to an up-sampling
        nonlinearity - the non-linearity to be used, e.g. tf.nn.relu
    OUTPUTS:
        y_out - output in the format [batch,length,channels]
    '''
    
    st_inv = tf.divide(1.0,st)
    szf = tf.shape(W)[0]
    nh_out = tf.shape(W)[2]
    
    if st<1.0:
        hidden1_pre = tf.cast((1/(szf*nh_out)),tf.float32)*tf.nn.conv1d(upsample(x_in,st_inv),W, strides=[1,1,1],padding='SAME')
        y_out = hidden1_pre
        if nonlinearity:
            y_out = nonlinearity(y_out)
    else:
        hidden1_pre = tf.cast((1/(szf*nh_out)),tf.float32)*tf.nn.conv1d(x_in,W, strides=[1,1,1],padding='SAME')
        if nonlinearity:
            y_out = nonlinearity(hidden1_pre)
        y_out = downsample(hidden1_pre,d=st)
    
    return y_out

def tf_conv1D_weights(szf,ch_in,ch_out,name=None):
    '''
    initialise 1D convolutional weights
    INPUTS:
        szf - size of the filters
        ch_in - number of channels in
        ch_out - number of channels out
    OPTIONAL INPUT:
        name - name to assign the weights
    OUTPUTS:
        W - initialised convolutional weights in the format [szf,ch_in,ch_out]
    '''
    
    W = tf.Variable(xavier_init([szf,ch_in,ch_out]), dtype=tf.float32, name=name)
    
    return W

def tf_conv2D_layer(x_in,W,st,nonlinearity):
    '''
    Run 2D convolutional Layer
    INPUTS:
        x_in - input in the format [batch,height,width,channels]
        W - convolutional weights, in the format [height,width,channels_in,channels_out]
        st - strieds for convolutional layer st>1.0 will lead to a down-sampling, while st<1.0 will lead to an up-sampling
        nonlinearity - the non-linearity to be used, e.g. tf.nn.sigmoid
    OUTPUTS:
    y_out - output in the format [batch,height,width,channels]
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
    '''
    initialise 2D convolutional weights
    INPUTS:
        szf - size of the filters
        ch_in - number of channels in
        ch_out - number of channels out
    OPTIONAL INPUT:
        name - name to assign the weights
    OUTPUTS:
        W - initialised convolutional weights in the format [szf,szf,ch_in,ch_out]
    '''
    
    W = tf.Variable(xavier_init([szf,szf,ch_in,ch_out]), dtype=tf.float32, name=name)
    
    return W

