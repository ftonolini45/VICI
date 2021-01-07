import tensorflow as tf
from neural_networks import layers
import numpy as np
import collections

def fc_network(x_in, W, nonlinearity, add_b=True, ID=0):
    '''
    run fully connected network
    INPUTS:
        x_in - input in the format [n_samples, dimensions]
        W - dictionary containing linear and additive weights (created with fc_make_weights)
        nonlinearity - non-linearity to be used, e.g. tf.nn.relu
    OPTIONAL INPUTS:
        add_b - whether to include additive weights b (True/False)
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        y_out - output of the network in the format [n_samples, dimensions]
    '''
    
    if add_b==True:
        num_layers_1 = tf.cast(len(W)/2,tf.int32)
    else:
        num_layers_1 = len(W)
    
    for i in range(num_layers_1):
        ni = i+1
    
        if add_b==True: 
            x_in = layers.tf_fc_layer(x_in,W['W_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],W['b_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],nonlinearity)
        else:
            x_in = tf.matmul(x_in, W['W_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)])
    
    y_out = x_in
    
    return y_out

def fc_make_weights(W_dict,n_in,N, add_b=True, ID=0):
    '''
    make weights for fully connected network
    INPUTS:
        W_dict - dictionary of weights to append the newly created ones to (if this is the first network, initialise it with 'w_dict={}')
        n_in - dimensionality of the input
        N - list or array of layers' dimensionalities
    OPTIONAL INPUTS:
        add_b - whether to include additive weights b (True/False)
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        W_dict - the weights' dictionary with the newly created weights appended to it
    '''
    
    num_layers_1 = tf.shape(N)[0]
    N = tf.concat([n_in,N],axis=0)
    
    for i in range(num_layers_1):
        ni = i+1
        
        W_dict['W_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_fc_weights_W(N[ni-1],N[ni])
        if add_b==True: 
            W_dict['b_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_fc_weights_b(N[ni-1],N[ni])
            
    return W_dict

def conv_1D_network(x_in, W, st, nonlinearity, sig_size=None, reshape_in=False, reshape_out=False, ID=0):
    '''
    run 1D convolutional network
    INPUTS:
        x_in - input in the format [n_samples, dimensions], if "reshape_in=True" or [n_samples, length, channels] if "reshape_in=False"
        W - dictionary containing convolutional weights (created with conv_1D_make_weights)
        st - list or array of strides for each layer
        nonlinearity - non-linearity to be used, e.g. tf.nn.relu
    OPTIONAL INPUTS:
        sig_size - length of the 1D signals. by default it is assumed length=dimensions
        reshape_in - whether to reshape the input from [n_samples, dimensions] to [n_samples, length, channels] (True/False)
        reshape_out - whether to reshape the output from [n_samples, length, channels] to [n_samples, dimensions] (True/False)
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        y_out - output of the network in the format [n_samples, dimensions] if "reshape_out=True" or [n_samples, length, channels] if "reshape_out=False"
    '''
    
    if reshape_in==True:
        x_in = layers.reshape_1D(x_in,sig_sz=sig_size)
    
    num_layers_1 = tf.shape(st)[0]
    
    for i in range(num_layers_1):
        ni = i+1
    
        x_in = layers.tf_conv1D_layer(x_in,W['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],st[ni-1],nonlinearity)

    if reshape_out==True:
        y_out = layers.flatten_from_1D(x_in)
    else:
        y_out = x_in
    
    return y_out

def conv_1D_make_weights(W_dict,ch_in,N_conv,fs, ID=0):
    '''
    make weights for 1D convolutional network
    INPUTS:
        W_dict - dictionary of weights to append the newly created ones to (if this is the first network, initialise it with 'w_dict={}')
        ch_in - number of channels of the input
        N_conv - list or array of numbers of channels of each layer
        fs - list or array of filter sizes for each layer
    OPTIONAL INPUTS:
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        W_dict - the weights' dictionary with the newly created weights appended to it
    '''
    
    num_layers_1 = tf.shape(N_conv)[0]
    N_conv = tf.concat([tf.expand_dims(ch_in,axis=0),N_conv],axis=0)
    
    for i in range(num_layers_1):
        
        ni = i+1
        W_dict['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_conv1D_weights(fs[ni-1],N_conv[ni-1],N_conv[ni], name='W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID))
        
    return W_dict

def conv_2D_network(x_in, W, st, nonlinearity, im_size=None, reshape_in=False, reshape_out=False, ID=0):
    '''
    run 2D convolutional network
    INPUTS:
        x_in - input in the format [n_samples, dimensions], if "reshape_in=True" or [n_samples, height, width, channels] if "reshape_in=False"
        W - dictionary containing convolutional weights (created with conv_1D_make_weights)
        st - list or array of strides for each layer
        nonlinearity - non-linearity to be used, e.g. tf.nn.relu
    OPTIONAL INPUTS:
        im_size - size of the input images [height, width]. by default it is assumed height=width=sqrt(dimensions)
        reshape_in - whether to reshape the input from [n_samples, dimensions] to [n_samples, height, width, channels] (True/False)
        reshape_out - whether to reshape the output from [n_samples, height, width, channels] to [n_samples, dimensions] (True/False)
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        y_out - output of the network in the format [n_samples, dimensions] if "reshape_out=True" or [n_samples, height, width, channels] if "reshape_out=False"
    '''
    
    if reshape_in==True:
        x_in = layers.reshape_2D(x_in,im_sz=im_size)
    
    num_layers_1 = tf.shape(st)[0]
    
    for i in range(num_layers_1):
        ni = i+1
    
        x_in = layers.tf_conv2D_layer(x_in,W['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],st[ni-1],nonlinearity)

    if reshape_out==True:
        y_out = layers.flatten_from_2D(x_in)
    else:
        y_out = x_in
    
    return y_out

def conv_2D_make_weights(W_dict,ch_in,N_conv,fs, ID=0):
    '''
    make weights for 2D convolutional network
    INPUTS:
        W_dict - dictionary of weights to append the newly created ones to (if this is the first network, initialise it with 'w_dict={}')
        ch_in - number of channels of the input
        N_conv - list or array of numbers of channels of each layer
        fs - list or array of filter sizes for each layer (1D array, filter size is assumed as symmetric)
    OPTIONAL INPUTS:
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        W_dict - the weights' dictionary with the newly created weights appended to it
    '''
    
    num_layers_1 = tf.shape(N_conv)[0]
    N_conv = tf.concat([tf.expand_dims(ch_in,axis=0),N_conv],axis=0)
    
    for i in range(num_layers_1):
        
        ni = i+1
        W_dict['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_conv2D_weights(fs[ni-1],N_conv[ni-1],N_conv[ni], name='W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID))
        
    return W_dict

def conv_2D_unet(x_in, W, st, nonlinearity, im_size=None, reshape_in=False, reshape_out=False, ID=0):
    '''
    run 2D convolutional Unet network
    INPUTS:
        x_in - input in the format [n_samples, dimensions], if "reshape_in=True" or [n_samples, height, width, channels] if "reshape_in=False"
        W - dictionary containing convolutional weights (created with conv_1D_make_weights)
        st - list or array of strides for each layer
        nonlinearity - non-linearity to be used, e.g. tf.nn.relu
    OPTIONAL INPUTS:
        im_size - size of the input images [height, width]. by default it is assumed height=width=sqrt(dimensions)
        reshape_in - whether to reshape the input from [n_samples, dimensions] to [n_samples, height, width, channels] (True/False)
        reshape_out - whether to reshape the output from [n_samples, height, width, channels] to [n_samples, dimensions] (True/False)
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        y_out - output of the network in the format [n_samples, dimensions] if "reshape_out=True" or [n_samples, height, width, channels] if "reshape_out=False"
    '''
    
    if reshape_in==True:
        x_in = layers.reshape_2D(x_in,im_sz=im_size)
    
    X = collections.OrderedDict()
    X['layer_{}'.format(0)] = x_in
    num_layers_1 = np.rint(np.round(np.shape(st)[0]/2)).astype(int)
    
    for i in range(num_layers_1):
        ni = i+1
    
        X['layer_{}'.format(ni)] = layers.tf_conv2D_layer(X['layer_{}'.format(ni-1)],W['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],st[ni-1],nonlinearity)

    num_layers_2 = np.rint(np.round(np.shape(st)[0]/2)).astype(int)
    
    for i in range(num_layers_2):
        ni = i + num_layers_1 + 1
        mi = num_layers_1 - i - 1
    
        stm = np.divide(np.prod(st[:ni]),np.prod(st[:mi]))
    
        x_d = layers.tf_conv2D_layer(X['layer_{}'.format(ni-1)],W['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)],st[ni-1],nonlinearity)
        x_r = layers.tf_conv2D_layer(X['layer_{}'.format(mi)],W['W_conv_h{}_to_h{}_ID{}'.format(mi,ni,ID)],stm,nonlinearity)
        X['layer_{}'.format(ni)] = x_d + x_r

    if reshape_out==True:
        y_out = layers.flatten_from_2D(X['layer_{}'.format(ni)])
    else:
        y_out = X['layer_{}'.format(ni)]
    
    return y_out

def conv_2D_unet_weights(W_dict,ch_in,N_conv_d,fs_d,fs_r,ID=0):
    '''
    make weights for 2D convolutional network
    INPUTS:
        W_dict - dictionary of weights to append the newly created ones to (if this is the first network, initialise it with 'w_dict={}')
        ch_in - number of channels of the input
        N_conv - list or array of numbers of channels of each layer
        fs_d - list or array of filter sizes for each layer through the direct channel (1D array, filter size is assumed as symmetric)
        fs_r - list or array of filter sizes for each recurrent layers (1D array, filter size is assumed as symmetric, must be half the size of fs_d)
    OPTIONAL INPUTS:
        ID - identifier number for the network (appended to the name of each weight)
    OUTPUTS:
        W_dict - the weights' dictionary with the newly created weights appended to it
    '''
    
    num_layers_1 = tf.cast(tf.shape(N_conv_d)[0]/2,tf.int32)
    N_conv_d = tf.concat([tf.expand_dims(ch_in,axis=0),N_conv_d],axis=0)
    
    for i in range(num_layers_1):
        
        ni = i+1
        W_dict['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_conv2D_weights(fs_d[ni-1],N_conv_d[ni-1],N_conv_d[ni], name='W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID))
        
    num_layers_2 = tf.cast(tf.shape(N_conv_d)[0]/2,tf.int32)
        
    for i in range(num_layers_2):
        ni = i + num_layers_1 + 1
        mi = num_layers_1 - i - 1
        
        W_dict['W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID)] = layers.tf_conv2D_weights(fs_d[ni-1],N_conv_d[ni-1],N_conv_d[ni], name='W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID))
        W_dict['W_conv_h{}_to_h{}_ID{}'.format(mi,ni,ID)] = layers.tf_conv2D_weights(fs_r[i],N_conv_d[mi],N_conv_d[ni], name='W_conv_h{}_to_h{}_ID{}'.format(ni-1,ni,ID))
    
    return W_dict