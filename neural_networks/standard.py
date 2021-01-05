'''
Neural networks for standard systems
'''

import numpy as np
import collections
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import linalg as tfl
from tools import vae_utils
from neural_networks import layers
from neural_networks import NN_utils
    
class Classifier(object):
    '''
    Class for simple classifier, taking  input x and outputting 
    class probability p(y|x)
    '''
    
    def __init__(self, name, n_x, n_y, N_h):
        '''
        Initialisation
        INPUTS:
        name - name to assign to the decoder
        n_x - dimensionality of the input
        n_z - dimensionality of latent space
        N_h - array of hidden units' dimensionalities in the format [Nhx,Nh1,Nh2,...,Nhn]
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights
        
        # Choice of non-linearity (tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
        self.nonlinearity = tf.nn.leaky_relu
        
    def compute_py(self,x):
        '''
        compute probability for each class
        INPUTS:
        x - input
        OUTPUTS:
        py - histogram of probabilities for each class
        '''
        
        hidden1_pre = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1']), self.weights['b_x_to_h1'])
        hidden_post = self.nonlinearity(hidden1_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        p_un = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_py'.format(ni)]), self.weights['b_h{}_to_py'.format(ni)])
        p_un = tf.nn.sigmoid(p_un) + 1e-6
        py = tfm.divide(p_un,tf.tile(tf.expand_dims(tfm.reduce_sum(p_un,axis=1),axis=1),[1,self.n_y]))
        
        return py

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()

        all_weights['W_x_to_h1'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_h[0]), dtype=tf.float32)
        all_weights['b_x_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-2], self.N_h[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_h[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_py'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_y), dtype=tf.float32)
        all_weights['b_h{}_to_py'.format(ni)] = tf.Variable(tf.zeros([self.n_y], dtype=tf.float32) * self.bias_start)
        
        return all_weights
    
class Gaussian_NN(object):
    '''
    Class for Gaussian neural network, taking input x and outputting 
    Gaussian distribution p(y|x)
    '''
    
    def __init__(self, name, n_x, n_y, N_h):
        '''
        Initialisation
        INPUTS:
        name - name to assign to the decoder
        n_x - dimensionality of the input
        n_y - dimensionality of output
        N_h - array of hidden units' dimensionalities in the format [Nhx,Nh1,Nh2,...,Nhn]
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights
        
        # Choice of non-linearity (tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
        self.nonlinearity = tf.nn.leaky_relu
        
    def compute_moments(self,x):
        '''
        compute moments of output Gaussian distribution
        INPUTS:
        x -  input
        OUTPUTS:
        mu_y - mean of output Gaussian distribution
        log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        hidden1_pre = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1']), self.weights['b_x_to_h1'])
        hidden_post = self.nonlinearity(hidden1_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muy'.format(ni)]), self.weights['b_h{}_to_muy'.format(ni)])
        mu_y = tf.nn.sigmoid(mu_y)
        log_sig_sq_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sy'.format(ni)]), self.weights['b_h{}_to_sy'.format(ni)])
        log_sig_sq_y = 100*(tf.nn.sigmoid(log_sig_sq_y/100)-0.5)

        return mu_y, log_sig_sq_y
    
    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()

        all_weights['W_x_to_h1'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_h[0]), dtype=tf.float32)
        all_weights['b_x_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-2], self.N_h[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_h[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_muy'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_y), dtype=tf.float32)
        all_weights['b_h{}_to_muy'.format(ni)] = tf.Variable(tf.zeros([self.n_y], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_sy'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_y), dtype=tf.float32)
        all_weights['b_h{}_to_sy'.format(ni)] = tf.Variable(tf.zeros([self.n_y], dtype=tf.float32) * self.bias_start)
        
        return all_weights
    
class Gaussian_CNN_2D(object):
    '''
    Class for Gaussian convolutionalneural network, taking input x and 
    outputting Gaussian distribution p(y|x)
    '''
    
    def __init__(self, name, n_x, n_y, N_h1, NF_h, N_h2, St1, St2, sz_f, sz_im):
        '''
        Initialisation
        INPUTS:
        name - name to assign to the decoder
        n_x - channels of the input
        n_y - channels of output
        N_h - array of number of channels in the hidden units [Nhx,Nh1,Nh2,...,Nhn]
        St - array of strides to use every operation (must be one longer then the above)
        sz_f - filters sizes in the format [H,W]
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.NF_h = NF_h
        self.Sz1 = NN_utils.compute_size(sz_im,St1)
        self.Sz2 = NN_utils.compute_size(self.Sz1[-1],St2)
        self.St = np.concatenate((St1,np.ones(np.shape(NF_h)[0]),St2),0)
        self.sz_f = sz_f
        self.sz_im = sz_im
        self.name = name
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights
        
        # Choice of non-linearity (tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
        self.nonlinearity = tf.nn.leaky_relu
        
    def compute_moments(self,xl):
        '''
        compute moments of output Gaussian distribution
        INPUTS:
        x -  input
        OUTPUTS:
        mu_y - mean of output Gaussian distribution
        log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        x, l = NN_utils.reshape_and_extract(xl,self.sz_im)
        
        hidden_post = layers.tf_conv_layer(x,self.weights['W_x_to_h1'],self.weights['b_x_to_h1'],self.St[0],self.nonlinearity)
        # print(tf.shape(hidden_post).numpy())
        
        num_layers_1 = np.shape(self.N_h1)[0]-1
        
        for i in range(num_layers_1):
            ni = i+2
        
            hidden_post = layers.tf_conv_layer(hidden_post,self.weights['W_h{}_to_h{}'.format(ni-1,ni)],self.weights['b_h{}_to_h{}'.format(ni-1,ni)],self.St[ni-1],self.nonlinearity)
            # print(tf.shape(hidden_post).numpy())
            
        hidden_post = NN_utils.flatten(hidden_post)
        hidden_post = tf.concat([hidden_post,l],axis=1)
        # print(tf.shape(hidden_post).numpy())
        
        num_layers_F = np.shape(self.NF_h)[0]
        
        for i in range(num_layers_F):
            ni = ni+1
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
            # print(tf.shape(hidden_post).numpy())
        
        hidden_post = NN_utils.reshape_to_images(hidden_post,self.Sz2[0,:])
        # print(tf.shape(hidden_post).numpy())
        
        num_layers_2 = np.shape(self.N_h2)[0]
        
        for i in range(num_layers_2):
            ni = ni+1
        
            hidden_post = layers.tf_conv_layer(hidden_post,self.weights['W_h{}_to_h{}'.format(ni-1,ni)],self.weights['b_h{}_to_h{}'.format(ni-1,ni)],self.St[ni-1],self.nonlinearity)
            # print(tf.shape(hidden_post).numpy())
        
        mu_y = layers.tf_conv_layer(hidden_post,self.weights['W_h{}_to_muy'.format(ni)],self.weights['b_h{}_to_muy'.format(ni)],1,self.nonlinearity)
        mu_y = tf.nn.sigmoid(mu_y)
        
        log_sig_sq_y = layers.tf_conv_layer(hidden_post,self.weights['W_h{}_to_sy'.format(ni)],self.weights['b_h{}_to_sy'.format(ni)],1,self.nonlinearity)
        log_sig_sq_y = 100*(tf.nn.sigmoid(log_sig_sq_y/100)-0.5)

        mu_y = NN_utils.flatten(mu_y)
        mu_y = tf.concat([mu_y,tf.zeros([tf.shape(mu_y)[0],1])],axis=1)
        log_sig_sq_y = NN_utils.flatten(log_sig_sq_y)
        log_sig_sq_y = tf.concat([log_sig_sq_y,tf.zeros([tf.shape(log_sig_sq_y)[0],1])],axis=1)

        return mu_y, log_sig_sq_y
    
    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        SZ1 = self.N_h1*self.Sz1[:,0]*self.Sz1[:,1]
        SZ1[-1] = SZ1[-1]+2
        SZF = self.NF_h
        SZ2 = self.N_h2*self.Sz2[:,0]*self.Sz2[:,1]
        SZ = np.concatenate((SZ1,SZF,SZ2),axis=0)
        
        nf = np.asarray(self.NF_h)
        nfi = np.rint(self.NF_h[-1]/(self.Sz2[0,0]*self.Sz2[0,1]))
        nfi = nfi.astype(int)
        nf[-1] = nfi
        N_h = tf.concat([self.N_h1,nf,self.N_h2],0)
        
        # print(N_h.numpy())
        # print(SZ)
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_x_to_h1'] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1],self.n_x,N_h[0]]), dtype=tf.float32)
        all_weights['b_x_to_h1'] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1],self.n_x,N_h[0]], dtype=tf.float32))
        
        num_layers_1 = np.shape(self.N_h1)[0]-1
        
        for i in range(num_layers_1):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1], N_h[ni-2], N_h[ni-1]]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1],N_h[ni-2], N_h[ni-1]], dtype=tf.float32))
            
        num_layers_F = np.shape(self.NF_h)[0]
        
        for i in range(num_layers_F):
            ni = ni+1
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(SZ[ni-2], SZ[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([SZ[ni-1]], dtype=tf.float32) * self.bias_start)
           
        num_layers_2 = np.shape(self.N_h2)[0]
            
        for i in range(num_layers_2):
            ni = ni+1
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1], N_h[ni-2], N_h[ni-1]]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1],N_h[ni-2], N_h[ni-1]], dtype=tf.float32))
        
        all_weights['W_h{}_to_muy'.format(ni)] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1], N_h[ni-1], self.n_y]), dtype=tf.float32)
        all_weights['b_h{}_to_muy'.format(ni)] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1], N_h[ni-1], self.n_y], dtype=tf.float32))
        
        all_weights['W_h{}_to_sy'.format(ni)] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1], N_h[ni-1], self.n_y]), dtype=tf.float32)
        all_weights['b_h{}_to_sy'.format(ni)] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1], N_h[ni-1], self.n_y], dtype=tf.float32))
        
        return all_weights

class Classification_CNN_2D(object):
    '''
    Class for Gaussian convolutionalneural network, taking input x and 
    outputting Gaussian distribution p(y|x)
    '''
    
    def __init__(self, name, n_x, n_y, N_h1, NF_h, St1, sz_f, sz_im):
        '''
        Initialisation
        INPUTS:
        name - name to assign to the decoder
        n_x - channels of the input
        n_y - channels of output
        N_h - array of number of channels in the hidden units [Nhx,Nh1,Nh2,...,Nhn]
        St - array of strides to use every operation (must be one longer then the above)
        sz_f - filters sizes in the format [H,W]
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.N_h1 = N_h1
        self.NF_h = NF_h
        self.Sz1 = NN_utils.compute_size(sz_im,St1)
        self.St = np.concatenate((St1,np.ones(np.shape(NF_h)[0])),0)
        self.sz_f = sz_f
        self.sz_im = sz_im
        self.name = name
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights
        
        # Choice of non-linearity (tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
        self.nonlinearity = tf.nn.leaky_relu
        
    def compute_py(self,xl):
        '''
        compute moments of output Gaussian distribution
        INPUTS:
        x -  input
        OUTPUTS:
        mu_y - mean of output Gaussian distribution
        log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        x, _ = NN_utils.reshape_and_extract(xl,self.sz_im)
        
        hidden_post = layers.tf_conv_layer(x,self.weights['W_x_to_h1'],self.weights['b_x_to_h1'],self.St[0],self.nonlinearity)
        # print(tf.shape(hidden_post).numpy())
        
        num_layers_1 = np.shape(self.N_h1)[0]-1
        
        for i in range(num_layers_1):
            ni = i+2
        
            hidden_post = layers.tf_conv_layer(hidden_post,self.weights['W_h{}_to_h{}'.format(ni-1,ni)],self.weights['b_h{}_to_h{}'.format(ni-1,ni)],self.St[ni-1],self.nonlinearity)
            # print(tf.shape(hidden_post).numpy())
            
        hidden_post = NN_utils.flatten(hidden_post)
        # print(tf.shape(hidden_post).numpy())
        
        num_layers_F = np.shape(self.NF_h)[0]
        
        for i in range(num_layers_F):
            ni = ni+1
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
            # print(tf.shape(hidden_post).numpy())
        
        p_un = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_py'.format(ni)]), self.weights['b_h{}_to_py'.format(ni)])
        p_un = tf.nn.sigmoid(p_un) + 1e-6
        py = tfm.divide(p_un,tf.tile(tf.expand_dims(tfm.reduce_sum(p_un,axis=1),axis=1),[1,self.n_y]))
        
        return py
    
    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        SZ1 = self.N_h1*self.Sz1[:,0]*self.Sz1[:,1]
        SZF = self.NF_h
        SZ = np.concatenate((SZ1,SZF),axis=0)
        
        N_h = tf.concat([self.N_h1,self.NF_h],0)
        
        # print(N_h.numpy())
        # print(SZ)
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_x_to_h1'] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1],self.n_x,N_h[0]]), dtype=tf.float32)
        all_weights['b_x_to_h1'] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1],self.n_x,N_h[0]], dtype=tf.float32))
        
        num_layers_1 = np.shape(self.N_h1)[0]-1
        
        for i in range(num_layers_1):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.random.uniform([self.sz_f[0],self.sz_f[1], N_h[ni-2], N_h[ni-1]]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.sz_f[0],self.sz_f[1],N_h[ni-2], N_h[ni-1]], dtype=tf.float32))
            
        num_layers_F = np.shape(self.NF_h)[0]
        
        for i in range(num_layers_F):
            ni = ni+1
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(SZ[ni-2], SZ[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([SZ[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_py'.format(ni)] = tf.Variable(vae_utils.xavier_init(SZ[ni-1], self.n_y), dtype=tf.float32)
        all_weights['b_h{}_to_py'.format(ni)] = tf.Variable(tf.zeros([self.n_y], dtype=tf.float32) * self.bias_start)
        
        return all_weights

    