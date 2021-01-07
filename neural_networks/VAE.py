'''
Neural networks for a VAE or conditional VAE
'''

import numpy as np
import collections
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import linalg as tfl
from neural_networks import NN_utils as vae_utils

class Decoder(object):
    '''
    Class for Gaussian decoder, taking latent variables z and outputting 
    Gaussian distribution p(x|z)
    '''
    
    def __init__(self, name, n_x, n_z, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the decoder
            n_x - dimensionality of the input
            n_z - dimensionality of latent space
            N_h - array of hidden units' dimensionalities in the format [Nhz,Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_z = n_z
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,z,constrain=True):
        '''
        compute moments of input/output Gaussian distribution
        INPUTS:
            z -  latent variable
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_x - mean of output Gaussian distribution
            log_sig_sq_x - log variance of output Gaussian distribution
        '''
        
        hidden1_pre = tfm.add(tfl.matmul(z, self.weights['W_z_to_h1']), self.weights['b_z_to_h1'])
        hidden_post = self.nonlinearity(hidden1_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_x = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_mux'.format(ni)]), self.weights['b_h{}_to_mux'.format(ni)])
        if constrain==True:
            mu_x = tf.nn.sigmoid(mu_x)
            
        log_sig_sq_x = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sx'.format(ni)]), self.weights['b_h{}_to_sx'.format(ni)])
        log_sig_sq_x = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_x/self.sig_lim)-0.5)

        return mu_x, log_sig_sq_x

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()

        all_weights['W_z_to_h1'] = tf.Variable(vae_utils.xavier_init(self.n_z, self.N_h[0]), dtype=tf.float32)
        all_weights['b_z_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-2], self.N_h[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_h[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_mux'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_x), dtype=tf.float32)
        all_weights['b_h{}_to_mux'.format(ni)] = tf.Variable(tf.zeros([self.n_x], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_sx'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_x), dtype=tf.float32)
        all_weights['b_h{}_to_sx'.format(ni)] = tf.Variable(tf.zeros([self.n_x], dtype=tf.float32) * self.bias_start)
        
        return all_weights
    
class Encoder(object):
    '''
    Class for Gaussian encoder, taking inputs/outputs x and outputting 
    Gaussian distribution q(z|x). Also used for obtaining conditional 
    prior p(z|y) or p(z|x) in CVAEs.
    '''
    
    def __init__(self, name, n_x, n_z, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the encoder
            n_x - dimensionality of the  input
            n_z - dimensionality of latent space
            N_h - array of hidden units' dimensionalities in the format [Nhx,Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_z = n_z
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,x):
        '''
        compute moments of latent Gaussian distribution
        INPUTS:
            x - conditional input
        OUTPUTS:
            mu_z - mean of latent Gaussian distribution
            log_sig_sq_z - log variance of latent Gaussian distribution
        '''
        
        hidden1_pre = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1']), self.weights['b_x_to_h1'])
        hidden_post = self.nonlinearity(hidden1_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muz'.format(ni)]), self.weights['b_h{}_to_muz'.format(ni)])
        log_sig_sq_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sz'.format(ni)]), self.weights['b_h{}_to_sz'.format(ni)])
        log_sig_sq_z = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_z/self.sig_lim)-0.5)

        return mu_z, log_sig_sq_z

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
        
        all_weights['W_h{}_to_muz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_muz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_sz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_sz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        return all_weights

class ConditionalDecoder(object):
    '''
    Class for Gaussian conditional decoder, taking inputs x, latent variable
    z and outputting Gaussian distribution p(y|z,x)
    '''
    
    def __init__(self, name, n_x, n_y, n_z, N_hx, N_hz, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of the conditional input
            n_y - dimensionality of the output
            n_z - dimensionality of latent space
            N_hx - array of hidden units' dimensionalities for the conditional input x in the format [Nhx1,Nhx2,...,Nhxn]
            N_hz - array of hidden units' dimensionalities for the latent variable z in the format [Nhz1,Nhz2,...,Nhzn]
            N_h - array of hidden units' dimensionalities for joint channels in the format [Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.N_hx = N_hx
        self.N_hz = N_hz
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,z,x,constrain=True):
        '''
        compute moments of output Gaussian distribution
        INPUTS:
            x - conditional input
            z - latent variable
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_y - mean of output Gaussian distribution
            log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        # Channel for latent variable alone
        hidden_pre_z = tfm.add(tfl.matmul(z, self.weights['W_z_to_h1z']), self.weights['b_z_to_h1z'])
        hidden_post_z = self.nonlinearity(hidden_pre_z)
        
        num_layers_middle_z = np.shape(self.N_hz)[0]-1
        for i in range(num_layers_middle_z):
            ni = i+2
        
            hidden_pre_z = tfm.add(tfl.matmul(hidden_post_z, self.weights['W_h{}z_to_h{}z'.format(ni-1,ni)]), self.weights['b_h{}z_to_h{}z'.format(ni-1,ni)])
            hidden_post_z = self.nonlinearity(hidden_pre_z)
        
        # Channel for conditional input alone
        hidden_pre_x = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1x']), self.weights['b_x_to_h1x'])
        hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        for i in range(num_layers_middle_x):
            ni = i+2
        
            hidden_pre_x = tfm.add(tfl.matmul(hidden_post_x, self.weights['W_h{}x_to_h{}x'.format(ni-1,ni)]), self.weights['b_h{}x_to_h{}x'.format(ni-1,ni)])
            hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        hidden_post = tf.concat([hidden_post_z,hidden_post_x],1)
        
        # Channel after combining the inputs
        hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h0_to_h1']), self.weights['b_h0_to_h1'])
        hidden_post = self.nonlinearity(hidden_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muy'.format(ni)]), self.weights['b_h{}_to_muy'.format(ni)])
        if constrain==True:
            mu_y = tf.nn.sigmoid(mu_y)
            
        log_sig_sq_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sy'.format(ni)]), self.weights['b_h{}_to_sy'.format(ni)])
        log_sig_sq_y = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_y/self.sig_lim)-0.5)

        return mu_y, log_sig_sq_y

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_z_to_h1z'] = tf.Variable(vae_utils.xavier_init(self.n_z, self.N_hz[0]), dtype=tf.float32)
        all_weights['b_z_to_h1z'] = tf.Variable(tf.zeros([self.N_hz[0]], dtype=tf.float32) * self.bias_start)

        num_layers_middle_z = np.shape(self.N_hz)[0]-1
        
        for i in range(num_layers_middle_z):
            ni = i+2
            
            all_weights['W_h{}z_to_h{}z'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hz[ni-2], self.N_hz[ni-1]), dtype=tf.float32)
            all_weights['b_h{}z_to_h{}z'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hz[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_x_to_h1x'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_hx[0]), dtype=tf.float32)
        all_weights['b_x_to_h1x'] = tf.Variable(tf.zeros([self.N_hx[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        
        for i in range(num_layers_middle_x):
            ni = i+2
            
            all_weights['W_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx[ni-2], self.N_hx[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx[ni-1]], dtype=tf.float32) * self.bias_start)

        
        all_weights['W_h0_to_h1'] = tf.Variable(vae_utils.xavier_init(self.N_hz[-1]+self.N_hx[-1], self.N_h[0]), dtype=tf.float32)
        all_weights['b_h0_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
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
    
class ConditionalEncoder(object):
    '''
    Class for Gaussian conditional encoder, taking inputs y and x and outputting latent variable
    distribution q(z|x,y)
    '''
    
    def __init__(self, name, n_x, n_y, n_z, N_hx, N_hy, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the conditional encoder
            n_x - dimensionality of the conditional input
            n_y - dimensionality of the input/output
            n_z - dimensionality of latent space
            N_hx - array of hidden units' dimensionalities for the conditional input x in the format [Nhx1,Nhx2,...,Nhxn]
            N_hy - array of hidden units' dimensionalities for the input/output y in the format [Nhy1,Nhy2,...,Nhyn]
            N_h - array of hidden units' dimensionalities for joint channels in the format [Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.N_hx = N_hx
        self.N_hy = N_hy
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,x,y):
        '''
        compute moments of latent Gaussian distribution
        INPUTS:
            x - conditional input
            y - output to encode
        OUTPUTS:
            mu_z - mean of output Gaussian distribution
            log_sig_sq_z - log variance of output Gaussian distribution
        '''
        
        # Channel for input/output alone
        hidden_pre_y = tfm.add(tfl.matmul(y, self.weights['W_y_to_h1y']), self.weights['b_y_to_h1y'])
        hidden_post_y = self.nonlinearity(hidden_pre_y)
        
        num_layers_middle_y = np.shape(self.N_hy)[0]-1
        for i in range(num_layers_middle_y):
            ni = i+2
        
            hidden_pre_y = tfm.add(tfl.matmul(hidden_post_y, self.weights['W_h{}y_to_h{}y'.format(ni-1,ni)]), self.weights['b_h{}y_to_h{}y'.format(ni-1,ni)])
            hidden_post_y = self.nonlinearity(hidden_pre_y)
        
        # Channel for conditional input alone
        hidden_pre_x = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1x']), self.weights['b_x_to_h1x'])
        hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        for i in range(num_layers_middle_x):
            ni = i+2
        
            hidden_pre_x = tfm.add(tfl.matmul(hidden_post_x, self.weights['W_h{}x_to_h{}x'.format(ni-1,ni)]), self.weights['b_h{}x_to_h{}x'.format(ni-1,ni)])
            hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        hidden_post = tf.concat([hidden_post_y,hidden_post_x],1)
        
        # Channel after combining the inputs
        hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h0_to_h1']), self.weights['b_h0_to_h1'])
        hidden_post = self.nonlinearity(hidden_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muz'.format(ni)]), self.weights['b_h{}_to_muz'.format(ni)])
        log_sig_sq_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sz'.format(ni)]), self.weights['b_h{}_to_sz'.format(ni)])
        log_sig_sq_z = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_z/self.sig_lim)-0.5)

        return mu_z, log_sig_sq_z

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_y_to_h1y'] = tf.Variable(vae_utils.xavier_init(self.n_y, self.N_hy[0]), dtype=tf.float32)
        all_weights['b_y_to_h1y'] = tf.Variable(tf.zeros([self.N_hy[0]], dtype=tf.float32) * self.bias_start)

        num_layers_middle_y = np.shape(self.N_hy)[0]-1
        
        for i in range(num_layers_middle_y):
            ni = i+2
            
            all_weights['W_h{}y_to_h{}y'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hy[ni-2], self.N_hy[ni-1]), dtype=tf.float32)
            all_weights['b_h{}y_to_h{}y'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hy[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_x_to_h1x'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_hx[0]), dtype=tf.float32)
        all_weights['b_x_to_h1x'] = tf.Variable(tf.zeros([self.N_hx[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        
        for i in range(num_layers_middle_x):
            ni = i+2
            
            all_weights['W_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx[ni-2], self.N_hx[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx[ni-1]], dtype=tf.float32) * self.bias_start)

        
        all_weights['W_h0_to_h1'] = tf.Variable(vae_utils.xavier_init(self.N_hy[-1]+self.N_hx[-1], self.N_h[0]), dtype=tf.float32)
        all_weights['b_h0_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-2], self.N_h[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_h[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_muz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_muz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_sz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_sz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        return all_weights
    
class DoubleConditionalDecoder(object):
    '''
    Class for Gaussian conditional decoder, taking inputs x and x2, latent variable
    z and outputting Gaussian distribution p(y|z,x,x2)
    '''
    
    def __init__(self, name, n_x, n_x2, n_y, n_z, N_hx, N_hx2, N_hz, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of the first conditional input
            n_x2 - dimensionality of the second conditional input
            n_y - dimensionality of the output
            n_z - dimensionality of latent space
            N_hx - array of hidden units' dimensionalities for the conditional input x in the format [Nhx1,Nhx2,...,Nhxn]
            N_hx2 - array of hidden units' dimensionalities for the conditional input x2 in the format [Nhxb1,Nhxb2,...,Nhxbn]
            N_hz - array of hidden units' dimensionalities for the latent variable z in the format [Nhz1,Nhz2,...,Nhzn]
            N_h - array of hidden units' dimensionalities for joint channels in the format [Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_x2 = n_x2
        self.n_y = n_y
        self.n_z = n_z
        self.N_hx = N_hx
        self.N_hx2 = N_hx2
        self.N_hz = N_hz
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,z,x,x2,constrain=True):
        '''
        compute moments of output Gaussian distribution
        INPUTS:
            x - conditional input
            x2 - conditional input
            z - latent variable
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_y - mean of output Gaussian distribution
            log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        # Channel for latent variable alone
        hidden_pre_z = tfm.add(tfl.matmul(z, self.weights['W_z_to_h1z']), self.weights['b_z_to_h1z'])
        hidden_post_z = self.nonlinearity(hidden_pre_z)
        
        num_layers_middle_z = np.shape(self.N_hz)[0]-1
        for i in range(num_layers_middle_z):
            ni = i+2
        
            hidden_pre_z = tfm.add(tfl.matmul(hidden_post_z, self.weights['W_h{}z_to_h{}z'.format(ni-1,ni)]), self.weights['b_h{}z_to_h{}z'.format(ni-1,ni)])
            hidden_post_z = self.nonlinearity(hidden_pre_z)
        
        # Channel for first conditional input alone
        hidden_pre_x = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1x']), self.weights['b_x_to_h1x'])
        hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        for i in range(num_layers_middle_x):
            ni = i+2
        
            hidden_pre_x = tfm.add(tfl.matmul(hidden_post_x, self.weights['W_h{}x_to_h{}x'.format(ni-1,ni)]), self.weights['b_h{}x_to_h{}x'.format(ni-1,ni)])
            hidden_post_x = self.nonlinearity(hidden_pre_x)
            
        # Channel for second conditional input alone
        hidden_pre_x2 = tfm.add(tfl.matmul(x2, self.weights['W_x2_to_h1x2']), self.weights['b_x2_to_h1x2'])
        hidden_post_x2 = self.nonlinearity(hidden_pre_x2)
        
        num_layers_middle_x2 = np.shape(self.N_hx2)[0]-1
        for i in range(num_layers_middle_x2):
            ni = i+2
        
            hidden_pre_x2 = tfm.add(tfl.matmul(hidden_post_x2, self.weights['W_h{}x2_to_h{}x2'.format(ni-1,ni)]), self.weights['b_h{}x2_to_h{}x2'.format(ni-1,ni)])
            hidden_post_x2 = self.nonlinearity(hidden_pre_x2)
        
        hidden_post = tf.concat([hidden_post_z,hidden_post_x,hidden_post_x2],1)
        
        # Channel after combining the inputs
        hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h0_to_h1']), self.weights['b_h0_to_h1'])
        hidden_post = self.nonlinearity(hidden_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muy'.format(ni)]), self.weights['b_h{}_to_muy'.format(ni)])
        if constrain==True:
            mu_y = tf.nn.sigmoid(mu_y)
            
        log_sig_sq_y = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sy'.format(ni)]), self.weights['b_h{}_to_sy'.format(ni)])
        log_sig_sq_y = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_y/self.sig_lim)-0.5)

        return mu_y, log_sig_sq_y

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_z_to_h1z'] = tf.Variable(vae_utils.xavier_init(self.n_z, self.N_hz[0]), dtype=tf.float32)
        all_weights['b_z_to_h1z'] = tf.Variable(tf.zeros([self.N_hz[0]], dtype=tf.float32) * self.bias_start)

        num_layers_middle_z = np.shape(self.N_hz)[0]-1
        
        for i in range(num_layers_middle_z):
            ni = i+2
            
            all_weights['W_h{}z_to_h{}z'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hz[ni-2], self.N_hz[ni-1]), dtype=tf.float32)
            all_weights['b_h{}z_to_h{}z'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hz[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_x_to_h1x'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_hx[0]), dtype=tf.float32)
        all_weights['b_x_to_h1x'] = tf.Variable(tf.zeros([self.N_hx[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        
        for i in range(num_layers_middle_x):
            ni = i+2
            
            all_weights['W_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx[ni-2], self.N_hx[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_x2_to_h1x2'] = tf.Variable(vae_utils.xavier_init(self.n_x2, self.N_hx2[0]), dtype=tf.float32)
        all_weights['b_x2_to_h1x2'] = tf.Variable(tf.zeros([self.N_hx2[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x2 = np.shape(self.N_hx2)[0]-1
        
        for i in range(num_layers_middle_x2):
            ni = i+2
            
            all_weights['W_h{}x2_to_h{}x2'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx2[ni-2], self.N_hx2[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x2_to_h{}x2'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx2[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h0_to_h1'] = tf.Variable(vae_utils.xavier_init(self.N_hz[-1]+self.N_hx[-1]+self.N_hx2[-1], self.N_h[0]), dtype=tf.float32)
        all_weights['b_h0_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
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
    
class DoubleConditionalEncoder(object):
    '''
    Class for Gaussian conditional encoder, taking inputs x, x2 and y, and outputting latent Gaussian distribution q(z|x,x2,y)
    '''
    
    def __init__(self, name, n_x, n_x2, n_y, n_z, N_hx, N_hx2, N_hy, N_h, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation
        INPUTS:
            name - name to assign to the conditional encoder
            n_x - dimensionality of the conditional input
            n_x2 - dimensionality of the second conditional input
            n_y - dimensionality of the input/output
            n_z - dimensionality of latent space
            N_hx - array of hidden units' dimensionalities for the conditional input x in the format [Nhx1,Nhx2,...,Nhxn]
            N_hx2 - array of hidden units' dimensionalities for the conditional input x2 in the format [Nhxb1,Nhxb2,...,Nhxbn]
            N_hy - array of hidden units' dimensionalities for the input/output y in the format [Nhy1,Nhy2,...,Nhyn]
            N_h - array of hidden units' dimensionalities for joint channels in the format [Nh1,Nh2,...,Nhn]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        self.n_x = n_x
        self.n_x2 = n_x2
        self.n_y = n_y
        self.n_z = n_z
        self.N_hx = N_hx
        self.N_hx2 = N_hx2
        self.N_hy = N_hy
        self.N_h = N_h
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = nonlinearity
        
    def compute_moments(self,y,x,x2):
        '''
        compute moments of latent Gaussian distribution
        INPUTS:
            x - conditional input
            y - output to encode
        OUTPUTS:
            mu_z - mean of output Gaussian distribution
            log_sig_sq_z - log variance of output Gaussian distribution
        '''
        
        # Channel for input/output alone
        hidden_pre_y = tfm.add(tfl.matmul(y, self.weights['W_y_to_h1y']), self.weights['b_y_to_h1y'])
        hidden_post_y = self.nonlinearity(hidden_pre_y)
        
        num_layers_middle_y = np.shape(self.N_hy)[0]-1
        for i in range(num_layers_middle_y):
            ni = i+2
        
            hidden_pre_y = tfm.add(tfl.matmul(hidden_post_y, self.weights['W_h{}y_to_h{}y'.format(ni-1,ni)]), self.weights['b_h{}y_to_h{}y'.format(ni-1,ni)])
            hidden_post_y = self.nonlinearity(hidden_pre_y)
        
        # Channel for conditional input alone
        hidden_pre_x = tfm.add(tfl.matmul(x, self.weights['W_x_to_h1x']), self.weights['b_x_to_h1x'])
        hidden_post_x = self.nonlinearity(hidden_pre_x)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        for i in range(num_layers_middle_x):
            ni = i+2
        
            hidden_pre_x = tfm.add(tfl.matmul(hidden_post_x, self.weights['W_h{}x_to_h{}x'.format(ni-1,ni)]), self.weights['b_h{}x_to_h{}x'.format(ni-1,ni)])
            hidden_post_x = self.nonlinearity(hidden_pre_x)
            
        # Channel for second conditional input alone
        hidden_pre_x2 = tfm.add(tfl.matmul(x2, self.weights['W_x2_to_h1x2']), self.weights['b_x2_to_h1x2'])
        hidden_post_x2 = self.nonlinearity(hidden_pre_x2)
        
        num_layers_middle_x2 = np.shape(self.N_hx2)[0]-1
        for i in range(num_layers_middle_x2):
            ni = i+2
        
            hidden_pre_x2 = tfm.add(tfl.matmul(hidden_post_x2, self.weights['W_h{}x2_to_h{}x2'.format(ni-1,ni)]), self.weights['b_h{}x2_to_h{}x2'.format(ni-1,ni)])
            hidden_post_x2 = self.nonlinearity(hidden_pre_x2)
        
        hidden_post = tf.concat([hidden_post_y,hidden_post_x,hidden_post_x2],1)
        
        # Channel after combining the inputs
        hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h0_to_h1']), self.weights['b_h0_to_h1'])
        hidden_post = self.nonlinearity(hidden_pre)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
        
            hidden_pre = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_h{}'.format(ni-1,ni)]), self.weights['b_h{}_to_h{}'.format(ni-1,ni)])
            hidden_post = self.nonlinearity(hidden_pre)
        
        mu_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_muz'.format(ni)]), self.weights['b_h{}_to_muz'.format(ni)])
        log_sig_sq_z = tfm.add(tfl.matmul(hidden_post, self.weights['W_h{}_to_sz'.format(ni)]), self.weights['b_h{}_to_sz'.format(ni)])
        log_sig_sq_z = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_z/self.sig_lim)-0.5)

        return mu_z, log_sig_sq_z

    def _create_weights(self):
        '''
        Initialise weights
        '''
        
        all_weights = collections.OrderedDict()
        
        all_weights['W_y_to_h1y'] = tf.Variable(vae_utils.xavier_init(self.n_y, self.N_hy[0]), dtype=tf.float32)
        all_weights['b_y_to_h1y'] = tf.Variable(tf.zeros([self.N_hy[0]], dtype=tf.float32) * self.bias_start)

        num_layers_middle_y = np.shape(self.N_hy)[0]-1
        
        for i in range(num_layers_middle_y):
            ni = i+2
            
            all_weights['W_h{}y_to_h{}y'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hy[ni-2], self.N_hy[ni-1]), dtype=tf.float32)
            all_weights['b_h{}y_to_h{}y'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hy[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_x_to_h1x'] = tf.Variable(vae_utils.xavier_init(self.n_x, self.N_hx[0]), dtype=tf.float32)
        all_weights['b_x_to_h1x'] = tf.Variable(tf.zeros([self.N_hx[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x = np.shape(self.N_hx)[0]-1
        
        for i in range(num_layers_middle_x):
            ni = i+2
            
            all_weights['W_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx[ni-2], self.N_hx[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x_to_h{}x'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx[ni-1]], dtype=tf.float32) * self.bias_start)

        
        all_weights['W_x2_to_h1x2'] = tf.Variable(vae_utils.xavier_init(self.n_x2, self.N_hx2[0]), dtype=tf.float32)
        all_weights['b_x2_to_h1x2'] = tf.Variable(tf.zeros([self.N_hx2[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle_x2 = np.shape(self.N_hx2)[0]-1
        
        for i in range(num_layers_middle_x2):
            ni = i+2
            
            all_weights['W_h{}x2_to_h{}x2'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_hx2[ni-2], self.N_hx2[ni-1]), dtype=tf.float32)
            all_weights['b_h{}x2_to_h{}x2'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_hx2[ni-1]], dtype=tf.float32) * self.bias_start)

        all_weights['W_h0_to_h1'] = tf.Variable(vae_utils.xavier_init(self.N_hy[-1]+self.N_hx[-1]+self.N_hx2[-1], self.N_h[0]), dtype=tf.float32)
        all_weights['b_h0_to_h1'] = tf.Variable(tf.zeros([self.N_h[0]], dtype=tf.float32) * self.bias_start)
        
        num_layers_middle = np.shape(self.N_h)[0]-1
        
        for i in range(num_layers_middle):
            ni = i+2
            
            all_weights['W_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-2], self.N_h[ni-1]), dtype=tf.float32)
            all_weights['b_h{}_to_h{}'.format(ni-1,ni)] = tf.Variable(tf.zeros([self.N_h[ni-1]], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_muz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_muz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        all_weights['W_h{}_to_sz'.format(ni)] = tf.Variable(vae_utils.xavier_init(self.N_h[ni-1], self.n_z), dtype=tf.float32)
        all_weights['b_h{}_to_sz'.format(ni)] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32) * self.bias_start)
        
        return all_weights
    