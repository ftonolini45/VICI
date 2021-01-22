import numpy as np
import collections
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import linalg as tfl
from neural_networks import NN_utils
from neural_networks import layers
from neural_networks import networks
    
class ForwardModel_Encoder(object):
    '''
    Forward model encoder q(w|x,y,y_til).
    '''
    
    def __init__(self, name, n_x, n_ch_y, n_ch_y_til, y_siz, y_til_siz, n_w, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of targets
            n_ch_y - number of channels in the high fidelity measurements
            n_ch_y_til - number of channels in the low fidelity measurements
            y_siz - size of high fideity capacitive images [height,width]
            y_til_siz - size of low fideity capacitive images [height,width]
            n_w - dimensionality of latent space
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y = n_ch_y
        self.n_ch_y_til = n_ch_y_til
        self.y_siz = y_siz
        self.y_til_siz = y_til_siz
        self.n_x = n_x
        self.n_w = n_w
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # the target signals x will go through a fully connected network:
        self.N_x = [40,20,20]
        
        # high fidelity measurements y will instead go through a conv2D network:
        self.N_y = [3,9,9] # number of channels for each hidden layer
        self.st_y = [1,2,1] # strides for each convolutional layer
        self.fs_y = [5,5,5] # filter sizes (they will be all square)
        
        # low fidelity measurements y_til will also go through a conv2D network:
        self.N_y_til = [3,9,9] # number of channels for each hidden layer
        self.st_y_til = [1,2,1] # strides for each convolutional layer
        self.fs_y_til = [5,5,5] # filter sizes (they will be all square)
        
        # finally, the flattened outputs of the convnets and the output of the fully connected network will be concatenated and passed through a fully connected network, so:
        self.N_comb = [100,70,40] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,y,x,y_til,constrain=True):
        '''
        compute moments of latent Gaussian distribution from targets, high-fidelity measurements and low-fidelity measurements
        INPUTS:
            x - targets
            y - high-fidelity measurements
            y_til -low fidelity measurements
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_w - mean of latent Gaussian distribution
            log_sig_sq_w - log variance of latent Gaussian distribution
        '''
        
        # we first pass x through a fully connected network:
        hidden_post_x = networks.fc_network(x, self.weights, tf.shape(self.N_x)[0], self.nonlinearity, ID=0)
        
        # then we pass y through a conv2D network:
        hidden_post_y = networks.conv_2D_network(y, self.weights, self.st_y, self.nonlinearity, im_size=self.y_siz, reshape_in=True, reshape_out=True, ID=1) 
        
        # then we pass y_til through a conv2D network:
        hidden_post_y_til = networks.conv_2D_network(y_til, self.weights, self.st_y_til, self.nonlinearity, im_size=self.y_til_siz, reshape_in=True, reshape_out=True, ID=2) 
        
        # now we concatenate the two results:
        hidden_post = tf.concat([hidden_post_x, hidden_post_y, hidden_post_y_til], axis=1)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, tf.shape(self.N_comb)[0], self.nonlinearity, ID=3)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_w = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=4) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_w = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=5) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_w = tf.nn.sigmoid(mu_w)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_w = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_w/self.sig_lim)-0.5)

        return mu_w, log_sig_sq_w

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_x, self.N_x, ID=0)
        
        # we can make the weights for the convolutional network taking y:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y,self.N_y,self.fs_y, ID=1)
        
        # we can make the weights for the convolutional network taking y_til:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y_til,self.N_y_til,self.fs_y_til, ID=2)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size.
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes_y = NN_utils.compute_size(self.y_siz,self.st_y) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y = im_sizes_y[-1,0]*im_sizes_y[-1,1]*self.N_y[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # we do the same for y_til
        im_sizes_y_til = NN_utils.compute_size(self.y_til_siz,self.st_y_til) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y_til = im_sizes_y_til[-1,0]*im_sizes_y_til[-1,1]*self.N_y_til[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # the size of the input to the last layer is then the sum of the above two and the dimensionality of the last fully connected layer that took z as input
        dim_input = siz_hidden_post_y + siz_hidden_post_y_til + self.N_x[-1]
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=3)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_w], add_b=False, ID=4)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_w], add_b=False, ID=5)
        
        return all_weights
    
class ForwardModel_ConditionalEncoder(object):
    '''
    Forward model conditional encoder p(w|x,y_til).
    '''
    
    def __init__(self, name, n_x, n_ch_y_til, y_til_siz, n_w, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of targets
            n_ch_y_til - number of channels in the low fidelity measurements
            y_til_siz - size of low fideity capacitive images [height,width]
            n_w - dimensionality of latent space
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y_til = n_ch_y_til
        self.y_til_siz = y_til_siz
        self.n_x = n_x
        self.n_w = n_w
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # the target signals x will go through a fully connected network:
        self.N_x = [40,20,20]
        
        # low fidelity measurements y_til will go through a conv2D network:
        self.N_y_til = [3,9,9] # number of channels for each hidden layer
        self.st_y_til = [1,2,1] # strides for each convolutional layer
        self.fs_y_til = [5,5,5] # filter sizes (they will be all square)
        
        # finally, the flattened outputs of the convnets and the output of the fully connected network will be concatenated and passed through a fully connected network, so:
        self.N_comb = [100,70,40] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,x,y_til,constrain=True):
        '''
        compute moments of latent Gaussian distribution from targets and low-fidelity measurements
        INPUTS:
            x - targets
            y_til -low fidelity measurements
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_w - mean of latent Gaussian distribution
            log_sig_sq_w - log variance of latent Gaussian distribution
        '''
        
        # we first pass x through a fully connected network:
        hidden_post_x = networks.fc_network(x, self.weights, tf.shape(self.N_x)[0], self.nonlinearity, ID=0)
        
        # then we pass y_til through a conv2D network:
        hidden_post_y_til = networks.conv_2D_network(y_til, self.weights, self.st_y_til, self.nonlinearity, im_size=self.y_til_siz, reshape_in=True, reshape_out=True, ID=1) 
        
        # now we concatenate the two results:
        hidden_post = tf.concat([hidden_post_x, hidden_post_y_til], axis=1)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, tf.shape(self.N_comb)[0], self.nonlinearity, ID=2)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_w = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=3) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_w = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=4) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_w = tf.nn.sigmoid(mu_w)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_w = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_w/self.sig_lim)-0.5)

        return mu_w, log_sig_sq_w

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_x, self.N_x, ID=0)
        
        # we can make the weights for the convolutional network taking y_til:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y_til,self.N_y_til,self.fs_y_til, ID=1)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size.
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes_y_til = NN_utils.compute_size(self.y_til_siz,self.st_y_til) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y_til = im_sizes_y_til[-1,0]*im_sizes_y_til[-1,1]*self.N_y_til[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # the size of the input to the last layer is then the sum of the above two and the dimensionality of the last fully connected layer that took z as input
        dim_input =  siz_hidden_post_y_til + self.N_x[-1]
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=2)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_w], add_b=False, ID=3)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_w], add_b=False, ID=4)
        
        return all_weights
    
class ForwardModel_Decoder(object):
    '''
    Forward model decoder p(y|x,y_til,w).
    '''
    
    def __init__(self, name, n_x, n_w, n_ch_y_til, y_til_siz, n_ch_y, y_siz, nonlinearity=tf.nn.leaky_relu, sig_lim=40):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of targets
            n_w - dimensionality of latent space
            n_ch_y_til - number of channels in the low fidelity measurements
            y_til_siz - size of low fideity capacitive images [height,width]
            n_ch_y - number of channels in the high fidelity measurements
            y_siz - size of high fideity capacitive images [height,width]
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y = n_ch_y
        self.n_ch_y_til = n_ch_y_til
        self.y_siz = y_siz
        self.y_til_siz = y_til_siz
        self.n_x = n_x
        self.n_w = n_w
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # the target signals x will go through a fully connected network:
        self.N_x = [20,40,80]
        
        # low fidelity measurements y_til will go through a conv2D network:
        self.N_y_til = [3,9,9] # number of channels for each hidden layer
        self.st_y_til = [1,2,1] # strides for each convolutional layer
        self.fs_y_til = [5,5,5] # filter sizes (they will be all square)
        
        # the latent variables w will go through a fully connected network:
        self.N_w = [20,40,80]
        
        # finally, the outputs of the three above will be concatenated to go through a conv2D layer to generate the output:
        self.N_comb = [9,6,3] # number of channels for each hidden layer
        self.st_comb = [0.5,1,1] # strides for each convolutional layer
        self.fs_comb = [5,5,5] # filter sizes (they will be all square)
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,w,x,y_til,constrain=True):
        '''
        compute moments of high-fidelity measurements Gaussian distribution from targets, low-fidelity measurements and latent variable
        INPUTS:
            x - targets
            y_til -low fidelity measurements
            w - latent variable
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_y - mean of output Gaussian distribution
            log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        # we first pass x through a fully connected network:
        hidden_post_x = networks.fc_network(x, self.weights, tf.shape(self.N_x)[0], self.nonlinearity, ID=0)
        
        # then we pass y_til through a conv2D network:
        hidden_post_y_til = networks.conv_2D_network(y_til, self.weights, self.st_y_til, self.nonlinearity, im_size=self.y_til_siz, reshape_in=True, reshape_out=False, ID=1) # note here reshape out is False
        
        # then we pass z through a fully connected layer:
        hidden_post_w = networks.fc_network(w, self.weights, tf.shape(self.N_w)[0], self.nonlinearity, ID=2)
        
        # now we reshape and concatenate the three results:
        hidden_post_x = layers.reshape_2D(hidden_post_x,im_sz=[8,5])
        hidden_post_w = layers.reshape_2D(hidden_post_w,im_sz=[8,5])
        hidden_post = tf.concat([hidden_post_x, hidden_post_w, hidden_post_y_til], axis=3)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last conv2D network
        hidden_post = networks.conv_2D_network(hidden_post, self.weights, self.st_comb, self.nonlinearity, im_size=[8,5], reshape_in=False, reshape_out=False, ID=3) # note here reshape in and out are both False
        
        # finally, we take the output of the last network and pass it through two linear filters to get mu and log_sigma_square
        mu_y = networks.conv_2D_network(hidden_post, self.weights, [1], nonlinearity=None, reshape_in=False, reshape_out=True, ID=4) # note here reshape in is False
        log_sig_sq_y = networks.conv_2D_network(hidden_post, self.weights, [1], nonlinearity=None, reshape_in=False, reshape_out=True, ID=5) # note here reshape in is False
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_y = tf.nn.sigmoid(mu_y)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_y = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_y/self.sig_lim)-0.5)

        return mu_y, log_sig_sq_y

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_x, self.N_x, ID=0)
        
        # we can make the weights for the convolutional network taking y_til:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y_til,self.N_y_til,self.fs_y_til, ID=1)
        
        # we make the weights for the fully connected network taking w:
        all_weights = networks.fc_make_weights(all_weights, self.n_w, self.N_w, ID=2)
        
        # now we calculate the number of channels that the input to the final conv2D layers will take
        n_ch_comb = tf.cast(tf.divide(self.N_x[-1],8*5),tf.int32) + tf.cast(tf.divide(self.N_w[-1],8*5),tf.int32) + self.N_y_til[-1]
        
        # we now make the weights for the last convolutional network:
        all_weights = networks.conv_2D_make_weights(all_weights,n_ch_comb,self.N_comb,self.fs_comb, ID=3)
        
        # lastly, we initialise the weights for the two single filters to get mu and log_sigma_square from the last layer
        all_weights = networks.conv_2D_make_weights(all_weights,self.N_comb[-1],[self.n_ch_y],[5], ID=4)
        all_weights = networks.conv_2D_make_weights(all_weights,self.N_comb[-1],[self.n_ch_y],[5], ID=5)
        
        return all_weights
    
class InverseModel_Encoder(object):
    '''
    Forward model encoder q(z|x,y).
    '''
    
    def __init__(self, name, n_x, n_ch_y, y_siz, n_z, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_x - dimensionality of targets
            n_ch_y - number of channels in the high fidelity measurements
            y_siz - size of high fideity capacitive images [height,width]
            n_z - dimensionality of latent space
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y = n_ch_y
        self.y_siz = y_siz
        self.n_x = n_x
        self.n_z = n_z
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # the target signals x will go through a fully connected network:
        self.N_x = [40,20,20]
        
        # high fidelity measurements y will instead go through a conv2D network:
        self.N_y = [3,9,9] # number of channels for each hidden layer
        self.st_y = [1,2,1] # strides for each convolutional layer
        self.fs_y = [5,5,5] # filter sizes (they will be all square)
        
        # finally, the flattened outputs of the convnets and the output of the fully connected network will be concatenated and passed through a fully connected network, so:
        self.N_comb = [100,70,40] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,y,x,constrain=True):
        '''
        compute moments of latent Gaussian distribution from targets and high-fidelity measurements
        INPUTS:
            x - targets
            y - high-fidelity measurements
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_z - mean of latent Gaussian distribution
            log_sig_sq_z - log variance of latent Gaussian distribution
        '''
        
        # we first pass x through a fully connected network:
        hidden_post_x = networks.fc_network(x, self.weights, tf.shape(self.N_x)[0], self.nonlinearity, ID=0)
        
        # then we pass y through a conv2D network:
        hidden_post_y = networks.conv_2D_network(y, self.weights, self.st_y, self.nonlinearity, im_size=self.y_siz, reshape_in=True, reshape_out=True, ID=1) 
        
        # now we concatenate the two results:
        hidden_post = tf.concat([hidden_post_x, hidden_post_y], axis=1)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, tf.shape(self.N_comb)[0], self.nonlinearity, ID=2)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_z = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=3) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_z = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=4) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_z = tf.nn.sigmoid(mu_z)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_z = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_z/self.sig_lim)-0.5)

        return mu_z, log_sig_sq_z

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_x, self.N_x, ID=0)
        
        # we can make the weights for the convolutional network taking y:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y,self.N_y,self.fs_y, ID=1)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size.
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes_y = NN_utils.compute_size(self.y_siz,self.st_y) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y = im_sizes_y[-1,0]*im_sizes_y[-1,1]*self.N_y[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # the size of the input to the last layer is then the sum of the above and the dimensionality of the last fully connected layer that took z as input
        dim_input = siz_hidden_post_y + self.N_x[-1]
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=2)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_z], add_b=False, ID=3)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_z], add_b=False, ID=4)
        
        return all_weights
    
class InverseModel_ConditionalEncoder(object):
    '''
    Forward model conditional encoder r(z|y).
    '''
    
    def __init__(self, name, n_ch_y, y_siz, n_z, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_ch_y - number of channels in the high fidelity measurements
            y_siz - size of high fideity capacitive images [height,width]
            n_z - dimensionality of latent space
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y = n_ch_y
        self.y_siz = y_siz
        self.n_z = n_z
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # high fidelity measurements y will go through a conv2D network:
        self.N_y = [3,9,9] # number of channels for each hidden layer
        self.st_y = [1,2,1] # strides for each convolutional layer
        self.fs_y = [5,5,5] # filter sizes (they will be all square)
        
        # finally, the flattened outputs of the convnets and the output of the fully connected network will be concatenated and passed through a fully connected network, so:
        self.N_comb = [100,70,40] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,y,constrain=True):
        '''
        compute moments of latent Gaussian prior distribution from high-fidelity measurements
        INPUTS:
            y - high-fidelity measurements
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_z - mean of latent Gaussian distribution
            log_sig_sq_z - log variance of latent Gaussian distribution
        '''
        
        # then we pass y through a conv2D network:
        hidden_post_y = networks.conv_2D_network(y, self.weights, self.st_y, self.nonlinearity, im_size=self.y_siz, reshape_in=True, reshape_out=True, ID=0) 
        hidden_post = hidden_post_y
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, tf.shape(self.N_comb)[0], self.nonlinearity, ID=1)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_z = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=2) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_z = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=3) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_z = tf.nn.sigmoid(mu_z)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_z = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_z/self.sig_lim)-0.5)

        return mu_z, log_sig_sq_z

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we can make the weights for the convolutional network taking y:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y,self.N_y,self.fs_y, ID=0)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size.
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes_y = NN_utils.compute_size(self.y_siz,self.st_y) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y = im_sizes_y[-1,0]*im_sizes_y[-1,1]*self.N_y[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # the size of the input to the last layer is then just the above
        dim_input = siz_hidden_post_y
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=1)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_z], add_b=False, ID=2)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_z], add_b=False, ID=3)
        
        return all_weights
    
class InverseModel_Decoder(object):
    '''
    Forward model encoder q(z|x,y).
    '''
    
    def __init__(self, name, n_z, n_ch_y, y_siz, n_x, nonlinearity=tf.nn.leaky_relu, sig_lim=30):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_z - dimensionality of latent space
            n_ch_y - number of channels in the high fidelity measurements
            y_siz - size of high fideity capacitive images [height,width]
            n_x - dimensionality of targets
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_y = n_ch_y
        self.y_siz = y_siz
        self.n_x = n_x
        self.n_z = n_z
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # the latent variable z will go through a fully connected network:
        self.N_z = [30,40,60]
        
        # high fidelity measurements y will instead go through a conv2D network:
        self.N_y = [3,9,9] # number of channels for each hidden layer
        self.st_y = [1,2,1] # strides for each convolutional layer
        self.fs_y = [5,5,5] # filter sizes (they will be all square)
        
        # finally, the flattened outputs of the convnets and the output of the fully connected network will be concatenated and passed through a fully connected network, so:
        self.N_comb = [200,70,30] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,z,y,constrain=True):
        '''
        compute moments of latent Gaussian distribution from targets and high-fidelity measurements
        INPUTS:
            z - latent variable
            y - high-fidelity measurements
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_x - mean of output target distribution
            log_sig_sq_x - log variance of output target distribution
        '''
        
        # we first pass x through a fully connected network:
        hidden_post_z = networks.fc_network(z, self.weights, tf.shape(self.N_z)[0], self.nonlinearity, ID=0)
        
        # then we pass y through a conv2D network:
        hidden_post_y = networks.conv_2D_network(y, self.weights, self.st_y, self.nonlinearity, im_size=self.y_siz, reshape_in=True, reshape_out=True, ID=1) 
        
        # now we concatenate the two results:
        hidden_post = tf.concat([hidden_post_z, hidden_post_y], axis=1)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, tf.shape(self.N_comb)[0], self.nonlinearity, ID=2)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_x = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=3) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_x = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, ID=4) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if constrain==True:
            mu_x = tf.nn.sigmoid(mu_x)
            
        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq_x = self.sig_lim*(tf.nn.sigmoid(log_sig_sq_x/self.sig_lim)-0.5)

        return mu_x, log_sig_sq_x

    def _create_weights(self):
        '''
        Initialise weights. each of the functions you can import from "networks" in the compute function above has a "make_weights" counterpart you need to call in here
        '''
        
        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()
        
        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_z, self.N_z, ID=0)
        
        # we can make the weights for the convolutional network taking y:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_y,self.N_y,self.fs_y, ID=1)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size.
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes_y = NN_utils.compute_size(self.y_siz,self.st_y) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_y = im_sizes_y[-1,0]*im_sizes_y[-1,1]*self.N_y[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        # the size of the input to the last layer is then the sum of the above and the dimensionality of the last fully connected layer that took z as input
        dim_input = siz_hidden_post_y + self.N_z[-1]
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=2)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_x], add_b=False, ID=3)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_x], add_b=False, ID=4)
        
        return all_weights