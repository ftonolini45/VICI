import numpy as np
import collections
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import linalg as tfl
from neural_networks import NN_utils
from neural_networks import layers
from neural_networks import networks
    
class ImageAndLatent_To_Vector(object):
    '''
    Example of a conditional decoder p(y|z,x) where x is an image and y and z are vectors.
    '''
    
    def __init__(self, name, n_ch_x, x_siz, n_y, n_z, nonlinearity=tf.nn.leaky_relu, sig_lim=10):
        '''
        Initialisation. This function should accept the external parameters, such as input and output sizes.
        INPUTS:
            name - name to assign to the conditional decoder
            n_ch_x - number of channels in the conditional input
            x_siz - size of the input images [height,width]
            n_y - dimensionality of the output
            n_z - dimensionality of latent space
        OPTIONAL INPUTS:
            nonlinearity - choice of non-linearity (e.g. tf.nn.relu/tf.nn.leaky_relu/tf.nn.elu)
            sig_lim - range to impose to the output log_sig_sq to avoid divergence.
        '''
        
        # casting the inputs as class global variables.
        self.n_ch_x = n_ch_x
        self.x_siz = x_siz
        self.n_y = n_y
        self.n_z = n_z
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.nonlinearity = nonlinearity
        
        # now define all the internal parameters of the networks used to build the model.
        
        # we want to pass x through a convolutional network, so we start by defining parameters for this:
        self.N_x = [3,9,30,60] # number of channels for each hidden layer
        self.st_x = [1,2,2,2] # strides for each convolutional layer
        self.fs = [5,7,7,5] # filter sizes (they will be all square)
        
        # instead, z will go through a fully connected layer, so:
        self.N_z = [10,20,20,30,40] # number of hidden units in each layer
        
        # finally, the flattened output of the convnet and the output of z will be concatenated and passed through a fully connected layer, so:
        self.N_comb = [200,100,50] # number of hidden units in each layer
        
        # now we initialise the weights and set them as a global variable (look below to see how to construct the weights)
        network_weights = self._create_weights()
        self.weights = network_weights
        
    def compute_moments(self,z,x,constrain=True):
        '''
        compute moments of output Gaussian distribution propagating z and x through our network
        INPUTS:
            x - conditional input in the format of flattened images [n_samples,dimensions], where dimensions = height x width x channels
            z - latent variable
        OPTIONAL INPUTS:
            constrain - whether to force the output mean to be between 0 and 1
        OUTPUTS:
            mu_y - mean of output Gaussian distribution
            log_sig_sq_y - log variance of output Gaussian distribution
        '''
        
        # we first pass x through a conv2D network we can call from "networks" as follows:
        hidden_post_x = networks.conv_2D_network(x, self.weights, self.st_x, self.nonlinearity, im_size=self.x_siz, reshape_in=True, reshape_out=True, ID=0) 
        # use a different ID number for each network, this way the function will automatically fetch the right weights from self.weights 
        # be careful to set reshape_in/out correctly. In this case, the image are fed to the network in [n_samples,dimensions] format and we wish to get them out in the same format, so both are set to true.
        
        # now onto z, which just goes through a fully connected network:
        hidden_post_z = networks.fc_network(z, self.weights, self.nonlinearity, ID=1)
        
        # now we concatenate the two results:
        hidden_post = tf.concat(hidden_post_x, hidden_post_z, axis=1)
        # if instead we were to concatenate images along the channels we wuould do this along axis 3 (again, careful at reshape_in/out in conv networks for this)
        
        # now we will input the concatenated vector into a last fully connected network
        hidden_post = networks.fc_network(hidden_post, self.weights, self.nonlinearity, ID=2)
        
        # finally, we take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu_y = networks.fc_network(hidden_post, self.weights, nonlinearity=None, add_b=False, ID=3) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        log_sig_sq_y = networks.fc_network(hidden_post, self.weights, nonlinearity=None, add_b=False, ID=4) # pass through a fully connected network made of one ayer without additive weights b and non-linearity
        
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
        
        # we can make the weights for the convolutional network taking x:
        all_weights = networks.conv_2D_make_weights(all_weights,self.n_ch_x,self.N_x,self.fs_x, ID=0)
        
        # now we make the weights for the fully connected network taking z:
        all_weights = networks.fc_make_weights(all_weights, self.n_z, self.N_z, ID=1)
        
        # to initialise the weights for the last fully connected networ, we need to know its input size (size of hidden_post_z + size of hidden_post_x).
        # there is a function in "NN_utils" to compute the size of the images at each conv layer we can use:
        im_sizes = NN_utils.compute_size(self.x_siz,self.st_x) # this computes a length(st_x) x 2 array where each row is the dimensions of the images at each hhidden layer
        siz_hidden_post_x = im_sizes[-1,0]*im_sizes[-1,1]*self.N_x[-1] # the size of hidden_post_x is the last layer's height*width*n_channels
        dim_input = siz_hidden_post_x + self.N_z[-1] # the size of the input to the last layer is then the sum of the above and the dimensionality of the last fully connected layer that took z as input
        
        # now we can make the weights for the fully connected network taking the concatenated layers:
        all_weights = networks.fc_make_weights(all_weights, dim_input, self.N_comb, ID=2)
        
        # lastly, we initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_y], add_b=False, ID=3)
        all_weights = networks.fc_make_weights(all_weights, self.N_comb[-1], [self.n_y], add_b=False, ID=4)
        
        return all_weights