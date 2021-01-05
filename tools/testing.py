'''
Testing Functions
'''

import tensorflow as tf
from tools import var_logger
from tools import costs
import numpy as np

def forward_model(targets, low_fidelity, decoder, encoder_c, load_dir='neural_networks/saved-weights/forward_model'):
    '''
    Function to run the multi-fidelity forward model
    INPUTS:
        targets - ground-truth targets
        low_fidelity - low-fidelity simulated measurements
        decoder - decoder of the forward model CVAE
        encoder_c - conditional encoder of the forward model CVAE
    OPTIONAL INPUTS:
        load_dir - directory from which to load the weights
    OUTPUTS:
        synthetic_measurements - sampled emulated measurements using the multi-fidelity forward model
    '''
    
    # Load the weights from the specified directory
    load_name_decoder = load_dir + '/dec.mat'
    load_name_encoder_c = load_dir + '/enc_c.mat'
    load_name_difference = load_dir + '/difference.mat'
    var_logger.restore_dict(load_name_decoder,decoder.weights)
    var_logger.restore_dict(load_name_encoder_c,encoder_c.weights)
    difference = var_logger.load_var(load_name_difference)
    
    if difference==True:
        constrain=False
    else:
        constrain=True
        
    x = tf.cast(targets,tf.float32)
    x2 = tf.cast(low_fidelity,tf.float32)
    
    # compute moments of p(z|x)
    mu_cz, log_sig_sq_cz = encoder_c.compute_moments(x,x2)
    
    # sample from q(z|x,y)
    z = costs.reparameterisation_trick(mu_cz, log_sig_sq_cz)
    
    # compute moments of p(y|z,x)
    mu_y, log_sig_sq_y = decoder.compute_moments(z,x,x2,constrain=constrain)
    
    # sample from p(y|z,x)
    y = costs.reparameterisation_trick(mu_y, log_sig_sq_y)
    
    synthetic_measurements = y.numpy()
    
    return synthetic_measurements 

def inverse_model(measurements, decoder, encoder_c, sample=False, load_dir='neural_networks/saved-weights/inverse_model'):
    '''
    Function to run the inverse model
    INPUTS:
        measurements - measurements to infer reconstructions from
        decoder - decoder of the inverse model CVAE
        encoder_c - conditional encoder of the inverse model CVAE
    OPTIONAL INPUTS:
        sample - if True, the samples are are sampled from p(x|z,y), if False the samples are the mean
        load_dir - directory from which to load the weights
    OUTPUTS:
        reconstruction_sample - sampled reconstruction
    '''
    
    x = tf.cast(measurements,tf.float32)
    
    # compute moments of p(z|x)
    mu_cz, log_sig_sq_cz = encoder_c.compute_moments(x)
    
    # sample from q(z|x,y)
    z = costs.reparameterisation_trick(mu_cz, log_sig_sq_cz)
    
    # compute moments of p(y|z,x)
    mu_y, log_sig_sq_y = decoder.compute_moments(z,x)
    
    if sample==True:
        # sample from p(y|z,x)
        y = costs.reparameterisation_trick(mu_y, log_sig_sq_y)
    else:
        y = mu_y
    
    reconstruction_sample = y.numpy()
    
    return reconstruction_sample

def forward_model_test(targets, low_fidelity, decoder, encoder_c, n_samp=10, load_dir='neural_networks/saved-weights/forward_model'):
    '''
    Function to run the multi-fidelity forward model several times and return samples along with mean and std
    INPUTS:
        measurements - measurements to infer reconstructions from
        decoder - decoder of the inverse model CVAE
        encoder_c - conditional encoder of the inverse model CVAE
    OPTIONAL INPUTS:
        n_samp = number of samples to compute
        load_dir - directory from which to load the weights
    OUTPUTS:
        INV_samples - array of samples from the inverse model (samples are along dim 2)
        mu - means of the samples
        sig - standard deviation of the samples
    '''
    
    for i in range(n_samp):
        
        fm_samples = forward_model(targets, low_fidelity, decoder, encoder_c, load_dir=load_dir)
        
        if i==0:
            FM_samples = np.zeros((np.shape(fm_samples)[0],np.shape(fm_samples)[1],n_samp))
        
        FM_samples[:,:,i] = fm_samples
    
    mu = np.mean(FM_samples,axis=2)
    sig = np.std(FM_samples,axis=2)
    
    return FM_samples, mu, sig

def inverse_model_test(measurements, decoder, encoder_c, n_samp=10, sample=False, load_dir='neural_networks/saved-weights/inverse_model'):
    '''
    Function to run the multi-fidelity forward model several times and return samples along with mean and std
    INPUTS:
        targets - ground-truth targets
        low_fidelity - low-fidelity simulated measurements
        decoder - decoder of the forward model CVAE
        encoder_c - conditional encoder of the forward model CVAE
    OPTIONAL INPUTS:
        n_samp = number of samples to compute
        sample - if True, the samples are are sampled from p(x|z,y), if False the samples are the mean
        load_dir - directory from which to load the weights
    OUTPUTS:
        INV_samples - array of samples from the forward model (samples are along dim 2)
        mu - means of the samples
        sig - standard deviation of the samples
    '''
    
    for i in range(n_samp):
        
        inv_samples = inverse_model(measurements, decoder, encoder_c, sample=sample, load_dir=load_dir)
        
        if i==0:
            INV_samples = np.zeros((np.shape(inv_samples)[0],np.shape(inv_samples)[1],n_samp))
        
        INV_samples[:,:,i] = inv_samples
    
    mu = np.mean(INV_samples,axis=2)
    sig = np.std(INV_samples,axis=2)
    
    return INV_samples, mu, sig
    

def classify(x, NN, load_dir='neural_networks/saved-weights/classifier'):
    '''
    Function to classify input x using pre-trained classifier NN
    INPUTS:
        x - inputs
        Ax - mask of missings in input
        NN - classifier model
    OPTIONAL INPUTS:
        load_dir - directory from which to load the classifier weights
    OUTPUTS:
        mu_y - generated output mean
        y_samp - sample of generated output
    '''
    
    # Load the weights from the specified directory
    load_name = load_dir + '/W'
    var_logger.restore_dict(load_name,NN.weights)
    
    x = tf.cast(x,tf.float32)
    
    # SMALL_CONSTANT = 1e-6
        
    # p(y|x)
    py = NN.compute_py(x)
    py = py.numpy()
    
    return py