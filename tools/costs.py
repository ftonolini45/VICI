'''
Cost functions
'''

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

def kl_normal(mu_1,log_sig_sq_1,mu_2,log_sig_sq_2):
    '''
    Element-wise KL divergence between two normal distributions
    INPUTS:
        mu_1 - mean of firat distribution
        log_sig_sq_1 - log variance of first diatribution
        mu_2 - mean of second distribution
        log_sig_sq_2 - log variance of second diatribution
    OUTPUTS:
        KL - element-wise KL divergence
    '''
    
    v_mean = mu_2 #2
    aux_mean = mu_1 #1
    v_log_sig_sq = log_sig_sq_2 #2
    aux_log_sig_sq = log_sig_sq_1 #1
    v_log_sig = tfm.log(tfm.sqrt(tfm.exp(v_log_sig_sq))) #2
    aux_log_sig = tfm.log(tfm.sqrt(tfm.exp(aux_log_sig_sq))) #1
    KL = v_log_sig-aux_log_sig+tf.divide(tfm.exp(aux_log_sig_sq)+tfm.square(aux_mean-v_mean),2.0*tfm.exp(v_log_sig_sq))-0.5

    return KL

def np_kl_normal(mu_1,log_sig_sq_1,mu_2,log_sig_sq_2):
    '''
    Element-wise KL divergence between two normal distributions
    INPUTS:
        mu_1 - mean of firat distribution
        log_sig_sq_1 - log variance of first diatribution
        mu_2 - mean of second distribution
        log_sig_sq_2 - log variance of second diatribution
    OUTPUTS:
        KL - element-wise KL divergence
    '''
    
    v_mean = mu_2 #2
    aux_mean = mu_1 #1
    v_log_sig_sq = log_sig_sq_2 #2
    aux_log_sig_sq = log_sig_sq_1 #1
    v_log_sig = np.log(np.sqrt(np.exp(v_log_sig_sq))) #2
    aux_log_sig = np.log(np.sqrt(np.exp(aux_log_sig_sq))) #1
    KL = v_log_sig-aux_log_sig+np.divide(np.exp(aux_log_sig_sq)+np.square(aux_mean-v_mean),2.0*np.exp(v_log_sig_sq))-0.5

    return KL

def kl_unit(mu,log_sig_sq):
    '''
    Element-wise KL divergence between normal distributions and unit normal distribution
    INPUTS:
        mu - mean of distribution
        log_sig_sq - log variance of diatribution
    OUTPUTS:
        KL - element-wise KL divergence
    '''
    
    KL = -0.5*(1 + log_sig_sq - tf.square(mu) - tf.exp(log_sig_sq))

    return KL

def reparameterisation_trick(mu, log_sig_sq):
    '''
    Sample from Gaussian such that it stays differentiable
    INPUTS:
        mu - mean of distribution
        log_sig_sq - log variance of diatribution
    OUTPUTS:
        samp - sample from distribution
    '''
    
    eps = tf.random.normal([tf.shape(mu)[0], tf.shape(mu)[1]], 0, 1., dtype=tf.float32)
    samp = tfm.add(mu, tfm.multiply(tfm.sqrt(tfm.exp(log_sig_sq)), eps))
    
    return samp

def gaussian_log_likelihood(x,mu_x,log_sig_sq_x,SMALL_CONSTANT = 1e-5):
    '''
    Element-wise Gaussian log likelihood
    INPUTS:
        x = points
        mu_x - means of Gaussians
        log_sig_sq_x - log variance of Gaussian
    OPTIONAL INPUTS:
        SMALL_CONSTANT - small constant to avoid taking the log of 0 or dividing by 0
    OUTPUTS:
        log_lik - element-wise log likelihood
    '''
    
    # -E_q(z|x) log(p(x|z))
    normalising_factor = - 0.5 * tfm.log(SMALL_CONSTANT+tfm.exp(log_sig_sq_x)) - 0.5 * np.log(2.0 * np.pi)
    square_diff_between_mu_and_x = tfm.square(mu_x - x)
    inside_exp = -0.5 * tfm.divide(square_diff_between_mu_and_x,SMALL_CONSTANT+tfm.exp(log_sig_sq_x))
    log_lik = normalising_factor + inside_exp
    
    return log_lik

def np_gaussian_log_likelihood(x,mu_x, log_sig_sq_x, SMALL_CONSTANT = 1e-7):
    '''
    Element-wise Gaussian log likelihood
    INPUTS:
        x = points
        mu_x - means of Gaussians
        log_sig_sq_x - log variance of Gaussian
    OPTIONAL INPUTS:
        SMALL_CONSTANT - small constant to avoid taking the log of 0 or dividing by 0
    OUTPUTS:
        log_lik - element-wise log likelihood
    '''
    
    # -E_q(z|x) log(p(x|z))
    normalising_factor = - 0.5 * np.log(SMALL_CONSTANT+np.exp(log_sig_sq_x)) - 0.5 * np.log(2.0 * np.pi)
    square_diff_between_mu_and_x = np.square(mu_x - x)
    inside_exp = -0.5 * np.divide(square_diff_between_mu_and_x,SMALL_CONSTANT+np.exp(log_sig_sq_x))
    log_lik = normalising_factor + inside_exp
    
    return log_lik

def cvae_cost(x, y, encoder, decoder, encoder_c):
    '''
    Cost function for conditional VAE
    INPUTS:
        x - inputs/conditions
        y - outputs to be reconstructed
        encoder - the neural network to be used for mapping x and y to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z and x to mu_y and log_sig_sq_y
        encoder_c - the neural network to be used for mapping x to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
    OUTPUTS:
        cost - the cost function
    '''
    
    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    
    # compute moments of p(z|x)
    mu_cz, log_sig_sq_cz = encoder_c.compute_moments(x)
    
    # compute moments of q(z|x,y)
    mu_z, log_sig_sq_z = encoder.compute_moments(x,y)
    
    # sample from q(z|x,y)
    z = reparameterisation_trick(mu_z, log_sig_sq_z)
    
    # compute moments of p(y|z,x)
    mu_y, log_sig_sq_y = decoder.compute_moments(z,x)
    
    # KL(q(z|x,y)|p(z|x))
    KLe = kl_normal(mu_z,log_sig_sq_z,mu_cz,log_sig_sq_cz)
    KLc = tfm.reduce_sum(KLe,1)
    KL = tfm.reduce_mean(tf.cast(KLc,tf.float32))
    
    # -E_q(z|y,x) log(p(y|z,x))
    reconstr_loss = -tfm.reduce_sum(gaussian_log_likelihood(y,mu_y,log_sig_sq_y), 1)
    cost_R = tfm.reduce_mean(reconstr_loss)
    
    # -ELBO
    cost = cost_R + KL 
    
    return cost

def dual_cvae_cost(x, x2, y, encoder, decoder, encoder_c, constrain=True):
    '''
    Cost function for conditional VAE with two conditions
    INPUTS:
        x - inputs/conditions
        x2 - second inputs/conditions
        y - outputs to be reconstructed
        encoder - the neural network to be used for mapping x, x2 and y to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z, x and x2 to mu_y and log_sig_sq_y
        encoder_c - the neural network to be used for mapping x and x2 to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
    OUTPUTS:
        cost - the cost function
    '''
    
    x = tf.cast(x,tf.float32)
    x2 = tf.cast(x2,tf.float32)
    y = tf.cast(y,tf.float32)
    
    # compute moments of p(z|x)
    mu_cz, log_sig_sq_cz = encoder_c.compute_moments(x,x2)
    
    # compute moments of q(z|x,y)
    mu_z, log_sig_sq_z = encoder.compute_moments(y,x,x2)
    
    # sample from q(z|x,y)
    z = reparameterisation_trick(mu_z, log_sig_sq_z)
    
    # compute moments of p(y|z,x)
    mu_y, log_sig_sq_y = decoder.compute_moments(z,x,x2,constrain=constrain)
    
    # KL(q(z|x,y)|p(z|x))
    KLe = kl_normal(mu_z,log_sig_sq_z,mu_cz,log_sig_sq_cz)
    KLc = tfm.reduce_sum(KLe,1)
    KL = tfm.reduce_mean(tf.cast(KLc,tf.float32))
    
    # -E_q(z|y,x) log(p(y|z,x))
    reconstr_loss = -tfm.reduce_sum(gaussian_log_likelihood(y,mu_y,log_sig_sq_y), 1)
    cost_R = tfm.reduce_mean(reconstr_loss)
    
    # -ELBO
    cost = cost_R + KL 
    
    return cost

def forward_model(targets, low_fidelity, high_fidelity, encoder, decoder, encoder_c, difference=False):
    '''
    Cost function for multi-fidelity forward model
    INPUTS:
        targets - ground-truths to be eventually reconstructed with the inverse model
        low_fidelity - low-fidelity measurements simulated with some analytical forward model
        high_fidelity - measured high fidelity measurements to be reproduced
        encoder - the neural network to be used for mapping targets, low_fidelity and high_fidelity to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z, targets and low_fidelity to mu_hf and log_sig_sq_hf
        encoder_c - the neural network to be used for mapping targets and low_fidelity to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
    OPTIONAL INPUTS:
        difference - whether to model the high-fidelity directly (False) or the difference between high and low fidelity (True)
    OUTPUTS:
        cost - the cost function
    '''
    
    if difference==False:
        cost = dual_cvae_cost(targets, low_fidelity, high_fidelity, encoder, decoder, encoder_c)
    else:
        cost = dual_cvae_cost(targets, low_fidelity, high_fidelity-low_fidelity, encoder, decoder, encoder_c,constrain=False)
    
    return cost

def inverse_model(measurements, targets, encoder, decoder, encoder_c):
    '''
    Cost function for inverse model
    INPUTS:
        measurements - the measurements to infer from
        targets - ground-truths to be eventually reconstructed with the inverse model
        encoder - the neural network to be used for mapping measurements and targets to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z and measurements to mu_targets and log_sig_sq_targets
        encoder_c - the neural network to be used for mapping measurements to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
    OUTPUTS:
        cost - the cost function
    '''
    
    cost = cvae_cost(measurements, targets, encoder, decoder, encoder_c)
    
    return cost