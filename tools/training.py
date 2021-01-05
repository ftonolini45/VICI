'''
Training Functions
'''

import numpy as np
import tensorflow as tf
from tools import var_logger
from tools import costs
from tools import testing

def forward_model(targets, low_fidelity, high_fidelity, encoder, decoder, encoder_c, params, difference=False, save_dir='neural_networks/saved-weights/forward_model'):
    '''
    Training function for multi-fidelity forward model. Trains the model and saves the trained weights.
    INPUTS:
        targets - ground-truths to be eventually reconstructed with the inverse model in the format [n_samples,dimensions]
        low_fidelity - low-fidelity measurements simulated with some analytical forward model in the format [n_samples,dimensions]
        high_fidelity - measured high fidelity measurements to be reproduced in the format [n_samples,dimensions]
        encoder - the neural network to be used for mapping targets, low_fidelity and high_fidelity to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z, targets and low_fidelity to mu_hf and log_sig_sq_hf
        encoder_c - the neural network to be used for mapping targets and low_fidelity to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
        params - optimisation parameters (see examples)
    OPTIONAL INPUTS:
        difference - whether to model the high-fidelity directly (False) or the difference between high and low fidelity (True)
        save dir - directory in which to save the trained multi-fidelity forward model weights (defaults to neural_networks/saved-weights/forward_model)
    OUTPUTS:
        cost_plot - batch training cost as a function of iterations
    '''
    
    # Initialise the array in which to save the cost plot
    cost_plot = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1))
    ni = -1
    
    # ADAM optimiser initialisation
    optimizer = tf.keras.optimizers.Adam(params['initial_training_rate'])
    
    # Start iterations
    for i in range(params['num_iterations']):
        
        # indices for the next batch
        next_indices = np.random.random_integers(np.shape(targets)[0],size=(params['batch_size']))-1
        
        # take inputs for the next batch
        tn = targets[next_indices, :]
        hfn = high_fidelity[next_indices, :]
        lfn = low_fidelity[next_indices, :]
        
        # Optimisation Step
        cost = lambda: costs.forward_model(tn, lfn, hfn, encoder, decoder, encoder_c, difference=difference)
        optimizer.minimize(cost,var_list = [encoder.weights,decoder.weights,encoder_c.weights])

        # Compute cost value over the whole training set every "report_interval" iterations
        if i % params['report_interval'] == 0:
                ni = ni+1
                
                # Compute cost
                cost = costs.forward_model(tn, lfn, hfn, encoder, decoder, encoder_c, difference=difference)
                
                # Put cost value in the plot (making sure it is a numpy and not tf)
                cost_plot[ni] = cost.numpy()
                
                # Print out values if needed
                if params['print_values']==True:
                    print(end='\r')
                    print('Training multi-fidelity forward model, Iteration:',i,'/',params['num_iterations'],', training batch cost value:',cost.numpy(), end="\r")
                    
                # Stop if we get a numerical evaluation error
                if np.isnan(cost)==True:
                    print('NaN Error! Iteration:',i,'/',params['num_iterations'])
                    break
        
        # Save weights every so many iterations into the weights dictionary           
        if i % params['save_interval'] == 0:
            save_name_encoder = save_dir + '/enc.mat'
            save_name_decoder = save_dir + '/dec.mat'
            save_name_encoder_c = save_dir + '/enc_c.mat'
            save_name_difference = save_dir + '/difference.mat'
            var_logger.save_dict(save_name_encoder,encoder.weights)
            var_logger.save_dict(save_name_decoder,decoder.weights)
            var_logger.save_dict(save_name_encoder_c,encoder_c.weights)
            var_logger.save_var(save_name_difference,difference)
                
    return cost_plot

def inverse_model(targets, low_fidelity, encoder, decoder, encoder_c, fm_decoder, fm_encoder_c, params, experimental_measurements=None, experimental_targets=None, load_dir='neural_networks/saved-weights/forward_model', save_dir='neural_networks/saved-weights/inverse_model'):
    '''
    Training function for inverse model. Trains the model and saves the trained weights.
    INPUTS:
        targets - ground-truths to be eventually reconstructed with the inverse model in the format [n_samples,dimensions]
        low_fidelity - low-fidelity measurements simulated with some analytical forward model in the format [n_samples,dimensions]
        encoder - the neural network to be used for mapping measurements and targets to mu_z and log_sig_sq_z
        decoder - the neural network to be used for mapping z and measurements to mu_targets and log_sig_sq_targets
        encoder_c - the neural network to be used for mapping measurements to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
        fm_decoder - the forward model decoder neural network to be used for mapping z, targets and low_fidelity to mu_hf and log_sig_sq_hf
        fm_encoder_c - the forward model encoder neural network to be used for mapping targets and low_fidelity to mu_cz and log_sig_sq_cz (conditional prior distribution in the latent space)
        params - optimisation parameters (see examples)
    OPTIONAL INPUTS:
        experimental_measurements - experimental measurements to be optionally included alongside forward model-emulated ones in training in the format [n_samples,dimensions]
        experimental_targets - ground truth targets corresponding to the bove measurements in the format [n_samples,dimensions]
        save_dir - directory in which to save the trained inverse model weights (defaults to 'neural_networks/saved-weights/inverse_model')
        load_dir - directory from which to load the trained inverse model weights (defaults to 'neural_networks/saved-weights/forward_model')
    OUTPUTS:
        cost_plot - batch training cost as a function of iterations
    '''
    
    if experimental_measurements:
        n_sim = np.shape(targets)[0]
        targets = np.concatenate((targets,experimental_targets),axis=0)
    
    # Initialise the array in which to save the cost plot
    cost_plot = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1))
    ni = -1
    
    # ADAM optimiser initialisation
    optimizer = tf.keras.optimizers.Adam(params['initial_training_rate'])
    
    # Start iterations
    for i in range(params['num_iterations']):
        
        # indices for the next batch
        next_indices = np.random.random_integers(np.shape(targets)[0],size=(params['batch_size']))-1
        
        # take inputs for the next batch
        tn = targets[next_indices, :]
        
        if experimental_measurements:
            if experimental_targets==None:
                print('Error: to use experimental measurements you must provide associated experimental targets in the input "experimental_targets"')
                break
                
            tn_sim = tn[next_indices<n_sim,:]
            tn_real = tn[next_indices>n_sim,:]
            next_indeces_real = next_indices[next_indices>n_sim]-n_sim
            next_indeces_sim = next_indices[next_indices<n_sim]
            mn_real = experimental_measurements[next_indeces_real,:]
            lfn_sim = low_fidelity[next_indeces_sim,:]
            mn_sim = testing.forward_model(tn, lfn_sim, fm_decoder, fm_encoder_c, load_dir=load_dir)
            tn = np.concatenate((tn_sim,tn_real),axis=0)
            mn = np.concatenate((mn_sim,mn_real),axis=0)
        else:
            lfn = low_fidelity[next_indices,:]
            mn = testing.forward_model(tn, lfn, fm_decoder, fm_encoder_c, load_dir=load_dir)

        # Optimisation Step
        cost = lambda: costs.inverse_model(mn, tn, encoder, decoder, encoder_c)
        optimizer.minimize(cost,var_list = [encoder.weights,decoder.weights,encoder_c.weights])

        # Compute cost value over the whole training set every "report_interval" iterations
        if i % params['report_interval'] == 0:
                ni = ni+1
                
                # Compute cost
                cost = costs.inverse_model(mn, tn, encoder, decoder, encoder_c)
                
                # Put cost value in the plot (making sure it is a numpy and not tf)
                cost_plot[ni] = cost.numpy()
                
                # Print out values if needed
                if params['print_values']==True:
                    print(end='\r')
                    print('Training inverse model, Iteration:',i,'/',params['num_iterations'],', training batch cost value:',cost.numpy(), end="\r")
                    
                # Stop if we get a numerical evaluation error
                if np.isnan(cost)==True:
                    print('NaN Error! Iteration:',i,'/',params['num_iterations'])
                    break
        
        # Save weights every so many iterations into the weights dictionary           
        if i % params['save_interval'] == 0:
            save_name_encoder = save_dir + '/enc.mat'
            save_name_decoder = save_dir + '/dec.mat'
            save_name_encoder_c = save_dir + '/enc_c.mat'
            var_logger.save_dict(save_name_encoder,encoder.weights)
            var_logger.save_dict(save_name_decoder,decoder.weights)
            var_logger.save_dict(save_name_encoder_c,encoder_c.weights)
                
    return cost_plot