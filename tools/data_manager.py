import numpy as np
import scipy.io as sio

def load_holographic(n_test=1000):
    '''
    load the holographic experiiments data. all outputs are normalised such that each image is rescaled from 0 to 1.
    OPTIONAL INPUTS:
        n_test - number of examples in the experimental set to use as test set
    OUTPUTS:
        x_exp - the groundtruth images passed through the experiemental system
        y_exp_hf - the experimental measurements recorded with the camera
        y_exp_lf - simulated measurements through a Fourier transform for the above
        x_exp_test - the groundtruth images passed through the experiemental system to be used as test
        y_exp_hf_test - the experimental measurements recorded with the camera to be used as test
        y_exp_lf_test - simulated measurements through a Fourier transform for the above to be used as test
        x_sim - a larger set of MNIST examples
        y_sim_lf - simulated measurements through a Fourier transform for the larger MNIST set
    '''
    
    data = sio.loadmat('data/holographic_data.mat')
    
    x_exp = data['experimental_targets']
    x_exp = normalise_data_set(x_exp)
    y_exp_hf = data['experimental_measurements_hf']
    y_exp_hf = normalise_data_set(y_exp_hf)
    y_exp_lf = data['experimental_measurements_lf']
    y_exp_lf = normalise_data_set(y_exp_lf)
    x_sim = data['simulated_targets']
    x_sim = normalise_data_set(x_sim)
    y_sim_lf = data['simulated_measurements_lf'] 
    y_sim_lf = normalise_data_set(y_sim_lf)
    
    x_exp_test = x_exp[-n_test:,:]
    y_exp_hf_test = y_exp_hf[-n_test:,:]
    y_exp_lf_test = y_exp_lf[-n_test:,:]
    x_exp = x_exp[:-n_test,:]
    y_exp_hf = y_exp_hf[:-n_test,:]
    y_exp_lf = y_exp_lf[:-n_test,:]
    
    return x_exp, y_exp_hf, y_exp_lf, x_exp_test, y_exp_hf_test, y_exp_lf_test, x_sim, y_sim_lf
    
def get_params(n_iterations=100001, rate=0.0001, batch_sz=30, report_interval=100, save_int=2000, print_val=True):
    '''
    Makes a dictionary with the training parameters
    OPTIONAL INPUTS:
        n_iterations - number of iterations
        rate - the initial training rate for the ADAM optimiser 
        batch_sz - batch size
        report_inerval - intervals in iterations at which to evaluate the cost and optionally print it
        save_int - intervals in iterations at which to save the weights being trained
        print_val - if True prints the cost and other measures every report_interval
    OUTPUTS:
        params - dictionary of parameters
    '''
    params = dict(
        print_values=print_val, # optionally print values every report interval
        num_iterations=n_iterations, # number of iterations inference model (inverse reconstruction)
        initial_training_rate=rate, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=batch_sz, # batch size inference model (inverse reconstruction)
        report_interval=report_interval, # interval at which to save objective function values and optionally print info during inference training
        save_interval=save_int, # interval at which to save inference model weights
    )
    return params

def normalise_data_set(x_gt, SC=0):
    '''
    normalises a data set such that all samples are between 0 and 1
    INPUTS:
        x_gt - the unnormalised data set
    OPTIONAL INPUTS:
        SC - a small constant that can be added, if needed, to avoid dividing by zero
    OUTPUTS:
        x_gt_new - the normalised data set
    '''
    
    x_gt_new = np.zeros(np.shape(x_gt))
    for i in range(np.shape(x_gt)[0]):
        xi = x_gt[i,:]
        min_i = np.amin(xi)
        max_i = np.amax(xi)
        xi = (xi-min_i)/(SC+max_i-min_i)
        x_gt_new[i,:] = xi
            
    return x_gt_new
    
        