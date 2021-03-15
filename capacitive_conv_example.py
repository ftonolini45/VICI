import numpy as np
import scipy.io as sio
from tools import training
from tools import testing
from tools import data_manager
from neural_networks import VAE_capacitive as VAE

def run_example(train_forward_model=True, train_inverse_model=True):

    with np.load("data/CapSensFrancescoData.npz") as data:
        x_exp = data['x_data_train_h']    # hand pose N x 5
        y_exp_hf = data['y_data_train_h']    # cap sense  N x 160 (16 x 10 images)
        y_exp_lf = data['y_data_train_lh']  # simulated capsense N x 160 (16 x 10 images)
        test_ind = [1101, 10037, 999, 1905, 2000, 5666]
        x_exp_test = data['x_data_test_h'][test_ind,:]
        y_exp_hf_test = data['y_data_test_h'][test_ind,:]
        y_exp_lf_test = data['y_data_test_lh'][test_ind,:]
        x_sim = data['x_data_train']        # these two are larger collections for training inverse
        y_sim_lf = data['y_data_train_l']    # simulated outputs

    # Define parameters for the optimisation of the forward model
    params_forward = data_manager.get_params(n_iterations=100001, rate=0.01, batch_sz=32)
    
    # Build networks for multi-fidelity forward model
    
    # Dimensions
    n_x = np.shape(x_exp)[1] # dimensionality of the targets
    n_ch_lf = 1 # channels in low fidelity measurements
    siz_lf = [16,10] # image size of low fidelity measurements
    n_ch_hf = 1 # channels in high fidelity measurements
    siz_hf = [16,10] # image size of high fidelity measurements
    n_w = 20 # latent dimensionality
    
    # Encoder
    encoder_fw = VAE.ForwardModel_Encoder('encoder_fw', n_x, n_ch_hf, n_ch_lf, siz_hf, siz_lf, n_w)
    
    # Decoder
    decoder_fw = VAE.ForwardModel_Decoder('decoder_fw', n_x, n_w, n_ch_lf, siz_lf,  n_ch_hf, siz_hf)
    
    # Conditional Encoder
    encoder_c_fw = VAE.ForwardModel_ConditionalEncoder('encoder_c_fw', n_x, n_ch_lf, siz_lf, n_w)
    
    if train_forward_model==True:
        
        # Train the multi-fidelity forward model
        cost_plot = training.forward_model(x_exp, y_exp_lf, y_exp_hf, encoder_fw, decoder_fw, encoder_c_fw, params_forward, warm_up=True, wu_start=10000, wu_end=20000)
    
    # Test the model by generating a few samples, mean and standard deviation
    samples, mu, sig = testing.forward_model_test(x_exp_test, y_exp_lf_test, decoder_fw, encoder_c_fw)
    
    # Save the results in a .mat file
    results = {}
    results['target'] = x_exp_test
    results['low_fidelity'] = y_exp_lf_test
    results['ground_truth'] = y_exp_hf_test
    results['samples'] = samples
    results['mean'] = mu
    results['standard_deviation'] = sig
    if train_forward_model==True:
        results['cost'] = cost_plot
    else:
        results['cost'] = 0
    sio.savemat('results/capacitive_forward_model_examples.mat',results)   

    # Define parameters for the optimisation of the inverse model
    params_inverse = data_manager.get_params(n_iterations=100001, rate=0.0001, batch_sz=32)    
    
    # Build networks for inverse model
    n_z = 20 # latent dimensionality
    
    # Encoder
    encoder_inv = VAE.InverseModel_Encoder('encoder_inv', n_x, n_ch_hf, siz_hf, n_z)
    
    # Decoder
    decoder_inv = VAE.InverseModel_Decoder('decoder_inv', n_z, n_ch_hf, siz_hf, n_x)
    
    # Conditional Encoder
    encoder_c_inv = VAE.InverseModel_ConditionalEncoder('encoder_c_inv', n_ch_hf, siz_hf, n_z)
        
    if train_inverse_model==True:
        
        # Train the inverse model
        cost_plot = training.inverse_model(x_sim, y_sim_lf, encoder_inv, decoder_inv, encoder_c_inv, decoder_fw, encoder_c_fw, params_inverse, warm_up=True, wu_start=10000, wu_end=20000)
        
    # Test the model by generating a few samples, mean and standard deviation
    samples, mu, sig = testing.inverse_model_test(y_exp_hf_test, decoder_inv, encoder_c_inv, sample=True)
    
    # Save the results in a .mat file
    results = {}
    results['target'] = x_exp_test
    results['measurements'] = y_exp_hf_test
    results['samples'] = samples
    results['mean'] = mu
    results['standard_deviation'] = sig
    if train_inverse_model==True:
        results['cost'] = cost_plot
    else:
        results['cost'] = 0
    sio.savemat('results/capacitive_inverse_model_examples.mat',results)
    
run_example()