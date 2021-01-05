import numpy as np
import scipy.io as sio
from tools import training
from tools import testing
from tools import data_manager
from neural_networks import VAE

def run_example(train_forward_model=True, train_inverse_model=True):

    # Load the MNIST holographic data-set (28x28 version)
    x_exp, y_exp_hf, y_exp_lf, x_exp_test, y_exp_hf_test, y_exp_lf_test, x_sim, y_sim_lf = data_manager.load_holographic(n_test=10)
        
    # Define parameters for the optimisation of the forward model
    params_forward = data_manager.get_params(n_iterations=200001, rate=0.0002)
    
    # Build networks for multi-fidelity forward model
    
    # Dimensions
    n_x = np.shape(x_exp)[1] # dimensionality of the targets
    n_lf = np.shape(y_exp_lf)[1] # dimensionality of the low fidelity measurements
    n_hf = np.shape(y_exp_hf)[1] # dimensionality of the high fidelity measurements
    n_z = 30 # latent dimensionality
    
    # Encoder
    # Network architecture
    N_x_encoder = [700,500] # numbers of units for the layers propagating the targets to the encoder common layer
    N_lf_encoder = [700,500] # numbers of units for the layers propagating the low fidelity measurements to the encoder common layer
    N_hf_encoder = [700,500] # numbers of units for the layers propagating the high fidelity measurements to the encoder common layer
    N_encoder = [500,400,300,200,100] # numbers of units for the layers propagating the common layer to the latent space
    # Initialise the encoder
    encoder_fw = VAE.DoubleConditionalEncoder('encoder_fw', n_x, n_lf, n_hf, n_z, N_x_encoder, N_lf_encoder, N_hf_encoder, N_encoder)
    
    # Decoder
    # Network architecture
    N_x_decoder = [700,500] # numbers of units for the layers propagating the targets to the decoder common layer
    N_lf_decoder = [700,500] # numbers of units for the layers propagating the low fidelity measurements to the decoder common layer
    N_z_decoder = [40,60,80,100,150,200] # numbers of units for the layers propagating the latent variable to the decoder common layer
    N_decoder = [700,800] # numbers of units for the layers propagating the common layer to the high fidelity output
    # Initialise the encoder
    decoder_fw = VAE.DoubleConditionalDecoder('decoder_fw', n_x, n_lf, n_hf, n_z, N_x_decoder, N_lf_decoder, N_z_decoder, N_decoder)
    
    # Conditional Encoder
    N_x_encoder_c = [700,500] # numbers of units for the layers propagating the low fidelity measurements to the conditional encoder common layer
    N_lf_encoder_c = [700,500] # numbers of units for the layers propagating the high fidelity measurements to the conditional encoder common layer
    N_encoder_c = [500,400,300,200,100] # numbers of units for the layers propagating the common layer to the latent space
    # Initialise the conditional encoder
    encoder_c_fw = VAE.ConditionalEncoder('encoder_c_fw', n_x, n_lf, n_z, N_x_encoder_c, N_lf_encoder_c, N_encoder_c)
    
    if train_forward_model==True:
        
        # Train the multi-fidelity forward model
        cost_plot = training.forward_model(x_exp, y_exp_lf, y_exp_hf, encoder_fw, decoder_fw, encoder_c_fw, params_forward)
    
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
    sio.savemat('results/holographic_forward_model_examples.mat',results)   
     
    # Define parameters for the optimisation of the inverse model
    params_inverse = data_manager.get_params(n_iterations=500001, rate=0.00001)    
    
    # Build networks for inverse model
    
    # Encoder
    # Network architecture
    N_x_encoder = [700,500] # numbers of units for the layers propagating the targets to the encoder common layer
    N_hf_encoder = [700,500] # numbers of units for the layers propagating the high fidelity measurements to the encoder common layer
    N_encoder = [500,400,300,200,100] # numbers of units for the layers propagating the common layer to the latent space
    # Initialise the encoder
    encoder_inv = VAE.ConditionalEncoder('encoder_inv', n_hf, n_x, n_z, N_hf_encoder, N_x_encoder, N_encoder)
    
    # Decoder
    # Network architecture
    N_hf_decoder = [700,500] # numbers of units for the layers propagating the low fidelity measurements to the decoder common layer
    N_z_decoder = [40,60,80,100,150,200] # numbers of units for the layers propagating the latent variable to the decoder common layer
    N_decoder = [700,800] # numbers of units for the layers propagating the common layer to the high fidelity output
    # Initialise the encoder
    decoder_inv = VAE.ConditionalDecoder('decoder_inv', n_hf, n_x, n_z, N_hf_decoder, N_z_decoder, N_decoder)
    
    # Conditional Encoder
    N_encoder_c = [500,400,300,200,100] # numbers of units for the layers propagating the common layer to the latent space
    # Initialise the conditional encoder
    encoder_c_inv = VAE.Encoder('encoder_c_inv', n_x, n_z, N_encoder_c)
        
    if train_inverse_model==True:
        
        # Train the inverse model
        cost_plot = training.inverse_model(x_sim, y_sim_lf, encoder_inv, decoder_inv, encoder_c_inv, decoder_fw, encoder_c_fw, params_inverse)
        
    # Test the model by generating a few samples, mean and standard deviation
    samples, mu, sig = testing.inverse_model_test(y_exp_hf_test, decoder_inv, encoder_c_inv)
    
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
    sio.savemat('results/holographic_inverse_model_examples.mat',results)

run_example()