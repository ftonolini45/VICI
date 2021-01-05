import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Plot Forward Model Results

n_plots = 3 # number of examples to plot
samp_index = [0,1,2,3]

# Load the test results for the multi-fidelity forward model
results = sio.loadmat('results/holographic_forward_model_examples.mat')
x_exp_test = results['target']
y_exp_lf_test = results['low_fidelity']
y_exp_hf_test = results['ground_truth']
samples = results['samples']
samples[samples<0.0]=0.0
mu = results['mean']
sig = results['standard_deviation']
cost_plot = results['cost'][0]

plt.figure()
plt.plot(cost_plot)
plt.title('Multi-Fidelity Forward Model Cost')

plt.figure()
for i in range(n_plots):
    
    target_i = x_exp_test[i,:]
    target_i = np.reshape(target_i,(28,28))
    
    lf_i = y_exp_lf_test[i,:]
    lf_i = np.reshape(lf_i,(28,28))
    
    hf_i = y_exp_hf_test[i,:]
    hf_i = np.reshape(hf_i,(28,28))
    
    mu_i = mu[i,:]
    mu_i = np.reshape(mu_i,(28,28))
    
    sig_i = sig[i,:]
    sig_i = np.reshape(sig_i,(28,28))
    
    samp_i = samples[i,:,:]
    samples_i = np.zeros((28*2,28*2))
    for j in range(4):
        samp_ij = samp_i[:,samp_index[j]]
        samp_ij = np.reshape(samp_ij,(28,28))
        if j<2:
            samples_i[j*28:(j+1)*28,0:28] = samp_ij
        else:
            samples_i[(j-2)*28:(j-1)*28,28:2*28] = samp_ij

    plt.subplot(n_plots,6,i*6+1)
    plt.imshow(target_i)
    plt.title('Target')
    
    plt.subplot(n_plots,6,i*6+2)
    plt.imshow(lf_i)
    plt.title('Low Fidelity Measurements')
    
    plt.subplot(n_plots,6,i*6+3)
    plt.imshow(hf_i)
    plt.title('High Fidelity Measurements')
    
    plt.subplot(n_plots,6,i*6+4)
    plt.imshow(mu_i)
    plt.title('Emulated Mean')
    
    plt.subplot(n_plots,6,i*6+5)
    plt.imshow(sig_i)
    plt.title('Emulated STD')
    
    plt.subplot(n_plots,6,i*6+6)
    plt.imshow(samples_i)
    plt.title('Emulated Samples')
    
# Plot Inverse Model Results

n_plots = 3 # number of examples to plot
samp_index = [0,1,2,3]

# Load the test results for the multi-fidelity forward model
results = sio.loadmat('results/holographic_inverse_model_examples.mat')
x_exp_test = results['target']
y_exp_hf_test = results['measurements']
samples = results['samples']
mu = results['mean']
sig = results['standard_deviation']
cost_plot = results['cost'][0]

plt.figure()
plt.plot(cost_plot)
plt.title('Inverse Model Model Cost')

plt.figure()
for i in range(n_plots):
    
    target_i = x_exp_test[i,:]
    target_i = np.reshape(target_i,(28,28))
    
    hf_i = y_exp_hf_test[i,:]
    hf_i = np.reshape(hf_i,(28,28))
    
    mu_i = mu[i,:]
    mu_i = np.reshape(mu_i,(28,28))
    
    sig_i = sig[i,:]
    sig_i = np.reshape(sig_i,(28,28))
    
    samp_i = samples[i,:,:]
    samples_i = np.zeros((28*2,28*2))
    for j in range(4):
        samp_ij = samp_i[:,samp_index[j]]
        samp_ij = np.reshape(samp_ij,(28,28))
        if j<2:
            samples_i[j*28:(j+1)*28,0:28] = samp_ij
        else:
            samples_i[(j-2)*28:(j-1)*28,28:2*28] = samp_ij

    plt.subplot(n_plots,5,i*5+1)
    plt.imshow(target_i)
    plt.title('Target')
    
    plt.subplot(n_plots,5,i*5+2)
    plt.imshow(hf_i)
    plt.title('High Fidelity Measurements')
    
    plt.subplot(n_plots,5,i*5+3)
    plt.imshow(mu_i)
    plt.title('Reconstructed Mean')
    
    plt.subplot(n_plots,5,i*5+4)
    plt.imshow(sig_i)
    plt.title('Reconstructed STD')
    
    plt.subplot(n_plots,5,i*5+5)
    plt.imshow(samples_i)
    plt.title('Reconstructed Samples')
    
        