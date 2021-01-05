'''
Functions to save and load weights as Matlab arrays from tensorflow2 sessions
'''

import scipy.io as sio
import tensorflow as tf

def save_dict(save_dir,D):
    '''
    Save function for dictionaries
    INPUTS: 
        save_dir - directory and file name (including extension) in which to save the weights
        D - tensorflow2 dictionary of weights
    '''
    
    DN = {}
    for key, value in D.items():
        DN[key] = value.numpy()
    
    sio.savemat(save_dir,DN)
    
def restore_dict(load_dir,D):
    '''
    Load function for dictionaries
    INPUTS: 
        load_dir - directory and file name (including extension) from which to load the weights
        D - tensorflow2 dictionary of weights
    '''
    
    DS = sio.loadmat(load_dir)
    
    for key, value in D.items():
        D[key] = tf.Variable(tf.constant(DS[key]), dtype=tf.float32)
        
def save_var(save_dir,v):
    '''
    Save function for variables
    INPUTS: 
        save_dir - directory and file name (including extension) in which to save the variable
        v - variable
    '''
    
    V = {}
    V['v'] = v
    
    sio.savemat(save_dir,V)
    
def load_var(load_dir):
    '''
    Load function for variables
    INPUTS: 
        load_dir - directory and file name (including extension) from which to load the variable
    OUTPUTS:
        v - variable
    '''
    
    V = sio.loadmat(load_dir)
    v = V['v']
    
    return v