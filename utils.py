import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import random
import torch
import torch.nn as nn
import torchvision
import sys
import matplotlib.pyplot as plt

def seed_everything(seed = 42):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_filtering(noisy_signal, sigma = 4):
    return gaussian_filter1d(noisy_signal, sigma)
    

def add_outliers(data, percentage):
    """
    Adds outliers to the data.
    
    Parameters:
        data (np.ndarray): The original data array.
        percentage (float): The percentage of data points to replace with outliers (0 to 100).
        
    Returns:
        np.ndarray: The data array with outliers added.
    """
    # Ensure percentage is valid
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    
    # Calculate the number of outliers to introduce
    n_outliers = int(len(data) * (percentage / 100))
    
    # Select random indices to replace with outliers
    outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
    
    # Generate random outlier values between 0 and 1
    outliers = np.random.uniform(0, 1, n_outliers)
    
    # Copy the original data to avoid modifying it in-place
    data_with_outliers = data.copy()
    
    # Replace selected indices with outliers
    data_with_outliers[outlier_indices] = outliers
    
    return data_with_outliers


def min_max(array, feature_range=(0, 1)):
    """
    Normalize a NumPy array using min-max scaling.

    Parameters:
        array (numpy.ndarray): Input array to normalize.
        feature_range (tuple): Desired range of transformed data (min, max).
    
    Returns:
        numpy.ndarray: Normalized array.
    """
    min_val, max_val = feature_range
    array_min = np.min(array)
    array_max = np.max(array)

    # Handle edge case where all values in the array are the same
    if array_min == array_max:
        return np.full_like(array, min_val)

    # Perform min-max normalization
    normalized = (array - array_min) / (array_max - array_min)  # Scale to [0, 1]
    normalized = normalized * (max_val - min_val) + min_val    # Scale to [min_val, max_val]

    return normalized



def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False

def guided_input(signal_noisy_np, device):

    return torch.from_numpy(gaussian_filtering(signal_noisy_np)).unsqueeze(0).unsqueeze(0).to(device,dtype=torch.float32)


def snr(signal, estimate):
    noise = signal - estimate
    signal_power = np.sum(signal**2)
    noise_power = np.sum(noise**2)
    return 10 * np.log10(signal_power / noise_power)

def rmse(signal, estimate):
    return np.sqrt(np.mean((signal - estimate) ** 2))

def mae(signal, estimate):
    return np.mean(np.abs(signal - estimate))