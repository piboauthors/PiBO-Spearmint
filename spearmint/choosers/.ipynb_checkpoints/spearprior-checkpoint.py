import sys
import os
from scipy.stats import norm
import numpy as np
import pandas as pd

class GaussianKDE():
    
    def __init__(self, data, bandwidth=False, one_dim=True):

        # create pdfs centered at different points in the input space
        self.data = data
        if not bandwidth:
            self.bandwidth = self.data.shape[0]**(-1./(1 + 4))
        else:
            self.bandwidth = bandwidth
            
        self.dists = np.array([norm(data[:, dim] / self.bandwidth) for dim in range(self.data.shape[1])])
        self.norm_factor = self.bandwidth * self.data.shape[0]
        
    def __call__(self, X, compute_grad=True, sep_dim=False):
        dim_probs = np.zeros(shape=X.shape)
        dim_derivs = np.zeros(shape=X.shape)
        for i, point in enumerate(X):
            for dim in range(self.data.shape[1]):
                dim_probs[i, dim] = np.sum(self.dists[dim].pdf(point[dim] / self.bandwidth))
                dim_derivs[i, dim] = -1 * np.sum((point[dim]-self.data[:,dim])/self.bandwidth * self.dists[dim].pdf(point[dim] / self.bandwidth))
        dim_probs /= self.norm_factor
        
        total_prob = np.ones(X.shape[0])
        total_derivs = np.ones(X.shape)
        
        for i, (dim_prob, dim_deriv) in enumerate(zip(dim_probs.T, dim_derivs.T)):
            total_prob *= dim_prob
            for dim in range(self.data.shape[1]):
                if dim == i:
                    total_derivs[:, dim] *= dim_deriv
                else:
                    total_derivs[:, dim] *= dim_prob 
        if sep_dim:
            return_prob = dim_probs
        else:
            return_prob = total_prob
            
        dim_derivs /= self.norm_factor
        if compute_grad:
            return return_prob, total_derivs
        return return_prob
    

def compute_prior_acq(model, pred, prior, acq_function, compute_grad, **kwargs):
    if not compute_grad:
        acq = acq_function(model, pred, compute_grad, **kwargs)
        p_values = prior(pred, compute_grad)
        return acq * p_values
    else:
        acq, acq_grad = acq_function(model, pred, compute_grad, **kwargs)
        p_values, p_grad = prior(pred, compute_grad)
        #print('Shape of EI:', ei.shape)
        #print('Shape of EI gradient:', ei_grad.shape)
        #print('Shape of prior:', p_values.shape)
        #print('Shape of prior gradient:', p_grad.shape)
        acq_values = acq * p_values
        acq_gradients = acq * p_grad +  acq_grad * p_values
        return acq_values, acq_gradients
    