import sys
import os
from os.path import dirname, join, abspath
from scipy.stats import norm
import numpy as np
import pandas as pd
from choosers.acquisition_functions import compute_ei
from time import time
import json

import numpy as np
from scipy.stats import gaussian_kde, beta, norm, beta, f, lognorm
from collections import OrderedDict

class Prior:
    
    def __init__(self, 
                 distributions_dict=None,
                 use_log_evaluation=True # not implemented yet
                ):
        
        # needed to be able to sample from distribution - used in the sample function
        self.estimated_pdf = []
        self.prior_floor = 1e-12
                    
        # needed to be able to efficiently call the function - one per dimension, used in __call__
        self.evaluate_functions = []
        self.ranges = []
        for key, value in distributions_dict.items():
            self.ranges.append(distributions_dict[key]['range'])
        self.ranges = np.array(self.ranges)
        self.distributions_dict = distributions_dict
        self.dims = len(distributions_dict.keys())

        self.means = {}
        self.stds = {}
        self.types = {}
        modes = {}
        # define lambda function (sum of univariates) for each dim
        # loop through the variables in the dict
        self.evaluate_functions = [None] * self.dims
        for i, (key, variable_data) in enumerate(distributions_dict.items()):
            self.means[key] = distributions_dict[key]['params']['mean']
            self.stds[key] = np.array(variable_data['params']['std'])
            
            self.types[key] = variable_data.get('type', 'gaussian')
            
            # this is what would have to change is we were to provide the prior beforehand - which makes sense
            if self.types[key] == 'gaussian':
                self.estimated_pdf.append([norm(mean, std) for mean, std in zip(self.means[key], self.stds[key])]) # the mean and std come in as an array with one element, an old relic from when we had a three-peaked branin prior
                self.evaluate_functions[i] = lambda x, means, stds: np.sum([norm(mean, std).pdf(x) for mean, std in zip(means, stds)], axis=0) / len(means)
                modes[key] = self.means[key][0]
            elif self.types[key] == 'categorical':
                self.estimated_pdf.append([norm(mean, std) for mean, std in zip(self.means[key], self.stds[key])]) # the mean and std come in as an array with one element, an old relic from when we had a three-peaked branin prior
                self.evaluate_functions[i] = lambda x, means, stds: np.sum([norm(mean, std).pdf(x) for mean, std in zip(means, stds)], axis=0) / len(means)
                modes[key] = variable_data['params']['mean'][0]
            elif self.types[key] == 'beta':
                self.estimated_pdf.append([beta(mean, std) for mean, std in zip(self.means[key], self.stds[key])]) # the mean and std come in as an array with one element, an old relic from when we had a three-peaked branin prior
                self.evaluate_functions[i] = lambda x, means, stds: np.sum([beta(mean, std).pdf(x) for mean, std in zip(means, stds)], axis=0) / len(means)
                modes[key] = distributions_dict[key]['params']['mode'][0]
            elif self.types[key] == 'lognorm':
                self.estimated_pdf.append([lognorm(mean, 0, std) for mean, std in zip(self.means[key], self.stds[key])]) # the mean and std come in as an array with one element, an old relic from when we had a three-peaked branin prior
                self.evaluate_functions[i] = lambda x, means, stds: np.sum([lognorm(mean, 0, std).pdf(x) for mean, std in zip(means, stds)], axis=0) / len(means)
                modes[key] = distributions_dict[key]['params']['mode'][0]
            else:
                raise OSError('Type of prior does not exist')
        
        mean_array = []
        for key in sorted(self.means.keys()):
            mean_array.append(modes[key])

        self.max_location = np.array(mean_array).reshape(1, -1)
        normalized_max_location = (self.max_location - self.ranges[:, 0]) / (self.ranges[:, 1] - self.ranges[:, 0])
        self.max = self(normalized_max_location)
        self.modes = np.array([])
        self.std_array = np.array([])
        self.dists_per_dim = np.array([0])

        for i, key in enumerate(sorted(self.means.keys())):
            
            self.modes = np.append(self.modes, modes[key])
            self.std_array = np.append(self.std_array, self.stds[key])
            self.dists_per_dim = np.append(self.dists_per_dim, len(self.means[key]))
        self.dists_per_dim = np.cumsum(self.dists_per_dim)

    def get_max_location(self):
        return self.modes
    
    def get_pdfs(self):
        return self.evaluate_functions
    
    def sample(self, size, normalize=True):
        oversampling_factor = 20000
        samples = np.zeros((size*oversampling_factor, self.dims))
            
        # for each dimension
        # random.choice on which pdf to use for each sample
        # for each pdf
        # sample the size required and store in the array
        for dim, pdfs in enumerate(self.estimated_pdf):
            pdf_choice = np.random.choice(len(pdfs), size=len(samples))
            for pdf_idx, pdf in enumerate(pdfs):
                n_samples_from_pdf = np.sum(pdf_choice == pdf_idx)
                samples[pdf_choice == pdf_idx, dim] = pdf.rvs(size=n_samples_from_pdf)
                
        in_bounds = np.array([True] * len(samples))
        
        # normalize samples to return values in 0, 1
        norm_samples = np.zeros(shape=(samples.shape))
        for dim, config in enumerate(self.distributions_dict.values()):
            lower, upper = config['range']
            if config.get('type', '') == 'categorical':
                    samples[:, dim] = samples[:, dim].round()
                    
            param_in_bounds = (samples[:, dim] >= lower) & (samples[:, dim] <= upper)
            in_bounds = param_in_bounds & in_bounds
            norm_samples[:, dim] = (samples[:, dim] - lower) / (upper - lower) 

        samples_in_bounds = samples[in_bounds]
        norm_samples = norm_samples[in_bounds]
        for dim, config in enumerate(self.distributions_dict.values()):
            if config.get('reverse', False):
                lower, upper = config['range']
                samples[:, dim] = upper - samples[:, dim]
                norm_samples[:, dim] = 1 - norm_samples[:, dim]
        
        if normalize:
            return norm_samples[in_bounds][0:size]
        return samples[in_bounds][0:size]
    
    def __call__(self, X, compute_grad=False):

        # everything comes in in range (0,1), and then gets scaled up to the proper range of the function
        X_scaled = np.zeros(X.shape)
        result_scaling = 1
        for i, X_col in enumerate(X.T):
            X_scaled[:, i] = X_col * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0]
            result_scaling *= (self.ranges[i][1] - self.ranges[i][0])
        
        # if multivariate - compute in one manner
        # if not - compute across dimensions and return (assume independence between dims)
        probabilities = np.ones(len(X))
        grads = np.ones((len(X), self.dims))
        # dimension-wise multiplication of the probabilities
        for i, key in enumerate(self.distributions_dict.keys()):
            if self.distributions_dict[key].get('reverse', False):
                X_scaled[:, i] = 1 - X_scaled[:, i] 
            probabilities *= self.evaluate_functions[i](X_scaled[:, i], self.means[key], self.stds[key])
            if compute_grad:
                for dim in range(self.dims):
                    if i == dim:

                        grads[:, dim] *= self.evaluate_derivs[i](X_scaled[:, i], self.means[key], self.stds[key])
                    else:
                        #pass
                        grads[:, dim] *= self.evaluate_functions[i](X_scaled[:, i], self.means[key], self.stds[key])
                
        if compute_grad:
            return probabilities.reshape(-1, 1) + self.prior_floor, grads
        return probabilities.reshape(-1, 1) + self.prior_floor
    
    def compute_grad(self, X):
        # evaluating all pdfs quickly in one dimension (kind of only relevant on branin)
        point_shaped = np.array([X] * self.dims).T.squeeze(1)
        computed_probs = norm(self.modes, self.std_array).pdf(point_shaped)
        computed_derivs = computed_probs * (self.modes - point_shaped) / np.power(self.std_array, 2)
        
        # all computed probs are there, can just divide the right ones to get grad?
        prob = 1
        derivs = np.ones((1, self.dims))
        indices = np.arange(self.dims)
        for i in range(self.dims):
            prob_dim = np.sum(computed_probs[i, self.dists_per_dim[i]:self.dists_per_dim[i+1]])
            prob *= prob_dim
            derivs[:, indices != i] *= prob_dim
            derivs[:, i] *= np.sum(computed_derivs[i, self.dists_per_dim[i]:self.dists_per_dim[i+1]])
        
        return prob + self.prior_floor, derivs


class PriorEI:

    def __init__(self, prior, interleave=1e12, beta = 1, decay='beta'):
        self.evals = 0
        self.prior = prior
        self.dims = self.prior.dims
        self.beta = beta
        self.interleave = interleave
        self.acq_function = compute_ei
        self.t = 1
        self.next_interleave = False
        self.decay = decay
    
    def __repr__(self):
        return 'beta: ' +  str(self.beta) + '\ninterleaving: '\
             + str(self.interleave) + '\ndecay: ' + self.decay
        
    def update(self):
        self.t += 1
        if self.decay == 'interleave':
            if np.random.uniform() < self.t/self.interleave:
                print('\nNext run is interleaving!')
                self.next_interleave = True
            else:
                self.next_interleave = False
        else:
            self.next_interleave = self.t % self.interleave == 0
            if self.next_interleave:    
                print('\nNext run is interleaving!')
    # compute the derivative of all the stuf    f - need to account for the derivative of the prior^prior strength too
    # then. check compute grad and return
    # prior derivative, prior strength derivative, EI derivative

    def _compute(self, model, X, ei_target, compute_grad):
        # REMOVE THIS RETURN STATEMENT TO GET IT TO WORK NORMALLY
        X = X.reshape(-1, self.dims)
        if self.decay == 'beta':
            power = self.beta/self.t
        else:
            power = self.beta/1
        
        if compute_grad:
            prior, prior_grad = self.prior.compute_grad(X)
            ei_X, ei_grad = self.acq_function(model, X, ei_target=ei_target, compute_grad=compute_grad)
            ei_grad = ei_grad.reshape(-1, self.dims)

            prior_ei_X = np.power(prior, power) * ei_X
            prior_ei_X_grad = np.array([ei_grad[:, dim] * np.power(prior, power) +\
                power * prior_grad[:, dim] * np.power(prior, power - 1) * ei_X for dim in range(self.dims)]).T

            if self.next_interleave:
                return ei_X, ei_grad
            return prior_ei_X, prior_ei_X_grad.flatten()

        else: 
            ei_X = self.acq_function(model, X, ei_target=ei_target, compute_grad=compute_grad)
            if self.next_interleave:
                return ei_X
            prior_ei_X = np.power(self.prior(X), power).reshape(-1) * ei_X
            
            #return ei_X
            # SWITCH THE RETURN STATEMENT ABOVE FOR THE ONE BELOW
            return prior_ei_X #, np.power(self.prior(X), power).reshape(-1), ei_X
    
    def sample_from_prior(self):
        return self.prior.sample(1, normalize=False )

    def reset_evals(self):
        self.evals = 0

    def get_evals(self):
        return self.evals

    def get_center(self):
        return self.prior.mean_array

    def get_max_location(self):
        return self.prior.get_max_location()
