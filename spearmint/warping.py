import os
import glob
import numpy as np
from scipy.stats import norm
import json

class WarpedSpace:
    
    # takes a prior as a dict and returns a scipy normal distribution
    @staticmethod
    def create_distribution(prior):
        means = []
        stds = []
        ranges = []

        for key in sorted(prior.keys()):
            mean = np.inf
            while mean > prior[key]['range'][1] or mean < prior[key]['range'][0]:
                mean = prior[key]['params']['mean'] + norm.rvs(0, prior[key].get('noise', 0))
            
            means.append(mean)
            stds.append(prior[key]['params']['std'][0])
            ranges.append(prior[key]['range'])

        return norm(np.array(means).reshape(1, -1), np.array(stds).reshape(1, -1))

    
    # class that computes the warped space, and through the call funtion extrapolates up
    # from warped space to function space - wraps around the original function
    
    def __init__(self, dist, ranges, objective):
        # ranges in numpy matrix, one row per dimension
        # dist needs to implement elements of a scipy distribution, i.e. pdf, cdf, ppf etc.
        self.dist = dist
        self.param_ranges = np.zeros((len(ranges), 2))
        for i, range_ in enumerate(ranges):
            self.param_ranges[i, 0] = range_[0]
            self.param_ranges[i, 1] = range_[1]
        self.get_warped_ranges()
        self.objective = objective
        
    def get_warped_ranges(self):
        # gives the coordinates in warped (0, 1)-space where the boundaries of the original space lie
        # we want this boundary to be represented as such in the warped space too - thus, we warp the
        # space again by minmax-scaling the warped space with these boundary values. Consequently, 
        # we get a limit on the warped space that (at largest) has the same boundaries as the original
        # space, and otherwise further enlarges the original search space. This makes is a truncated 
        # gaussian even if the prior is set at the very edge
        # in the case where the entire prior fits within the search space, we need boundaries for where
        # numerical issues occur - i.e. not letting the algorithm go more than 8:ish standard deviations
        # away for any dimension
        self.boundaries = np.zeros(self.param_ranges.shape)
        for i, range_ in enumerate(self.param_ranges.T):
            self.boundaries[:, i] = self.dist.cdf(np.array(range_))
        
        #increment boundaries with smallest possible value to avoid inverting bach to infinity
        self.boundaries[:, 0] = self.boundaries[:, 0] + 2e-16
        self.boundaries[:, 1] = self.boundaries[:, 1] - 2e-16
        
    def get_original_range(self, X):
        # input - an X in range 0, 1 irregardless of problem
        # this needs to be shrinked linearly to the range which is allowed to still be in range
        # Thus, we get inverse cdf (floor + X * (floor of w.s. - ceiling w.s.) )
        X_scaled = np.zeros(X.shape)
        
        for dim in range(X.shape[1]):
            X_scaled[:, dim] = self.boundaries[dim, 0] + X[:, dim] * (self.boundaries[dim, 1] - self.boundaries[dim, 0])
        
        # this probably won't work in higher dimensions
        X_unwarped = self.dist.ppf(X_scaled)
        for dim in range(X.shape[1]):
            assert np.all(X_unwarped[:, dim] >= self.param_ranges[dim, 0]) 
            assert np.all(X_unwarped[:, dim] <= self.param_ranges[dim, 1])
        return X_unwarped
        
    def __call__(self, X):
        X_original = self.get_original_range(X)
        return self.objective(X_original)