'''
Created on 18 May 2016

@author: simpson
'''

from heatmapbcc import GPGrid
import numpy as np
from scipy.stats import norm

class GPPref(GPGrid):
    '''
    Preference learning with GP, with variational inference implementation.
    
    Redefines:
    - Calculations of the Jacobian, referred to as self.G
    - Nonlinear forward model, "sigmoid"
    - Process_observations:
     - Observations, self.z. Observations now consist not of a count at a point, but two points and a label. 
     - self.obsx and self.obsy refer to all the locations mentioned in the observations.
     - self.Q is the observation covariance at the observed locations, i.e. the  
    - Lower bound?
    '''
    sigma = 1 # controls the observation noise. Is this equivalent to the output scale of f? I.e. doesn't it have the 
    # same effect by controlling the amount of noise that is permissable at each point? If so, I think we can fix this
    # to 1.
    # By approximating the likelihood distribution with a Gaussian, the covariance of the approximation is the
    # inverse Hessian of the negative log likelihood. Through moment matching self.Q with the likelihood covariance,
    # we can compute sigma?
    
    obs_v = [] # the first items in each pair -- index to the observation coordinates in self.obsx and self.obsy
    obs_u = [] # the second items in each pair -- indices to the observations in self.obsx and self.obsy
    
    def __init__(self, nx, ny, mu0=[], shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1):
        
        if not np.any(mu0):
            mu0 = 0            
        
        super(GPPref, self).__init__(nx, ny, mu0, shape_s0, rate_s0, s_initial, shape_ls, rate_ls, ls_initial, 
                                     force_update_all_points, n_lengthscales)
        
        
    def init_prior_mean_f(self, z0):
        self.mu0 = z0 # for preference learning, we pass in the latent mean directly  
    
    def f_to_z(self, f=None, v=None, u=None):
        '''
        f - should be of shape nobs x 1
        '''
        if not np.any(f):
            f = self.obs_f            
        if not np.any(v):
            v = self.obs_v
        if not np.any(u):
            u = self.obs_u
        if np.any(self.obs_v) and np.any(self.obs_u):   
            return f[v, :] - f.T[:, u] / (np.sqrt(2 * np.pi) * self.sigma)
        else: # provide the complete set of pairs
            return f - f.T / (np.sqrt(2 * np.pi) * self.sigma)
    
    def forward_model(self, f=None, v=None, u=None):
        # select only the pairs we have observed preferences for
        z = self.f_to_z(f, v, u)
        # gives an NobsxNobs matrix
        return norm.cdf(z)
    
    def update_jacobian(self, G_update_rate=1.0):
        z = self.f_to_z()
        phiz = self.forward_model()
        J = norm.pdf(z) / (phiz * np.sqrt(2) * self.sigma)
        J = J[:, np.newaxis]
        s = (J == self.obs_v[np.newaxis, :]) - (J == self.obs_u[np.newaxis, :])
        J = J * s 
        
        self.G = G_update_rate * s + (1 - G_update_rate) * self.G
        return phiz
    
    def observations_to_z(self):
        obs_probs = self.obs_values/self.obs_total_counts
        self.z = obs_probs        
    
    def init_obs_f(self):
        # Mean is just initialised to its prior here. Could be done faster?
        self.obs_f = np.zeros((len(self.obsx), 1)) + self.mu0
    
    def estimate_obs_noise(self):
        # Noise in observations
        self.obs_mean = (self.obs_values + self.nu0[1]) / (self.obs_total_counts + np.sum(self.nu0))
        var_obs_mean = self.obs_mean * (1-self.obs_mean) / (self.obs_total_counts + 1) # uncertainty in obs_mean
        self.Q = np.diagflat((self.obs_mean * (1 - self.obs_mean) - var_obs_mean) / self.obs_total_counts)
        
    def predict_obs(self, variance_method='rough', expectedlog=False, return_not=False):
        ''' 
        Need to change the behaviour from GPGrid because in this function we give distributions over the preference
        labels at observed pairs of locations, whereas the function predict() returns predictions over the latent
        function at a set of locations. This means that we cannot now use just post_rough or post_sample to obtain 
        predictions in this method. 
        '''
        if variance_method=='rough' and not expectedlog:
            z = self.f_to_z()
            m_post = self.forward_model()
            not_m_post = 1 - m_post
            v_post = - 2.0 * self.sigma / (norm.pdf(z)**2/m_post**2 + norm.pdf(z)*z/m_post) # use the inverse hessian
        else:
            # this should sample different values of obs_f and put them into the forward model
            v = np.diag(self.obs_C)[:, np.newaxis]
            f_samples = norm.rvs(loc=self.obs_f, scale=np.sqrt(v), size=(len(self.obs_f.flatten()), 1000))
            z = self.f_to_z(f_samples, self.obs_v, self.obs_u)
            rho_samples = self.forward_model()#
            rho_not_samples = 1 - rho_samples            
            if expectedlog:
                rho_samples = np.log(rho_samples)
                rho_not_samples = np.log(rho_not_samples)
            
            m_post = np.mean(rho_samples, axis=1)[:, np.newaxis]
            not_m_post = np.mean(rho_not_samples, axis=1)[:, np.newaxis]
            v_post = np.var(rho_samples, axis=1)[:, np.newaxis]

        if return_not:
            return m_post, not_m_post, v_post
        else:
            return m_post, v_post          
        
    def post_rough(self, f, v):
        ''' 
        When making predictions, we want to predict the latent value of each observed data point. Thus we  
        return the expected value of f, which is simply its mean. There is no need to apply the nonlinear function as
        we are not trying to predict the probability of a preference label here.
        '''
        m_post = f
        not_m_post = -f
        v_post = v
        
        return m_post, not_m_post, v_post
    
    def post_sample(self, f, v, expectedlog): 
        ''' 
        When making predictions, we want to predict the latent value of each observed data point. Thus we  
        return the expected value of f, which is simply its mean. There is no need to apply the nonlinear function as
        we are not trying to predict the probability of a preference label here.
        '''
        m_post = f
        not_m_post = -f
        v_post = v
        
        return m_post, not_m_post, v_post         

if __name__ == '__main__':
    pass