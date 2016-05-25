'''
Created on 18 May 2016

@author: simpson
'''

from gpgrid import GPGrid, coord_arr_to_1d, coord_arr_from_1d
import numpy as np
from scipy.stats import norm
from scipy.sparse import coo_matrix
import logging

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
    
    pref_v = [] # the first items in each pair -- index to the observation coordinates in self.obsx and self.obsy
    pref_u = [] # the second items in each pair -- indices to the observations in self.obsx and self.obsy
    
    def __init__(self, nx, ny, mu0=[], shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1):
        
        if not np.any(mu0):
            mu0 = 0            
        
        super(GPPref, self).__init__(nx, ny, mu0, shape_s0, rate_s0, s_initial, shape_ls, rate_ls, ls_initial, 
                                     force_update_all_points, n_lengthscales)
        
        
    def init_prior_mean_f(self, z0):
        self.mu0 = z0 # for preference learning, we pass in the latent mean directly  
    
    def init_obs_prior(self):
        nsamples = 50000
        f_samples = norm.rvs(loc=self.mu0, scale=np.sqrt(1.0/self.s), size=nsamples)
        v_samples = np.random.choice(nsamples, nsamples)
        u_samples = np.random.choice(nsamples, nsamples)
        rho_samples = self.forward_model(f_samples, v_samples, u_samples)
        rho_mean = np.mean(rho_samples)
        rho_var = np.var(rho_samples)
        # find the beta parameters
        a_plus_b = 1.0 / (rho_var / (rho_mean*(1 - rho_mean))) - 1
        a = a_plus_b * rho_mean
        b = a_plus_b * (1 - rho_mean)
        #b = 1.0
        #a = 1.0
        self.nu0 = np.array([b, a])
        logging.debug("Prior parameters for the observed pairwise preference variance are: %s" % str(self.nu0))           
    
    def init_obs_f(self):
        # Mean probability at observed points given local observations
        self.obs_f = np.zeros((self.obs_coords.shape[0], 1))
    
    def forward_model(self, f=None, v=None, u=None, return_g_f=False):
        '''
        f - should be of shape nobs x 1
        
        This returns the probability that each pair has a value of 1, which is referred to as Phi(z) 
        in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper. 
        In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with 
        value 1, so use a different label here.
        '''        
        if not np.any(f):
            f = self.obs_f            
        if not np.any(v):
            v = self.pref_v
        if not np.any(u):
            u = self.pref_u

        if f.ndim < 2:
                f = f[:, np.newaxis]
        
        if np.any(v) and np.any(u):   
            g_f = f[v, :] - f[u, :] / (np.sqrt(2 * np.pi) * self.sigma) # gives an NobsxNobs matrix
        else: # provide the complete set of pairs
            g_f = f - f.T / (np.sqrt(2 * np.pi) * self.sigma)    
                
        phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the 
        # With the standard GP density classifier, we can skip this step because
        # g_f is already a probability and Phi(z) is a Bernoulli distribution.
        if return_g_f:
            return phi, g_f
        else:
            return phi 
    
    def update_jacobian(self, G_update_rate=1.0):
        phi, g_mean_f = self.forward_model(return_g_f=True) # first order Taylor series approximation
        J = norm.pdf(g_mean_f) / (phi * np.sqrt(2) * self.sigma)
        obs_idxs = np.arange(self.obs_coords.shape[0])[np.newaxis, :]
        s = (self.pref_v[:, np.newaxis]==obs_idxs).astype(int) - (self.pref_u[:, np.newaxis]==obs_idxs).astype(int)
        J = J * s 
        if not np.any(self.G):
            self.G = J
        else:        
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
        return phi
    
    # If this is the same as parent class, can the initialisation also be the same init_obs_f?
    #def observations_to_z(self):
    #    obs_probs = self.obs_values/self.obs_total_counts
    #    self.z = obs_probs        
    
    #def init_obs_f(self):
    #    # Mean is just initialised to its prior here. Could be done faster?
    #    self.obs_f = np.zeros((self.obs_coords.shape[0], 1)) + self.mu0
    
#     def estimate_obs_noise(self):
#         #Noise in observations
#         var_obs_mean = self.obs_mean * (1-self.obs_mean) / (self.obs_total_counts + 1) # uncertainty in obs_mean
#         self.Q = np.diagflat((self.obs_mean * (1 - self.obs_mean) - var_obs_mean) / self.obs_total_counts)
       
    def get_unique_locations(self, obs_coords_0, obs_coords_1):
        coord_rows_0 = coord_arr_to_1d(obs_coords_0)
        coord_rows_1 = coord_arr_to_1d(obs_coords_1)
        _, uidxs, pref_vu = np.unique([coord_rows_0, coord_rows_1], return_index=True, return_inverse=True) # get unique locations
        
        # Record the coordinates of all points that were compared
        obs_coords = [coord_rows_0, coord_rows_1][uidxs]
       
        # Record the indexes into the list of coordinates for the pairs that were compared 
        pref_v = pref_vu[:len(coord_rows_0)]
        pref_u = pref_vu[len(coord_rows_0):]
        
        return obs_coords, pref_v, pref_u  
       
    def count_observations(self, obs_coords, n_obs, poscounts, totals):
        '''
        obs_coords - a tuple with two elements, the first containing the list of coordinates for the first items in each
        pair, and the second containing the coordinates of the second item in the pair.
        '''        
        obs_coords_0 = np.array(obs_coords[0])
        obs_coords_1 = np.array(obs_coords[1])
        if obs_coords_0.dtype=='int': # duplicate locations should be merged and the number of duplicates counted
            # Ravel the coordinates
            ravelled_coords_0 = coord_arr_to_1d(obs_coords_0)
            ravelled_coords_1 = coord_arr_to_1d(obs_coords_1) 
            
            # SWAP PAIRS SO THEY ALL HAVE LOWEST COORD FIRST so we can count prefs for duplicate location pairs
            # get unique keys
            uravelled_coords, keys = np.unique([ravelled_coords_0, ravelled_coords_1], return_inverse=True)
            keys_0 = keys[:len(ravelled_coords_0)]
            keys_1 = keys[len(ravelled_coords_0):]
            idxs_to_swap = keys_0 < keys_1
            
            swap_coords_0 = keys_0[idxs_to_swap]
            keys_0[idxs_to_swap] = keys_1[idxs_to_swap]
            keys_1[idxs_to_swap] = swap_coords_0
            
            grid_obs_counts = coo_matrix((totals, (keys_0, keys_1)) ).toarray()            
            grid_obs_pos_counts = coo_matrix((poscounts, (keys_0, keys_1)) ).toarray()
                                                              
            nonzero_v, nonzero_u = grid_obs_counts.nonzero() # coordinate key pairs with duplicate pairs removed
            ukeys, pref_vu = np.unique([nonzero_v, nonzero_u], return_inverse=True) # get unique locations
            
            # Record the coordinates of all points that were compared
            self.obs_coords = coord_arr_from_1d(uravelled_coords[ukeys], obs_coords_0.dtype, 
                                                dims=(len(ukeys), obs_coords_0.shape[1]))
            
            # Record the indexes into the list of coordinates for the pairs that were compared 
            self.pref_v = pref_vu[:len(nonzero_v)]
            self.pref_u = pref_vu[len(nonzero_v):]
                   
            # Return the counts for each of the observed pairs
            return grid_obs_pos_counts[nonzero_v, nonzero_u], grid_obs_counts[nonzero_v, nonzero_u]
                    
        elif obs_coords_0.dtype=='float': # Duplicate locations are not merged
            self.obs_coords, self.pref_v, self.pref_u = self.get_unique_locations()
            
            return poscounts, totals # these remain unaltered as we have not de-duplicated            
            
    def fit(self, pair_item_1_coords, pair_item_2_coords, obs_values, totals=None, process_obs=True, update_s=True):
        super(GPPref, self).fit((pair_item_1_coords, pair_item_2_coords), obs_values, totals, process_obs, update_s)  
        
    def predict(self, pair_item_0_coords=None, pair_item_1_coords=None, variance_method='rough', max_block_size=1e5, expectedlog=False, return_not=False):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        # if no output_coords provided, give predictions at the fitted locations
        if not np.any(pair_item_0_coords) and not np.any(pair_item_1_coords):
            return self.predict_obs(variance_method, expectedlog, return_not)
        
        output_coords, out_pref_v, out_pref_u = self.get_unique_locations(pair_item_0_coords, pair_item_1_coords)
        
        nblocks, noutputs = self.init_output_arrays(output_coords, max_block_size, variance_method)
                
        for block in range(nblocks):
            if self.verbose:
                logging.debug("GPGrid predicting block %i of %i" % (block, nblocks))            
            self.predict_block(block, max_block_size, noutputs)
        
        noutprefs = self.pair_item_0.shape[0]
        
        if variance_method=='sample':
            m_post = np.empty((noutprefs, 1), dtype=float)
            not_m_post = np.empty((noutprefs, 1), dtype=float)
            v_post = np.empty((noutprefs, 1), dtype=float)        
                    
        # Approximate the expected value of the variable transformed through the sigmoid.
        nblocks = int(np.ceil(float(noutprefs) / max_block_size))
        for block in range(nblocks):
            maxidx = (block + 1) * max_block_size
            if maxidx > noutputs:
                maxidx = noutputs
            blockidxs = np.arange(block * max_block_size, maxidx, dtype=int)            
            
            if variance_method == 'sample' or expectedlog:
                m_post[blockidxs, :], v_post[blockidxs, :], not_m_post[blockidxs, :] = \
                    self.post_sample(self.f[blockidxs, :], self.v[blockidxs, :], expectedlog, out_pref_v, out_pref_u)                
        
        if variance_method == 'rough' and not expectedlog:
            m_post, v_post, not_m_post = self.post_rough(self.f, self.v, out_pref_v, out_pref_u)
        elif variance_method == 'rough':
            logging.warning("Switched to using sample method as expected log requested. No quick method is available.")
            
        if return_not:
            return m_post, not_m_post, v_post
        else:
            return m_post, v_post     
        
    def post_rough(self, f_mean, f_var, pref_v=None, pref_u=None):
        ''' 
        When making predictions, we want to predict the probability of each listed preference pair.
        Use a solution given by applying the forward model to the mean of the latent function -- 
        ignore the uncertainty in f itself, considering only the uncertainty due to the noise sigma.
        '''
        if not np.any(pref_v):
            pref_v = self.pref_v
        if not np.any(pref_u):
            pref_u = self.pref_u
        
        m_post, g_f = self.forward_model(f_mean, pref_v, pref_u, return_g_f=True)
        not_m_post = 1 - m_post
        v_post = - 2.0 * self.sigma / (norm.pdf(g_f)**2/m_post**2 + norm.pdf(g_f)*g_f/m_post) # use the inverse hessian
        
        return m_post, not_m_post, v_post
    
    def post_sample(self, f_mean, f_var, expectedlog, pref_v=None, pref_u=None): 
        ''' 
        When making predictions, we want to predict the probability of each listed preference pair. 
        Use sampling to handle the nonlinearity. 
        '''
        if not np.any(pref_v):
            pref_v = self.pref_v
        if not np.any(pref_u):
            pref_u = self.pref_u
        
        # this should sample different values of obs_f and put them into the forward model
        #v = np.diag(self.obs_C)[:, np.newaxis]
        f_samples = norm.rvs(loc=f_mean, scale=np.sqrt(f_var), size=(len(self.f_mean.flatten()), 1000))
        rho_samples = self.forward_model(f_samples, self.pref_v, self.pref_u)#
        rho_not_samples = 1 - rho_samples            
        if expectedlog:
            rho_samples = np.log(rho_samples)
            rho_not_samples = np.log(rho_not_samples)
        
        m_post = np.mean(rho_samples, axis=1)[:, np.newaxis]
        not_m_post = np.mean(rho_not_samples, axis=1)[:, np.newaxis]
        v_post = np.var(rho_samples, axis=1)[:, np.newaxis]
        
        return m_post, not_m_post, v_post         

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    
    
    # Generate some data
    
    nx = 100
    ny = 100
    
    ls = [10, 10]
    
    sigma = 10
    
    N = 100
    P = 500 # number of pairs
    
    from scipy.stats import multivariate_normal as mvn

    def matern_3_2(xvals, yvals, ls):
        Kx = np.abs(xvals) * 3**0.5 / ls[0]
        Kx = (1 + Kx) * np.exp(-Kx)
        Ky = np.abs(yvals) * 3**0.5 / ls[1]
        Ky = (1 + Ky) * np.exp(-Ky)
        if Ky.shape[1] == 1:
            Ky = Ky.T     
        K = Kx * Ky
        return K
    
    # Some random feature values
    xvals = np.random.choice(nx, N, replace=True)[:, np.newaxis]
    yvals = np.random.choice(ny, N, replace=True)[:, np.newaxis]
    
    K = matern_3_2(xvals, yvals, ls)
    
    f = mvn.rvs(cov=K) # zero mean
    
    # generate pairs indices
    pair1idxs = np.random.choice(N, P, replace=True)
    pair2idxs = np.random.choice(N, P, replace=True)
     
    # generate the noisy function values for the pairs
    f1noisy = norm.rvs(loc=f[pair1idxs], scale=sigma)
    f2noisy = norm.rvs(loc=f[pair2idxs], scale=sigma)
    
    # generate the discrete labels from the noisy preferences
    prefs = f1noisy > f2noisy
    
    # Create a GPPref model
    model = GPPref(nx, ny, 0, 1, 1, ls_initial=[10, 10])
    pair1coords = np.concatenate((xvals[pair1idxs, :], yvals[pair1idxs, :]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs, :], yvals[pair2idxs, :]), axis=1)    
    model.fit(pair1coords, pair2coords, prefs)
    
    # Predict at the test locations
    fpred, vpred = model.predict((xvals, yvals), variance_method='sample')
    fpred = fpred.flatten()
    vpred = vpred.flatten()
    
    # Evaluate the accuracy of the predictions
    print "RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
    print "NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
    
    # turn the values into predictions of preference pairs.
    pair1idxs_test = np.random.choice(N, P, replace=True)
    pair2idxs_test = np.random.choice(N, P, replace=True)
    
    rho = model.forward_model(f, pair1idxs_test, pair2idxs_test)
    rho_pred = model.forward_model(fpred, pair1idxs_test, pair2idxs_test)
    
    print "Brier score of %f" % np.sqrt(np.mean((rho-rho_pred)**2))
    print "Cross entropy error of %f" % -np.sum(rho * np.log(rho_pred) + (1-rho) * np.log(1 - rho_pred))    
    
#     from sklearn.metrics import f1_score, roc_auc_score
#     print "F1 score of %f" % f1_score(rho, rho_pred)
#     print "Accuracy of %f" % np.mean(rho==rho_pred)
#     print "ROC of %f" % roc_auc_score(rho, rho_pred)
    