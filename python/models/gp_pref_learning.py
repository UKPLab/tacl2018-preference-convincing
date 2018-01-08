'''
Created on 18 May 2016

@author: simpson
'''
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from scipy.sparse import coo_matrix, issparse, hstack
import logging
from gp_classifier_vb import coord_arr_to_1d, coord_arr_from_1d, temper_extreme_probs
from gp_classifier_svi import GPClassifierSVI

def get_unique_locations(obs_coords_0, obs_coords_1, mu_0, mu_1):
    if issparse(obs_coords_0) or issparse(obs_coords_1):
        uidxs_0 = []
        pref_vu = []
        
        for r, row in enumerate(obs_coords_0):
            print "%i out of %i" % (r, obs_coords_0.shape[0])
            idx = row == obs_coords_0[uidxs_0]
            if not np.sum(idx):
                uidxs_0.append(r)
                pref_vu.append(len(uidxs_0) - 1)
            else:
                pref_vu.append(np.argwhere(idx)[0])
                
        len_0 = obs_coords_0.shape[0]
        uidxs_1 = []
        for r, row in enumerate(obs_coords_1):
            print "%i out of %i" % (r, obs_coords_0.shape[0])
            idx = row == obs_coords_0[uidxs_0]
            if not np.sum(idx):
                idx = row == obs_coords_1[uidxs_1]
                if not np.sum(idx):
                    uidxs_1.append(r + len_0)
                    pref_vu.append(len(uidxs_1) - 1)
                else:
                    pref_vu.append(np.argwhere(idx)[0] + len_0)
            else:
                pref_vu.append(np.argwhere(idx)[0])
                
        # convert urows to a sparse matrix
        obs_coords = hstack((obs_coords_0[uidxs_0], obs_coords_1[uidxs_1]), format='csc')
        uidxs = np.concatenate((uidxs_0, np.array(uidxs_1) + len_0))
    else:                  
        coord_rows_0 = coord_arr_to_1d(obs_coords_0)
        coord_rows_1 = coord_arr_to_1d(obs_coords_1)
        all_coord_rows = np.concatenate((coord_rows_0, coord_rows_1), axis=0)
        _, uidxs, pref_vu = np.unique(all_coord_rows, return_index=True, return_inverse=True) # get unique locations
        
        # Record the coordinates of all points that were compared
        obs_coords = np.concatenate((obs_coords_0, obs_coords_1), axis=0)[uidxs]
   
    # Record the indexes into the list of coordinates for the pairs that were compared 
    pref_v = pref_vu[:obs_coords_0.shape[0]]
    pref_u = pref_vu[obs_coords_0.shape[0]:]
    
    mu_vu = np.concatenate((mu_0, mu_1), axis=0)[uidxs]
    
    return obs_coords, pref_v, pref_u, mu_vu

def pref_likelihood(fmean, fvar=None, subset_idxs=[], v=[], u=[], return_g_f=False):
    '''
    f - should be of shape nobs x 1
    
    This returns the probability that each pair has a value of 1, which is referred to as Phi(z) 
    in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper. 
    In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with 
    value 1, so use a different label here.
    '''        
    if len(subset_idxs):
        if len(v) and len(u):
            # keep only the pairs that reference two items in the subet
            pair_subset = np.in1d(v, subset_idxs) & np.in1d(u, subset_idxs)
            v = v[pair_subset]
            u = u[pair_subset]
        else:
            fmean = fmean[subset_idxs]  

    if fmean.ndim < 2:
        fmean = fmean[:, np.newaxis]
        
    if fvar is None:
        fvar = 2.0
    else:
        if fvar.ndim < 2:
            fvar = fvar[:, np.newaxis]        
        fvar += 2.0
    
    # TODO: should we include posterior covariance of f? Would this mean replacing 2 with 2 + var(u) + var(v) - cov(uv)?
    if len(v) and len(u):
        g_f = (fmean[v, :] - fmean[u, :]) / np.sqrt(fvar) # / np.sqrt(self.s)) # gives an NobsxNobs matrix
    else: # provide the complete set of pairs
        g_f = (fmean - fmean.T) / np.sqrt(fvar) # / np.sqrt(self.s))  # the maths shows that s cancels out -- it's already 
        # included in our estimates of f, which are scaled by s. However, the prior mean mu0 should also be scaled
        # to match, but this should happen automatically if we learn s, I think. 
            
    phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the 
    # With the standard GP density classifier, we can skip this step because
    # g_f is already a probability and Phi(z) is a Bernoulli distribution.
    if return_g_f:
        return phi, g_f
    else:
        return phi
    
class GPPrefLearning(GPClassifierSVI):
    '''
    Preference learning with GP, with variational inference implementation. Can use stochastic variational inference.
    
    Redefines:
    - Calculations of the Jacobian, referred to as self.G
    - Nonlinear forward model, "sigmoid"
    - Process_observations:
     - Observations, self.z. Observations now consist not of a count at a point, but two points and a label. 
     - self.obsx and self.obsy refer to all the locations mentioned in the observations.
     - self.Q is the observation covariance at the observed locations, i.e. the  
    - Lower bound?
    '''
    
    pref_v = [] # the first items in each pair -- index to the observation coordinates in self.obsx and self.obsy
    pref_u = [] # the second items in each pair -- indices to the observations in self.obsx and self.obsy
    
    item_features = None
    
    def __init__(self, ninput_features, mu0=0, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=0.1, ls_initial=None, 
        kernel_func='matern_3_2', kernel_combination='*',
        max_update_size=10000, ninducing=500, use_svi=True, delay=10, forgetting_rate=0.7, verbose=False, fixed_s=False):
        
        # We set the function scale and noise scale to the same value so that we assume apriori that the differences
        # in preferences can be explained by noise in the preference pairs or the latent function. Ordering patterns 
        # will change this balance in the posterior.  
        
        #self.sigma = 1 # controls the observation noise. Equivalent to the output scale of f? I.e. doesn't it have the 
        # same effect by controlling the amount of noise that is permissible at each point? If so, I think we can fix this
        # to 1.
        # By approximating the likelihood distribution with a Gaussian, the covariance of the approximation is the
        # inverse Hessian of the negative log likelihood. Through moment matching self.Q with the likelihood covariance,
        # we can compute sigma?
        
        if shape_s0 <= 0:
            shape_s0 = 0.5
        if rate_s0 <= 0:
            rate_s0 = 0.5
        
        super(GPPrefLearning, self).__init__(ninput_features, mu0, shape_s0, rate_s0, shape_ls, rate_ls, ls_initial, 
         kernel_func, kernel_combination,
         max_update_size, ninducing, use_svi, delay, forgetting_rate, verbose=verbose, fixed_s=fixed_s)
    
    # Initialisation --------------------------------------------------------------------------------------------------
    def _init_prior_mean_f(self, z0):
        self.mu0_default = z0 # for preference learning, we pass in the latent mean directly  
    
    def _init_obs_prior(self):
        # TODO: are we missing adding the uncertainty in the prior mu0? 
        # to make a and b smaller and put more weight onto the observations, increase v_prior by increasing rate_s0/shape_s0
        if self.use_svi:
            Kstar = self.K_nm
            Ks_starstar = self.K_nm.dot(self.K_nm.T) * self.rate_s0/self.shape_s0
        else:
            Kstar = self.K
            Ks_starstar = self.K * self.rate_s0/self.shape_s0
            
        samples = norm.rvs(loc=self.mu0, scale=np.sqrt(self.rate_s0/self.shape_s0), size=(self.mu0.shape[0], 1000))
        samples = self.forward_model(samples, v=self.pref_v, u=self.pref_u, return_g_f=False)
        v_post = np.var(samples, axis=1)[:, np.newaxis]
        v_post = temper_extreme_probs(v_post, zero_only=True)            
            
        _, _, v_prior = self._post_sample(self.mu0, Kstar, False, self.mu0, self.pref_v, self.pref_u)
        # since the sampling method uses a small sample size, the mean can be a little wrong, e.g. not 0.5 when mu0 is 
        # 0. This is okay for the variance but use the method below to correct the mean. 
        m_prior, not_m_prior = self._post_rough(self.mu0, Ks_starstar, self.pref_v, self.pref_u)
        
        v_post[m_prior * (1 - not_m_prior) <= 1e-7] = 1e-8
        
        # find the beta parameters
        a_plus_b = 1.0 / (v_prior / (m_prior*not_m_prior)) - 1
        a = (a_plus_b * m_prior)
        b = (a_plus_b * not_m_prior)

        self.nu0 = np.array([b, a])
        #if self.verbose:
        #    logging.debug("Prior parameters for the observed pairwise preference variance are: %s" % str(self.nu0))           
    
    def _init_obs_f(self):
        # Mean probability at observed points given local observations
        self.obs_f = np.zeros((self.n_locs, 1)) + self.mu0
        
    def _init_obs_mu0(self, mu0):
        if mu0 is None or not len(mu0):
            self.mu0 = np.zeros((self.n_locs, 1)) + self.mu0_default
        else:
            self.mu0 = mu0
            self.mu0_1 = self.mu0[self.pref_v, :]
            self.mu0_2 = self.mu0[self.pref_u, :]
            
        self.Ntrain = self.pref_u.size 
            
    # Input data handling ---------------------------------------------------------------------------------------------

    def _count_observations(self, obs_coords, _, poscounts, totals):
        '''
        obs_coords - a tuple with two elements, the first containing the list of coordinates for the first items in each
        pair, and the second containing the coordinates of the second item in the pair.
        '''                
        obs_coords_0 = np.array(obs_coords[0])
        obs_coords_1 = np.array(obs_coords[1])
        if obs_coords_0.ndim == 1:
            obs_coords_0 = obs_coords_0[:, np.newaxis]
        if obs_coords_1.ndim == 1:
            obs_coords_1 = obs_coords_1[:, np.newaxis]
                
        # duplicate locations should be merged and the number of duplicates counted
        #poscounts = poscounts.astype(int)
        totals = totals.astype(int)  
                               
        if self.item_features is not None:
            self.obs_uidxs = np.arange(self.item_features.shape[0])
            self.pref_v = obs_coords_0.flatten()
            self.pref_u = obs_coords_1.flatten()
            self.obs_coords = self.item_features
            self.obs_uidxs = np.arange(self.item_features.shape[0])
            return poscounts, totals             
        else:
            # TODO: This code could be merged with get_unique_locations()
            ravelled_coords_0 = coord_arr_to_1d(obs_coords_0)# Ravel the coordinates
            ravelled_coords_1 = coord_arr_to_1d(obs_coords_1) 
        
            # get unique keys
            all_ravelled_coords = np.concatenate((ravelled_coords_0, ravelled_coords_1), axis=0)
            uravelled_coords, origidxs, keys = np.unique(all_ravelled_coords, return_index=True, return_inverse=True)
            
            keys_0 = keys[:len(ravelled_coords_0)]
            keys_1 = keys[len(ravelled_coords_0):]

            # SWAP PAIRS SO THEY ALL HAVE LOWEST COORD FIRST so we can count prefs for duplicate location pairs
            idxs_to_swap = keys_0 < keys_1
            swap_coords_0 = keys_0[idxs_to_swap]
            poscounts[idxs_to_swap] = totals[idxs_to_swap] - poscounts[idxs_to_swap]            
            
            keys_0[idxs_to_swap] = keys_1[idxs_to_swap]
            keys_1[idxs_to_swap] = swap_coords_0
            
            grid_obs_counts = coo_matrix((totals, (keys_0, keys_1)) ).toarray()            
            grid_obs_pos_counts = coo_matrix((poscounts, (keys_0, keys_1)) ).toarray()
                                                              
            nonzero_v, nonzero_u = grid_obs_counts.nonzero() # coordinate key pairs with duplicate pairs removed
                        
            nonzero_all = np.concatenate((nonzero_v, nonzero_u), axis=0)
            ukeys, pref_vu = np.unique(nonzero_all, return_inverse=True) # get unique locations
            
            self.obs_uidxs = origidxs[ukeys] # indexes of unique observation locations into the original input data
            
            # Record the coordinates of all points that were compared
            self.obs_coords = coord_arr_from_1d(uravelled_coords[ukeys], obs_coords_0.dtype, 
                                            dims=(len(ukeys), obs_coords_0.shape[1]))
        
            # Record the indexes into the list of coordinates for the pairs that were compared 
            self.pref_v = pref_vu[:len(nonzero_v)]
            self.pref_u = pref_vu[len(nonzero_v):]
               
            # Return the counts for each of the observed pairs
            pos_counts = grid_obs_pos_counts[nonzero_v, nonzero_u]
            total_counts = grid_obs_counts[nonzero_v, nonzero_u]
            return pos_counts, total_counts
            
    # Mapping between latent and observation spaces -------------------------------------------------------------------
              
    def forward_model(self, fmean=None, fvar=None, subset_idxs=[], v=[], u=[], return_g_f=False):
        '''
        f - should be of shape nobs x 1
        
        This returns the probability that each pair has a value of 1, which is referred to as Phi(z) 
        in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper. 
        In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with 
        value 1, so use a different label here.
        '''        
        if fmean is None:
            fmean = self.obs_f
        if len(v) == 0:
            v = self.pref_v
        if len(u) == 0:
            u = self.pref_u
            
        return pref_likelihood(fmean, fvar, subset_idxs, v, u, return_g_f)
    
    def _compute_jacobian(self, data_idx_i=None):
        phi, g_mean_f = self.forward_model(return_g_f=True) # first order Taylor series approximation
        J = 1 / (2*np.pi)**0.5 * np.exp(-g_mean_f**2 / 2.0) * np.sqrt(0.5)
        
        obs_idxs = np.arange(self.n_locs)[np.newaxis, :]
        
        if data_idx_i is not None and hasattr(self, 'data_obs_idx_i') and len(self.data_obs_idx_i): 
            obs_idxs = obs_idxs[:, data_idx_i]
            J = J[self.data_obs_idx_i, :]
            s = (self.pref_v[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int) -\
                                                    (self.pref_u[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int)
        else:    
            s = (self.pref_v[:, np.newaxis]==obs_idxs).astype(int) - (self.pref_u[:, np.newaxis]==obs_idxs).astype(int)
            
        J = J * s
        
        return phi, J
    
    def _update_jacobian(self, G_update_rate=1.0):            
        phi, J = self._compute_jacobian(self.data_idx_i)
        
        if self.G is None or not np.any(self.G) or self.G.shape != J.shape: 
            # either G has not been initialised, or is from different observations:
            self.G = J
        else:        
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
            
        return phi
    
    # Log Likelihood Computation ------------------------------------------------------------------------------------- 
        
    def _logpt(self):
        rho = self.forward_model(self.obs_f, None, v=self.pref_v, u=self.pref_u, return_g_f=False)
        rho = temper_extreme_probs(rho)
        logrho = np.log(rho)
        lognotrho = np.log(1 - rho)
        
        return logrho, lognotrho  
    
    # Training methods ------------------------------------------------------------------------------------------------  
            
    def fit(self, items1_coords=None, items2_coords=None, item_features=None, preferences=None, totals=None, 
            process_obs=True, mu0=None, K=None, optimize=False, input_type='binary'):
        '''
        preferences -- Preferences by default can be 1 = item 1 is preferred to item 2, 
        or 0 = item 2 is preferred to item 1, 0.5 = no preference, or values in between.
        For preferences that are not normalised to between 0 and 1, the value of input_type needs to be set. 
        input_type -- can be 'binary', meaning preferences must be [0,1], or 'zero-centered' meaning that value 1 
        indicates item 1 is preferred, value -1 indicates item 2 is preferred, and 0 indicates no preference. The value
        are converted internally to [0,1]. 
        '''
        pref_values_in_input = np.unique(preferences)
        if process_obs and input_type == 'binary' and np.sum((pref_values_in_input < 0) | (pref_values_in_input > 1)):
            raise ValueError('Binary input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif process_obs and input_type == 'zero-centered' and np.sum((pref_values_in_input < -1) | (pref_values_in_input > 1)):
            raise ValueError('Zero-centered input preferences specified but the data contained the values %s' % pref_values_in_input)
        elif process_obs and input_type == 'zero-centered':
            #convert them to [0,1]
            preferences += 1
            preferences /= 2.0
        elif process_obs and input_type != 'binary':
            raise ValueError('input_type for preference labels must be either "binary" or "zero-centered"') 
            
        if item_features is not None: # keep the old item features if we pass in none
            self.item_features = item_features
            
        super(GPPrefLearning, self).fit((items1_coords, items2_coords), preferences, totals, process_obs, 
                                        mu0=mu0, K=K, optimize=optimize)  
        
    def _update_sample_idxs(self):
        nobs = self.obs_f.shape[0]
        
        if not self.fixed_sample_idxs:
            self.data_obs_idx_i = 0
        
            while not np.sum(self.data_obs_idx_i): # make sure we don't choose indices that have not been compared
                self.data_idx_i = np.sort(np.random.choice(nobs, self.update_size, replace=False))
                self.data_obs_idx_i = np.in1d(self.pref_v, self.data_idx_i) & np.in1d(self.pref_u, self.data_idx_i)
        else:
            self.data_obs_idx_i = np.in1d(self.pref_v, self.data_idx_i) & np.in1d(self.pref_u, self.data_idx_i)                            
            
    # Prediction methods ---------------------------------------------------------------------------------------------
    def predict(self, out_feats=None, item_0_idxs=None, item_1_idxs=None, out_1_feats=None, K_star=None, K_starstar=None,  
                expectedlog=False, return_not=False, mu0_out=None, mu0_out_1=None,
                reuse_output_kernel=False, return_var=True):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        
        If using items_0_idxs and items_1_idxs, out_1_feats can be set to None so that both sets of indexes look up values in out_feats 
        '''
        if item_0_idxs is None and item_1_idxs is None and out_1_feats is not None and out_feats is not None:
            out_feats, item_0_idxs, item_1_idxs, mu0_out = get_unique_locations(out_feats, out_1_feats, mu0_out, mu0_out_1)
            out_1_feats = None # the object is no longer needed.
        elif item_0_idxs is None and item_1_idxs is None and out_1_feats is None and out_feats is not None:
            # other combinations are invalid
            logging.error('Invalid combination of parameters for predict(): please supply either (items_0_idxs AND \
            items_1_idxs AND out_feats) OR (items_0_idxs AND \
            items_1_idxs AND K_star AND K_starstar) OR (out_feats AND out_1_feats)')
            return

        if item_0_idxs is not None and item_1_idxs is not None and out_1_feats is not None and out_feats is not None:
            out_feats = np.concatenate((out_feats, out_1_feats), axis=0)
            mu0_out = np.concatenate((mu0_out, mu0_out_1))

        # predict f for all the rows in out_feats or K_star if these variables are not None, otherwise error.
        f, C = self.predict_f(out_feats, None, K_star, K_starstar, mu0_out, reuse_output_kernel, 
                              full_cov=True)

        m_post, not_m_post = self._post_rough(f, C, item_0_idxs, item_1_idxs)
        if return_var:
            _, _, v_post = self._post_sample(f, self.K_star, False, mu0_out, item_0_idxs, item_1_idxs)
        
        if expectedlog:
            m_post = np.log(m_post)
            not_m_post = np.log(not_m_post)
            
        if return_not:
            if return_var:
                return m_post, not_m_post, v_post
            else:
                return m_post, not_m_post
        elif return_var:
            return m_post, v_post
        else:
            return m_post
            
    def _post_sample(self, f_mean, f_cov, expectedlog, mu=0, v=None, u=None):
        if v is None:
            v = self.pref_v
        if u is None:
            u = self.pref_u
        if mu is None:
            mu = 0
            
        if len(f_mean) == 0:
            f_mean = self.obs_f
            f_cov = self.obs_C
            
        # since we don't have f_cov
        if self.use_svi:
            #sample the inducing points because we don't have full covariance matrix. In this case, f_cov should be Ks_nm
            f_samples = mvn.rvs(mean=self.um_minus_mu0.flatten(), cov=self.uS, size=1000).T
            f_samples = f_cov.dot(self.invK_mm).dot(f_samples) + mu
        else:
            f_samples = mvn.rvs(mean=f_mean.flatten(), cov=f_cov, size=1000).T
             
        g_f = (f_samples[v, :] - f_samples[u, :])  / np.sqrt(2)
        phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the 
        phi = temper_extreme_probs(phi)
        if expectedlog:
            phi = np.log(phi)
            notphi = np.log(1-phi)
        else:
            notphi = 1 - phi
        
        m_post = np.mean(phi, axis=1)[:, np.newaxis]
        not_m_post = np.mean(notphi, axis=1)[:, np.newaxis]
        v_post = np.var(phi, axis=1)[:, np.newaxis]
        v_post = temper_extreme_probs(v_post, zero_only=True)
        v_post[m_post * (1 - not_m_post) <= 1e-7] = 1e-8 # fixes extreme values to sensible values. Don't think this is needed and can lower variance?

        return m_post, not_m_post, v_post 

    def _post_rough(self, f_mean, f_cov, pref_v=None, pref_u=None):
        ''' 
        When making predictions, we want to predict the probability of each listed preference pair.
        Use a solution given by applying the forward model to the mean of the latent function -- 
        ignore the uncertainty in f itself, considering only the uncertainty due to the noise sigma.
        '''
        if pref_v is None:
            pref_v = self.pref_v
        if pref_u is None:
            pref_u = self.pref_u
        
        # TODO: since we introduced the softening using f_cov, m_post is too uncertain. Perhaps try increasing rate_s to remedy this.
        m_post = self.forward_model(f_mean, f_cov[pref_v, pref_v] + f_cov[pref_u, pref_u] + f_cov[pref_v, pref_u] 
                                    + f_cov[pref_u, pref_v], v=pref_v, u=pref_u, return_g_f=False)
        m_post = temper_extreme_probs(m_post)
        not_m_post = 1 - m_post
            
        return m_post, not_m_post