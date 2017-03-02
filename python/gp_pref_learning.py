'''
Created on 18 May 2016

@author: simpson
'''

from gp_classifier_vb import coord_arr_to_1d, coord_arr_from_1d, temper_extreme_probs

#supply_update_size = True
#from gp_classifier_svi import GPGridSVI as GPClassifierVB

#supply_update_size = False
#from gp_classifier_vb import GPClassifierVB
supply_update_size = False
from gp_classifier_svi import GPClassifierSVI

import numpy as np
from scipy.stats import norm
from scipy.sparse import coo_matrix
import logging

def get_unique_locations(obs_coords_0, obs_coords_1):
    coord_rows_0 = coord_arr_to_1d(obs_coords_0)
    coord_rows_1 = coord_arr_to_1d(obs_coords_1)
    all_coord_rows = np.concatenate((coord_rows_0, coord_rows_1), axis=0)
    _, uidxs, pref_vu = np.unique(all_coord_rows, return_index=True, return_inverse=True) # get unique locations
    
    # Record the coordinates of all points that were compared
    obs_coords = np.concatenate((obs_coords_0, obs_coords_1), axis=0)[uidxs]
   
    # Record the indexes into the list of coordinates for the pairs that were compared 
    pref_v = pref_vu[:obs_coords_0.shape[0]]
    pref_u = pref_vu[obs_coords_0.shape[0]:]
    
    return obs_coords, pref_v, pref_u

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
    
    def __init__(self, dims, mu0=0, shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1, max_update_size=1000, ninducing=500, 
                 use_svi=True):
        
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
        
        super(GPPrefLearning, self).__init__(dims, mu0, shape_s0, rate_s0, s_initial, shape_ls, rate_ls, ls_initial, 
                                     force_update_all_points, n_lengthscales, max_update_size, ninducing, use_svi)
        
    def init_prior_mean_f(self, z0):
        self.mu0_default = z0 # for preference learning, we pass in the latent mean directly  
    
    def init_obs_prior(self):
        m_prior, not_m_prior, v_prior = self.post_rough(self.mu0, self.rate_s0/self.shape_s0, self.pref_v, self.pref_u)

        # find the beta parameters
        a_plus_b = 1.0 / (v_prior / (m_prior*(not_m_prior))) - 1
        a = (a_plus_b * m_prior)
        b = (a_plus_b * not_m_prior)

        self.nu0 = np.array([b, a])
        if self.verbose:
            logging.debug("Prior parameters for the observed pairwise preference variance are: %s" % str(self.nu0))           
    
    def init_obs_f(self):
        # Mean probability at observed points given local observations
        self.obs_f = np.zeros((self.n_locs, 1)) + self.mu0
        self.Ntrain = self.pref_u.size 
        
    def init_obs_mu0(self, mu0):
        self.mu0 = np.zeros((self.n_locs, 1)) + self.mu0_default
        
        if mu0 is not None and mu0[0] is not None and mu0[1] is not None:
            self.mu0_1 = mu0[0]
            self.mu0_2 = mu0[1]
            
            all_mu0 = np.concatenate((self.mu0_1, self.mu0_2), axis=0)
            self.mu0 = all_mu0[self.original_idxs] # the means corresponding to uravelled_coords
            self.mu0_1 = self.mu0[self.pref_v, :]
            self.mu0_2 = self.mu0[self.pref_u, :]
            
#     def init_s(self):
#         self.shape_s = self.shape_s0
#         self.rate_s = self.rate_s0            
#         self.s = self.shape_s / self.rate_s        
#         self.Elns = psi(self.shape_s) - np.log(self.rate_s)
#         
#         self.Ks = self.K / self.s
#         self.obs_C = self.K / self.s
#         
#         self.old_s = self.s
#         if self.verbose:
#             logging.debug("Setting the initial precision scale to s=%.3f" % self.s)    
#             
    def forward_model(self, f=[], subset_idxs=[], v=[], u=[], return_g_f=False):
        '''
        f - should be of shape nobs x 1
        
        This returns the probability that each pair has a value of 1, which is referred to as Phi(z) 
        in the chu/ghahramani paper, and the latent parameter referred to as z in the chu/ghahramani paper. 
        In this work, we use z to refer to the observations, i.e. the fraction of comparisons of a given pair with 
        value 1, so use a different label here.
        '''        
        if len(f) == 0:
            f = self.obs_f            
        if len(v) == 0:
            v = self.pref_v
        if len(u) == 0:
            u = self.pref_u
            
        if len(subset_idxs):
            if len(v) and len(u):
                # keep only the pairs that reference two items in the subet
                pair_subset = np.in1d(v, subset_idxs) & np.in1d(u, subset_idxs)
                v = v[pair_subset]
                u = u[pair_subset]
            else:
                f = f[subset_idxs]  

        if f.ndim < 2:
            f = f[:, np.newaxis]
        
        if len(v) and len(u):   
            g_f = (f[v, :] - f[u, :]) / np.sqrt(2) # / np.sqrt(self.s)) # gives an NobsxNobs matrix
        else: # provide the complete set of pairs
            g_f = (f - f.T) / np.sqrt(2) # / np.sqrt(self.s))  # the maths shows that s cancels out -- it's already 
            # included in our estimates of f, which are scaled by s. However, the prior mean mu0 should also be scaled
            # to match, but this should happen automatically if we learn s, I think. 
                
        phi = norm.cdf(g_f) # the probability of the actual observation, which takes g_f as a parameter. In the 
        # With the standard GP density classifier, we can skip this step because
        # g_f is already a probability and Phi(z) is a Bernoulli distribution.
        if return_g_f:
            return phi, g_f
        else:
            return phi
    
    def update_sample_idxs(self):
        nobs = self.obs_f.shape[0]        
        self.data_idx_i = np.random.choice(nobs, self.update_size, replace=False)
        self.data_obs_idx_i = np.in1d(self.pref_v, self.data_idx_i) & np.in1d(self.pref_u, self.data_idx_i)        
    
    def update_jacobian(self, G_update_rate=1.0):
        phi, g_mean_f = self.forward_model(return_g_f=True) # first order Taylor series approximation
            
        J = 1 / (2*np.pi)**0.5 * np.exp(-g_mean_f**2 / 2.0) * np.sqrt(0.5)
        obs_idxs = np.arange(self.n_locs)[np.newaxis, :]
        
        if len(self.data_obs_idx_i): 
            obs_idxs = obs_idxs[:, self.data_idx_i]
            J = J[self.data_obs_idx_i, :]
            s = (self.pref_v[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int) -\
                                                    (self.pref_u[self.data_obs_idx_i, np.newaxis]==obs_idxs).astype(int)
        else:    
            s = (self.pref_v[:, np.newaxis]==obs_idxs).astype(int) - (self.pref_u[:, np.newaxis]==obs_idxs).astype(int)
            
        J = J * s 
        
        if self.G is None or not np.any(self.G) or self.G.shape != J.shape: 
            # either G has not been initialised, or is from different observations:
            self.G = J
        else:        
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
            
        return phi
    
    def count_observations(self, obs_coords, n_obs, poscounts, totals):
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
                            
        if obs_coords_0.dtype=='int': # duplicate locations should be merged and the number of duplicates counted
            poscounts = poscounts.astype(int)
            totals = totals.astype(int)        
            
            # Ravel the coordinates
            ravelled_coords_0 = coord_arr_to_1d(obs_coords_0)
            ravelled_coords_1 = coord_arr_to_1d(obs_coords_1) 
            
            # SWAP PAIRS SO THEY ALL HAVE LOWEST COORD FIRST so we can count prefs for duplicate location pairs
            # get unique keys
            all_ravelled_coords = np.concatenate((ravelled_coords_0, ravelled_coords_1), axis=0)
            uravelled_coords, origidxs, keys = np.unique(all_ravelled_coords, return_index=True, return_inverse=True)
            keys_0 = keys[:len(ravelled_coords_0)]
            keys_1 = keys[len(ravelled_coords_0):]
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
            
            self.original_idxs = origidxs[ukeys] # indexes of unique observation locations into the original input data
            
            # Record the coordinates of all points that were compared
            self.obs_coords = coord_arr_from_1d(uravelled_coords[ukeys], obs_coords_0.dtype, 
                                                dims=(len(ukeys), obs_coords_0.shape[1]))
            
            # Record the indexes into the list of coordinates for the pairs that were compared 
            self.pref_v = pref_vu[:len(nonzero_v)]
            self.pref_u = pref_vu[len(nonzero_v):]
                   
            # Return the counts for each of the observed pairs
            return grid_obs_pos_counts[nonzero_v, nonzero_u], grid_obs_counts[nonzero_v, nonzero_u]
                    
        elif obs_coords_0.dtype=='float':
            self.obs_coords, self.pref_v, self.pref_u = get_unique_locations(obs_coords_0, obs_coords_1)
            
            return poscounts, totals # these remain unaltered as we have not de-duplicated            
            
    def fit(self, items_1_coords, items_2_coords, obs_values, totals=None, process_obs=True, update_s=True, mu0_1=None,
            mu0_2=None):
        super(GPPrefLearning, self).fit((items_1_coords, items_2_coords), obs_values, totals, process_obs, update_s, 
                                mu0=(mu0_1, mu0_2))  
        
    def predict_f(self, items_coords=[], max_block_size=1e5, mu0_output=None):
        nblocks, noutputs = self.init_output_arrays(items_coords, max_block_size)
                
        if mu0_output is not None and len(mu0_output):
            self.mu0_output = mu0_output
        else:
            self.mu0_output = np.zeros((noutputs, 1)) + self.mu0_default
                
        for block in range(nblocks):
            if self.verbose:
                logging.debug("GPClassifierVB predicting block %i of %i" % (block, nblocks))            
            self.predict_block(block, max_block_size, noutputs)
        
        return self.f, self.v

    def predict(self, items_0_coords=[], items_1_coords=[], max_block_size=1e5, 
                expectedlog=False, return_var=True, return_not=False, mu0_output1=None, mu0_output2=None):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        # if no output_coords provided, give predictions at the fitted locations
        if not len(items_0_coords) and not len(items_1_coords):
            return_args = self.predict_obs('rough', False, return_not)
            if expectedlog:
                return_args[0] = np.log(return_args[0])
                if len(return_args) == 3:
                    return_args[1] = np.log(return_args[1])
            return return_args
        
        if not isinstance(items_0_coords, np.ndarray):
            items_0_coords = np.array(items_0_coords)
        if items_0_coords.ndim==2 and items_0_coords.shape[1]!=len(self.dims) and items_0_coords.shape[0]==len(self.dims):
            items_0_coords = items_0_coords.T
            
        if not len(items_1_coords):
            items_1_coords = items_0_coords.copy()
            
        if items_1_coords.ndim==2 and items_1_coords.shape[1]!=len(self.dims) and \
                            items_1_coords.shape[0]==len(self.dims):
            items_1_coords = items_1_coords.T       
        
        output_coords, out_pref_v, out_pref_u = get_unique_locations(items_0_coords, items_1_coords)
        nblocks, noutputs = self.init_output_arrays(output_coords, max_block_size)
                
        self.mu0_output = np.zeros((noutputs, 1)) + self.mu0_default
        if mu0_output1 is not None:
            self.mu0_output[out_pref_v, :] = mu0_output1
        if mu0_output2 is not None:
            self.mu0_output[out_pref_u, :] = mu0_output2
                
        for block in range(nblocks):
            if self.verbose:
                logging.debug("GPClassifierVB predicting block %i of %i" % (block, nblocks))            
            self.predict_block(block, max_block_size, noutputs)

        m_post, not_m_post, v_post = self.post_rough(self.f, self.v, out_pref_v, out_pref_u)
        
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
        
    def post_rough(self, f_mean, f_var=None, pref_v=None, pref_u=None):
        ''' 
        When making predictions, we want to predict the probability of each listed preference pair.
        Use a solution given by applying the forward model to the mean of the latent function -- 
        ignore the uncertainty in f itself, considering only the uncertainty due to the noise sigma.
        '''
        if pref_v is None:
            pref_v = self.pref_v
        if pref_u is None:
            pref_u = self.pref_u
        
        m_post = self.forward_model(f_mean, v=pref_v, u=pref_u, return_g_f=False)
        m_post = temper_extreme_probs(m_post)
        
        not_m_post = 1 - m_post

        if f_var is not None:
            f_samples = norm.rvs(loc=f_mean, scale=np.sqrt(f_var), size=(f_mean.shape[0], 5000))
            rho_samples = self.forward_model(f_samples, v=pref_v, u=pref_u, return_g_f=False)
            v_post = np.var(rho_samples, axis=1)[:, np.newaxis]
            v_post = temper_extreme_probs(v_post, zero_only=True)
            v_post[m_post * (1 - not_m_post) <= 1e-7] = 1e-8 # important to make sure our fixes for extreme values lead
            # to sensible values
            return m_post, not_m_post, v_post            
        else:        
            return m_post, not_m_post
        
    def logpt(self):
        rho = self.forward_model(self.obs_f)
        rho = temper_extreme_probs(rho)
        logrho_rough = np.log(rho)
        lognotrho_rough = np.log(1 - rho)   
        #logging.debug("Approximation error in rho =%.4f" % np.max(np.abs(logrho - logrho_rough)))
        #logging.debug("Approximation error in notrho =%.4f" % np.max(np.abs(lognotrho - lognotrho_rough)))
        return logrho_rough, lognotrho_rough     

def matern_3_2(distances, ls):
    K = np.zeros(distances.shape)
    for d in range(distances.shape[2]):
        K[:, :, d] = np.abs(distances[:, :, d]) * 3**0.5 / ls[d]
        K[:, :, d] = (1 + K[:, :, d]) * np.exp(-K[:, :, d])
    K = np.prod(K, axis=2)
    return K

def matern_3_2_from_raw_vals(vals, ls):
    distances = np.zeros((vals[0].size, vals[0].size, vals.shape[0]))
    for i, xvals in enumerate(vals):
        if xvals.ndim == 1:
            xvals = xvals[:, np.newaxis]
        elif xvals.shape[0] == 1 and xvals.shape[1] > 1:
            xvals = xvals.T
        xdists = xvals - xvals.T
        distances[:, :, i] = xdists
    K = matern_3_2(distances, ls)
    return K


def gen_synthetic_prefs(f_prior_mean=None, nx=100, ny=100):
    # f_prior_mean should contain the means for all the grid squares
    
    # Generate some data
    ls = [10, 10]
    sigma = 0.1 
    N = 1000
    P = 10000 # number of pairs for training
    s = 1 # inverse precision scale for the latent function.
    
    # Some random feature values
    xvals = np.random.choice(nx, N, replace=True)[:, np.newaxis]
    yvals = np.random.choice(ny, N, replace=True)[:, np.newaxis]
    # remove repeated coordinates
    for coord in range(N):
        
        while np.sum((xvals==xvals[coord]) & (yvals==yvals[coord])) > 1:
            xvals[coord] = np.random.choice(nx, 1)
            yvals[coord] = np.random.choice(ny, 1)           
        
    K = matern_3_2_from_raw_vals(np.array([xvals, yvals]), ls)
    from scipy.stats import multivariate_normal as mvn
    if f_prior_mean is None:
        f = mvn.rvs(cov=K/s) # zero mean        
    else:
        f = mvn.rvs(mean=f_prior_mean[xvals, yvals].flatten(), cov=K/s) # zero mean
    
    # generate pairs indices
    pair1idxs = np.random.choice(N, P, replace=True)
    pair2idxs = np.random.choice(N, P, replace=True)
    
    # remove indexes of pairs that compare the same data points -- the correct answer is trivial
    while(np.sum(pair1idxs==pair2idxs)):
        matchingidxs = pair1idxs==pair2idxs
        pair2idxs[matchingidxs] = np.random.choice(N, np.sum(matchingidxs), replace=True)
      
    # generate the noisy function values for the pairs
    f1noisy = norm.rvs(scale=sigma, size=P) + f[pair1idxs]
    f2noisy = norm.rvs(scale=sigma, size=P) + f[pair2idxs]
    
    # generate the discrete labels from the noisy preferences
    prefs = f1noisy > f2noisy
    
    return N, nx, ny, prefs, xvals, yvals, pair1idxs, pair2idxs, f, K

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    
    
    from scipy.stats import kendalltau
    
    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)
    
    N, nx, ny, prefs, xvals, yvals, pair1idxs, pair2idxs, f, _ = gen_synthetic_prefs()
    pair1coords = np.concatenate((xvals[pair1idxs, :], yvals[pair1idxs, :]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs, :], yvals[pair2idxs, :]), axis=1)   
        
    # separate training and test data
    Ptest = 100
    testpairs = np.random.choice(pair1coords.shape[0], Ptest, replace=False)
    testidxs = np.zeros(pair1coords.shape[0], dtype=bool)
    testidxs[testpairs] = True
    trainidxs = np.invert(testidxs)
    
    xvals_test = np.array([xvals[pair1idxs[testidxs]], xvals[pair2idxs[testidxs]]]).flatten()
    yvals_test = np.array([yvals[pair1idxs[testidxs]], yvals[pair2idxs[testidxs]]]).flatten()
    _, uidxs = np.unique(coord_arr_to_1d(np.concatenate((xvals_test[:, np.newaxis], yvals_test[:, np.newaxis]), axis=1)), 
              return_index=True)
    xvals_test = xvals_test[uidxs][:, np.newaxis]
    yvals_test = yvals_test[uidxs][:, np.newaxis]
    f_test = np.concatenate((f[pair1idxs[testidxs]], f[pair2idxs[testidxs]]))[uidxs]
    
    # Create a GPPrefLearning model
    model = GPPrefLearning([nx, ny], mu0=0, shape_s0=1, rate_s0=1, ls_initial=[10, 10], max_update_size=100, 
                           ninducing=200, use_svi=True)    
    #model.verbose = True
    model.max_iter_VB = 1000
    model.min_iter_VB = 5
    model.uselowerbound = True
    #model.conv_threshold_G = 1e-8
    #model.conv_check_freq = 1
    #model.conv_threshold = 1e-3 # the difference must be less than 1% of the value of the lower bound
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
    
    model.fit(pair1coords[trainidxs], pair2coords[trainidxs], prefs[trainidxs])
    print "Final lower bound: %f" % model.lowerbound()
    
    # Predict at the test locations
    fpred, vpred = model.predict_f(np.concatenate((xvals_test, yvals_test), axis=1))
    
    # Compare the observation point values with the ground truth
    obs_coords_1d = coord_arr_to_1d(model.obs_coords)
    test_coords_1d = coord_arr_to_1d(np.concatenate((xvals, yvals), axis=1))
    f_obs = [f[(test_coords_1d==obs_coords_1d[i]).flatten()][0] for i in range(model.obs_coords.shape[0])]
    print "Kendall's tau (observations): %.3f" % kendalltau(f_obs, model.obs_f.flatten())[0]
        
    # Evaluate the accuracy of the predictions
    #print "RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
    #print "NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
    print "Kendall's tau (test): %.3f" % kendalltau(f_test, fpred)[0] 
        
    t = (f[pair1idxs[testidxs]] > f[pair2idxs[testidxs]]).astype(int)
    rho_pred, var_rho_pred = model.predict(pair1coords[testidxs], pair2coords[testidxs])
    rho_pred = rho_pred.flatten()
    t_pred = np.round(rho_pred)
    
    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    print "Brier score of %.3f" % np.sqrt(np.mean((t-rho_pred)**2))
    print "Cross entropy error of %.3f" % -np.sum(t * np.log(rho_pred) + (1-t) * np.log(1 - rho_pred))    
    
    from sklearn.metrics import f1_score, roc_auc_score
    print "F1 score of %.3f" % f1_score(t, t_pred)
    print "Accuracy of %.3f" % np.mean(t==t_pred)
    print "ROC of %.3f" % roc_auc_score(t, rho_pred)
    
    # looks like we don't correct the modified order of points in predict() that occurs in the unique locations function 
    