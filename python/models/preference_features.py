'''
TODO 1: lower bound should use inducing points in SVI option

Preference learning model for identifying relevant input features of items and people, plus finding latent 
characteristics of items and people. Can be used to predict preferences or rank items, therefore could be part of
a recommender system. In this case the method uses both collaborative filtering and item-based similarity.

For preference learning, we use a GP as in Chu and Ghahramani. This has been modified to use VB updates to integrate 
into the complete VB framework and allow an SVI adaptation for scalability. 

For finding the latent factors, we modify the model of Archambeau and Bach 2009 to consider correlations between 
items and people using a GP kernel. We assume only one "view" (from their terminology) that best fits the preferences.
Multiple views could be worthwhile if predicting multiple preference functions?
According to A&B 2009, the generative model for probabilistic projection includes several techniques as special cases:
 - diagonal priors on y gives probabilistic factor analysis
 - isotropic priors give probabilistic PCA
 - our model doesn't allow other specific types, but is intended to be used more generally instead
From A&B'09: "PCA suffers from the fact that each principal component is a linear combination of all the original 
variables. It is thus often difficult to interpret the results." A sparse representation would be easier to interpret. 
The aim is to use as few components as are really necessary, and allow priors to determine a trade-off between sparsity
(and interpretability; possibly also avoidance of overfitting/better generalisation) and accuracy of the low-dimensional
representation (avoiding loss). In preference learning, we would like to be able to predict unseen values in a person-item
matrix when we have limited, noisy data. A sparse representation seems applicable as it would avoid overfitting and 
allow personalised noise models to represent complexity so that the shared latent features are more easily interpretable
as common characteristics. 

Our implementation here is similar to the inverse Gamma prior on the weight precision
proposed by A&B'09, but we use a gamma prior that is conjugate to the Gaussian instead. This makes inference simpler
but may have the disadvantage of not enforcing sparseness so strictly -- it is not clear from A&B'09 why they chose
the non-conjugate option. They also use completely independent scale variables for each weight in the weight matrix,
i.e. for each item x factor pair. We have correlations between items with similar features through a kernel function, 
but we also use a common scale parameter for each feature. This induces sparsity over the features, i.e. reduces the
number of features used but means that all items will have an entry for all the important features. It's unclear what
difference this makes -- perhaps features that are weakly supported by small amounts of data for one item will be pushed
to zero by A&B approach, while our approach will allow them to vary more since the feature is important for other items.
The A&B approach may make more sense for representing rare but important features; our approach would not increase their 
scale so that items that do possess the feature will not be able to have a large value and the feature may disappear?
Empirical tests may be needed here. 

The approach is similar to Khan et al. 2014. "Scalable Collaborative Bayesian Preference Learning" but differs in that
we also place priors over the weights and model correlations between different items and between different people.
Our use of priors that encourage sparseness in the features is also different. 

Observed features -- why is it good to use them as inputs to latent features? 
-- we assume some patterns in the observations are common to multiple people, and these manifest as latent features
-- we can use the GP model to map observations to latent features to handle sparsity of data for each item
and person
-- the GP will model dependencies between the input features
An alternative would be a flat model, where the input features for items were added to columns of w, 
and the input features of people created new rows in y. This may make it easier to learn which features are relevant,
but does not help with sparse features because we could not use a GP to smooth and interpolate between items, so 
would need mode observed preference pairs for each item and person to determine their latent feature values.  

For testing effects of no. inducing points, forgetting rate, update size, delay, it would be useful to see accuracy and 
convergence rate.

Created on 2 Jun 2016

@author: simpson
'''

import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import multivariate_normal as mvn
import logging
from gp_classifier_vb import matern_3_2, diagonal, sq_exp_cov, deriv_matern_3_2, deriv_sq_exp_cov
from gp_pref_learning import GPPrefLearning, get_unique_locations, pref_likelihood
from scipy.linalg import cholesky, solve_triangular, block_diag
from scipy.special import gammaln, psi
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans

def expec_output_scale(shape_s0, rate_s0, N, cholK, f_mean, m, f_cov):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N
    L_expecFF = solve_triangular(cholK, f_cov + f_mean.dot(f_mean.T) - m.dot(f_mean.T) -f_mean.dot(m.T) + m.dot(m.T), 
                                 trans=True, overwrite_b=True, check_finite=False)
    LT_L_expecFF = solve_triangular(cholK, L_expecFF, overwrite_b=True, check_finite=False)
    rate_s = rate_s0 + 0.5 * np.trace(LT_L_expecFF) 
    
    return shape_s/rate_s, shape_s, rate_s

def expec_output_scale_svi(shape_s0, rate_s0, N, invK, f_mean, m, invK_f_cov):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N
    invK_expecFF = invK_f_cov + invK.dot( (f_mean - m).dot(f_mean.T - m.T) )
    rate_s = rate_s0 + 0.5 * np.trace(invK_expecFF) 
    
    return shape_s, rate_s

def lnp_output_scale(shape_s0, rate_s0, shape_s, rate_s):
    s = shape_s / rate_s
    Elns = psi(shape_s) - np.log(rate_s)
    
    logprob_s = - gammaln(shape_s0) + shape_s0 * np.log(rate_s0) + (shape_s0-1) * Elns - rate_s0 * s
    return logprob_s            
        
def lnq_output_scale(shape_s, rate_s):
    s = shape_s / rate_s
    Elns = psi(shape_s) - np.log(rate_s)
    
    lnq_s = - gammaln(shape_s) + shape_s * np.log(rate_s) + (shape_s-1) * Elns - rate_s * s
    return lnq_s
    
def svi_update_gaussian(invQi_y, mu0_n, mu_u, K_mm, invK_mm, K_nm, K_im, K_nn, invQi, prev_invS, prev_invSm, vb_iter, 
                        delay, forgetting_rate, N, update_size):
    Lambda_factor1 = invK_mm.dot(K_im.T)
    Lambda_i = Lambda_factor1.dot(invQi).dot(Lambda_factor1.T)
    
    # calculate the learning rate for SVI
    rho_i = (vb_iter + delay) ** (-forgetting_rate)
    #print "\rho_i = %f " % rho_i
    
    # weighting. Lambda and 
    w_i = N / float(update_size)
    
    # S is the variational covariance parameter for the inducing points, u. 
    # Canonical parameter theta_2 = -0.5 * S^-1.
    # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over 
    # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.  
    invS = (1 - rho_i) * prev_invS + rho_i * (w_i * Lambda_i + invK_mm)
    
    # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y  
#     invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * invK_mm.dot(K_im.T).dot(invQi).dot(y)
    invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * Lambda_factor1.dot(invQi_y)
    
    # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
    L_invS = cholesky(invS.T, lower=True, check_finite=False)
    B = solve_triangular(L_invS, invK_mm.T, lower=True, check_finite=False)
    A = solve_triangular(L_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
    invK_mm_S = A.T
    
    fhat_u = solve_triangular(L_invS, invSm, lower=True, check_finite=False)
    fhat_u = solve_triangular(L_invS, fhat_u, lower=True, trans=True, check_finite=False, overwrite_b=True)
    fhat_u += mu_u
    
    covpair =  K_nm.dot(invK_mm)
    covpair_uS = K_nm.dot(invK_mm_S)
    fhat = covpair_uS.dot(invSm) + mu0_n
    C = K_nn + (covpair_uS - covpair.dot(K_mm)).dot(covpair.T)
    return fhat, C, invS, invSm, fhat_u, invK_mm_S

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, nitem_features, nperson_features=0, mu0=0, mu0_y=0, shape_s0=1, rate_s0=1, shape_ls=1, rate_ls=100, 
                 ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=3, use_fa=False,  
                 kernel_func='matern_3_2', max_update_size=10000, ninducing=500, use_svi=True, forgetting_rate=0.9, 
                 delay=1.0):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        # if use_svi is switched off, we revert to the standard (parent class) VB implementation
        self.use_svi = use_svi
        self.use_svi_people = False # this gets switched on later if we have features and correlations between people
        
        self.people = None
        self.pref_gp = {}
        self.nitem_features = nitem_features
        self.nperson_features = nperson_features
        self.mu0 = mu0 # these are abstract latent functions, not related to any real-world variables: mu0=0 by default
        # other means should be provided later so that we can put priors on type of person liking type of object
        
        # for preference learning, the scale of the function is divided out when making predictions. Allowing it to vary
        # a lot means the function can grow unnecessarily in certain situations. However, higher values of s will mean 
        # smaller covariance relative to the noise, Q, and allow less learning from the observations. Increasing s is 
        # therefore similar to increasing nu0 in IBCC.  The relative sizes of s also determine how noisy each person's 
        # preferences are -- relationship between s_f and s_t and var(wy). For example, two people could have broadly
        # the same preferences over feature space, so same length scales, but one person has more variation/outliers,
        # so is more noisy. Another person who deviates a lot more from everyone else will also have a higher variance.
        # However, in preference learning, if s is allowed to vary a lot, the variance could grow for some people more 
        # than others based on little information, and make some people's f functions dominate over others -- thus a 
        # reasonably high shape_sf0 value is recommended, or setting shape_sf0/rate_sf0 >> shape_st0 / rate_st0 so that
        # the personal noise is not dominant; however shapes should not be too high so that there is no variation 
        # between people. Perhaps this is best tuned using ML2?
        # In practice: smallish precision scales s seem to work well, but small shape_s0 values and very small s values
        # should be avoided as they lead to errors.
        self.shape_sf0 = shape_s0
        self.rate_sf0 = rate_s0
                
        # For the latent means and components, the relative sizes of the scales controls how much the components can 
        # vary relative to the overall f, i.e. how much they learn from f. A high s will mean that the wy & t functions 
        # have smaller scale relative to f, so they will be less fitted to f. By default we assume a common prior.   
        self.shape_sw0 = shape_s0
        self.rate_sw0 = rate_s0
                            
        self.shape_sy0 = shape_s0
        self.rate_sy0 = rate_s0 #* 100 --> this made it detect features, but not very well.
    
        # if the scale doesn't matter, then let's fix the mean to be scaled to one? However, fixing t's scale and not
        # the noise scale in f means that since preference learning can collapse toward very large scales, the noise
        # can grow large and the shared mean t has less influence. So it makes sense to limit noise and  
        self.shape_st0 = shape_s0
        self.rate_st0 = rate_s0
        
        # y has different length-scales because it is over user features space
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        self.ls = ls
        
        self.shape_lsy = shape_lsy
        self.rate_lsy = rate_lsy
        self.lsy = lsy  
        
        self.t_mu0 = 0
        if use_svi:
            self.t_mu0_u = 0
        
        self.conv_threshold = 1e-1
        self.max_iter = 100
        self.min_iter = 3
        if self.use_svi:
            self.n_converged = 10 # number of iterations while apparently converged (avoids numerical errors)
        else:
            self.n_converged = 3
        self.vb_iter = 0
        
        self.verbose = verbose
        
        self.Nfactors = nfactors
        
        self.use_fa = use_fa # flag to indicate whether to use the simple factor analysis ML update instead of the VB GP
        
        self.max_update_size = max_update_size # maximum number of data points to update in each SVI iteration
        
        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = forgetting_rate
        self.delay = delay # delay must be at least 1
        
        # number of inducing points
        self.ninducing = ninducing
        
        self._select_covariance_function(kernel_func)
        
        self.matches = {} # indexes of the obs_coords in the child noise GPs 
            
    def _select_covariance_function(self, cov_type):
        self.cov_type = cov_type
        if cov_type == 'diagonal':
            self.kernel_func = diagonal
            def zerocovder(distance, ls, dim): 
                return 0
            self.kernel_der = zerocovder 
        elif cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2
            self.kernel_der = deriv_matern_3_2
        elif cov_type == 'sq_exp':
            self.kernel_func = sq_exp_cov
            self.kernel_der = deriv_sq_exp_cov
        else:
            logging.error('GPClassifierVB: Invalid covariance type %s' % cov_type)        
    
        
    def _init_params(self):
        self.N = len(self.obs_coords)
        self.Npeople = np.max(self.people).astype(int) + 1
        
        self.f = np.zeros((self.Npeople, self.N))
        self.w_cov = np.diag(np.ones(self.N*self.Nfactors)) # use ones to avoid divide by zero
        self.y_cov = np.diag(np.ones(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
        
        self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0
        self.t = np.zeros((self.N, 1))
        self.t_cov = np.diag(np.ones(self.N))

        # put all prefs into a single GP to get a good initial mean estimate t -- this only makes sense if we can also 
        #estimate w y in a sensibel way, e.g. through factor analysis?        
        #self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs)
        
        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0
        self.shape_sy = np.zeros(self.Nfactors) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors) + self.rate_sy0
        self.shape_st = self.shape_st0
        self.rate_st = self.rate_st0                
        
        self.invKf = {}
        self.coordidxs = {}
        
        if self.new_obs:
            for person in self.people:
                self.pref_gp[person] = GPPrefLearning(self.nitem_features, self.mu0, self.shape_sf0, self.rate_sf0,
                                        self.shape_ls, self.rate_ls, self.ls, use_svi=self.use_svi, delay=self.delay, 
                                        forgetting_rate=self.forgetting_rate)
                self.pref_gp[person]._select_covariance_function(self.cov_type)
                self.pref_gp[person].max_iter_VB = 1
                self.pref_gp[person].min_iter_VB = 1
                self.pref_gp[person].max_iter_G = 5
                self.pref_gp[person].verbose = self.verbose
                
        distances = np.zeros((self.N, self.N, self.nitem_features))
        for d in range(self.nitem_features):
            distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
        # kernel used by t
        self.ls = np.zeros(self.nitem_features) + self.ls
        self.K = self.kernel_func(distances, self.ls)
        self.item_distances = distances
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        if not self.use_svi:
            self.invK = np.linalg.inv(self.K)
        
        # kernel used by w
        blocks = [self.K for _ in range(self.Nfactors)]
        self.Kw = block_diag(*blocks)
        self.sw_matrix = np.ones(self.Kw.shape) * self.shape_sw0 / self.rate_sw0
                
        # kernel used by y  
        if self.person_features is None:
            self.Ky_block = np.diag(np.ones(self.Npeople))
        else:
            self.lsy = np.zeros(self.nperson_features) + self.lsy  
            pdistances = np.zeros((self.Npeople, self.Npeople, self.nperson_features))
            for d in range(self.nperson_features):
                pdistances[:, :, d] = self.person_features[:, d:d+1] - self.person_features[:, d:d+1].T
            self.Ky_block = self.kernel_func(pdistances, self.lsy)
            self.person_distances = distances
           
        blocks = [self.Ky_block for _ in range(self.Nfactors)]
        self.Ky = block_diag(*blocks) 
        self.cholKy = cholesky(self.Ky_block, overwrite_a=False, check_finite=False)
        self.sy_matrix = np.ones(self.Ky.shape) * self.shape_sy0 / self.rate_sy0     

        # Factor Analysis
        if self.use_fa:                        
            self.fa = FactorAnalysis(n_components=self.Nfactors)        
        elif self.use_svi:
            self._choose_inducing_points()
        
        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them,
        # i.e. the cluster identifiability problem
        self.y = mvn.rvs(np.zeros(self.Nfactors * self.Npeople), cov=self.Ky / self.sy_matrix).reshape((self.Nfactors, 
                                                                                                        self.Npeople))
        if not self.use_svi:
            self.w = mvn.rvs(np.zeros(self.Nfactors * self.N), cov=self.Kw / self.sw_matrix).reshape((self.Nfactors, 
                                                                                                      self.N)).T    
        else:
            self.w = np.zeros((self.N, self.Nfactors))
        self.wy = self.w.dot(self.y)
        
    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        nobs = self.obs_coords.shape[0]
        
        self.update_size = self.max_update_size # number of inducing points in each stochastic update        
        if self.update_size > nobs:
            self.update_size = nobs 
           
        # For w and t               
        if self.ninducing > self.obs_coords.shape[0]:
            self.ninducing = self.obs_coords.shape[0]
        
        self.w_invSm = np.zeros((self.ninducing * self.Nfactors, 1), dtype=float)# theta_1
        self.w_invS = np.zeros((self.ninducing * self.Nfactors, self.ninducing * self.Nfactors), dtype=float) # theta_2

        self.t_invSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1
        self.t_invS = np.zeros((self.ninducing, self.ninducing), dtype=float) # theta_2             
                
        init_size = 300
        if self.ninducing > init_size:
            init_size = self.ninducing
        kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing)
        kmeans.fit(self.obs_coords)
        
        self.inducing_coords = kmeans.cluster_centers_

        self.t_mu0_u = np.zeros((self.ninducing, 1)) + self.t_mu0_u
        
        mm_dist = np.zeros((self.ninducing, self.ninducing, self.nitem_features))
        nm_dist = np.zeros((nobs, self.ninducing, self.nitem_features))
        for d in range(self.nitem_features):
            mm_dist[:, :, d] = self.inducing_coords[:, d:d+1].T - self.inducing_coords[:, d:d+1]
            nm_dist[:, :, d] = self.inducing_coords[:, d:d+1].T - self.obs_coords[:, d:d+1].astype(float)
         
        self.K_mm = self.kernel_func(mm_dist, self.ls) + 1e-6 * np.eye(self.ninducing) # jitter
        self.invK_mm = np.linalg.inv(self.K_mm)
        self.K_nm = self.kernel_func(nm_dist, self.ls)
        
        blocks = [self.K_mm for _ in range(self.Nfactors)]
        self.Kw_mm = block_diag(*blocks)
        blocks = [self.invK_mm for _ in range(self.Nfactors)]
        self.invKw_mm = block_diag(*blocks)
        blocks = [self.K_nm for _ in range(self.Nfactors)]
        self.Kw_nm = block_diag(*blocks)        
                
        #self.w_u = mvn.rvs(np.zeros(self.Nfactors * self.ninducing), cov=self.Kw_mm).reshape((self.Nfactors, self.ninducing)).T
        #self.w_u *= (self.shape_sw/self.rate_sw)[np.newaxis, :]
        self.w_u = np.zeros((self.ninducing, self.Nfactors))
        self.t_u = np.zeros((self.ninducing, 1))                 
        self.f_u = {}
                
        for person in self.pref_gp:
            self.pref_gp[person].fix_inducing_points(self.inducing_coords)
                
        # sort this out when we call updates to s
        #self.shape_s = self.shape_s0 + 0.5 * self.ninducing # update this because we are not using n_locs data points -- needs replacing?

        # For y
        if self.person_features is not None:
            self.use_svi_people = True
                        
            self.y_update_size = self.max_update_size # number of inducing points in each stochastic update            
            if self.y_update_size > self.Npeople:
                self.y_update_size = self.Npeople       
            
            self.y_ninducing = self.ninducing           
            if self.y_ninducing > self.people.shape[0]:
                self.y_ninducing = self.people.shape[0]
            
            self.y_u = np.zeros((self.Nfactors, self.y_ninducing))
            
            init_size = 300
            if self.y_ninducing > init_size:
                init_size = self.y_ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.y_ninducing)
            kmeans.fit(self.person_features)
            
            self.y_inducing_coords = kmeans.cluster_centers_
    
            self.y_invSm = np.zeros((self.y_ninducing * self.Nfactors, 1), dtype=float)# theta_1
            self.y_invS = np.zeros((self.y_ninducing * self.Nfactors, self.y_ninducing * self.Nfactors), dtype=float) # theta_2
    
            mm_dist = np.zeros((self.y_ninducing, self.y_ninducing, self.nperson_features))
            # should be okay up to here.    
            nm_dist = np.zeros((self.Npeople, self.y_ninducing, self.nperson_features))
            for d in range(self.nperson_features):
                mm_dist[:, :, d] = self.y_inducing_coords[:, d:d+1].T - self.y_inducing_coords[:, d:d+1]
                nm_dist[:, :, d] = self.y_inducing_coords[:, d:d+1].T - self.person_features[:, d:d+1].astype(float)
             
            self.Ky_mm_block = self.kernel_func(mm_dist, self.lsy)
            self.Ky_mm_block += 1e-6 * np.eye(len(self.Ky_mm_block)) # jitter 
            blocks = [self.Ky_mm_block for _ in range(self.Nfactors)]
            self.Ky_mm = block_diag(*blocks)            
            
            self.invKy_mm_block = np.linalg.inv(self.Ky_mm_block)
            blocks = [self.invKy_mm_block for _ in range(self.Nfactors)]
            self.invKy_mm = block_diag(*blocks)  
            
            self.Ky_nm = self.kernel_func(nm_dist, self.lsy)
            blocks = [self.Ky_nm for _ in range(self.Nfactors)]
            self.Ky_nm = block_diag(*blocks)
            
        else:
            self.use_svi_people = False
            
    def fit(self, personIDs=None, items_1_coords=None, items_2_coords=None, preferences=None, person_features=None, 
            _optimize=False, maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):
        '''
        Learn the model with data as follows:
        personIDs - a list of the person IDs of the people who expressed their preferences
        items_1_coords - coordinates of the first items in the pairs being compared
        items_2_coords - coordinates of the second items in each pair being compared
        preferences - the values, 0 or 1 to express that item 1 was preferred to item 2.
        '''
        if _optimize:
            return self._optimize(personIDs, items_1_coords, items_2_coords, preferences, person_features, maxfun, 
                                 use_MAP, nrestarts, input_type)
        
        
        if personIDs is not None:
            self.new_obs = True # there are people we haven't seen before            
            # deal only with the original IDs to simplify prediction steps and avoid conversions 
            self.people = np.unique(personIDs)
            self.personIDs = personIDs               
            self.obs_coords, self.pref_v, self.pref_u, self.obs_uidxs = get_unique_locations(items_1_coords, items_2_coords)
            self.person_features = person_features # rows per person, columns for feature values
            
            self.items_1_coords = items_1_coords
            self.items_2_coords = items_2_coords
            self.preferences = np.array(preferences)
                  
        else:  
            self.new_obs = False # do we have new data? If so, reset everything. If not, don't reset the child GPs.
            items_1_coords = self.items_1_coords
            items_2_coords = self.items_2_coords
            preferences = self.preferences
 
        self.input_type = input_type
        self._init_params()
        
        # reset the iteration counters
        self.vb_iter = 0    
        diff = np.inf
#         old_w = np.inf
        old_lb = -np.inf
        converged_count = 0
        while (self.vb_iter < self.min_iter) or (((diff > self.conv_threshold) or (converged_count < self.n_converged)) 
                                                 and (self.vb_iter < self.max_iter)):
            if self.use_svi:
                self._update_sample()
                
            # run a VB iteration
            # compute preference latent functions for all workers
            self._expec_f()
            
            # compute the preference function means
            self._expec_t()            
            
            # find the personality components
            self._expec_w()
             
#             diff = np.max(old_w - self.w)
#             logging.debug( "Difference in latent item features: %f" % diff)
#             old_w = self.w

            # Don't use lower bound here, it doesn't really make sense when we use ML for some parameters
            if not self.use_fa:
                lb = self.lowerbound()
                if self.verbose:
                    logging.debug('Iteration %i: lower bound = %.5f, difference = %.5f' % (self.vb_iter, lb, lb-old_lb))
                diff = lb - old_lb
                old_lb = lb

            self.vb_iter += 1
            
            if diff <= self.conv_threshold:
                converged_count += 1
            
        logging.debug( "Preference personality model converged in %i iterations." % self.vb_iter )

    def _optimize(self, personIDs, items_1_coords, items_2_coords, preferences, person_features=None, 
                 maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):

        max_iter = self.max_iter
        self.max_iter = 1 # set this temporarily
        self.fit(personIDs, items_1_coords, items_2_coords, preferences, person_features, input_type=input_type)
        self.max_iter = max_iter

        for d, ls in enumerate(self.ls):
            min_nlml = np.inf
            best_opt_hyperparams = None
            best_iter = -1            
            
            logging.debug("Optimising item length-scale for %i dimension" % d)
            
            nfits = 0 # number of calls to fit function
            
            # optimise each length-scale sequentially in turn
            for r in range(nrestarts):
                if ls == 1:
                    logging.warning("Changing item length-scale of 1 to 2 to avoid optimisation problems.")
                    ls = 2.0
            
                initialguess = np.log(ls) 
                logging.debug("Initial item length-scale guess for dimension %i in restart %i: %.3f" % (d, r, ls))
        
                #ftol = self.conv_threshold * 1e2
                #logging.debug("Ftol = %.5f" % ftol)
#                 opt_hyperparams, nlml, _, _, _ = minimize(self.neg_marginal_likelihood, initialguess, 
#                       args=('item', d, use_MAP,), method='Nelder-Mead', options={'maxfev':maxfun, 'fatol':ftol, 
#                         'xatol':ls * 1e100, 'return_all':True})

                # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML 
                # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg, 
                # or trust-ncg, which may be faster still?
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', d, use_MAP,), 
                    jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun})

                opt_hyperparams = res['x']
                nlml = res['fun']
                nfits += res['nfev']
                
                if nlml < min_nlml:
                    min_nlml = nlml
                    best_opt_hyperparams = opt_hyperparams
                    best_iter = r
                    
                # choose a new lengthscale for the initial guess of the next attempt
                ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls)
    
            if best_iter < r:
                # need to go back to the best result
                self.ls[d] = np.exp(best_opt_hyperparams)
                if d == len(self.ls) - 1 and person_features is None:
                    self.neg_marginal_likelihood(best_opt_hyperparams, 'item', d, use_MAP=False)

            logging.debug("Chosen item length-scale %.5f for dimension %i. Last initialguess=%.5f, used %i evals of \
            NLML over %i restarts" % (self.ls[d], d, np.exp(initialguess), nfits, nrestarts))

        if person_features is None:
            return self.ls, self.lsy, -min_nlml

        for e, lsy in enumerate(self.lsy):
            min_nlml = np.inf
            best_opt_hyperparams = None
            best_iter = -1            
                        
            nfits = 0 # number of calls to fit function

            logging.debug("Optimising item length-scale for %i dimension" % e)
            
            # optimise each length-scale sequentially in turn
            for r in range(nrestarts):
                if lsy == 1:
                    logging.warning("Changing person length-scale of 1 to 2 to avoid optimisation problems.")
                    lsy = 2.0
            
                initialguess = np.log(lsy) 
                logging.debug("Initial person length-scale guess for dimension %i in restart %i: %.3f" % (e, r, lsy))
        
                ftol = self.conv_threshold * 1e2
                logging.debug("Ftol = %.5f" % ftol)
#                 opt_hyperparams, nlml, _, _, _ = minimize(self.neg_marginal_likelihood, initialguess, 
#                       args=('person', e, use_MAP,), method='Nelder-Mead', options={'maxfev':maxfun, 'fatol':ftol, 
#                         'xatol':ls * 1e100, 'return_all':True})
                # replace Nelder-Mead, as above.
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('person', e, use_MAP,),  
                  jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'return_all':True})
                opt_hyperparams = res['x']
                nlml = res['fun']
                nfits += res['nfev']
                                                
                if nlml < min_nlml:
                    min_nlml = nlml
                    best_opt_hyperparams = opt_hyperparams
                    best_iter = r
                    
                # choose a new lengthscale for the initial guess of the next attempt
                lsy = gamma.rvs(self.shape_lsy, scale=1.0/self.rate_lsy)
    
            if best_iter < r:
                # need to go back to the best result
                self.lsy[e] = np.exp(best_opt_hyperparams)
                if e == len(self.ls) - 1:
                    self.neg_marginal_likelihood(best_opt_hyperparams, 'person', e, use_MAP=False)

            logging.debug("Chosen person length-scale %.5f for dimension %i. Last initialguess=%.5f, used %i evals of \
            NLML over %i restarts" % (self.lsy[e], e, np.exp(initialguess), nfits, nrestarts))

        logging.debug("Optimal hyper-parameters: item = %s, person = %s" % (self.ls, self.lsy))   
        return self.ls, self.lsy, -min_nlml # return the log marginal likelihood

    def neg_marginal_likelihood(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        if lstype=='item':
            self.ls[dimension] = np.exp(hyperparams)
        elif lstype=='person':
            self.lsy[dimension] = np.exp(hyperparams)
        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf
                
        # make sure we start again -- fit should set the value of parameters back to the initial guess
        self.fit()
        marginal_log_likelihood = self.lowerbound()        
        if use_MAP:
            log_model_prior = self.ln_modelprior()        
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood
            
        if lstype=='person':
            logging.debug("LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.lsy[dimension]))
        elif lstype=='item':
            logging.debug("LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.ls[dimension]))
            
        return -lml
    
    def nml_jacobian(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        
        needs_fitting = self.people is None
        if lstype=='item':
            if self.ls[dimension] != np.exp(hyperparams):
                self.ls[dimension] = np.exp(hyperparams)
                needs_fitting = True
        elif lstype=='person':
            if self.lsy[dimension] != np.exp(hyperparams):
                self.lsy[dimension] = np.exp(hyperparams)
                needs_fitting = True
        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf
                
        # make sure we start again -- fit should set the value of parameters back to the initial guess
        if needs_fitting:
            self.fit()

        # compute the gradient
        # all of the terms below can use the same function except der_logpf. This should follow the MAP estimate from 
        # chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        #gradient when converged due to the coordinate ascent method.
        if lstype == 'item':
            dKdls = self.kernel_der(self.item_distances, self.ls, dimension) 

            # try to make the s scale cancel as much as possible
            invK_w = solve_triangular(self.cholK, self.w, trans=True, check_finite=False)
            invK_w = solve_triangular(self.cholK, invK_w, check_finite=False)
            
            invK_dkdls = solve_triangular(self.cholK, dKdls, trans=True, check_finite=False)
            invK_dkdls = solve_triangular(self.cholK, invK_dkdls, check_finite=False)            
            
            der_logpw_logqw = 0
            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + (self.N * f)
                invK_wf = invK_w[:, f]
                der_logpw_logqw += 0.5 * (np.sum(invK_wf.T.dot(dKdls).dot(invK_wf) * (self.shape_sw[f] / self.rate_sw[f]))
                                           - np.trace(self.w_cov[fidxs, :][:, fidxs].dot(invK_dkdls)))
            
            der_logpy_logqy = 0

            invK_t = solve_triangular(self.cholK, self.t, trans=True, check_finite=False)
            invK_t = solve_triangular(self.cholK, invK_t, check_finite=False)
            der_logpt_logqt = 0.5 * np.sum(invK_t.T.dot(dKdls).dot(invK_t) * (self.shape_st / self.rate_st))
            der_logpt_logqt -= 0.5 * np.trace(self.t_cov.dot(invK_dkdls))
            
            der_logpf_logqf = 0
            for p in self.pref_gp:
                der_logpf_logqf = self.pref_gp[p].lowerbound_gradient(dimension)
            
        elif lstype == 'person':
            der_logpw_logqw = 0
            
            # try to make the s scale cancel as much as possible
            invK_y = solve_triangular(self.cholKy, self.y.T, trans=True, check_finite=False)
            invK_y = solve_triangular(self.cholKy, invK_y, check_finite=False)
            
            invK_dkdls = solve_triangular(self.cholK, dKdls, trans=True, check_finite=False)
            invK_dkdls = solve_triangular(self.cholK, invK_dkdls, check_finite=False)            
            
            der_logpy_logqy = 0
            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)
                invK_yf = invK_y[:, f]
                der_logpy_logqy += 0.5 * np.sum(invK_yf.T.dot(dKdls).dot(invK_yf) * (self.shape_sy[f] / self.rate_sy[f]))
                der_logpy_logqy -= 0.5 * np.trace(self.y_cov[fidxs, :][:, fidxs].dot(invK_dkdls))            
            
            der_logpt_logqt = 0
            der_logpf_logqf = 0
            
        mll_jac = der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf

        if use_MAP: # gradient of the log prior
            log_model_prior_grad = self.ln_modelprior_grad()        
            lml_jac = mll_jac + log_model_prior_grad
        else:
            lml_jac = mll_jac
        logging.debug("Jacobian of LML: %f" % lml_jac)
        if self.verbose:
            logging.debug("...with item length-scales = %s, person length-scales = %s" % (self.ls, self.lsy))
        return -lml_jac # negative because the objective function is also negated
 
    def predict(self, personids, items_1_coords, items_2_coords, person_features=None):
        Npairs = len(personids)
        results = np.zeros(Npairs)
         
        upeople = np.unique(personids)
        for p in upeople:            
            pidxs = personids == p
            Npairs_p = np.sum(pidxs)
            
            if p in self.people:
                y = self.y[:, p:p+1] 
            elif self.person_features is None:
                y = np.zeros((self.Nfactors, 1))
            else:
                if self.use_svi_people:
                    distances = np.zeros((1, self.y_ninducing, self.nperson_features))
                    for d in range(self.nperson_features):
    
                        distances[:, :, d] = person_features[p, d:d+1] - self.y_inducing_coords[:, d:d+1].T
                    Ky = self.kernel_func(distances, self.lsy)                    
                    # use kernel to compute y
                    invKy_train = self.Ky_mm_block
                    y_train = self.y_u.reshape(self.Nfactors, self.Npeople).T
                else:
                    #distances for y-space
                    distances = np.zeros((1, self.Npeople, self.nperson_features))
                    for d in range(self.nperson_features):
                        distances[:, :, d] = person_features[p, d:d+1] - self.person_features[:, d:d+1].T        
                    # kernel between p and people already seen
                    Ky = self.kernel_func(distances, self.lsy)
                    invKy_train = np.linalg.inv(self.Ky_block)
                    y_train = self.y.T
                
                # use kernel to compute y
                y = Ky.dot(invKy_train).dot(y_train)
                y *= self.rate_sy / self.shape_sy      
                y = y.T
                
            coords_1 = items_1_coords[pidxs]
            coords_2 = items_2_coords[pidxs]
            
            # this could be made more efficient because duplicate locations are computed separately!
            # distances for t-space
            if self.use_svi:
                distances1 = np.zeros((Npairs_p, self.ninducing, self.nitem_features))
                distances2 = np.zeros((Npairs_p, self.ninducing, self.nitem_features))
                for d in range(self.nitem_features):
                    distances1[:, :, d] = coords_1[:, d:d+1] - self.inducing_coords[:, d:d+1].T
                    distances2[:, :, d] = coords_2[:, d:d+1] - self.inducing_coords[:, d:d+1].T
                    
                # kernel between pidxs and t
                K1 = self.kernel_func(distances1, self.ls)
                K2 = self.kernel_func(distances2, self.ls)            
            
                # use kernel to compute t. 
                t1 = K1.dot(self.invK_mm).dot(self.t_u)
                t2 = K2.dot(self.invK_mm).dot(self.t_u)

                # kernel between pidxs and w -- use kernel to compute w. Don't need Kw_mm block-diagonal matrix
                w1 = K1.dot(self.invK_mm).dot(self.w_u)   
                w2 = K2.dot(self.invK_mm).dot(self.w_u)                    
            else:                                
                distances1 = np.zeros((Npairs_p, self.N, self.nitem_features))
                distances2 = np.zeros((Npairs_p, self.N, self.nitem_features))
                for d in range(self.nitem_features):
                    distances1[:, :, d] = coords_1[:, d:d+1] - self.obs_coords[:, d:d+1].T
                    distances2[:, :, d] = coords_2[:, d:d+1] - self.obs_coords[:, d:d+1].T
                
                # kernel between pidxs and t
                K1 = self.kernel_func(distances1, self.ls)
                K2 = self.kernel_func(distances2, self.ls)            
            
                # use kernel to compute t
                invKt = solve_triangular(self.cholK, self.t, trans=True, check_finite=False)
                invKt = solve_triangular(self.cholK, invKt, overwrite_b=True, check_finite=False)

                t1 = K1.dot(invKt)
                t2 = K2.dot(invKt)

                # kernel between pidxs and w -- use kernel to compute w
                invKw = solve_triangular(self.cholK, self.w, trans=True, check_finite=False)
                invKw = solve_triangular(self.cholK, invKw, overwrite_b=True, check_finite=False)
                
                w1 = K1.dot(invKw)
                w2 = K2.dot(invKw)   
            
            wy_1p = w1.dot(y)
            wy_2p = w2.dot(y)
            mu0_1 = wy_1p + t1
            mu0_2 = wy_2p + t2                                        
            if p in self.people:            
                pref_gp_p = self.pref_gp[p]
                results[pidxs] = pref_gp_p.predict(coords_1, coords_2, 
                                                  mu0_output1=mu0_1, mu0_output2=mu0_2, return_var=False).flatten()
            else:
                mu0 = np.concatenate((mu0_1, mu0_2), axis=0)
                results[pidxs] = pref_likelihood(f=mu0, subset_idxs=[], v=np.arange(len(mu0_1)), 
                                                                      u=np.arange(len(mu0_1), len(mu0_1)+len(mu0_2)))
        return results
        
    def _expec_f(self):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''
        for p in self.pref_gp:
            plabelidxs = self.personIDs == p
            if self.items_1_coords is not None:
                items_1_p = self.items_1_coords[plabelidxs]
                items_2_p = self.items_2_coords[plabelidxs]
                prefs_p = self.preferences[plabelidxs]
            else: # no data passed in -- likely because it has already been provided in a previous function call 
                items_1_p = None
                items_2_p = None
                prefs_p = None
            
            if self.vb_iter == 0 or self.new_obs:
                # take the initial prior so that we can calculate Q correctly -- use the prior mean as an approximation
                # to integrating over the prior. We don't want to use a posterior or the random initialisation of wy.
                mu0_output = self.t_mu0.copy()
            else:
                mu0_output = self.wy[:, p:p+1] + self.t
            
            mu0_1 = mu0_output[self.pref_v[plabelidxs], :]
            mu0_2 = mu0_output[self.pref_u[plabelidxs], :]
            
            if not self.new_obs and self.vb_iter==0:
                self.pref_gp[p]._init_params((mu0_1, mu0_2))
            self.pref_gp[p].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs, 
                                input_type=self.input_type)                
            # find the index of the coords in coords_p in self.obs_coords
            # coordsidxs[p] needs to correspond to data points in same order as invKf[p]
            if p not in self.coordidxs:
                internal_coords_p = self.pref_gp[p].obs_coords
                self.matches[p] = np.ones((internal_coords_p.shape[0], self.N), dtype=bool)
                for dim in range(internal_coords_p.shape[1]):
                    self.matches[p] = self.matches[p] & np.equal(internal_coords_p[:, dim:dim+1], 
                                                                           self.obs_coords[:, dim:dim+1].T)
                self.coordidxs[p] = np.sort(np.argwhere(self.matches[p])[:, 1])
            
                if not self.use_svi:
                    self.invKf[p] = np.linalg.inv(self.pref_gp[p].K) 

            f, _ = self.pref_gp[p].predict_f(items_coords=self.obs_coords, mu0_output=mu0_output)
            self.f[p, :] = f.flatten()
            
            if self.use_svi:
                self.f_u[p] = self.pref_gp[p].um_minus_mu0 + self.w_u.dot(self.y[:, p:p+1]) + self.t_u
            
            if self.verbose:    
                logging.debug( "Expec_f for person %i out of %i. s=%.3f" % (p, len(self.pref_gp.keys()), self.pref_gp[p].s) )
                
        self.new_obs = False # don't process the observations again unless fit() is called

        if self.verbose:
            logging.debug('Updated q(f)')
             
    def _expec_w(self):
        '''
        Compute the expectation over the latent features of the items and the latent personality components
        '''
        if self.use_fa:
            self.y = self.fa.fit_transform(self.f).T
            self.w = self.fa.components_.T
            self.wy = self.w.dot(self.y)
            return
            
        # Put a GP prior on w with covariance K/gamma and mean 0
        if self.use_svi:
            N = self.ninducing
        else:
            N = self.N
        
        x = np.zeros((N, self.Nfactors))
        Sigma = np.zeros((N * self.Nfactors, N * self.Nfactors))

        Nobs_counter = 0
        Nobs_counter_i = 0

        for p in self.pref_gp:
            pidxs = self.coordidxs[p]
            y_p = self.y[:, p:p+1]

            if self.use_svi:
                Nobs_counter += len(pidxs)
                psample = np.in1d(pidxs, self.data_idx_i)
                pidxs = pidxs[psample]
                Nobs_counter_i += len(pidxs)
                if not len(pidxs):
                    continue # not yet tested but seemed to be missing
                pidxs = np.arange(N) # save values for each inducing point
                # need to correctly include mu0 of the child GP. Requires same inducing points in child GP
                prec_p = self.pref_gp[p].inv_Ks_mm
                invQ_f = self.pref_gp[p].inv_Ks_mm.dot(self.f_u[p] - self.t_u)
                x += y_p.T * invQ_f
            else:
                prec_p = self.invKf[p] * self.pref_gp[p].s
                invQ_f = prec_p.dot(self.f[p:p+1, pidxs].T - self.t[pidxs])
                 
                # add the means for this person's observations to the list of observations, x 
                x[pidxs, :] += y_p.T * invQ_f
            
            # add the covariance for this person's observations as a block in the covariance matrix Sigma
            Sigma_p = np.zeros((N * self.Nfactors, N * self.Nfactors))
            yidxs = p + self.Npeople * np.arange(self.Nfactors)
            Sigma_yscaling = y_p.dot(y_p.T) + self.y_cov[yidxs, :][:, yidxs] # covariance between people?
            
            for f in range(self.Nfactors):
                for g in range(self.Nfactors):
                    Sigma_p_rows = np.zeros((len(pidxs), N * self.Nfactors))
                    Sigma_p_rows[:, pidxs + g * N] = prec_p * Sigma_yscaling[f, g]
                    Sigma_p[pidxs + f * N, :] += Sigma_p_rows
                        
            Sigma += Sigma_p
                            
        x = x.T.flatten()[:, np.newaxis]
        
        if not self.use_svi:
            # w_cov is same shape as K with rows corresponding to (f*N) + n where f is factor index from 0 and 
            # n is data point index
            
            self.w_cov = np.linalg.inv(np.linalg.inv(self.Kw  / self.sw_matrix) + Sigma)
            self.w = self.w_cov.dot(x)
            
            self.w = np.reshape(self.w, (self.Nfactors, self.N)).T # w is N x Nfactors    
            
            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + (self.N * f)
                _, self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, self.N, 
                                self.cholK, self.w[:, f:f+1], np.zeros((self.N, 1)), self.w_cov[fidxs, :][:, fidxs])
                self.sw_matrix[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]            
            
        else: # SVI implementation
            self.w, self.w_cov, self.w_invS, self.w_invSm, self.w_u, self.invKws_mm_S = svi_update_gaussian(x, 0, 0,
                self.Kws_mm, self.inv_Kws_mm, self.Kws_nm, self.Kws_mm, self.Kw / self.sw_matrix, Sigma, self.w_invS, 
                self.w_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)                
        
            self.w = np.reshape(self.w, (self.Nfactors, self.N)).T # w is N x Nfactors    
            self.w_u = np.reshape(self.w_u, (self.Nfactors, self.ninducing)).T # w is N x Nfactors    
            
            for f in range(self.Nfactors):
                fidxs = np.arange(N) + (N * f)
                self.shape_sw[f], self.rate_sw[f] = expec_output_scale_svi(self.shape_sw0, self.rate_sw0, 
                        self.ninducing, self.invK_mm, self.w_u[:, f:f+1], np.zeros((self.ninducing, 1)), 
                        self.invKws_mm_S[fidxs, :][:, fidxs] / self.shape_sw[f] * self.rate_sw[f])
                fidxs = np.arange(self.N) + (self.N * f)
                self.sw_matrix[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]
        
        self._expec_y()
        self.wy = self.w.dot(self.y)    
        return

    def _expec_y(self):
        '''
        Compute expectation over the personality components using VB
        '''
        Npeople = self.Npeople  
                    
        Sigma = np.zeros((self.Nfactors * Npeople, self.Nfactors * Npeople))
        x = np.zeros((Npeople, self.Nfactors))

        Nobs_counter = 0
        Nobs_counter_i = 0

        for p in self.pref_gp:
            pidxs = self.coordidxs[p]                       
            Nobs_counter += len(pidxs)
            if self.use_svi_people and p not in self.pdata_idx_i:
                continue
            
            Nobs_counter_i += len(pidxs)
            if self.use_svi:
                prec_f_p = self.pref_gp[p].inv_Ks_mm
                w_cov = self.Kws_mm.dot(self.invKws_mm_S)
                w = self.w_u
                N = self.ninducing
                pidxs = np.arange(N)
                invQ_f = prec_f_p.dot(self.f_u[p] - self.t_u)
            else:
                prec_f_p = self.invKf[p] * self.pref_gp[p].s          
                w_cov = self.w_cov
                w = self.w[pidxs, :]
                N = self.N
                invQ_f = prec_f_p.dot(self.f[p, pidxs][:, np.newaxis] - self.t[pidxs, :]) 
                
            covterm = np.zeros((self.Nfactors, self.Nfactors))
            for f in range(self.Nfactors): 
                w_cov_idxs = pidxs + (f * N)
                w_cov_f = w_cov[w_cov_idxs, :]
                for g in range(self.Nfactors):
                    w_cov_idxs = pidxs + (g * N)
                    covterm[f, g] = np.sum(prec_f_p * w_cov_f[:, w_cov_idxs])
            Sigma_p = w.T.dot(prec_f_p).dot(w) + covterm
                
            sigmaidxs = np.arange(self.Nfactors) * self.Npeople + p
            Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
            Sigmarows[:, sigmaidxs] =  Sigma_p
            Sigma[sigmaidxs, :] += Sigmarows             
              
            x[p, :] = w.T.dot(invQ_f).T
                
        x = x.T.flatten()[:, np.newaxis]
                    
        if not self.use_svi_people:
            # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0 
            # and p is person index
            self.y_cov = np.linalg.inv(np.linalg.inv(self.Ky  / self.sy_matrix) + Sigma)
            self.y = self.y_cov.dot(x)
           
            # y is Nfactors x Npeople            
            self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
                
            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)
                _, self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople, 
                            self.cholKy, self.y[f:f+1, :].T, np.zeros((self.Npeople, 1)), self.y_cov[fidxs, :][:, fidxs])    
                self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]                
        else: # SVI implementation
            self.y, self.y_cov, self.y_invS, self.y_invSm, self.y_u, self.invKys_mm_S = svi_update_gaussian(x, 0, 0, 
                self.Kys_mm, self.inv_Kys_mm, self.Kys_nm, self.Kys_nm, self.Ky / self.sy_matrix, Sigma, self.y_invS, 
                self.y_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)
        
            # y is Nfactors x Npeople            
            self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
            self.y_u = np.reshape(self.y_u, (self.Nfactors, self.Npeople))
                
            for f in range(self.Nfactors):
                fidxs = np.arange(self.y_ninducing) + (self.y_ninducing * f)
                self.shape_sy[f], self.rate_sy[f] = expec_output_scale_svi(self.shape_sy0, self.rate_sy0, 
                    self.y_ninducing, self.invKy_mm_block, self.y_u[f:f+1, :].T, np.zeros((self.Npeople, 1)), 
                    self.invKys_mm_S[fidxs, :][:, fidxs] / self.shape_sy[f] * self.rate_sy[f])    
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)
                self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]    

    def _expec_t(self):
        if self.use_fa:
            self.t = self.fa.mean_[:, np.newaxis]
            return
        if self.use_svi:
            N = self.ninducing
        else:
            N = self.N
            
        Sigma = np.zeros((N, N))
        x = np.zeros((N, 1))
        
        Nobs_counter = 0
        Nobs_counter_i = 0
        
        #size_added = 0
        for p in self.pref_gp:
            pidxs = self.coordidxs[p]
            Nobs_counter += len(pidxs)
            
            if self.use_svi:
                psample = np.in1d(pidxs, self.data_idx_i)
                pidxs = pidxs[psample] # indexes to read the observation data from
                Nobs_counter_i += len(pidxs)                
                if not len(pidxs):
                    continue # not yet tested but seemed to be missing
                pidxs = np.arange(N) # save values for each inducing point
                Sigma_p = self.pref_gp[p].inv_Ks_mm
                
                # add the means for this person's observations to the list of observations, x 
                invQ_f = self.pref_gp[p].inv_Ks_mm.dot(self.f_u[p] - self.w_u.dot(self.y[:, p:p+1]))
            else:            
                Sigma_p = self.invKf[p] * self.pref_gp[p].s
                # add the means for this p's observations to the list of observations, x 
                f_obs = self.f[p:p+1, pidxs].T - self.wy[pidxs, p:p+1]
                invQ_f = Sigma_p.dot(f_obs)
                
            sigmarows = np.zeros((len(pidxs), N))
            sigmarows[:, pidxs] = Sigma_p
            Sigma[pidxs, :] += sigmarows
            
            x[pidxs, :] += invQ_f 
                
        if not self.use_svi:
            invKts = self.invK * self.shape_st / self.rate_st
            self.t = invKts.dot(self.t_mu0) + x
            self.t_cov = np.linalg.inv(Sigma + invKts)
            self.t = self.t_cov.dot(self.t)

            _, self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.N, self.cholK, 
                                                            self.t, np.zeros((self.N, 1)), self.t_cov)

        else:
            # SVI implementation
            self.t, self.t_cov, self.t_invS, self.t_invSm, self.t_u, self.invKts_mm_S = svi_update_gaussian(x, 
                self.t_mu0, self.t_mu0_u, self.Kts_mm, self.inv_Kts_mm, self.Kts_nm, self.Kts_mm, 
                self.K*self.rate_st/self.shape_st, Sigma, self.t_invS, self.t_invSm, self.vb_iter, self.delay, 
                self.forgetting_rate, Nobs_counter, Nobs_counter_i)

            self.shape_st, self.rate_st = expec_output_scale_svi(self.shape_st0, self.rate_st0, self.ninducing, 
                self.invK_mm, self.t_u, np.zeros((self.ninducing, 1)), self.invKts_mm_S / self.shape_st * self.rate_st)
        
    def _update_sample(self):
        self._update_sample_idxs()
        
        sw_mm = np.zeros((self.Nfactors * self.ninducing, self.Nfactors * self.ninducing))
        sw_nm = np.zeros((self.Nfactors * self.N, self.Nfactors * self.ninducing))
        for f in range(self.Nfactors):
            fidxs = np.arange(self.ninducing) + (self.ninducing * f)
            sw_mm[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]
            fidxs = np.arange(self.N) + (self.N * f)
            sw_nm[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]
            
        st = self.shape_st / self.rate_st
                    
        self.Kws_mm = self.Kw_mm / sw_mm
        self.inv_Kws_mm  = self.invKw_mm * sw_mm
        self.Kws_nm = self.Kw_nm  / sw_nm

        self.Kts_mm = self.K_mm / st
        self.inv_Kts_mm  = self.invK_mm * st
        self.Kts_nm = self.K_nm / st   
        
        if self.use_svi_people:
            sy_mm = np.zeros((self.Nfactors * self.y_ninducing, self.Nfactors * self.y_ninducing))
            sy_nm = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.y_ninducing))
            for f in range(self.Nfactors):
                fidxs = np.arange(self.y_ninducing) + (self.y_ninducing * f)
                sy_mm[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)
                sy_nm[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]
            
            self.Kys_mm = self.Ky_mm / sy_mm
            self.inv_Kys_mm  = self.invKy_mm * sy_mm
            self.Kys_nm = self.Ky_nm / sy_nm
        
    def _update_sample_idxs(self):
        self.data_idx_i = np.sort(np.random.choice(self.N, self.update_size, replace=False))        
        
        if self.use_svi_people:
            self.pdata_idx_i = np.sort(np.random.choice(self.Npeople, self.y_update_size, replace=False))        
                    
    def lowerbound(self):
        f_terms = 0
        y_terms = 0
        
        for p in self.pref_gp:
            f_terms += self.pref_gp[p].lowerbound()
            if self.verbose:
                logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
            
        if self.use_svi:
            logpw = mvn.logpdf(self.w_u.T.flatten(), cov=self.Kws_mm)
            logqw = mvn.logpdf(self.w_u.T.flatten(), mean=self.w_u.T.flatten(), cov=self.Kws_mm.dot(self.invKws_mm_S))

            logpt = mvn.logpdf(self.t_u.flatten(), cov=self.Kts_mm)
            logqt = mvn.logpdf(self.t_u.flatten(), mean=self.t_u.flatten(), cov=self.Kts_mm.dot(self.invKts_mm_S))    
        else:
            logpw = mvn.logpdf(self.w.T.flatten(), cov=self.Kw / self.sw_matrix, allow_singular=True)
            logqw = mvn.logpdf(self.w.T.flatten(), mean=self.w.T.flatten(), cov=self.w_cov, allow_singular=True)
            
            logpt = mvn.logpdf(self.t.flatten(), mean=self.t_mu0.flatten(), cov=self.K * self.rate_st / self.shape_st)
            logqt = mvn.logpdf(self.t.flatten(), mean=self.t.flatten(), cov=self.t_cov)    

        if self.use_svi_people:
            logpy = mvn.logpdf(self.y_u.flatten(), cov=self.Kys_mm)
            logqy = mvn.logpdf(self.y_u.flatten(), mean=self.y_u.flatten(), cov=self.Kys_mm.dot(self.invKys_mm_S))
        else:                        
            logpy = mvn.logpdf(self.y.flatten(), cov=self.Ky / self.sy_matrix)
            logqy = mvn.logpdf(self.y.flatten(), mean=self.y.flatten(), cov=self.y_cov)
        
        logps_y = 0
        logqs_y = 0
        logps_w = 0
        logqs_w = 0        
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f])
                    
            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f])
        
        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st) 
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st)        
        
        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms += logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw/self.rate_sw))        
            #logging.debug("logpw: %.2f" % logpw)       
            #logging.debug("logqw: %.2f" % logqw)
            #logging.debug("logps_w: %.2f" % logps_w)
            #logging.debug("logqs_w: %.2f" % logqs_w)   
            #logging.debug("wy: %s" % self.wy)
            #logging.debug("E[w]: %s" % self.w)       
            #logging.debug("cov(w): %s" % self.w_cov)  
            
            logging.debug('s_y=%s' % (self.shape_sy/self.rate_sy))
            #logging.debug("logpy: %.2f" % logpy)
            #logging.debug("logqy: %.2f" % logqy)
            #logging.debug("logps_y: %.2f" % logps_y)
            #logging.debug("logqs_y: %.2f" % logqs_y)
            
            logging.debug('s_t=%.2f' % (self.shape_st/self.rate_st))        
            #logging.debug("t_cov: %s" % self.t_cov)
        

        lb = f_terms + t_terms + w_terms + y_terms
        if self.verbose:
            logging.debug( "Iteration %i: Lower bound = %.3f, fterms=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f" % 
                       (self.vb_iter, lb, f_terms, w_terms, y_terms, t_terms) )
        
        if self.verbose:
            for p in self.people:                                                              
                logging.debug("f_%i: %.2f, %.2f" % (p, np.min(self.f[p, :]), np.max(self.f[p, :])))
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))
                
        return lb
    
    def ln_modelprior(self):
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls*np.log(self.rate_ls) \
                   + (self.shape_ls-1)*np.log(self.ls) - self.ls*self.rate_ls
                   
        lnp_gpy = - gammaln(self.shape_lsy) + self.shape_lsy*np.log(self.rate_lsy) \
                   + (self.shape_lsy-1)*np.log(self.lsy) - self.lsy*self.rate_lsy
                                      
        return np.sum(lnp_gp) + np.sum(lnp_gpy)    
    
    def pickle_me(self, filename):
        import pickle
        from copy import  deepcopy
        with open (filename, 'w') as fh:
            m2 = deepcopy(self)
            for p in m2.pref_gp:
                m2.pref_gp[p].kernel_func = None # have to do this to be able to pickle
            pickle.dump(m2, fh)        