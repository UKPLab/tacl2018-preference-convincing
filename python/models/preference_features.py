'''
TODO: SVI and non-SVI implementations give different answers
TODO: SVI results seem not to converge with small datasets
TODO: non-SVI results seem to have overly small w and y -- perhaps need broader priors (smaller s)
TODO: SVI results seem to vary hugely between runs or on different computers despite same seeds -- need to verify this is true
TODO: Lower bound can go down with SVI -- what about without SVI? Bug in LB or in the SVI algorithm? 
TODO: Lower bound for noise SVI should be computed inside GP code -- is this possible? I think we can just add the missing cov_mu term.

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
from scipy.stats import multivariate_normal as mvn, norm
import logging
from gp_classifier_vb import matern_3_2_from_raw_vals, derivfactor_matern_3_2_from_raw_vals
from gp_pref_learning import GPPrefLearning, get_unique_locations, pref_likelihood
from scipy.linalg import block_diag
from scipy.special import gammaln, psi
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

def expec_output_scale(shape_s0, rate_s0, N, invK, f_mean, m, invK_f_cov=None, f_cov=None):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N
    
    if invK_f_cov is None:
        if f_cov is None:
            logging.error('Provide either invK_f_cov or f_cov')
            return
        invK_f_cov = invK.dot(f_cov)
    
    invK_expecFF = invK_f_cov + invK.dot( (f_mean - m).dot(f_mean.T - m.T) )
    rate_s = rate_s0 + 0.5 * np.trace(invK_expecFF) 
    
    return shape_s, rate_s

def lnp_output_scale(shape_s0, rate_s0, shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)
    
    logprob_s = - gammaln(shape_s0) + shape_s0 * np.log(rate_s0) + (shape_s0-1) * Elns - rate_s0 * s
    return logprob_s            
        
def lnq_output_scale(shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)
    
    lnq_s = - gammaln(shape_s) + shape_s * np.log(rate_s) + (shape_s-1) * Elns - rate_s * s
    return lnq_s
    
def update_gaussian(invK, s, Sigma, x):
    cov = np.linalg.inv((invK * s) + Sigma)
    m = cov.dot(x)
    return m, cov
    
def svi_update_gaussian(invQi_y, mu0_n, mu_u, K_mm, invK_mm, K_nm, Lambda_factor1, K_nn, invQi, prev_invS, prev_invSm, 
                        vb_iter, delay, forgetting_rate, N, update_size):

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
    #L_invS = cholesky(invS.T, lower=True, check_finite=False)
    #B = solve_triangular(L_invS, invK_mm.T, lower=True, check_finite=False)
    #A = solve_triangular(L_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
    #invK_mm_S = A.T
    S = np.linalg.inv(invS)
    invK_mm_S = invK_mm.dot(S)
    
    #fhat_u = solve_triangular(L_invS, invSm, lower=True, check_finite=False)
    #fhat_u = solve_triangular(L_invS, fhat_u, lower=True, trans=True, check_finite=False, overwrite_b=True)
    fhat_u = S.dot(invSm)
    fhat_u += mu_u
    
    # TODO: move the K_mm.T.dot(K_nm.T) computation out    
    covpair_uS = K_nm.dot(invK_mm_S)
    fhat = covpair_uS.dot(invSm) + mu0_n
    if K_nn is None:
        C = None
    else:
        covpair =  K_nm.dot(invK_mm)    
        C = K_nn + (covpair_uS - covpair.dot(K_mm)).dot(covpair.T)
    return fhat, C, invS, invSm, fhat_u, invK_mm_S, S

def expec_pdf_gaussian(K, invK, Elns, N, s, f, mu, f_cov, mu_cov):
    '''
    Expected value of the PDF with respect to the function values with expectation f, the mean values with expectation
    mu, and the inverse covariance scale with expectation s.
    
    Parameters
    ----------
    
    K : covariance matrix (without scaling)
    invK : inverse of the covariance matrix (without scaling)
    Elns : expected log of the covariance inverse scale factor
    N : number of data points
    s : expected covariance inverse scale factor
    f : expected function values
    mu : expected mean values
    f_cov : covariance of the function values
    mu_cov : covariance of the mean parameters; this is needed if the mean is a model parameter inferred using VB
    '''
    _, logdet_K = np.linalg.slogdet(K)
    logdet_Ks = - np.sum(N * Elns) + logdet_K
    invK_expecF = (s * invK).dot(f_cov + (f-mu).dot((f-mu).T) + mu_cov)
    logpf = 0.5 * (- np.log(2*np.pi) * N - logdet_Ks - np.trace(invK_expecF))
    
    return logpf

def expec_q_gaussian(f_cov, D):
    _, logdet_C = np.linalg.slogdet(f_cov)
    logqf = 0.5 * (- np.log(2*np.pi)*D - logdet_C - D)
    return logqf    

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1, 
                 shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20, 
                 use_common_mean_t=True, uncorrelated_noise=False, kernel_func='matern_3_2',
                 max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        self.people = None
        self.pref_gp = {}
        self.nitem_features = nitem_features
        self.nperson_features = nperson_features
        self.mu0 = mu0 # these are abstract latent functions, not related to any real-world variables: mu0=0 by default
        # other means should be provided later so that we can put priors on type of person liking type of object

        self.ninducing = ninducing
        self.max_update_size = max_update_size # maximum number of data points to update in each SVI iteration of noise GPs

        shape_s0 = float(shape_s0)
        rate_s0 = float(rate_s0)

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
        self.rate_sf0 = rate_s0 / 2.0 # split variance -- noise and components
                
        # For the latent means and components, the relative sizes of the scales controls how much the components can 
        # vary relative to the overall f, i.e. how much they learn from f. A high s will mean that the wy & t functions 
        # have smaller scale relative to f, so they will be less fitted to f. By default we assume a common prior.   
        # Since the mean of f is formed by wy+t, we may wish to make the scale of f and wy+t similar. Since wy is a 
        # product of gaussian-distributed random variables, the variance of wy is defined in the equations in 
        # "Products and Convolutions of Gaussian Probability Density Functions", P.A. Bromiley 2003 (updated 2014).
        # To make var(wy) = var(f), assume var(w) = var(y), then var(w)=var(y) = 2 * var(f). We use this as our 
        # default, but halve var(wy) so that var(wy)=var(t). In practice, since var(f) represents noise, we may set the 
        # prior for f to be smaller --> this leads us to use big values for var(w) and var(y).
        self.shape_sw0 = shape_s0 
        self.rate_sw0 = rate_s0
                            
        self.shape_sy0 = shape_s0
        self.rate_sy0 =  rate_s0 ** 0.5
    
        # if the scale doesn't matter, then let's fix the mean to be scaled to one? However, fixing t's scale and not
        # the noise scale in f means that since preference learning can collapse toward very large scales, the noise
        # can grow large and the shared mean t has less influence. So it makes sense to limit noise and  
        self.shape_st0 = shape_s0
        self.rate_st0 = rate_s0 / 2.0
        
        if not use_common_mean_t: # make var(wy+t) = var(f) 
            self.rate_sw0 *= np.sqrt(2.0)
            self.rate_sy0 *= np.sqrt(2.0)
        
        # y has different length-scales because it is over user features space
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        
        if ls is not None:
            self.n_wlengthscales = len(np.array([ls]).flatten()) # can pass in a single length scale to be used for all dimensions
        else:
            self.n_wlengthscales = self.nitem_features
        self.ls = ls
        
        self.shape_lsy = shape_lsy
        self.rate_lsy = rate_lsy
        self.lsy = lsy  
        if lsy is not None:
            self.n_ylengthscales = len(np.array([lsy]).flatten()) # can pass in a single length scale to be used for all dimensions
        else:
            self.n_ylengthscales = self.nperson_features        
        
        self.t_mu0 = 0
        
        self.conv_threshold = 1e-3
        self.max_iter = 1000
        self.min_iter = 3
        self.n_converged = 10 # number of iterations while apparently converged (avoids numerical errors)
        self.vb_iter = 0
        
        self.verbose = verbose
        
        self.Nfactors = nfactors
        
        self.use_t = use_common_mean_t
        self.uncorrelated_noise = uncorrelated_noise
        
        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = forgetting_rate
        self.delay = delay # delay must be at least 1
                
        self._select_covariance_function(kernel_func)
        
        self.matches = {} # indexes of the obs_coords in the child noise GPs 
        
    def _select_covariance_function(self, cov_type):
        self.cov_type = cov_type
        if cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2_from_raw_vals
            self.kernel_der = derivfactor_matern_3_2_from_raw_vals
        # the other kernels no longer work because they need to use kernel functions that work with the raw values
        else:
            logging.error('PreferenceComponents: Invalid covariance type %s' % cov_type)        
    
    def _init_w(self):
        self.K = self.kernel_func(self.obs_coords, self.ls) + np.eye(self.N) * 1e-6
        self.invK = np.linalg.inv(self.K)
    
        # kernel used by w
        blocks = [self.K for _ in range(self.Nfactors)]
        self.Kw = block_diag(*blocks)
        self.invKw = np.linalg.inv(self.Kw)
        
        self.sw_matrix = np.ones(self.Kw.shape) * self.shape_sw0 / self.rate_sw0
        
        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them,
        # i.e. the cluster identifiability problem        
        #self.w = mvn.rvs(np.zeros(self.Nfactors * self.N), cov=self.Kw / self.sw_matrix).reshape((self.Nfactors, self.N)).T
        self.w = np.zeros((self.N, self.Nfactors))
        #self.w_cov = self.Kw / self.sw_matrix 
        self.w_cov = np.diag(np.ones(self.N*self.Nfactors)) # use ones to avoid divide by zero
        
        self.Sigma_w = np.zeros((self.N, self.N, self.Nfactors))
        
    def _init_y(self):
        if self.person_features is None: 
            self.invKy = np.diag(np.ones(self.Npeople * self.Nfactors)) # they are all ones
        else:
            self.lsy = np.zeros(self.nperson_features) + self.lsy  
            self.Ky_block = self.kernel_func(self.person_features, self.lsy) + np.eye(self.Npeople) * 1e-6
            self.invKy_block = np.linalg.inv(self.Ky_block)
    
            blocks = [self.Ky_block for _ in range(self.Nfactors)]
            self.Ky = block_diag(*blocks) 
            self.invKy = np.linalg.inv(self.Ky)
        
        # needs the brackets to get the right shape for some reason
        self.sy_matrix = np.ones(self.invKy.shape) * (self.shape_sy0 / float(self.rate_sy0))
        
        #self.y = mvn.rvs(np.zeros(self.Npeople), self.Ky_block, self.Nfactors)
        #self.y /= (self.shape_sy/self.rate_sy)[:, None]
        
        self.y = np.zeros((self.Nfactors, self.Npeople), dtype=float)
        self.y[np.mod(np.arange(self.Npeople), self.Nfactors), np.arange(self.Npeople)] = 1.0
        #self.y /= np.max(self.y)
        #self.y = self.y * self.rate_sy[:, np.newaxis] / self.shape_sy[:, np.newaxis]
        
        #self.y = np.ones((self.Nfactors, self.Npeople))
        
        #self.y_cov = self.Ky / self.sy_matrix
        self.y_cov = np.diag(np.ones(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
        #self.y_cov = np.diag(np.zeros(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
                
        self.Sigma_y = np.zeros((self.Npeople, self.Npeople, self.Nfactors))

    def _init_t(self):
        self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0
        self.t = np.copy(self.t_mu0)     
        self.t_cov = self.K / (self.shape_st / self.rate_st)#np.diag(np.ones(self.N))
        self.Sigma_t = np.zeros((self.N, self.N))
        
    def _init_obs(self, p):
        if p not in self.coordidxs:
            internal_coords_p = self.pref_gp[p].obs_coords
            self.matches[p] = np.ones((internal_coords_p.shape[0], self.N), dtype=bool)
            for dim in range(internal_coords_p.shape[1]):
                self.matches[p] = self.matches[p] & np.equal(internal_coords_p[:, dim:dim+1], 
                                                                       self.obs_coords[:, dim:dim+1].T)
            self.coordidxs[p] = np.sort(np.argwhere(np.sum(self.matches[p], 0))).flatten()        
        
        if p in self.invKf:
            return
        
        if self.pref_gp[p].K is None or len(self.pref_gp[p].K) == 0: # other conditions make this redundant: self.use_noise_svi? 
            self.pref_gp[p].K = self.pref_gp[p].kernel_func(self.pref_gp[p].obs_coords, self.pref_gp[p].ls, 
                                                                operator=self.pref_gp[p].kernel_combination)
        self.invKf[p] = np.linalg.inv(self.pref_gp[p].K)          
            
    def _init_f(self, use_noise_svi):
        self.f = np.zeros((self.N, self.Npeople))        
        self.invKf = {}
        self.coordidxs = {}
        self.wyt_cov = np.zeros((self.Npeople, self.N, self.N))
        
        for person in self.people:
            self.pref_gp[person] = GPPrefLearning(self.nitem_features, self.mu0, self.shape_sf0, self.rate_sf0,
                                    self.shape_ls, self.rate_ls, self.ls, 
                                    use_svi=use_noise_svi, delay=self.delay, 
                                    forgetting_rate=self.forgetting_rate, 
                                    kernel_func='diagonal' if self.uncorrelated_noise else self.cov_type,
                                    ninducing=self.N if self.uncorrelated_noise else 500)
            self.pref_gp[person].max_iter_VB = 1
            self.pref_gp[person].min_iter_VB = 1
            self.pref_gp[person].max_iter_G = 5
            self.pref_gp[person].verbose = self.verbose
            self.pref_gp[person].conv_threshold = 1e-3
            self.pref_gp[person].conv_check_freq = 1
                
    def _init_params(self, use_noise_svi=False):       
        self.N = self.obs_coords.shape[0]
        
        if self.person_features is not None:
            self.Npeople = self.person_features.shape[0]
        else:              
            self.Npeople = np.max(self.people).astype(int) + 1
                    
        if self.Nfactors is None or self.Npeople < self.Nfactors: # not enough items or people
            self.Nfactors = self.Npeople
                    
        # put all prefs into a single GP to get a good initial mean estimate t -- this only makes sense if we can also 
        #estimate w y in a sensibel way, e.g. through factor analysis?        
        #self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs)
        
        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0
        self.shape_sy = np.zeros(self.Nfactors) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors) + self.rate_sy0
        self.shape_st = self.shape_st0
        self.rate_st = self.rate_st0                
                
        if self.new_obs:
            self._init_f(use_noise_svi)
                
        self.ls = np.zeros(self.nitem_features) + self.ls
        
        self._init_w()
        self._init_y()
        self._init_t()
                    
        self.wy = np.zeros((self.N, self.Npeople)) # initialise to priors so that child noise GPs estimate obs noise correctly
        self.wyt = np.zeros((self.N, self.Npeople))
        #self.wy = self.w.dot(self.y) 
        #self._update_wy_plus_t()
        
    def fit(self, personIDs=None, items_1_coords=None, items_2_coords=None, item_features=None, 
            preferences=None, person_features=None, optimize=False, maxfun=20, use_MAP=False, nrestarts=1, 
            input_type='binary', use_lb=False):
        '''
        Learn the model with data as follows:
        personIDs - a list of the person IDs of the people who expressed their preferences
        items_1_coords - if item_features is None, these should be coordinates of the first items in the pairs being 
        compared, otherwise these should be indexes into the item_features vector
        items_2_coords - if item_features is None, these should be coordinates of the second items in each pair being 
        compared, otherwise these should be indexes into the item_features vector
        item_features - feature values for the items. Can be None if the items_x_coords provide the feature values as
        coordinates directly.
        preferences - the values, 0 or 1 to express that item 1 was preferred to item 2.
        '''
        if optimize:
            return self._optimize(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features, 
                            maxfun, use_MAP, nrestarts, input_type)
        
        if personIDs is not None:
            self.new_obs = True # there are people we haven't seen before            
            # deal only with the original IDs to simplify prediction steps and avoid conversions 
            self.people = np.unique(personIDs)
            self.personIDs = personIDs           
            if item_features is None:
                self.obs_coords, self.pref_v, self.pref_u, self.obs_uidxs = get_unique_locations(items_1_coords, items_2_coords)
            else:
                self.obs_coords = np.array(item_features, copy=False)
                self.pref_v = np.array(items_1_coords, copy=False)
                self.pref_u = np.array(items_2_coords, copy=False)
            
            if person_features is not None:
                self.person_features = np.array(person_features, copy=False) # rows per person, columns for feature values
                self.nperson_features = self.person_features.shape[1]
            else:
                self.person_features = None
                self.nperson_features = 0 
            self.preferences = np.array(preferences, copy=False)
        else:  
            self.new_obs = False # do we have new data? If so, reset everything. If not, don't reset the child GPs.
 
        self.input_type = input_type
        self._init_params()
        
        # reset the iteration counters
        self.vb_iter = 0    
        diff = np.inf
        old_w = np.inf
        old_y = np.inf
        old_lb = -np.inf
        converged_count = 0
        while (self.vb_iter < self.min_iter) or (((diff > self.conv_threshold) or (converged_count < self.n_converged)) 
                                                 and (self.vb_iter < self.max_iter)):
            # run a VB iteration
            # compute preference latent functions for all workers
            self._expec_f()
            
            if self.use_t: # compute the preference function means -- assumes a bias toward certain items shared by all
                self._expec_t()            
             
            # find the personality components
            self._expec_w()          
            
            lb = self.lowerbound()
            difflb = lb - old_lb
            logging.debug('Iteration %i: lower bound = %.5f, difference = %.5f' % (self.vb_iter, lb, difflb))
            
            diffwy = np.max((np.max(np.abs(old_w - self.w)), np.max(np.abs(old_y - self.y))))
            logging.debug( "Max difference in latent features: %f at %i iterations" % (diffwy, self.vb_iter))
            
            if not use_lb:
                diff = diffwy
            else:
                diff = difflb                

            old_w = self.w
            old_y = self.y            
            old_lb = lb

            self.vb_iter += 1
            
            if diff <= self.conv_threshold:
                converged_count += 1
            elif diff > self.conv_threshold and converged_count > 0:
                converged_count -= 1
                
            # update covariance terms relating to wy+t. These are used in lower bound and by child noise GPs
            self._update_wy_plus_t()                
            
        logging.debug( "Preference personality model converged in %i iterations." % self.vb_iter )

    def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None, 
                 maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):

        max_iter = self.max_iter
        self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features, input_type=input_type)
        self.max_iter = max_iter

        min_nlml = np.inf
        best_opt_hyperparams = None
        best_iter = -1            
            
        logging.debug("Optimising item length-scale for all dimensions")
            
        nfits = 0 # number of calls to fit function
            
        # optimise each length-scale sequentially in turn
        for r in range(nrestarts):    
            # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML 
            # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg, 
            # or trust-ncg, which may be faster still?
            if person_features is None:
                initialguess = np.log(self.ls)
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))                
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, use_MAP,), 
                   jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'gtol': 0.1 / self.nitem_features})
            else:
                initialguess = np.append(np.log(self.ls), np.log(self.lsy))
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
                logging.debug("Initial person length-scale guess in restart %i: %s" % (r, self.lsy))
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('both', -1, use_MAP,), 
                   jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'gtol': 0.1 / self.nitem_features})
                
            opt_hyperparams = res['x']
            nlml = res['fun']
            nfits += res['nfev']
            
            if nlml < min_nlml:
                min_nlml = nlml
                best_opt_hyperparams = opt_hyperparams
                best_iter = r
                
            # choose a new lengthscale for the initial guess of the next attempt
            if r < nrestarts - 1:
                self.ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls, size=len(self.ls))
                if person_features is not None:
                    self.lsy = gamma.rvs(self.shape_lsy, scale=1.0/self.rate_lsy, size=len(self.lsy))  

        if best_iter < r:
            # need to go back to the best result
            if person_features is None: # don't do this if further optimisation required anyway
                self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, use_MAP=False)

        logging.debug("Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
        if self.person_features is not None:
            logging.debug("Chosen person length-scale %s, used %i evals of NLML over %i restarts" % (self.lsy, nfits, nrestarts))
            
        logging.debug("Optimal hyper-parameters: item = %s, person = %s" % (self.ls, self.lsy))   
        return self.ls, self.lsy, -min_nlml # return the log marginal likelihood

    def neg_marginal_likelihood(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        if lstype=='item':
            if dimension == -1 or self.n_wlengthscales == 1:
                self.ls[:] = np.exp(hyperparams)
            else:
                self.ls[dimension] = np.exp(hyperparams)
        elif lstype=='person':
            if dimension == -1 or self.n_ylengthscales == 1:
                self.lsy[:] = np.exp(hyperparams)
            else:
                self.lsy[dimension] = np.exp(hyperparams)
        elif lstype=='fa':
            new_Nfactors = int(np.round(np.exp(hyperparams)))
        elif lstype=='both' and dimension <= 0: # can be zero if single length scales or -1 to do all
            # person and item
            self.ls[:] = np.exp(hyperparams[:self.nitem_features])
            self.lsy[:] = np.exp(hyperparams[self.nitem_features:])
        else:
            logging.error("Invalid length-scale type for optimization.")
        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf
                
        # make sure we start again -- fit should set the value of parameters back to the initial guess
        if lstype!='fa' or new_Nfactors != self.Nfactors: #don't rerun if the number of factors is same.
            self.fit()
        marginal_log_likelihood = self.lowerbound()        
        if use_MAP:
            log_model_prior = self.ln_modelprior()        
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood
            
        if lstype=='person':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.lsy))
            else:
                logging.debug("LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.lsy[dimension]))
        elif lstype=='item':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.ls))
            else:
                logging.debug("LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.ls[dimension]))
        elif lstype == 'both':
                logging.debug("LML: %f, item length-scales = %s, person length-scales = %s" % (lml, self.ls, self.lsy))
        return -lml
    
    def _gradient_dim_people_terms(self, dimension):
        dKdls = self.Ky * self.kernel_der(self.person_features, self.lsy, dimension) 
        # try to make the s scale cancel as much as possible
        invK_y = self.invKy.dot(self.y.T)
        invKs_C = self.sy_matrix * self.invKy.dot(self.y_cov)
        N = self.Npeople
        
        return dKdls, invK_y, invKs_C, N
    
    def _gradient_dim(self, lstype, d, dimension):
        der_logpw_logqw = 0
        der_logpy_logqy = 0
        der_logpt_logqt = 0
        der_logpf_logqf = 0
        
        # compute the gradient. This should follow the MAP estimate from chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
            dKdls = self.K * self.kernel_der(self.obs_coords, self.ls, dimension) 
            # try to make the s scale cancel as much as possible
            invK_w = self.invK.dot(self.w)
            invKs_C = self.sw_matrix * self.invKw.dot(self.w_cov)
            N = self.N
                
            for f in range(self.Nfactors):
                fidxs = np.arange(N) + (N * f)
                invK_wf = invK_w[:, f]
                invKs_C_f = invKs_C[fidxs, :][:, fidxs] 
                sw = self.shape_sw[f] / self.rate_sw[f]
                Sigma_w_f = self.Sigma_w[:, :, f]
                der_logpw_logqw += 0.5 * (invK_wf.T.dot(dKdls).dot(invK_wf) * sw - 
                                    np.trace(invKs_C_f.dot(Sigma_w_f).dot(dKdls / sw)))
            
            if self.use_t:
                invKs_t = self.invK.dot(self.t) * self.shape_st / self.rate_st
                invKs_C = self.shape_st / self.rate_st * self.invKt.dot(self.t_cov)
        
                der_logpt_logqt = 0.5 * (invKs_t.T.dot(dKdls).dot(invKs_t) - 
                            np.trace(invKs_C.dot(self.Sigma_t).dot(dKdls / self.shape_st * self.rate_st)))
                
            for p in self.pref_gp:
                der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
            
        elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):               
            if self.person_features is None:
                pass          
            else:
                dKdls, invK_y, invKs_C, N = self._gradient_dim_people_terms(dimension)
            
            for f in range(self.Nfactors):
                fidxs = np.arange(N) + (N * f)
                invK_yf = invK_y[:, f]
                invKs_C_f = invKs_C[fidxs, :][:, fidxs]                     
                sy = self.shape_sy[f] / self.rate_sy[f]
                Sigma_y_f = self.Sigma_y[fidxs, :][:, fidxs]             
                der_logpy_logqy += 0.5 * (invK_yf.T.dot(dKdls).dot(invK_yf) * sy - 
                                    np.trace(invKs_C_f.dot(Sigma_y_f).dot(dKdls / sy)))
                         
        return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf
    
    def nml_jacobian(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        
        needs_fitting = self.people is None
        
        if lstype=='item':
            if dimension == -1 or self.n_wlengthscales == 1:
                if np.any(np.abs(self.ls - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True            
                    self.ls[:] = np.exp(hyperparams)
                dimensions = np.arange(len(self.ls))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True            
                    self.ls[dimension] = np.exp(hyperparams)
                dimensions = [dimension]
        elif lstype=='person':
            if dimension == -1 or self.n_ylengthscales == 1:
                if np.any(np.abs(self.lsy - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True            
                    self.lsy[:] = np.exp(hyperparams)        
                dimensions = np.arange(len(self.lsy))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True            
                    self.lsy[dimension] = np.exp(hyperparams)
                dimensions = [dimension]
        elif lstype=='both' and dimension <= 0:
            
            hyperparams_w = hyperparams[:self.nitem_features]
            hyperparams_y = hyperparams[self.nitem_features:]
            
            if np.any(np.abs(self.ls - np.exp(hyperparams_w)) > 1e-4):
                needs_fitting = True
                self.ls[:] = np.exp(hyperparams_w)
            
            if np.any(np.abs(self.lsy - np.exp(hyperparams_y)) > 1e-4):
                needs_fitting = True
                self.lsy[:] = np.exp(hyperparams_y)
            
            dimensions = np.append(np.arange(len(self.ls)), np.arange(len(self.lsy)))
        else:
            logging.error("Invalid optimization setup.")

        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf
                
        # make sure we start again -- fit should set the value of parameters back to the initial guess
        if needs_fitting:
            self.fit()

        #num_jobs = multiprocessing.cpu_count()
        #mll_jac = Parallel(n_jobs=num_jobs)(delayed(self._gradient_dim)(lstype, d, dim)
        #                                      for d, dim in enumerate(dimensions))
        #mll_jac = np.array(mll_jac, order='F')
        mll_jac = np.zeros(len(dimensions), dtype=float)
        for d, dim in enumerate(dimensions):
            mll_jac[d] = self._gradient_dim(lstype, d, dim)
        
        if len(mll_jac) == 1: # don't need an array if we only compute for one dimension
            mll_jac = mll_jac[0]
        elif (lstype == 'item' and self.n_wlengthscales == 1) or (lstype == 'person' and self.n_ylengthscales == 1):
            mll_jac = np.sum(mll_jac)
        elif lstype == 'both':
            if self.n_wlengthscales == 1:
                mll_jac[:self.nitem_features] = np.sum(mll_jac[:self.nitem_features])
            if self.n_ylengthscales == 1:
                mll_jac[self.nitem_features:] = np.sum(mll_jac[self.nitem_features:])

        if use_MAP: # gradient of the log prior
            log_model_prior_grad = self.ln_modelprior_grad()        
            lml_jac = mll_jac + log_model_prior_grad
        else:
            lml_jac = mll_jac
        logging.debug("Jacobian of LML: %s" % lml_jac)
        if self.verbose:
            logging.debug("...with item length-scales = %s, person length-scales = %s" % (self.ls, self.lsy))
        return -lml_jac # negative because the objective function is also negated
 
    def predict_item_feats(self, items_coords, item_features=None):
        if item_features is None:
            coords = items_coords
        else:
            coords = item_features[items_coords]
               
        K1 = self.kernel_func(coords, self.ls, self.obs_coords)
        invKw = self.invK.dot(self.w)
        w1 = K1.dot(invKw)
            
        return w1
            
    def predict(self, personids, items_1_coords, items_2_coords, item_features=None, person_features=None):
        Npairs = len(personids)
        predicted_prefs = np.zeros(Npairs)
        upeople = np.unique(personids)
        
        if item_features is None:
            coords_1 = items_1_coords
            coords_2 = items_2_coords
        else:
            coords_1 = item_features[items_1_coords]
            coords_2 = item_features[items_2_coords]
         
        if person_features is None and self.person_features:
            logging.debug('No person features provided -- assuming same people as during training')
            reuse_training_people = True
            person_features = self.person_features
        else:
            reuse_training_people = False                  
         
        y = self._predict_y(person_features, len(upeople))
            
        # this could be made more efficient because duplicate locations are computed separately!
        # distances for t-space
        t1, w1 = self._predict_w_t(coords_1)
        t2, w2 = self._predict_w_t(coords_2)
        
        wy_1p = w1.dot(y)
        wy_2p = w2.dot(y)
        mu0_1 = wy_1p + t1
        mu0_2 = wy_2p + t2         
         
        for p in upeople:            
            pidxs = personids == p
            npairs_p = np.sum(pidxs)
                                    
            if p in self.people and reuse_training_people:            
                pref_gp_p = self.pref_gp[p]
                predicted_prefs[pidxs] = pref_gp_p.predict(coords_1[pidxs, :], coords_2[pidxs, :], 
                                      mu0_output1=mu0_1[pidxs, p:p+1], mu0_output2=mu0_2[pidxs, p:p+1], return_var=False).flatten()
            else:
                mu0 = np.concatenate((mu0_1[pidxs, p:p+1], mu0_2[pidxs, p:p+1]), axis=0)
                predicted_prefs[pidxs] = pref_likelihood(f=mu0, subset_idxs=[], 
                                     v=np.arange(npairs_p), u=np.arange(npairs_p, npairs_p*2)).flatten()
                
        return predicted_prefs
    
    def predict_f(self, personids, items_1_coords, item_features=None, person_features=None):
        N = items_1_coords.shape[0]
        predicted_f = np.zeros(N)
        upeople = np.unique(personids)
         
        if item_features is None:
            coords_1 = items_1_coords
        else:
            coords_1 = item_features[items_1_coords]         

        if person_features is None and self.person_features:
            logging.debug('No person features provided -- assuming same people as during training')
            person_features = self.person_features
            reuse_training_people = True
        else:
            reuse_training_people = False
         
        y = self._predict_y(person_features, len(upeople))
         
        t1, w1 = self._predict_w_t(coords_1)

        wy_1p = w1.dot(y)
        mu0_1 = wy_1p + t1
         
        if not reuse_training_people:
            return mu0_1[np.arange(N), personids]
         
        for p in upeople:            
            pidxs = personids == p

            if p in self.people:            
                pref_gp_p = self.pref_gp[p]
                predicted_f[pidxs] = pref_gp_p.predict_f(coords_1[pidxs, :], mu0_output=mu0_1[pidxs, p:p+1])[0].flatten()
            else:
                predicted_f[pidxs] = mu0_1[pidxs, 0]
                
        return predicted_f
    
    def _predict_w_t(self, coords_1):
        # kernel between pidxs and t
        K1 = self.kernel_func(coords_1, self.ls, self.obs_coords)
    
        if self.use_t:
            # use kernel to compute t
            invKt = self.invK.dot(self.t)
            t1 = K1.dot(invKt)
        else:
            t1 = np.zeros((coords_1.shape[0], 1))
            
        # kernel between pidxs and w -- use kernel to compute w
        invKw = self.invK.dot(self.w)
        w1 = K1.dot(invKw)
        
        return t1, w1        
    
    def _predict_y(self, person_features, Npeople):
        if person_features is None:
            y = np.ones((self.Nfactors, Npeople))
        else:
            #distances for y-space. Kernel between p and people already seen
            Ky = self.kernel_func(person_features, self.lsy, self.person_features)
            invKy_train = self.invKy_block
            y_train = self.y.T
            
            # use kernel to compute y
            y = Ky.dot(invKy_train).dot(y_train)
            #y *= self.rate_sy / self.shape_sy # why was this line used before? It seems wrong.      
            y = y.T   
            
        return y        
        
    def _expec_f_p(self, p, mu0_output):
        f, _ = self.pref_gp[p].predict_f(
                items_coords=self.coordidxs[p] if self.vb_iter==0 else None,
                items_features=self.obs_coords if self.vb_iter==0 else None, 
                mu0_output=mu0_output, 
                reuse_output_kernel=True)
        self.f[self.coordidxs[p], p] = f.flatten()        
        
    def _update_wy_plus_t(self):
        self.wyt = self.wy + self.t
        self.wyt_cov[:, :, :] = 0
        
        for p in self.people:
            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + (f * self.N)
                pidx = f * self.Npeople + p
                self.wyt_cov[p] += self.w_cov[fidxs, :][:, fidxs] * self.y_cov[pidx, pidx] + \
                                self.w_cov[fidxs, :][:, fidxs] * self.y[f, p:p+1]**2 + \
                                self.w[:, f] * self.y_cov[pidx, pidx] * self.w[:, f].T
            if self.use_t:
                self.wyt_cov[p] += self.t_cov
            
        return self.wyt_cov        
        
    def _expec_f(self):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''                           
        for p in self.pref_gp:
            if self.verbose:    
                logging.debug( "Running expec_f for person %i..." % p )
            plabelidxs = self.personIDs == p
                        
            self.pref_gp[p].fit(self.pref_v[plabelidxs], self.pref_u[plabelidxs], self.obs_coords, 
                                self.preferences[plabelidxs], mu0=self.wyt[:, p:p+1], 
                                cov_mu0=self.wyt_cov[p], process_obs=self.new_obs, input_type=self.input_type)                
            
            # find the index of the coords in coords_p in self.obs_coords
            # coordsidxs[p] needs to correspond to data points in same order as invKf[p]
            self._init_obs(p)
            
            self._expec_f_p(p, self.wyt[:, p:p+1])
                
            if self.verbose:    
                logging.debug( "Expec_f for person %i out of %i. s=%.3f" % (p, len(self.pref_gp.keys()), self.pref_gp[p].s) )

        self.new_obs = False # don't process the observations again unless fit() is called

        if self.verbose:
            logging.debug('Updated q(f)')
    
    def _compute_sigma_w_p(self, N, y_p, yidxs, prec_p, pidxs):
        # add the covariance for this person's observations as a block in the covariance matrix Sigma
        Sigma_p = np.zeros((N * self.Nfactors, N * self.Nfactors))
        if self.y_cov.ndim > 1:
            Sigma_yscaling = y_p.dot(y_p.T) + self.y_cov[yidxs, :][:, yidxs] # covariance between people?
        else:
            Sigma_yscaling = y_p.dot(y_p.T)
            Sigma_yscaling[range(self.Nfactors), range(self.Nfactors)] += self.y_cov # covariance between people?
        
        Sigma_w = np.zeros((N, N, self.Nfactors))
        
        for f in range(self.Nfactors):
            for g in range(self.Nfactors):
                Sigma_p_fg = prec_p * Sigma_yscaling[f, g]
                Sigma_p_rows = np.zeros((len(pidxs), N * self.Nfactors))
                Sigma_p_rows[:, pidxs + g * N] = Sigma_p_fg
                Sigma_p[pidxs + f * N, :] += Sigma_p_rows

                if f==g:
                    Sigma_w[:, :, f] = Sigma_p_fg
                    
        return Sigma_p, Sigma_w
                 
    def _expec_w(self):
        '''
        Compute the expectation over the latent features of the items and the latent personality components
        '''
        # Put a GP prior on w with covariance K/gamma and mean 0
        N = self.N
        Sigma = np.zeros((N * self.Nfactors, N * self.Nfactors))
        x = np.zeros((N, self.Nfactors))

        for p in self.pref_gp:
            pidxs = self.coordidxs[p]
            y_p = self.y[:, p:p+1]
            yidxs = p + self.Npeople * np.arange(self.Nfactors)

            prec_p = self.invKf[p] * self.pref_gp[p].s
            invQ_f = prec_p.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :])     
            # add the means for this person's observations to the list of observations, x 
            x[pidxs, :] += y_p.T * invQ_f 
            
            Sigma_p, Sigma_w_p = self._compute_sigma_w_p(N, y_p, yidxs, prec_p, pidxs)
            Sigma += Sigma_p
            self.Sigma_w += Sigma_w_p
                            
        x = x.T.flatten()[:, np.newaxis]
        # w_cov is same shape as K with rows corresponding to (f*N) + n where f is factor index from 0 and 
        # n is data point index
        
        self.w, self.w_cov = update_gaussian(self.invKw, self.sw_matrix, Sigma, x)
        self.w = np.reshape(self.w, (self.Nfactors, self.N)).T # w is N x Nfactors   
        
        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + (self.N * f)
            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, self.N, 
                            self.invK, self.w[:, f:f+1], np.zeros((self.N, 1)), f_cov=self.w_cov[fidxs, :][:, fidxs])
            
            self.sw_matrix[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]            
        
        self._expec_y()
        self.wy = self.w.dot(self.y)    
        return
    
    def _compute_sigma_y_p(self, N, w, w_cov, prec_f, pidxs):
        covterm = np.zeros((self.Nfactors, self.Nfactors))
        for f in range(self.Nfactors): 
            w_cov_idxs = pidxs + (f * N)
            w_cov_f = w_cov[w_cov_idxs, :]
            for g in range(self.Nfactors):
                w_cov_idxs = pidxs + (g * N)
                covterm[f, g] = np.sum(prec_f * w_cov_f[:, w_cov_idxs])
        return w.T.dot(prec_f).dot(w) + covterm

    def _expec_y(self):
        '''
        Compute expectation over the personality components using VB
        '''
        Npeople = self.Npeople  
        Sigma = np.zeros((self.Nfactors * Npeople, self.Nfactors * Npeople))
        x = np.zeros((Npeople, self.Nfactors))

        pidx = 0
        for p in self.pref_gp:
            pidxs = self.coordidxs[p]                       
            prec_f = self.invKf[p] * self.pref_gp[p].s
            w_cov = self.w_cov
            w = self.w[pidxs, :]
            N = self.N
            
            invQ_f = prec_f.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :]) 
            
            sigmaidxs = np.arange(self.Nfactors) * Npeople  + pidx
            Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
            Sigmarows[:, sigmaidxs] =  self._compute_sigma_y_p(N, w, w_cov, prec_f, pidxs)
            Sigma[sigmaidxs, :] += Sigmarows             
              
            x[pidx, :] = w.T.dot(invQ_f).T
            pidx += 1
                
        x = x.T.flatten()[:, np.newaxis]
        self.Sigma_y = Sigma
        
        # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0 
        # and p is person index
        self.y_cov = np.linalg.inv(self.invKy * self.sy_matrix + Sigma)
        self.y = self.y_cov.dot(x)
       
        # y is Nfactors x Npeople            
        self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
            
        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + (self.Npeople * f)
            self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople, 
                        self.invKy_block, self.y[f:f+1, :].T, np.zeros((self.Npeople, 1)), f_cov=self.y_cov[fidxs, :][:, fidxs])
            
            self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f] # sy_rows
 
    def _expec_t(self):
        if not self.use_t:
            return

        N = self.N    
        Sigma = np.zeros((N, N))
        x = np.zeros((N, 1))
        
        for p in self.pref_gp:
            pidxs = self.coordidxs[p]
            prec_f = self.invKf[p] * self.pref_gp[p].s
            invQ_f = prec_f.dot(self.f[pidxs, p:p+1] - self.wy[pidxs, p:p+1])
            x[pidxs, :] += invQ_f
                
            sigmarows = np.zeros((len(pidxs), N))
            sigmarows[:, pidxs] = prec_f
            Sigma[pidxs, :] += sigmarows

        self.Sigma_t = Sigma
                
        invKts = self.invK * self.shape_st / self.rate_st
        self.t = invKts.dot(self.t_mu0) + x
        self.t_cov = np.linalg.inv(Sigma + invKts)
        self.t = self.t_cov.dot(self.t)

        self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.N, self.invK, 
                                                        self.t, np.zeros((self.N, 1)), f_cov=self.t_cov)
                
    def lowerbound(self):
        f_terms = 0
        for p in self.pref_gp:
            f_terms += self.pref_gp[p].lowerbound()
             
            if self.verbose:
                logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
                
            break # do only one person for debugging
            
        Elnsw = psi(self.shape_sw) - np.log(self.rate_sw)
        Elnsy = psi(self.shape_sy) - np.log(self.rate_sy)
        Elnst = psi(self.shape_st) - np.log(self.rate_st)
        
        sw = self.shape_sw / self.rate_sw
        sy = self.shape_sy / self.rate_sy
        st = self.shape_st / self.rate_st
            
        logpw = expec_pdf_gaussian(self.Kw, self.invKw, Elnsw, self.N*self.Nfactors, self.sw_matrix, 
                                   self.w.T.reshape(self.N * self.Nfactors, 1), 0, self.w_cov, 0) 
        logqw = expec_q_gaussian(self.w_cov, self.N * self.Nfactors) 
        
        if self.use_t:
            logpt = expec_pdf_gaussian(self.K, self.invK, Elnst, self.N, st, self.t, self.t_mu0, self.t_cov, 0)
            logqt = expec_q_gaussian(self.t_cov, self.N) 
        else:
            logpt = 0
            logqt = 0        

        logpy = expec_pdf_gaussian(self.Ky, self.invKy, Elnsy, self.Npeople*self.Nfactors, self.sy_matrix, 
                                       self.y.reshape(self.Npeople * self.Nfactors, 1), 0, self.y_cov, 0) 
        logqy = expec_q_gaussian(self.y_cov, self.Npeople*self.Nfactors) 

        logps_y = 0
        logqs_y = 0
        logps_w = 0
        logqs_w = 0        
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])
                    
            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])
        
        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst) 
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)
    
        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = f_terms + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw/self.rate_sw))        
            logging.debug('s_y=%s' % (self.shape_sy/self.rate_sy))
            logging.debug('s_t=%.2f' % (self.shape_st/self.rate_st))
            
        logging.debug('fterms=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (f_terms, w_terms, y_terms, t_terms))
    
        if self.verbose:
            logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
        
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
            
            
class PreferenceNoComponentFactors(PreferenceComponents):
    '''
    Class for preference learning with multiple users, where each user has a GP. No sharing of information between users
    and no latent components.
    '''
    def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1, 
                 shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20, 
                 use_common_mean_t=True, uncorrelated_noise=False, use_noise_svi=True, kernel_func='matern_3_2',
                 max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
        
        if uncorrelated_noise:
            uncorrelated_noise = False 
            logging.warning("It doesn't make sense to run this without correlations at the personal GP level -- turning\
            uncorrelated_noise off")
        
        PreferenceComponents.__init__(self, nitem_features, nperson_features, mu0, shape_s0, rate_s0, 
                 shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors, 
                 use_common_mean_t, uncorrelated_noise, use_noise_svi, kernel_func,
                 max_update_size, ninducing, forgetting_rate, delay)
        self.use_t = False
    
    def _init_w(self):
        self.w = np.zeros((self.N, self.Nfactors))        
            
    def _init_y(self):
        self.y = np.zeros((self.Nfactors, self.Npeople))
        
    def _predict_w_t(self, coords_1):
        # kernel between pidxs and t
        w1 = np.zeros((coords_1.shape[0], self.Nfactors))
        t1 = np.zeros((coords_1.shape[0], 1))
        
        return t1, w1        
        
    def _predict_y(self, _, Npeople):
        return super(PreferenceNoComponentFactors, self)._predict_y(None, Npeople)        
        
    def _gradient_dim(self, lstype, d, dimension):
        der_logpf_logqf = 0
        
        if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
            for p in self.pref_gp:
                der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
                                     
        return der_logpf_logqf
    
    def _expec_f_p(self, p, mu0_output):
        f, _ = self.pref_gp[p].predict_f(
                items_coords=self.coordidxs[p] if self.vb_iter==0 else None,
                items_features=self.obs_coords if self.vb_iter==0 else None, 
                mu0_output=mu0_output, 
                reuse_output_kernel=True)
        self.f[self.coordidxs[p], p] = f.flatten()      
    
    def _expec_w(self):
        return
    
    def lowerbound(self):
        f_terms = 0
        
        for p in self.pref_gp:
            f_terms += self.pref_gp[p].lowerbound()
            if self.verbose:
                logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
            
        lb = f_terms
        
        if self.verbose:
            logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
        
        return lb    
            
class PreferenceComponentsFA(PreferenceComponents):
    # Factor Analysis
    def _init_w(self):
        self.fa = FactorAnalysis(n_components=self.Nfactors)  
        self.w = np.zeros((self.N, self.Nfactors))

    def _init_y(self):
        self.y = np.ones((self.Nfactors, self.Npeople))
        
    def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None, 
                 maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):

        max_iter = self.max_iter
        self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features, input_type=input_type)
        self.max_iter = max_iter

        min_nlml = np.inf
        best_opt_hyperparams = None
        best_iter = -1            
            
        logging.debug("Optimising item length-scale for all dimensions")
            
        nfits = 0 # number of calls to fit function
            
        # optimise each length-scale sequentially in turn
        for r in range(nrestarts):    
            # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML 
            # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg, 
            # or trust-ncg, which may be faster still?
            initialguess = np.log(self.ls)
            logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))                
            res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, use_MAP,), 
                   jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'gtol': 0.1 / self.nitem_features})

            opt_hyperparams = res['x']
            nlml = res['fun']
            nfits += res['nfev']
            
            if nlml < min_nlml:
                min_nlml = nlml
                best_opt_hyperparams = opt_hyperparams
                best_iter = r
                
            # choose a new lengthscale for the initial guess of the next attempt
            if r < nrestarts - 1:
                self.ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls, size=len(self.ls))

        if best_iter < r:
            # need to go back to the best result
            self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, use_MAP=False)

        logging.debug("Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
            
        initialguess = np.log(self.Nfactors)
        res = minimize(self.neg_marginal_likelihood, initialguess, args=('fa', -1, use_MAP,), 
               method='Nelder-Mead', options={'maxfev':maxfun, 'xatol':np.mean(self.ls) * 1e100, 'return_all':True})
        min_nlml = res['fun']
        logging.debug("Optimal number of factors = %s, with initialguess=%i and %i function evals" % (self.Nfactors,
                                                                       int(np.exp(initialguess)), res['nfev']))     

        logging.debug("Optimal hyper-parameters: item = %s" % (self.ls))               
        return self.ls, self.lsy, -min_nlml

    def _gradient_dim(self, lstype, d, dimension):
        der_logpw_logqw = 0
        der_logpy_logqy = 0
        der_logpt_logqt = 0
        der_logpf_logqf = 0
        
        # compute the gradient. This should follow the MAP estimate from chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):                    
            for p in self.pref_gp:
                der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
            
        elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):               
            return 0  
                         
        return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf
    
    def _predict_w_t(self, coords_1):
        if self.cov_type == 'matern_3_2':
            kernel = Matern(self.ls)
        else:
            logging.error('Kernel not implemented for FA')
            return 0
        
        w1 = np.zeros((coords_1.shape[0], self.Nfactors))
        for f in range(self.Nfactors):
            w_gp = GPR(kernel=kernel, optimizer=None)
            w_gp.fit(self.obs_coords, self.w[:, f])
            w1[:, f] = w_gp.predict(coords_1, return_std=False)
            
        t_gp = GPR(kernel, optimizer=None)
        t_gp.fit(self.obs_coords, self.t)
        t1 = t_gp.predict(coords_1, return_std=False)
        
        return t1, w1
          
    def _predict_y(self, person_features, Npeople):
        
        y1 = np.zeros((self.Nfactors, Npeople))        
        if person_features is None:
            return y1
        
        if self.cov_type == 'matern_3_2':
            kernel = Matern(self.ls)
        else:
            logging.error('Kernel not implemented for FA')
            return 0
                
        for f in range(self.Nfactors):
            y_gp = GPR(kernel=kernel, optimizer=None)
            y_gp.fit(self.person_features, self.y[f, :])
            y1[f, :] = y_gp.predict(person_features, return_std=False)
            
        return y1
          
    def _expec_w(self):
        '''
        Compute the expectation over the latent features of the items and the latent personality components
        '''
        self.y = self.fa.fit_transform(self.f.T).T
        self.w = self.fa.components_.T
        self.wy = self.w.dot(self.y)
        return

    def _expec_t(self):
        self.t = self.fa.mean_[:, np.newaxis]
        return
    
    def lowerbound(self):
        f_terms = 0
        
        for p in self.pref_gp:
            f_terms += self.pref_gp[p].lowerbound()
            if self.verbose:
                logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
            
        lb = np.sum(self.fa.score_samples(self.f.T)) + f_terms    
        if self.verbose:
            logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
        return lb

class PreferenceComponentsSVI(PreferenceComponents):
    
    def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1, 
                 shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20, 
                 use_common_mean_t=True, uncorrelated_noise=False, use_noise_svi=True, kernel_func='matern_3_2',
                 max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
        
        self.use_svi_people = False # this gets switched on later if we have features and correlations between people
        self.t_mu0_u = 0
        
        self.use_noise_svi = use_noise_svi
        if uncorrelated_noise:
            logging.info('Switching off SVI for the noise model because we are using uncorrelated noise.')
            self.use_noise_svi = False        

        super(PreferenceComponentsSVI, self).__init__(nitem_features, nperson_features, mu0, shape_s0, rate_s0, 
                 shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors, use_common_mean_t, 
                 uncorrelated_noise, True, kernel_func, max_update_size, ninducing, forgetting_rate, delay)
        
    def _init_w(self):
        self.w = np.zeros((self.N, self.Nfactors))
        
    def _init_y(self):
        if not self.use_svi_people:
            return super(PreferenceComponentsSVI, self)._init_y()
        
        #self.y = np.mod(np.arange(self.Npeople), self.Nfactors).astype(float) + 1
        #self.y /= np.max(self.y)
        self.y = self.Ky_nm_block.dot(self.invKy_mm_block).dot(self.y_u.T).T
        
        if not self.use_svi_people:        
            self.y_cov = np.diag(np.ones(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
        else:
            self.y_cov = np.array(self.rate_sy / self.shape_sy).flatten()
            self.Sigma_y = np.zeros((self.Npeople, self.Npeople, self.Nfactors))
        
    def _init_t(self):
        self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0
        self.t = np.copy(self.t_mu0)     
        self.t_cov = self.K / (self.shape_st / self.rate_st)#np.diag(np.ones(self.N))
        self.Sigma_t = np.zeros((self.N, self.N))
        
    def _init_params(self):       
        if self.person_features is not None:
            self.use_svi_people = True
        super(PreferenceComponentsSVI, self)._init_params(self.use_noise_svi)
        self._choose_inducing_points()
        
    # merge this with below... should not be possible to use SVI with the child GPs without SVI for the factors.
    # make sure correct setting is used for the child GPs in their constructor...
    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        nobs = self.obs_coords.shape[0]
        
        self.update_size = self.max_update_size # number of observed points in each stochastic update        
        if self.update_size > nobs:
            self.update_size = nobs 
            
        if self.ninducing > nobs:
            self.ninducing = nobs
            self.inducing_coords = self.obs_coords
        else:
            init_size = 300
            if self.ninducing < init_size:
                init_size = self.ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing, random_state=345)
            kmeans.fit(self.obs_coords)
        
            self.inducing_coords = kmeans.cluster_centers_            
        
        self.w_invSm = np.zeros((self.ninducing * self.Nfactors, 1), dtype=float)# theta_1
        self.w_invS = np.zeros((self.ninducing * self.Nfactors, self.ninducing * self.Nfactors), dtype=float) # theta_2

        self.t_invSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1
        self.t_invS = np.diag(np.ones(self.ninducing, dtype=float)) # theta_2

        self.t_mu0_u = np.zeros((self.ninducing, 1)) + self.t_mu0_u
        
        self.K_mm = self.kernel_func(self.inducing_coords, self.ls) + 1e-6 * np.eye(self.ninducing) # jitter
        self.invK_mm = np.linalg.inv(self.K_mm)
        self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords)

        #If the child GPs use SVI, they can use a common set of inducing points to save computing the 
        #kernels multiple times. This may not help if each person has only a few observations. 
        if self.use_noise_svi and not self.uncorrelated_noise:
            for person in self.pref_gp:
                self.pref_gp[person].init_inducing_points(self.inducing_coords, self.K_mm, self.invK_mm, self.K_nm)
        
        self.Lambda_factor_t = np.diag(np.ones(self.K_mm.shape[0])) # self.invK_mm.dot(self.K_mm.T) 
        
        blocks = [self.K_mm for _ in range(self.Nfactors)]
        self.Kw_mm = block_diag(*blocks)
        blocks = [self.invK_mm for _ in range(self.Nfactors)]
        self.invKw_mm = block_diag(*blocks)
        blocks = [self.K_nm for _ in range(self.Nfactors)]
        self.Kw_nm = block_diag(*blocks)
        
        if self.use_noise_svi:
            # if the inducing points are the same for the noise GPs, this matrix is diagonal
            self.Lambda_factor_w = np.diag(np.ones(self.Kw_mm.shape[0]))
            # if not, we don't want to save the whole matrix because it will be too large -- use stochastic sampling.
         
        self.w_u = np.zeros((self.ninducing, self.Nfactors))       
        #self.w_u = mvn.rvs(np.zeros(self.Nfactors * self.ninducing), cov=self.Kw_mm).reshape((self.Nfactors, self.ninducing)).T
        #self.w_u *= (self.shape_sw/self.rate_sw)[np.newaxis, :]
        #self.w_u = 2 * (np.random.rand(self.ninducing, self.Nfactors) - 0.5) * self.rate_sw / self.shape_sw #np.zeros((self.ninducing, self.Nfactors))
        self.t_u = np.zeros((self.ninducing, 1))
        self.f_u = np.zeros((self.ninducing, self.Npeople))
                
        if not self.uncorrelated_noise:
            for person in self.pref_gp:
                self.pref_gp[person].init_inducing_points(self.inducing_coords, self.K_mm, self.invK_mm, self.K_nm)
                
        # sort this out when we call updates to s
        #self.shape_s = self.shape_s0 + 0.5 * self.ninducing # update this because we are not using n_locs data points -- needs replacing?

        # For y
        if self.person_features is None:
            self.use_svi_people = False
            return 
    
        self.y_update_size = self.max_update_size # number of inducing points in each stochastic update            
        if self.y_update_size > self.Npeople:
            self.y_update_size = self.Npeople       
        
        self.y_ninducing = self.ninducing           
        if self.y_ninducing > self.people.shape[0]:
            self.y_ninducing = self.people.shape[0]
        
        init_size = 300
        if self.y_ninducing > init_size:
            init_size = self.y_ninducing
        kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.y_ninducing)
        kmeans.fit(self.person_features)
        
        self.y_inducing_coords = kmeans.cluster_centers_

        self.y_invSm = np.zeros((self.y_ninducing * self.Nfactors, 1), dtype=float)# theta_1
        self.y_invS = np.diag(np.ones(self.y_ninducing * self.Nfactors, dtype=float)) # theta_2

        self.Ky_mm_block = self.kernel_func(self.y_inducing_coords, self.lsy)
        self.Ky_mm_block += 1e-6 * np.eye(len(self.Ky_mm_block)) # jitter 
        blocks = [self.Ky_mm_block for _ in range(self.Nfactors)]
        self.Ky_mm = block_diag(*blocks)            
        
        self.invKy_mm_block = np.linalg.inv(self.Ky_mm_block)
        blocks = [self.invKy_mm_block for _ in range(self.Nfactors)]
        self.invKy_mm = block_diag(*blocks)  
        
        self.Ky_nm_block = self.kernel_func(self.person_features, self.lsy, self.y_inducing_coords)
        blocks = [self.Ky_nm_block for _ in range(self.Nfactors)]
        self.Ky_nm = block_diag(*blocks)
        
        self.Lambda_factor_y = self.invKy_mm.dot(self.Ky_nm.T)
        
        self.y_u = mvn.rvs(np.zeros(self.y_ninducing), self.Ky_mm_block, self.Nfactors)
        self.y_u /= (self.shape_sy / self.rate_sy)[:, None]

        #np.random.rand(self.Nfactors, self.y_ninducing) - 0.5 #np.zeros((self.Nfactors, self.y_ninducing))        
        
    def _gradient_dim(self, lstype, d, dimension):
        der_logpw_logqw = 0
        der_logpy_logqy = 0
        der_logpt_logqt = 0
        der_logpf_logqf = 0
        
        # compute the gradient. This should follow the MAP estimate from chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
            dKdls = self.K_mm * self.kernel_der(self.inducing_coords, self.ls, dimension) 
            # try to make the s scale cancel as much as possible
            invK_w = self.invK_mm.dot(self.w_u)
            invKs_C = self.invKws_mm_S
            N = self.ninducing
                
            self._compute_sigma_w()
                
            for f in range(self.Nfactors):
                fidxs = np.arange(N) + (N * f)
                invK_wf = invK_w[:, f]
                invKs_C_f = invKs_C[fidxs, :][:, fidxs] 
                sw = self.shape_sw[f] / self.rate_sw[f]
                Sigma_w_f = self.Sigma_w[:, :, f]
                der_logpw_logqw += 0.5 * (invK_wf.T.dot(dKdls).dot(invK_wf) * sw - 
                                    np.trace(invKs_C_f.dot(Sigma_w_f).dot(dKdls / sw)))
            
            if self.use_t:
                invKs_t = self.inv_Kts_mm.dot(self.t_u)
                invKs_C = self.invKts_mm_S
            
                self._compute_sigma_t()
            
                der_logpt_logqt = 0.5 * (invKs_t.T.dot(dKdls).dot(invKs_t) - 
                            np.trace(invKs_C.dot(self.Sigma_t).dot(dKdls / self.shape_st * self.rate_st)))
                
            for p in self.pref_gp:
                der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
            
        elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):               
            if self.person_features is None:
                pass          
            elif not self.use_svi_people:
                dKdls, invK_y, invKs_C, N = self._gradient_dim_people_terms(dimension)
            else:
                dKdls = self.Ky_mm * self.kernel_der(self.y_inducing_coords, self.lsy, dimension) 
                invK_y = self.invKy_mm_block.dot(self.y_u.T)
                invKs_C = self.invKys_mm_S
                N = self.y_ninducing                                             
            
                self._compute_sigma_y()
            
            for f in range(self.Nfactors):
                fidxs = np.arange(N) + (N * f)
                invK_yf = invK_y[:, f]
                invKs_C_f = invKs_C[fidxs, :][:, fidxs]                     
                sy = self.shape_sy[f] / self.rate_sy[f]
                Sigma_y_f = self.Sigma_y[:, :, f]             
                der_logpy_logqy += 0.5 * (invK_yf.T.dot(dKdls).dot(invK_yf) * sy - 
                                    np.trace(invKs_C_f.dot(Sigma_y_f).dot(dKdls / sy)))
                         
        return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf        
        
    def predict_item_feats(self, items_coords, item_features=None):
        if item_features is None:
            coords = items_coords
        else:
            coords = item_features[items_coords]
               
        K1 = self.kernel_func(coords, self.ls, self.inducing_coords)
        w1 = K1.dot(self.invK_mm).dot(self.w_u)
            
        return w1    
    
    def _predict_w_t(self, coords_1):
    
        # kernel between pidxs and t
        K1 = self.kernel_func(coords_1, self.ls, self.inducing_coords)
    
        # use kernel to compute t. 
        t1 = K1.dot(self.invK_mm).dot(self.t_u)

        # kernel between pidxs and w -- use kernel to compute w. Don't need Kw_mm block-diagonal matrix
        w1 = K1.dot(self.invK_mm).dot(self.w_u)   
        
        return t1, w1
    
    def _predict_y(self, person_features, Npeople):
        if self.use_svi_people and person_features is not None:
            Ky = self.kernel_func(person_features, self.lsy, self.y_inducing_coords)                    
            # use kernel to compute y
            invKy_train = self.invKy_mm_block
            y_train = self.y_u.reshape(self.Nfactors, self.y_ninducing).T
            
            # use kernel to compute y
            y = Ky.dot(invKy_train).dot(y_train)
            return y.T
        else:
            return super(PreferenceComponentsSVI, self)._predict_y(person_features, Npeople)
            
    def _expec_f_p(self, p, mu0_output):
        if not self.use_noise_svi:
            f, _ = self.pref_gp[p].predict_f(
                    items_coords=self.coordidxs[p] if self.vb_iter==0 else None,
                    items_features=self.obs_coords if self.vb_iter==0 else None, 
                    mu0_output=mu0_output, 
                    reuse_output_kernel=True)
            self.f[self.coordidxs[p], p] = f.flatten()             
            self.f_u[:, p:p+1] = self.w_u.dot(self.y[:, p:p+1]) + self.t_u
            return
        
        f, _ = self.pref_gp[p].predict_f(items_coords=None, 
                                 items_features=self.inducing_coords if self.vb_iter==0 else None, 
                                 mu0_output=self.w_u.dot(self.y[:, p:p+1]) + self.t_u, reuse_output_kernel=True)
        self.f_u[:, p] = f.flatten()
            
    def _update_wy_plus_t(self, p):
        super(PreferenceComponentsSVI, self)._update_wy_plus_t(p)
        
        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + (f * self.N)
            pidx = f * self.Npeople + p
            self.pref_gp[p].cov_mu0_mm += self.wS[fidxs, :][:, fidxs] * self.y_cov[pidx, pidx] + \
                            self.wS[fidxs, :][:, fidxs] * self.y[f, p:p+1]**2 + \
                            self.w_u[:, f]**2 * self.y_cov[pidx, pidx]
        if self.use_t:
            self.pref_gp[p].cov_mu0_mm += self.tS
        
    def _expec_f(self):
        self._update_sample()
        super(PreferenceComponentsSVI, self)._expec_f()              
        
    def _compute_sigma_w(self):
        self.Sigma_w = np.zeros((self.ninducing, self.ninducing, self.Nfactors))
        psample = np.arange(self.ninducing)
        
        for p in self.pref_gp:
            prec_p = self.invK_mm * self.pref_gp[p].s
            y_p = self.y[:, p:p+1] 
            
            if self.use_svi_people:
                yidxs = p + self.y_ninducing * np.arange(self.Nfactors)
            else:
                yidxs = p + self.Npeople * np.arange(self.Nfactors)            
                           
            _, Sigma_w_p = self._compute_sigma_w_p(self.ninducing, y_p, yidxs, prec_p, psample)
            self.Sigma_w += Sigma_w_p
        
    def _expec_w(self):
        '''
        Compute the expectation over the latent features of the items and the latent personality components
        '''
        # Put a GP prior on w with covariance K/gamma and mean 0
        if self.use_noise_svi:
            N = self.ninducing
            Nobs_counter = 1
            Nobs_counter_i = 1
        else:
            N = self.update_size
            Nobs_counter = 0
            Nobs_counter_i = 0
            
        x = np.zeros((N, self.Nfactors))
        Sigma = np.zeros((N * self.Nfactors, N * self.Nfactors))

        for p in self.pref_gp:
            if self.use_svi_people and p not in self.pdata_idx_i:
                continue
                        
            pidxs = self.coordidxs[p]
            y_p = self.y[:, p:p+1]
            if self.use_noise_svi:
                psample = np.arange(N)
                prec_p = self.invK_mm * self.pref_gp[p].s
                invQ_f = prec_p.dot(self.f_u[:, p:p+1] - self.t_u)
                           
                x += y_p.T * invQ_f                
            else:
                Nobs_counter += len(pidxs)
                psample = np.in1d(self.data_idx_i, pidxs)
                pidxs = self.data_idx_i[psample]
                if not len(pidxs):
                    continue 
                Nobs_counter_i += len(pidxs)            
                
                prec_p = self.invKf[p] * self.pref_gp[p].s
                invQ_f = prec_p.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :])     
                x_p = y_p.T * invQ_f
                x[psample, :] += x_p

            if self.use_svi_people:
                yidxs = p + self.y_ninducing * np.arange(self.Nfactors)
            else:
                yidxs = p + self.Npeople * np.arange(self.Nfactors)
                            
            Sigma_p, _ = self._compute_sigma_w_p(N, y_p, yidxs, prec_p, psample)
            Sigma += Sigma_p
            
        x = x.T.flatten()[:, np.newaxis]
        
        if self.use_noise_svi:
            Lambda_factor_w_i = self.Lambda_factor_w
        else:
            widxs_i = (np.tile(self.data_idx_i[:,None], (1,self.Nfactors)) + np.arange(self.Nfactors)[None,:]).flatten()
            Lambda_factor_w_i = self.invKw_mm.dot(self.Kw_nm[widxs_i, :].T)
        
        self.w, _, self.w_invS, self.w_invSm, self.w_u, self.invKws_mm_S, self.wS = svi_update_gaussian(x, 0, 0,
            self.Kws_mm, self.inv_Kws_mm, self.Kws_nm, Lambda_factor_w_i, None, Sigma, self.w_invS, 
            self.w_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)                
    
        self.w = np.reshape(self.w, (self.Nfactors, self.N)).T # w is N x Nfactors    
        self.w_u = np.reshape(self.w_u, (self.Nfactors, self.ninducing)).T # w is N x Nfactors    
        
        for f in range(self.Nfactors):
            fidxs = np.arange(self.ninducing) + (self.ninducing * f)
            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, 
                    self.ninducing, self.invK_mm, self.w_u[:, f:f+1], np.zeros((self.ninducing, 1)), 
                    self.invKws_mm_S[fidxs, :][:, fidxs] / self.shape_sw[f] * self.rate_sw[f])
            fidxs = np.arange(self.N) + (self.N * f)
        
        self._expec_y()
        self.wy = self.w.dot(self.y)    

    def _compute_sigma_y(self):
        self.Sigma_y = np.zeros((self.y_ninducing, self.y_ninducing, self.Nfactors))
        pidxs = np.arange(self.ninducing)
        for p in self.pref_gp:
            prec_f = self.invK_mm * self.pref_gp[p].s                
            self.Sigma_y[np.arange(self.y_ninducing), np.arange(self.y_ninducing), :] += self._compute_sigma_y_p(
                    self.ninducing, self.w_u, self.Kws_mm.dot(self.invKws_mm_S), prec_f, pidxs)[
                        np.arange(self.Nfactors), np.arange(self.Nfactors)]

    def _expec_y(self):
        '''
        Compute expectation over the personality components using VB
        '''
        Npeople = self.Npeople  
        Sigma = np.zeros((self.Nfactors * Npeople, self.Nfactors * Npeople))
        x = np.zeros((Npeople, self.Nfactors))

        Nobs_counter = 0
        Nobs_counter_i = 0

        pidx = 0
        
        w_cov = self.Kws_mm.dot(self.invKws_mm_S)
        w = self.w_u
        N = self.ninducing
        
        for p in self.pref_gp:
            pidxs = self.coordidxs[p]                       
            Nobs_counter += len(pidxs)
            if self.use_svi_people and p not in self.pdata_idx_i:
                continue            
            Nobs_counter_i += len(pidxs)
            pidxs = np.arange(N)
            
            if self.use_noise_svi:
                prec_f = self.invK_mm * self.pref_gp[p].s                
                invQ_f = prec_f.dot(self.f_u[:, p:p+1] - self.t_u)
            else:
                prec_f = self.invKf[p] * self.pref_gp[p].s
                invQ_f = prec_f.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :]) 
                
            Sigma_p = self._compute_sigma_y_p(N, w, w_cov, prec_f, pidxs)
            sigmaidxs = np.arange(self.Nfactors) * Npeople + pidx
            Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
            Sigmarows[:, sigmaidxs] = Sigma_p
            Sigma[sigmaidxs, :] += Sigmarows
            
            x[pidx, :] = w.T.dot(invQ_f).T
            pidx += 1
                
        x = x.T.flatten()[:, np.newaxis]
        
        if not self.use_svi_people:
            self.Sigma_y = Sigma 
            
            # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0 
            # and p is person index
            self.y_cov = np.linalg.inv(self.invKy * self.sy_matrix + Sigma)
            self.y = self.y_cov.dot(x)
           
            # y is Nfactors x Npeople            
            self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
                
            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)
                self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, 
                                                    self.Npeople, self.invKy, self.y[f:f+1, :].T,
                                                    np.zeros((self.Npeople, 1)), f_cov=self.y_cov[fidxs, :][:, fidxs])
                
                self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f] # sy_rows
        else: # SVI implementation
            self.y, _, self.y_invS, self.y_invSm, self.y_u, self.invKys_mm_S, self.yS = svi_update_gaussian(x, 0, 0, 
                self.Kys_mm, self.inv_Kys_mm, self.Kys_nm, self.Lambda_factor_y, None, Sigma, self.y_invS, 
                self.y_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)
        
            # y is Nfactors x Npeople            
            self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
            self.y_u = np.reshape(self.y_u, (self.Nfactors, self.y_ninducing))
            self.y_cov = self.Ky_nm.dot(self.Kys_mm.dot(self.invKys_mm_S)).dot(self.Ky_nm.T)
            
            for f in range(self.Nfactors):
                fidxs = np.arange(self.y_ninducing) + (self.y_ninducing * f)
                self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, 
                    self.y_ninducing, self.invKy_mm_block, self.y_u[f:f+1, :].T, np.zeros((self.y_ninducing, 1)), 
                    self.invKys_mm_S[fidxs, :][:, fidxs] / self.shape_sy[f] * self.rate_sy[f])    
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)

    def _compute_sigma_t(self):
        self.Sigma_t = np.zeros((self.ninducing, self.ninducing))
        
        for p in self.pref_gp:
            prec_f = self.invK_mm * self.pref_gp[p].s
            self.Sigma_t += prec_f            

    def _expec_t(self):
        if not self.use_t:
            return

        if self.use_noise_svi:
            N = self.ninducing
            Nobs_counter = 1
            Nobs_counter_i = 1
        else:
            N = self.update_size
            Nobs_counter = 0
            Nobs_counter_i = 0
    
        Sigma = np.zeros((N, N))
        x = np.zeros((N, 1))
        
        #size_added = 0
        for p in self.pref_gp:
            pidxs = self.coordidxs[p]
            
            if self.use_noise_svi:
                prec_f = self.invK_mm * self.pref_gp[p].s
                invQ_f = prec_f.dot(self.f_u[:, p:p+1] - self.w_u.dot(self.y[:, p:p+1]))                
                x += invQ_f
                Sigma += prec_f
            else:
                Nobs_counter += len(pidxs)
                psample = np.in1d(self.data_idx_i, pidxs)
                pidxs = self.data_idx_i[psample]
                if not len(pidxs):
                    continue 
                Nobs_counter_i += len(pidxs)            
                
                prec_f = self.pref_gp[p].s
                invQ_f = (self.f[pidxs, p:p+1] - self.t[pidxs, :]) * prec_f
                x[psample, :] += invQ_f
            
                sigmarows = np.zeros((np.sum(psample), N))
                sigmarows[:, psample] = prec_f
                Sigma[psample, :] += sigmarows

        if self.use_noise_svi:
            Lambda_factor_t = self.Lambda_factor_t
        else:
            Lambda_factor_t = self.invK_mm.dot(self.K_nm[self.data_idx_i, :].T)
                
        self.t, _, self.t_invS, self.t_invSm, self.t_u, self.invKts_mm_S, self.tS = svi_update_gaussian(x, 
            self.t_mu0, self.t_mu0_u, self.Kts_mm, self.inv_Kts_mm, self.Kts_nm, Lambda_factor_t, 
            None, Sigma, self.t_invS, self.t_invSm, self.vb_iter, self.delay, 
            self.forgetting_rate, Nobs_counter, Nobs_counter_i)

        self.t_cov_u = self.Kts_mm.dot(self.invKts_mm_S)

        self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.ninducing, 
            self.invK_mm, self.t_u, np.zeros((self.ninducing, 1)), self.invKts_mm_S / self.shape_st * self.rate_st)    

    def _update_sample(self):
        self._update_sample_idxs()
        
        sw_mm = np.zeros((self.Nfactors * self.ninducing, self.Nfactors * self.ninducing), dtype=float)
        sw_nm = np.zeros((self.Nfactors * self.N, self.Nfactors * self.ninducing), dtype=float)
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
            sy_mm = np.zeros((self.Nfactors * self.y_ninducing, self.Nfactors * self.y_ninducing), dtype=float)
            sy_nm = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.y_ninducing), dtype=float)
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
            
        logpw = mvn.logpdf(self.w_u.T.flatten(), cov=self.Kws_mm) # this line is slow
        logqw = mvn.logpdf(self.w_u.T.flatten(), mean=self.w_u.T.flatten(), cov=self.Kws_mm.dot(self.invKws_mm_S), 
                           allow_singular=True) # this line is slow

        if self.use_t:
            logpt = mvn.logpdf(self.t_u.flatten(), cov=self.Kts_mm)
            logqt = mvn.logpdf(self.t_u.flatten(), mean=self.t_u.flatten(), cov=self.Kts_mm.dot(self.invKts_mm_S))
        else:
            logpt = 0
            logqt = 0        

        if self.use_svi_people:
            logpy = mvn.logpdf(self.y_u.flatten(), cov=self.Kys_mm)
            logqy = mvn.logpdf(self.y_u.flatten(), mean=self.y_u.flatten(), cov=self.Kys_mm.dot(self.invKys_mm_S), 
                               allow_singular=True)
        else:
            if self.person_features is not None: 
                logpy = mvn.logpdf(self.y.flatten(), cov=self.Ky / self.sy_matrix)
            else:
                logpy = 0
                for f in range(self.Nfactors):
                    logpy += np.sum(norm.logpdf(self.y[f, :], scale=np.sqrt(self.rate_sy[f] / self.shape_sy[f])))
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

        lb = f_terms + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw/self.rate_sw))        
            logging.debug('s_y=%s' % (self.shape_sy/self.rate_sy))
            logging.debug('s_t=%.2f' % (self.shape_st/self.rate_st))        
            logging.debug('fterms=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (f_terms, w_terms, y_terms, t_terms))
    
        if self.verbose:
            logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
        
        if self.verbose:
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))
                
        return lb