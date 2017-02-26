'''
TODO: find out whether we correctly model correlations between people.

TODO: length-scale learning through MLII

TODO: some of the matrix inversions should use Cholesky. E.g. for Kw and Ky

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

Created on 2 Jun 2016

@author: simpson
'''

from gppref import GPPref, gen_synthetic_prefs, get_unique_locations, matern_3_2, matern_3_2_from_raw_vals
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import multivariate_normal as mvn
import logging
from gpgrid import coord_arr_to_1d
from scipy.linalg import cholesky, solve_triangular, block_diag
from scipy.special import gammaln, psi

def expec_output_scale(shape_s0, rate_s0, N, cholK, f_mean, m, f_cov):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N
    L_expecFF = solve_triangular(cholK, f_cov + f_mean.dot(f_mean.T) - m.dot(f_mean.T) -f_mean.dot(m.T) + m.dot(m.T), 
                                 trans=True, overwrite_b=True, check_finite=False)
    LT_L_expecFF = solve_triangular(cholK, L_expecFF, overwrite_b=True, check_finite=False)
    rate_s = rate_s0 + 0.5 * np.trace(LT_L_expecFF) 
    
    return shape_s/rate_s, shape_s, rate_s

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

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, item_dims, person_dims=0, mu0=0, mu0_y=0, shape_s0=1, rate_s0=1, shape_ls=1, rate_ls=100, 
                 ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=3, use_fa=False):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        self.dims = item_dims
        self.person_dims = person_dims
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
        self.rate_sy0 = rate_s0   
    
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
        
        self.conv_threshold = 1e-1
        self.max_iter = 100
        self.min_iter = 3
        
        self.verbose = verbose
        
        self.Nfactors = nfactors
        
        self.use_fa = use_fa # flag to indicate whether to use the simple factor analysis ML update instead of the VB GP
        
    def init_params(self):
        self.Npeople = np.max(self.people).astype(int) + 1
        
        self.f = np.zeros((self.Npeople, self.N))
        #self.w = np.ones((self.N, self.Nfactors)) #np.zeros((self.N, self.Nfactors))
        self.w_cov = np.diag(np.ones(self.N*self.Nfactors)) # use ones to avoid divide by zero
        #self.y = np.ones((self.Nfactors, self.Npeople)) #np.random.rand(self.Nfactors, self.Npeople) # use ones to avoid divide by zero
        self.y_cov = np.diag(np.ones(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
        
        self.t = np.zeros((self.N, 1))
        self.t_cov = np.diag(np.ones(self.N))
        
        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0
        self.shape_sy = np.zeros(self.Nfactors) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors) + self.rate_sy0
        self.shape_st = self.shape_st0
        self.rate_st = self.rate_st0                
        
        self.invKf = {}
        self.coordidxs = {}
        
        for person in self.people:
            self.pref_gp[person] = GPPref(self.dims, self.mu0, self.shape_sf0, self.rate_sf0, None,
                                                self.shape_ls, self.rate_ls, self.ls)
            self.pref_gp[person].select_covariance_function('matern_3_2')
            self.pref_gp[person].max_iter_VB = 1
            self.pref_gp[person].min_iter_VB = 1
            self.pref_gp[person].max_iter_G = 5
            self.pref_gp[person].verbose = self.verbose
            self.pref_gp[person].update_s = False # don't update s in the first round. Wait until we have computed f 
            #against a reasonable mean estimate t
             
        self.new_obs = True # cause the pref GPs to process the new observations in the first iteration
                        
        distances = np.zeros((self.N, self.N, len(self.dims)))
        for d in range(len(self.dims)):
            distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
        # kernel used by t
        self.K = matern_3_2(distances, self.ls)
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        self.invK = np.linalg.inv(self.K)
        
        # kernel used by w
        blocks = [self.K for _ in range(self.Nfactors)]
        self.Kw = block_diag(*blocks)
        self.sw_matrix = np.ones(self.Kw.shape)
                
        # kernel used by y  
        if self.person_features == None:
            Ky = np.diag(np.ones(self.Npeople))
        else:
            distances = np.zeros((self.Npeople, self.Npeople, len(self.person_dims)))
            for d in range(len(self.person_dims)):
                distances[:, :, d] = self.person_features[:, d:d+1] - self.person_features[:, d:d+1].T        
            Ky = matern_3_2(distances, self.lsy)
           
        blocks = [Ky for _ in range(self.Nfactors)]
        self.Ky = block_diag(*blocks) 
        self.cholKy = cholesky(Ky, overwrite_a=False, check_finite=False)
        self.sy_matrix = np.ones(self.Ky.shape)

        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them,
        # i.e. the cluster identifiability problem
        self.w = mvn.rvs(np.zeros(self.Nfactors * self.N), cov=self.Kw).reshape((self.Nfactors, self.N)).T 
        self.y = mvn.rvs(np.zeros(self.Nfactors * self.Npeople), cov=self.Ky).reshape((self.Nfactors, self.Npeople))
        self.wy = self.w.dot(self.y)
        
        # Factor Analysis
        if self.use_fa:                        
            self.fa = FactorAnalysis(n_components=self.Nfactors)        
        
    def fit(self, personIDs, items_1_coords, items_2_coords, preferences, person_features=None):
        '''
        Learn the model with data as follows:
        personIDs - a list of the person IDs of the people who expressed their preferences
        items_1_coords - coordinates of the first items in the pairs being compared
        items_2_coords - coordinates of the second items in each pair being compared
        preferences - the values, 0 or 1 to express that item 1 was preferred to item 2.
        '''
        
        # deal only with the original IDs to simplify prediction steps and avoid conversions 
        self.people = np.unique(personIDs)
        self.pref_gp = {}
        
        self.obs_coords, self.pref_v, self.pref_u = get_unique_locations(items_1_coords, items_2_coords)
        
        self.person_features = person_features # rows per person, columns for feature values         
        
        self.N = len(self.obs_coords)
        
        self.Nlabels = 0
        for person in self.people:
            pidxs = personIDs==person
            self.Nlabels += np.unique([self.pref_v[pidxs], self.pref_u[pidxs]]).shape[0]
        
        self.init_params()
            
        niter = 0
        diff = np.inf
        old_x = np.inf
        old_lb = -np.inf
        while (niter < self.min_iter) | ((diff > self.conv_threshold) and (niter < self.max_iter)):
            # run a VB iteration
            # compute preference latent functions for all workers
            self.expec_f(personIDs, items_1_coords, items_2_coords, preferences)
            
            # compute the preference function means
            self.expec_t()            
            
            # find the personality components
            self.expec_w(personIDs)
             
            diff = np.max(old_x - self.w)
            logging.debug( "Difference in latent personality features: %f" % diff)
            old_x = self.w

            # Don't use lower bound here, it doesn't really make sense when we use ML for some parameters
            if not self.use_fa:
                lb = self.lowerbound()
                logging.debug('Lower bound = %.5f, difference = %.5f' % (lb, lb-old_lb ))
                diff = lb - old_lb
                old_lb = lb

            niter += 1
            
        logging.debug( "Preference personality model converged in %i iterations." % niter )
        
    def predict(self, personids, items_1_coords, items_2_coords, person_features=None):
        Npairs = len(personids)
        results = np.zeros(Npairs)
         
        upeople = np.unique(personids)
        for p in upeople:            
            pidxs = personids == p
            Npairs_p = np.sum(pidxs)
            
            if p in self.people:
                y = self.y[:, p:p+1] 
            else:
                if self.person_features == None:
                    Ky = np.zeros((1, self.Npeople))
                else:
                    #distances for y-space
                    distances = np.zeros((1, self.Npeople, len(self.person_dims)))
                    for d in range(len(self.person_dims)):
                        distances[:, :, d] = person_features[p, d:d+1] - self.person_features[:, d:d+1].T        
                    # kernel between p and people already seen
                    Ky = matern_3_2(distances, self.lsy)
                # use kernel to compute y    
                Ky = Ky * self.rate_sy[np.newaxis, :] / self.shape_sy[np.newaxis, :]               
                y = Ky.dot(self.y.T).T
            
            coords_1 = items_1_coords[pidxs]
            coords_2 = items_2_coords[pidxs]
            
            # this could be made more efficient because duplicate locations are computed separately!
            # distances for t-space
            distances1 = np.zeros((Npairs_p, self.N, len(self.dims)))
            distances2 = np.zeros((Npairs_p, self.N, len(self.dims)))
            for d in range(len(self.dims)):
                distances1[:, :, d] = coords_1[:, d:d+1] - self.obs_coords[:, d:d+1].T
                distances2[:, :, d] = coords_2[:, d:d+1] - self.obs_coords[:, d:d+1].T
                
            # kernel between pidxs and t
            K1 = matern_3_2(distances1, self.ls)
            K2 = matern_3_2(distances1, self.ls)            
            
            K1t = K1 * self.rate_st / self.shape_st
            K2t = K2 * self.rate_st / self.shape_st
            
            # use kernel to compute t
            t1 = K1t.dot(self.t)
            t2 = K2t.dot(self.t)

            # kernel between pidxs and w -- use kernel to compute w
            w1 = K1.dot(self.w) * self.rate_sw[np.newaxis, :] / self.shape_sw[np.newaxis, :]   
            w2 = K2.dot(self.w) * self.rate_sw[np.newaxis, :] / self.shape_sw[np.newaxis, :]   
                        
            wy_1p = w1.dot(y)
            wy_2p = w2.dot(y)
            mu0_1 = wy_1p + t1
            mu0_2 = wy_2p + t2
            
            if p in self.people:
                pref_gp_p = self.pref_gp[p]
            else:
                # create a new pref GP for a new person
                pref_gp_p = GPPref(self.dims, self.mu0, self.shape_sf0, self.rate_sf0, None, self.shape_ls, 
                                   self.rate_ls, self.ls)
                pref_gp_p.select_covariance_function('matern_3_2')
                pref_gp_p.max_iter_VB = 1
                pref_gp_p.min_iter_VB = 1
                pref_gp_p.max_iter_G = 5
                pref_gp_p.verbose = self.verbose
                pref_gp_p.update_s = False # don't update s in the first round. Wait until we have computed f             
            
            results[pidxs] = pref_gp_p.predict(coords_1, coords_2, 
                                                  mu0_output1=mu0_1, mu0_output2=mu0_2, return_var=False).flatten()
            
        return results
        
    def expec_f(self, personids, items_1_coords, items_2_coords, preferences):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''
        for person in self.pref_gp:
            plabelidxs = personids == person
            items_1_p = items_1_coords[plabelidxs]
            items_2_p = items_2_coords[plabelidxs]
            prefs_p = preferences[plabelidxs]
            
            if self.new_obs:
                # take the initial prior so that we can calculate Q correctly -- use the prior mean as an approximation
                # to integrating over the prior. We don't want to use a posterior or the random initialisation of wy.
                mu0_output = np.zeros((self.N, 1)) + self.t_mu0
            else:
                mu0_output = self.wy[:, person:person+1] + self.t
            
            mu0_1 = mu0_output[self.pref_v[plabelidxs], :]
            mu0_2 = mu0_output[self.pref_u[plabelidxs], :]
            
            self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs)
            # find the index of the coords in coords_p in self.obs_coords
            if person not in self.coordidxs:
                internal_coords_p = self.pref_gp[person].obs_coords
                matches = np.ones((internal_coords_p.shape[0], self.N), dtype=bool)
                for dim in range(internal_coords_p.shape[1]):
                    matches = matches & np.equal(internal_coords_p[:, dim:dim+1], self.obs_coords[:, dim:dim+1].T)
                self.coordidxs[person] = np.argwhere(matches)[:, 1]
            
                self.invKf[person] = np.linalg.inv(self.pref_gp[person].K) 
            
            f, _ = self.pref_gp[person].predict_f(items_coords=self.obs_coords, mu0_output=mu0_output)
            self.f[person, :] = f.flatten()
        
            if self.verbose:    
                logging.debug( "Expec_f for person %i out of %i" % (person, len(self.pref_gp.keys())) )
                
        self.new_obs = False # don't process the observations again unless fit() is called

        if self.verbose:
            logging.debug('Updated q(f)')
             
    def expec_w(self, personids):
        '''
        Compute the expectation over the latent features of the items and the latent personality components
        '''
        if self.use_fa:
            self.y = self.fa.fit_transform(self.f).T
            self.w = self.fa.components_.T
            self.wy = self.w.dot(self.y)
        else:
            # Put a GP prior on w with covariance K/gamma and mean 0
            x = np.zeros((self.N, self.Nfactors))
            Sigma = np.zeros((self.N * self.Nfactors, self.N * self.Nfactors))

            for person in self.pref_gp:
                pidxs = self.coordidxs[person]                
                prec_p = self.invKf[person] * self.pref_gp[person].s #[npidxs, npidxs]
                y_p = self.y[:, person:person+1]
                
                # add the means for this person's observations to the list of observations, x 
                x[pidxs, :] += y_p.T * prec_p.dot(self.f[person:person+1, pidxs].T - self.t[pidxs])
                
                # add the covariance for this person's observations as a block in the covariance matrix Sigma
                Sigma_p = np.zeros((self.N * self.Nfactors, self.N * self.Nfactors))
                yidxs = person + self.Npeople * np.arange(self.Nfactors)
                Sigma_yscaling = y_p.dot(y_p.T) + self.y_cov[yidxs, :][:, yidxs] # covariance between people?
                for f in range(self.Nfactors):
                    for g in range(self.Nfactors):
                        Sigma_p_rows = np.zeros((len(pidxs), self.N * self.Nfactors))
                        Sigma_p_rows[:, pidxs + g * self.N] = prec_p * Sigma_yscaling[f, g]
                        Sigma_p[pidxs + f * self.N, :] += Sigma_p_rows
                            
                Sigma += Sigma_p
                    
            x = x.T.flatten()[:, np.newaxis]
                    
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
            
            self.expec_y(personids)
            self.wy = self.w.dot(self.y)                

    def expec_y(self, personids):
        '''
        Compute expectation over the personality components using VB
        '''
        Sigma = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.Npeople))
        x = np.zeros((self.Npeople, self.Nfactors))

        for person in self.pref_gp:
            pidxs = self.coordidxs[person]           
            
            # the means for this person's observations 
            prec_f_p = self.invKf[person] * self.pref_gp[person].s
            
            # np.zeros((self.Nfactors * len(pidxs), self.Nfactors * len(pidxs))) do we need to factorise w_cov into two NF x N factors?
            # the data points are not independent given y. The factors are independent?
            covterm = np.zeros((self.Nfactors, self.Nfactors))
            for f in range(self.Nfactors): 
                w_cov_idxs = pidxs + (f * self.N)
                w_cov_f = self.w_cov[w_cov_idxs, :]
                for g in range(self.Nfactors):
                    w_cov_idxs = pidxs + (g * self.N)
                    covterm[f, g] = np.sum(prec_f_p * w_cov_f[:, w_cov_idxs])
            Sigma_p = self.w[pidxs, :].T.dot(prec_f_p).dot(self.w[pidxs, :]) + covterm
                
            sigmaidxs = np.arange(self.Nfactors) * self.Npeople + person
            Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
            Sigmarows[:, sigmaidxs] =  Sigma_p
            Sigma[sigmaidxs, :] += Sigmarows             
              
            x[person, :] = self.w[pidxs, :].T.dot(prec_f_p).dot(self.f[person, pidxs][:, np.newaxis] 
                                                                - self.t[pidxs, :]).T
                
        x = x.T.flatten()[:, np.newaxis]
                
        # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0 
        # and p is person index
        #self.y, self.y_cov = vb_gp_regression(0, K, shape_s0, rate_s0, Sigma, x, cholK)
        self.y_cov = np.linalg.inv(np.linalg.inv(self.Ky  / self.sy_matrix) + Sigma)
        self.y = self.y_cov.dot(x)
        self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
            
        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + (self.Npeople * f)
            _, self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople, 
                        self.cholKy, self.y[f:f+1, :].T, np.zeros((self.Npeople, 1)), self.y_cov[fidxs, :][:, fidxs])    
            self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]    
        # y is Nfactors x Npeople     
         
        
    def expec_t(self):
        if self.use_fa:
            self.t = self.fa.mean_[:, np.newaxis]
        else:
            t_prec = self.invK * self.shape_st * self.rate_st
            self.t = t_prec.dot(np.zeros((self.N, 1)) + self.t_mu0)
            
            #size_added = 0
            for person in self.pref_gp:
                pidxs = self.coordidxs[person]
                sigmarows = np.zeros((len(pidxs), self.N))
                
                Sigma_p = np.linalg.inv(self.pref_gp[person].Ks)
                sigmarows[:, pidxs] = Sigma_p
                t_prec[pidxs, :] += sigmarows 
                
                # add the means for this person's observations to the list of observations, x 
                f_obs = self.pref_gp[person].obs_f - self.wy[pidxs, person:person+1]
                self.t[pidxs, :] += Sigma_p.dot(f_obs)
                
            self.t_cov = np.linalg.inv(t_prec)
            self.t = self.t_cov.dot(self.t)

            #self.t, self.t_cov = vb_gp_regression(m, K, shape_s0, rate_s0, Sigma, x, cholK=cholK)
            _, self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.N, self.cholK, 
                                                                self.t, np.zeros((self.N, 1)), self.t_cov)
        #logging.debug('Updated q(t). Biggest noise value = %f' % np.max(np.abs(self.t.T - self.f)))
        
    def lowerbound(self):
        f_terms = 0
        y_terms = 0
        
        for person in self.pref_gp:
            f_terms += self.pref_gp[person].lowerbound()
            if self.verbose:
                logging.debug('s_f^%i=%.2f' % (person, self.pref_gp[person].s))
            
        logpy = mvn.logpdf(self.y.flatten(), mean=np.zeros(self.Ky.shape[0]), cov=self.Ky / self.sy_matrix)
        logqy = mvn.logpdf(self.y.flatten(), mean=self.y.flatten(), cov=self.y_cov)
        logps_y = 0
        logqs_y = 0
        for f in range(self.Nfactors):
            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f])
        y_terms += logpy - logqy + logps_y - logqs_y
        if self.verbose:
            logging.debug('s_y=%s' % (self.shape_sy/self.rate_sy))
            #logging.debug("logpy: %.2f" % logpy)
            #logging.debug("logqy: %.2f" % logqy)
            #logging.debug("logps_y: %.2f" % logps_y)
            #logging.debug("logqs_y: %.2f" % logqs_y)
            
        logpt = mvn.logpdf(self.t.flatten(), mean=np.zeros(self.N) + self.t_mu0, cov=self.K * self.rate_st / self.shape_st)
        logqt = mvn.logpdf(self.t.flatten(), mean=self.t.flatten(), cov=self.t_cov)
        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st) 
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st)
        t_terms = logpt - logqt + logps_t - logqs_t
        if self.verbose:         
            logging.debug('s_t=%.2f' % (self.shape_st/self.rate_st))        
            #logging.debug("t_cov: %s" % self.t_cov)
        
        #logging.debug("wy: %s" % self.wy)
        #logging.debug("E[w]: %s" % self.w)       
        #logging.debug("cov(w): %s" % self.w_cov)       

        logpw = mvn.logpdf(self.w.T.flatten(), mean=np.zeros(self.Kw.shape[0]), cov=self.Kw / self.sw_matrix)
        logqw = mvn.logpdf(self.w.T.flatten(), mean=self.w.T.flatten(), cov=self.w_cov)
        logps_w = 0
        logqs_w = 0
        for f in range(self.Nfactors):   
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f])
        
        w_terms = logpw - logqw + logps_w - logqs_w
        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw/self.rate_sw))        
            #logging.debug("logpw: %.2f" % logpw)       
            #logging.debug("logqw: %.2f" % logqw)
            #logging.debug("logps_w: %.2f" % logps_w)
            #logging.debug("logqs_w: %.2f" % logqs_w)    
                
        lb = f_terms + t_terms + w_terms + y_terms
        if self.verbose:
            logging.debug( "Lower bound = %.3f, fterms=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f" % 
                       (lb, f_terms, w_terms, y_terms, t_terms) )
        
        if self.verbose:
            for person in self.people:                                                              
                logging.debug("f_%i: %.2f, %.2f" % (person, np.min(self.f[person, :]), np.max(self.f[person, :])))
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))
                
        return lb
    
    def pickle_me(self, filename):
        import pickle
        from copy import  deepcopy
        with open (filename, 'w') as fh:
            m2 = deepcopy(self)
            for p in m2.pref_gp:
                m2.pref_gp[p].kernel_func = None # have to do this to be able to pickle
            pickle.dump(m2, fh)        
        
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    

    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(10)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    Npeople = 200
    Ptest = 20
    pair1idxs = []
    pair2idxs = []
    prefs = []
    personids = []
    xvals = []
    yvals = []
    
    nx = 5
    ny = 5
    
    # generate a common prior:
    ls = [10, 5]
    xvals = np.arange(nx)[:, np.newaxis]
    xvals = np.tile(xvals, (1, ny)).flatten()
    yvals = np.arange(ny)[np.newaxis, :]
    yvals = np.tile(yvals, (nx, 1)).flatten()
    Kt = matern_3_2_from_raw_vals(np.array([xvals, yvals]), ls)
    t = mvn.rvs(cov=Kt).reshape(nx, ny)
    
    Nfactors = 2
    
    Ky = matern_3_2_from_raw_vals(np.arange(Npeople)[np.newaxis, :], [2])
    
    w = np.zeros((nx * ny, Nfactors))
    y = np.zeros((Nfactors, Npeople))
    for f in range(Nfactors):
        w[:, f] = mvn.rvs(cov=Kt).flatten()
        y[f, :] = mvn.rvs(cov=Ky)
    
    for p in range(Npeople):
        
        y_p = y[:, p:p+1]
        wy_p = w.dot(y_p).reshape((nx, ny))
        
        f_prior_mean = t + wy_p
        
        _, nx, ny, prefs_p, xvals_p, yvals_p, pair1idxs_p, pair2idxs_p, f, K = gen_synthetic_prefs(f_prior_mean, nx, ny)
        pair1idxs = np.concatenate((pair1idxs, pair1idxs_p + len(xvals))).astype(int)
        pair2idxs = np.concatenate((pair2idxs, pair2idxs_p + len(yvals))).astype(int)
        prefs = np.concatenate((prefs, prefs_p)).astype(int)
        personids = np.concatenate((personids, np.zeros(len(pair1idxs_p)) + p)).astype(int)
        xvals = np.concatenate((xvals, xvals_p.flatten()))
        yvals = np.concatenate((yvals, yvals_p.flatten()))

    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 


    testpairs = np.random.choice(pair1coords.shape[0], Ptest, replace=False)
    testidxs = np.zeros(pair1coords.shape[0], dtype=bool)
    testidxs[testpairs] = True
    trainidxs = np.invert(testidxs)
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    model = PreferenceComponents([nx,ny], ls=ls, nfactors=Nfactors + 5, use_fa=False)
    model.verbose = False
    model.fit(personids[trainidxs], pair1coords[trainidxs], pair2coords[trainidxs], prefs[trainidxs])
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1coords[testidxs], pair2coords[testidxs], )
    
    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    from sklearn.metrics import accuracy_score
    
    print 'Accuracy: %f' % accuracy_score(prefs[testidxs], np.round(results))
    
#     from scipy.stats import kendalltau
#      
#     for p in range(Npeople):
#         logging.debug( "Personality features of %i: %s" % (p, str(model.w[p])) )
#         for q in range(Npeople):
#             logging.debug( "Distance between personalities: %f" % np.sqrt(np.sum(model.w[p] - model.w[q])**2)**0.5 )
#             logging.debug( "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0] )
#              
    
    # visualise the results
    import matplotlib.pyplot as plt
                
    cmap = plt.get_cmap('jet')                
    cmap._init()    
    
    # t
    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = model.t.flatten()
    scale = np.sqrt(model.rate_st/model.shape_st)
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: t (item mean)')

    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = np.sqrt(np.diag(model.t_cov))
    scale = np.std(tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]])
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
    plt.title('STD at training points: t (item mean)')

    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = t[model.obs_coords[:, 0], model.obs_coords[:, 1]].flatten()
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: t (item mean)')    
    
    # y
    plt.figure()
    ymap = model.y.T
    scale = np.sqrt(model.rate_sy[np.newaxis, :]/model.shape_sy[np.newaxis, :])
    ymap /= scale
    plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]], 
               aspect=Nfactors / float(ymap.shape[0]), vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: y (latent features for people)')

    plt.figure()
    ymap = y.T
    plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]], 
               aspect=Nfactors / float(ymap.shape[0]), vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: y (latent features for people')      
       
    # w
    for f in range(model.Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = model.w[:, f]
        scale = np.sqrt(model.rate_sw[f]/model.shape_sw[f])
        wmap /= scale
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('predictions at training points: w_%i (latent feature for items)' %f)

        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = np.sqrt(model.w_cov[np.arange(model.N*f, model.N*(f+1)), 
                                                                                   np.arange(model.N*f, model.N*(f+1))])        
        scale = np.std(wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]])
        wmap /= scale
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]], aspect=None, vmin=-2, 
                   vmax=2, interpolation='none', filterrad=0.01)
        plt.title('STD at training points: w_%i (latent feature for items)' %f)

    for f in range(Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = w[np.ravel_multi_index((model.obs_coords[:, 0], 
                                                                       model.obs_coords[:, 1]), dims=(nx, ny)), f]
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('ground truth at training points: w_%i (latent feature for items' % f)  