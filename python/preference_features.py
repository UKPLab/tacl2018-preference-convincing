'''
Created on 2 Jun 2016

@author: simpson
'''

from gppref import GPPref, gen_synthetic_prefs, get_unique_locations
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import multivariate_normal as mvn
import logging
from gpgrid import coord_arr_to_1d
from scipy.linalg import cholesky, solve_triangular

def vb_gp_regression(m, K, shape_s0, rate_s0, Sigma, y, cholK=[], max_iter=20, conv_threshold=1e-5):
    
    shape_s = shape_s0
    rate_s = rate_s0
    
    nIt = 0
    diff = np.inf
    
    if not np.any(cholK):
        cholK = cholesky(K, overwrite_a=False, check_finite=False)
    
    while (diff > conv_threshold) and (nIt < max_iter):
        s = shape_s / rate_s
        
        Ks = K / s + 1e-6
        L = cholesky(Ks + Sigma, lower=True, check_finite=False, overwrite_a=True)
        B = solve_triangular(L, (y - m), lower=True, overwrite_b=True, check_finite=False)
        A = solve_triangular(L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
        f_mean = Ks.dot(A) + m # need to add the prior mean here?
        
        V = solve_triangular(L, Ks.T, lower=True, overwrite_b=True, check_finite=False)
        Cov_f = Ks - V.T.dot(V)
        
        # learn the output scale with VB
        shape_s = shape_s0 + 0.5 * y.shape[0]
        L_expecFF = solve_triangular(cholK, Cov_f + f_mean.dot(f_mean.T) - m.dot(f_mean.T) -f_mean.dot(m.T) + m.dot(m.T), 
                                     trans=True, overwrite_b=True, check_finite=False)
        LT_L_expecFF = solve_triangular(cholK, L_expecFF, overwrite_b=True, check_finite=False)
        rate_s = rate_s0 + 0.5 * np.trace(LT_L_expecFF) 
        
        nIt += 1
        diff = np.abs(shape_s / rate_s - s)
    
    return f_mean, Cov_f, shape_s, rate_s 

def matern_3_2(distances, ls):
    K = np.zeros(distances.shape)
    for d in range(distances.shape[2]):
        K[:, :, d] = np.abs(distances[:, :, d]) * 3**0.5 / ls[d]
        K[:, :, d] = (1 + K[:, :, d]) * np.exp(-K[:, :, d])
    K = np.prod(K, axis=2)
    return K

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, dims, mu0=[], shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls=100, shape_lsy=10, rate_lsy=0.1, lsy=100, verbose=False, nfactors=3, use_fa=False):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        self.dims = dims
        self.mu0 = mu0
        
        self.sigmasq_t = 100
        
        self.shape_sf0 = shape_s0
        self.rate_sf0 = rate_s0
        
        self.shape_sw0 = shape_s0
        self.rate_sw0 = rate_s0
                            
        self.shape_sy0 = shape_s0
        self.rate_sy0 = rate_s0   
        
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
        
        self.conv_threshold = 1e-3
        self.max_iter = 100
        self.min_iter = 10
        
        self.verbose = verbose
        
        self.Nfactors = nfactors
        
        self.use_fa = use_fa # flag to indicate whether to use the simple factor analysis ML update instead of the VB GP
        
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
        
        self.Npeople = np.max(self.people).astype(int) + 1
        self.t_mu = np.zeros((self.N, 1))
        
        self.f = np.zeros((self.Npeople, self.N))
        self.w = np.zeros((self.N, self.Nfactors))
        self.y = np.ones((self.Nfactors, self.Npeople)) # use ones to avoid divide by zero
        self.y_cov = np.ones((self.Npeople*self.Nfactors, self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
        self.wy = np.zeros((self.N, self.Npeople))
        self.t = np.zeros((self.N, 1))
        
        self.invKf = {}
        self.invKsf = {}
        self.coordidxs = {}
        
        for person in self.people:
            self.pref_gp[person] = GPPref(self.dims, self.mu0, self.shape_sf0, self.rate_sf0, None,
                                                self.shape_ls, self.rate_ls, self.ls)
            self.pref_gp[person].select_covariance_function('matern_3_2')
            self.pref_gp[person].max_iter_VB = 20
            self.pref_gp[person].min_iter_VB = 2
            self.pref_gp[person].max_iter_G = 5
            self.pref_gp[person].verbose = self.verbose
                        
        distances = np.zeros((self.N, self.N, len(self.dims)))
        for d in range(len(self.dims)):
            distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
        # kernel used by w and t
        self.K = matern_3_2(distances, self.ls)
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        
        # kernel used by y???    
        if self.person_features == None:
            self.Ky = np.diag(np.ones(self.Npeople))
        else:
            distances = np.zeros((self.Npeople, self.Npeople, len(self.dims)))
            for d in range(len(self.dims)):
                distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T        
            self.Ky = matern_3_2(distances, self.lsy)
        self.cholKy = cholesky(self.Ky, overwrite_a=False, check_finite=False)
                
        self.fa = FactorAnalysis(n_components=self.Nfactors)
            
        niter = 0
        diff = np.inf
        old_x = np.inf
        #lb = 0
        while ((niter < self.min_iter) and (diff > 0)) | ((diff > self.conv_threshold) and (niter < self.max_iter)):
            # run a VB iteration
            # compute preference latent functions for all workers
            self.expec_f(personIDs, items_1_coords, items_2_coords, preferences)
            
            # find the personality components
            self.expec_w(personIDs)
            
            # compute the preference function means
            self.expec_t()
             
            diff = np.max(old_x - self.w)
            logging.debug( "Difference in latent personality features: %f" % diff)
            old_x = self.w
            
            for person in self.people:
                logging.debug( "Variance 1/s: %f." % (1.0/self.pref_gp[person].s))
                
#             for v in self.fa.noise_variance_:
#                 logging.debug("variance from FA estimate: %f" % v)

             
            # Don't use lower bound here, it doesn't really make sense when we use ML for some parameters
            #old_lb = lb
            #lb = self.lowerbound()
            #logging.debug('Lower bound = %.5f, difference = %.5f' % (lb, lb-old_lb))        
            
            niter += 1
            
        logging.debug( "Preference personality model converged in %i iterations." % niter )
        
    def predict(self, personids, items_0_coords, items_1_coords):
        Npairs = len(personids)
         
        results = np.zeros(Npairs)
         
        # convert items_0_coords and items_1_coords to local indices
        obs_1d = coord_arr_to_1d(self.obs_coords)
        items_0_1d = coord_arr_to_1d(items_0_coords)
        items_1_1d = coord_arr_to_1d(items_1_coords)
        
        matches_0 = obs_1d[:, np.newaxis]==items_0_1d[np.newaxis, :]
        items_0_local = np.argmax(matches_0, axis=0)
        items_0_local[np.sum(matches_0, axis=0)==0] = self.mu0
        
        matches_1 = obs_1d[:, np.newaxis]==items_1_1d[np.newaxis, :]
        items_1_local = np.argmax(matches_1, axis=0)
        items_1_local[np.sum(matches_1, axis=0)==0] = self.mu0
         
        upeople = np.unique(personids)
        for p in upeople:            
            pidxs = personids == p
            
            if p not in self.people:
                if self.verbose:
                    logging.warning('Cannot predict for this person %i' % p)
                results[pidxs] = 0.5
                continue
            
            mu0 = self.wy[:, p:p+1] + self.t
            mu0_1 = mu0[items_0_local[pidxs].flatten(), :] # need to translate coords to local first
            mu0_2 = mu0[items_1_local[pidxs].flatten(), :] # need to translate coords to local first
            results[pidxs] = self.pref_gp[p].predict(items_0_coords[pidxs], items_1_coords[pidxs], 
                                                  mu0_output1=mu0_1, mu0_output2=mu0_2, return_var=False).flatten()
            
        return results
        
    def expec_f(self, personids, items_1_coords, items_2_coords, preferences):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''
        for person in self.pref_gp:
            pidxs = personids == person
            items_1_p = items_1_coords[pidxs]
            items_2_p = items_2_coords[pidxs]
            prefs_p = preferences[pidxs]
            
            mu0_output = self.wy[:, person:person+1] + self.t
            
            mu0_1 = mu0_output[self.pref_v[pidxs], :]
            mu0_2 = mu0_output[self.pref_u[pidxs], :]
            
            self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2)
            
            # find the index of the coords in coords_p in self.obs_coords
            if person not in self.coordidxs:
                internal_coords_p = self.pref_gp[person].obs_coords
                matches = np.ones((internal_coords_p.shape[0], self.N), dtype=bool)
                for dim in range(internal_coords_p.shape[1]):
                    matches = matches & np.equal(internal_coords_p[:, dim:dim+1], self.obs_coords[:, dim:dim+1].T)
                self.coordidxs[person] = np.argwhere(matches)[:, 1]
            
                if person not in self.invKf:            
                    self.invKf[person] = np.linalg.inv(self.pref_gp[person].K) 
                self.invKsf[person] = self.invKf[person] / self.pref_gp[person].s 
            
            f, _ = self.pref_gp[person].predict_f(items_coords=self.obs_coords, mu0_output=mu0_output)
            self.f[person, :] = f.flatten()
        
            if self.verbose:    
                logging.debug( "Expec_f for person %i out of %i" % (person, len(self.pref_gp.keys())) )
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
            shape_s0 = self.shape_sw0
            rate_s0 = self.rate_sw0
            
            x = np.zeros((self.N, self.Nfactors))
            self.w = np.zeros((self.N, self.Nfactors))
            
            Sigma = np.zeros((self.Nlabels * self.Nfactors, self.Nlabels * self.Nfactors))
            K = np.zeros((self.Nlabels * self.Nfactors, self.Nlabels * self.Nfactors))
            
            for f in range(self.Nfactors):    
                Krows = np.zeros((self.N, self.Nfactors*self.N))
                Krows[:, f*self.N:(f+1)*self.N] = self.K
                K[f*self.N:(f+1)*self.N, :] = Krows 
                    
            cholK = cholesky(K, overwrite_a=False, check_finite=False)            
            
            #for n in range(self.N):
                
            for person in self.pref_gp:
                pidxs = self.coordidxs[person]                
                prec_p = self.invKsf[person]#[npidxs, npidxs]
                
                # add the means for this person's observations to the list of observations, x 
                x[pidxs, :] += self.y[:, person:person+1].T * prec_p.dot(self.f[person:person+1, pidxs].T - self.t[pidxs])
                
                # add the covariance for this person's observations as a block in the covariance matrix Sigma
                Sigma_p = np.zeros((self.N * self.Nfactors, self.N * self.Nfactors))
                Sigma_yscaling = (self.y.dot(self.y.T) + 
                    self.y_cov[person+self.Npeople*np.arange(self.Nfactors), :]
                                [:, person+self.Npeople*np.arange(self.Nfactors)])
                for f in range(self.Nfactors):
                    for g in range(self.Nfactors):
                        Sigma_p_rows = np.zeros((len(pidxs), self.N * self.Nfactors))
                        Sigma_p_rows[:, pidxs + g * self.N] = prec_p * Sigma_yscaling[f, g]
                        Sigma_p[pidxs + f*self.N, :] += Sigma_p_rows
                            
                Sigma += Sigma_p
                    
            x = x.T.flatten()[:, np.newaxis]
                    
            # w_cov is same shape as K with rows corresponding to (f*N) + n where f is factor index from 0 and 
            # n is data point index
            #self.w, self.w_cov, _, _ = vb_gp_regression(np.zeros((K.shape[0], 1)), K, shape_s0, rate_s0, Sigma, x, cholK)
            
            self.w_cov = np.linalg.inv(np.linalg.inv(K) + Sigma)
            self.w = self.w_cov.dot(x)
            
            # w is N x Nfactors
            self.w = np.reshape(self.w, (self.N, self.Nfactors))
            
            self.expec_y(personids)
            self.wy = self.w.dot(self.y)    

    def expec_y(self, personids):
        '''
        Compute expectation over the personality components using VB
        '''
        shape_s0 = self.shape_sy0
        rate_s0 = self.rate_sy0
        
        K = np.zeros((self.Npeople * self.Nfactors, self.Nfactors * self.Npeople))
        Sigma = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.Npeople))
        
        x = np.zeros((self.Npeople, self.Nfactors))
        
        for f in range(self.Nfactors):    
            Krows = np.zeros((self.Npeople, self.Nfactors*self.Npeople))
            Krows[:, f*self.Npeople:(f+1)*self.Npeople] = self.Ky
            K[f*self.Npeople:(f+1)*self.Npeople, :] = Krows 

        for person in self.pref_gp:
            pidxs = self.coordidxs[person]           
            
            # the means for this person's observations 
            prec_f_p = self.invKsf[person]
            
            # np.zeros((self.Nfactors * len(pidxs), self.Nfactors * len(pidxs))) do we need to factorise w_cov into two NF x N factors?
            # the data points are not independent given y. The factors are independent?
            covterm = np.zeros((self.Nfactors, self.Nfactors))
            for f in range(self.Nfactors): 
                w_cov_idxs = pidxs + (f * self.N)
                covterm[f, f] = np.sum(prec_f_p * self.w_cov[w_cov_idxs, :][:, w_cov_idxs])
            Sigma_p = self.w[pidxs, :].T.dot(prec_f_p).dot(self.w[pidxs, :]) + covterm
                
            sigmaidxs = np.arange(self.Nfactors) * self.Npeople + person
            Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
            Sigmarows[:, sigmaidxs] =  Sigma_p
            Sigma[sigmaidxs, :] += Sigmarows             
              
            x[person, :] = self.w[pidxs, :].T.dot(prec_f_p).dot(self.f[person, pidxs][:, np.newaxis] 
                                                                - self.t[pidxs, :]).T
                
        cholK = cholesky(K, overwrite_a=False, check_finite=False)
                
        x = x.T.flatten()[:, np.newaxis]
                
        # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0 
        # and p is person index
        #self.y, self.y_cov = vb_gp_regression(0, K, shape_s0, rate_s0, Sigma, x, cholK)
        self.y_cov = np.linalg.inv(np.linalg.inv(K) + Sigma)
        self.y = self.y_cov.dot(x)
        # y is Nfactors x Npeople     
        self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
        
    def expec_t(self):
        if self.use_fa:
            self.t = self.fa.mean_[:, np.newaxis]
        else:
            m = self.t_mu0
            K = self.K
            cholK = self.cholK
            
            shape_s0 = self.shape_st0
            rate_s0 = self.rate_st0
            
            x = np.zeros((0, 1))
            
            obs_size = 0            
            for person in self.pref_gp:
                obs_size += self.pref_gp[person].n_locs
            
#             Sigma = np.zeros((obs_size, obs_size))
#             K = np.zeros((self.N, obs_size))
            
            self.t_cov = np.linalg.inv(self.K.copy())
            self.t = self.t_cov.dot(np.zeros((self.N, 1)) + self.t_mu0)
            
            #size_added = 0
            for person in self.pref_gp:
                pidxs = self.coordidxs[person]
                
                sigmarows = np.zeros((len(pidxs), self.N))
                #ycovidxs = np.arange(self.Nfactors) * self.Npeople + person
                # wcovidxs = np.arange(self.Nfactors) * self.N + 
                Sigma_p = np.linalg.inv(self.pref_gp[person].Ks)
                # + np.linalg.inv(self.w[pidxs, :].dot(self.y_cov[
                #    ycovidxs, :][:, ycovidxs]).dot(self.w[pidxs, :].T))# variance of wy?
                sigmarows[:, pidxs] = Sigma_p
                self.t_cov[pidxs, :] += sigmarows 
                
                # add the means for this person's observations to the list of observations, x 
                f_obs = self.pref_gp[person].obs_f - self.w[pidxs, :].dot(self.y[:, person:person+1])
                
                self.t[pidxs, :] += Sigma_p.dot(f_obs)
                
                
                #x = np.concatenate((x, f_obs), axis=0)
                
#                 # add the covariance for this person's observations as a block in the covariance matrix Sigma
#                 size_p = self.pref_gp[person].n_locs
#                 Sigma_rows = np.zeros((size_p, obs_size))
#                 Sigma_rows[:, size_added:size_added+size_p] = self.pref_gp[person].Ks
#                 Sigma[size_added:size_added+size_p, :] = Sigma_rows 
#                 
#                 K[pidxs, size_added:size_added+size_p] = self.pref_gp[person].Ks
#                 
#                 size_added += size_p
                
            self.t_cov = np.linalg.inv(self.t_cov)
            self.t = self.t_cov.dot(self.t)
            #self.t, self.t_cov = vb_gp_regression(m, K, shape_s0, rate_s0, Sigma, x, cholK=cholK)
            
        logging.debug('Updated q(w). Biggest noise value = %f' % np.max(np.abs(self.t.T - self.f)))
        
    def lowerbound(self):
        f_terms = 0
        t_terms = 0
        
        for person in self.pref_gp:
            f_terms += self.pref_gp[person].lowerbound()
            logging.debug('s=%.2f' % self.pref_gp[person].s)
            
        for n in range(self.N):
            t_terms_p = mvn.logpdf(self.t[:, n], mean=np.zeros(self.t.shape[0]), cov=np.eye(self.t.shape[0]))
            t_terms_q = mvn.logpdf(self.t[:, n], mean=self.t[:, n], cov=self.fa.components_.T * self.fa.components_)
            t_terms += t_terms_p - t_terms_q
                        
        t_terms = np.sum(t_terms)
        
        lb = f_terms + t_terms
        
        logging.debug( "Lower bound = %.3f, fterms=%.3f, tterms=%.3f" % (lb, f_terms, t_terms) )
        
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
        np.random.seed(1)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    Npeople = 3
    pair1idxs = []
    pair2idxs = []
    prefs = []
    personids = []
    xvals = []
    yvals = []
    for p in range(Npeople):
        _, nx, ny, prefs_p, xvals_p, yvals_p, pair1idxs_p, pair2idxs_p, f, K = gen_synthetic_prefs()
        pair1idxs = np.concatenate((pair1idxs, pair1idxs_p + len(xvals))).astype(int)
        pair2idxs = np.concatenate((pair2idxs, pair2idxs_p + len(yvals))).astype(int)
        prefs = np.concatenate((prefs, prefs_p)).astype(int)
        personids = np.concatenate((personids, np.zeros(len(pair1idxs_p)) + p)).astype(int)
        xvals = np.concatenate((xvals, xvals_p.flatten()))
        yvals = np.concatenate((yvals, yvals_p.flatten()))

    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 

    Ptest = 6
    testpairs = np.random.choice(pair1coords.shape[0], Ptest, replace=False)
    testidxs = np.zeros(pair1coords.shape[0], dtype=bool)
    testidxs[testpairs] = True
    trainidxs = np.invert(testidxs)
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    model = PreferenceComponents([nx,ny], mu0=0, shape_s0=1.0, rate_s0=1.0, ls=[10,10], nfactors=7, use_fa=False)
    model.fit(personids[trainidxs], pair1coords[trainidxs], pair2coords[trainidxs], prefs[trainidxs])
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1coords[testidxs], pair2coords[testidxs])
    
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
    