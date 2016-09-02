'''
Created on 2 Jun 2016

@author: simpson
'''

from gppref import GPPref, gen_synthetic_prefs, get_unique_locations
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import norm, multivariate_normal as mvn
import logging
from scipy.sparse import coo_matrix
from gpgrid import coord_arr_to_1d

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, dims, mu0=[], shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, verbose=False, nfactors=3):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        self.dims = dims
        self.mu0 = mu0
        
        self.sigmasq_t = 100
        
        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        self.s_initial = s_initial
        
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        self.ls_initial = ls_initial
        
        self.conv_threshold = 1e-5
        self.max_iter = 100
        
        self.cov_type = 'matern_3_2'
        
        self.verbose = verbose
        
        self.nfactors = nfactors
        
    def fit(self, personIDs, items_1_coords, items_2_coords, preferences):
        '''
        Learn the model with data as follows:
        personIDs - a list of the person IDs of the people who expressed their preferences
        items_1_coords - coordinates of the first items in the pairs being compared
        items_2_coords - coordinates of the second items in each pair being compared
        preferences - the values, 0 or 1 to express that item 1 was preferred to item 2.
        '''
        
        # deal only with the original IDs to simplify prediction steps and avoid conversions 
        self.people = np.unique(personIDs)
        self.gppref_models = {}
        
        self.obs_coords, self.pref_u, self.pref_v = get_unique_locations(items_1_coords, items_2_coords)
        
        self.N = len(self.obs_coords)
        self.t_cov = np.diag(self.sigmasq_t * np.ones(self.N)).astype(float)
        self.mu = np.zeros((self.N, 1))
        self.Npeople = np.max(self.people) + 1
        self.f = {}
        self.t = np.zeros((self.Npeople, self.N))
        for p, person in enumerate(self.people):
            self.gppref_models[person] = GPPref(self.dims, self.mu0, self.shape_s0, self.rate_s0, self.s_initial, 
                                                self.shape_ls, self.rate_ls, self.ls_initial)
            self.gppref_models[person].select_covariance_function(self.cov_type)
            self.gppref_models[person].max_iter_VB = 1
            self.gppref_models[person].min_iter_VB = 1
            self.gppref_models[person].max_iter_G = 1            
            
            if p==0: # initialise the output prior covariance, do this only once  
                distances = np.zeros((self.N, self.N, len(self.dims)))
                for d in range(len(self.dims)):
                    distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
                self.Kpred = self.gppref_models[person].kernel_func(distances)
            
        self.fa = FactorAnalysis(n_components=self.nfactors)
            
        niter = 0
        diff = np.inf
        old_x = 0
        lb = 0
        while diff > self.conv_threshold and niter < self.max_iter:
            # run a VB iteration
            # compute preference latent functions for all workers
            self.expec_f(personIDs, items_1_coords, items_2_coords, preferences)
            
            diff = 0
            # compute the preference function means
            self.expec_t()
            # find the personality components from the preference function means
            self.expec_x()
             
            diff = np.max(old_x - self.x)
            logging.debug( "Difference in latent personality features: %f" % diff)
            old_x = self.x
             
            old_lb = lb
            lb = self.lowerbound()
            logging.debug('Lower bound = %.5f, difference = %.5f' % (lb, lb-old_lb))
            
            niter += 1
            
        logging.debug( "Preference personality model converged in %i iterations." % niter )
        
    def predict(self, personids, items_0_coords, items_1_coords, variance_method='rough'):
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
            
            mu0_1 = self.t[p, items_0_local[pidxs]] # need to translate coords to local first
            mu0_2 = self.t[p, items_1_local[pidxs]] # need to translate coords to local first
            results[pidxs], _ = self.gppref_models[p].predict(items_0_coords[pidxs], items_1_coords[pidxs], 
                                                           variance_method, mu0_output1=mu0_1, mu0_output2=mu0_2)
            
        return results
        
    def expec_t(self):
        '''
        Compute the expectation over the preference function mean
        '''
        
        invK = np.linalg.inv(self.Kpred)
        for person in self.gppref_models:
            invKs = invK * self.gppref_models[person].s            
            
            C = np.linalg.inv(self.t_cov + invKs)# covariance of t
            self.t[person, :] = C.dot(self.t_cov.dot(self.mu) + invKs.dot(self.f[person])).T
            
            if self.verbose:
                logging.debug( "Expec_t for person %i out of %i" % (person, len(self.gppref_models.keys())) )
        logging.debug('Updated q(t)')
    
    def expec_f(self, personids, items_1_coords, items_2_coords, preferences):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''
        for person in self.gppref_models:
            pidxs = personids == person
            items_1_p = items_1_coords[pidxs]
            items_2_p = items_2_coords[pidxs]
            prefs_p = preferences[pidxs]
            
            mu0_1 = self.t[person, self.pref_v[pidxs]][:, np.newaxis]
            mu0_2 = self.t[person, self.pref_u[pidxs]][:, np.newaxis]
            mu0_output1 = self.t[person, :][:, np.newaxis]
            
            self.gppref_models[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2)
            self.gppref_models[person].predict(items_0_coords=self.obs_coords, variance_method='sample', 
                                               mu0_output1=mu0_output1)
            self.f[person] = self.gppref_models[person].f 
        
            if self.verbose:    
                logging.debug( "Expec_f for person %i out of %i" % (person, len(self.gppref_models.keys())) )
        logging.debug('Updated q(f)')
             
    def expec_x(self):
        '''
        Compute the expectation over the personality components.
        '''
        
        self.x = self.fa.fit_transform(self.t)
        self.cov_t = self.fa.components_.T.dot(self.fa.components_) + np.diag(self.fa.noise_variance_)
        self.mu = self.fa.mean_[:, np.newaxis]
        
        logging.debug('Updated q(x)')
        
    def lowerbound(self):
        f_terms = 0
        t_terms = 0
        
        invKpred_p = np.linalg.inv(self.Kpred)
        
        for person in self.gppref_models:
            f_terms += self.gppref_models[person].lowerbound()
            
            invKs = invKpred_p * self.gppref_models[person].s
                       
            t_terms_p = mvn.logpdf(self.t[person, :], mean=self.fa.mean_, cov=self.t_cov)
            t_terms_q = mvn.logpdf(self.t[person, :], mean=self.t[person, :], cov=np.linalg.inv(self.t_cov + invKs))
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
            for p in m2.gppref_models:
                m2.gppref_models[p].kernel_func = None # have to do this to be able to pickle
            pickle.dump(m2, fh)        
        
if __name__ == '__main__':
    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    Npeople = 5
    pair1idxs = []
    pair2idxs = []
    prefs = []
    personids = []
    xvals = []
    yvals = []
    for p in range(Npeople):
        N, Ptest, prefs_p, nx, ny, xvals_p, yvals_p, pair1idxs_p, pair2idxs_p, f = gen_synthetic_prefs()
        pair1idxs = np.concatenate((pair1idxs, pair1idxs_p)).astype(int)
        pair2idxs = np.concatenate((pair2idxs, pair2idxs_p)).astype(int)
        prefs = np.concatenate((prefs, prefs_p)).astype(int)
        personids = np.concatenate((personids, np.zeros(Ptest) + p)).astype(int)
        xvals = np.concatenate((xvals, xvals_p.flatten()))
        yvals = np.concatenate((yvals, yvals_p.flatten()))

    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 
    
    model = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10])
    model.fit(personids, pair1coords, pair2coords, prefs)
    
    from scipy.stats import kendalltau
     
    for p in range(Npeople):
        logging.debug( "Personality features of %i: %s" % (p, str(model.x[p])) )
        for q in range(Npeople):
            logging.debug( "Distance between personalities: %f" % np.sqrt(np.sum(model.x[p] - model.x[q])**2)**0.5 )
            logging.debug( "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0] )
             
    