'''
Created on 2 Jun 2016

@author: simpson
'''

from gppref import GPPref, gen_synthetic_prefs, get_unique_locations
import numpy as np
from sklearn.decomposition import FactorAnalysis

class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, dims, mu0=[], shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None):
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
        
    def fit(self, personIDs, items_1_coords, items_2_coords, preferences):
        '''
        Learn the model with data as follows:
        personIDs - a list of the person IDs of the people who expressed their preferences
        items_1_coords - coordinates of the first items in the pairs being compared
        items_2_coords - coordinates of the second items in each pair being compared
        preferences - the values, 0 or 1 to express that item 1 was preferred to item 2.
        '''
        
        self.people = np.unique(personIDs)
        self.gppref_models = {}
        
        self.obs_coords, _, _ = get_unique_locations(items_1_coords, items_2_coords)
        
        self.N = len(self.obs_coords)
        self.sigmasq_t = self.sigmasq_t * np.eye(self.N).astype(float)
        self.Npeople = len(self.people)
        self.f = {}
        self.t = np.zeros((self.Npeople, self.N))
        self.Wx_plus_mu = np.zeros((self.Npeople, self.N)) + self.mu0
        for p, person in enumerate(self.people):
            self.gppref_models[person] = GPPref(self.dims, self.mu0, self.shape_s0, self.rate_s0, self.s_initial, 
                                                self.shape_ls, self.rate_ls, self.ls_initial)
            if p==0: # initialise the output prior covariance, do this only once  
                distances = np.zeros((self.N, self.N, len(self.dims)))
                for d in range(len(self.dims)):
                    distances[:, :, d] = self.obs_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
                self.Kpred = self.gppref_models[person].kernel_func(distances)
            
        self.fa = FactorAnalysis(n_components=3)
            
        niter = 0
        diff = np.inf
        old_x = 0
        while diff > self.conv_threshold and niter < self.max_iter:
            # run a VB iteration
            # compute preference latent functions for all workers
            self.expec_f(personIDs, items_1_coords, items_2_coords, preferences)
            # compute the preference function means
            self.expec_t()
            # find the personality components from the preference function means
            self.expec_x()
            
            diff = np.max(old_x - self.x)
            print "Difference in latent personality features: %f" % diff 
            old_x = self.x
            
            niter += 1
            
        print "Preference personality model converged in %i iterations." % niter 
        
    def expec_t(self):
        '''
        Compute the expectation over the preference function mean
        '''
        for person in self.gppref_models:
            Kpred_p = self.Kpred / self.gppref_models[person].s
            invKs = np.linalg.inv(Kpred_p)
            inv_sigmasq_t = np.linalg.inv(self.sigmasq_t)
            C = np.linalg.inv(inv_sigmasq_t + invKs)# covariance of t
            self.t[person, :] = C.dot(inv_sigmasq_t.dot(self.Wx_plus_mu[person:person+1, :].T) + invKs.dot(self.f[person])).T
    
    def expec_f(self, personids, items_1_coords, items_2_coords, preferences):
        '''
        Compute the expectation over each worker's latent preference function values for the set of objects.
        '''
        for person in self.gppref_models:
            pidxs = personids == person
            items_1_p = items_1_coords[pidxs]
            items_2_p = items_2_coords[pidxs]
            prefs_p = preferences[pidxs]
            
            self.gppref_models[person].fit(items_1_p, items_2_p, prefs_p)
            self.gppref_models[person].predict(items_0_coords=self.obs_coords, variance_method='sample')
            self.f[person] = self.gppref_models[person].f 
             
    def expec_x(self):
        '''
        Compute the expectation over the personality components.
        '''
        self.x = self.fa.fit_transform(self.t)
        self.Wx_plus_mu = self.x.dot(self.fa.components_) + self.fa.mean_[np.newaxis, :]
        print self.Wx_plus_mu
        self.sigmasq_t = np.diag(self.fa.noise_variance_)
        
if __name__ == '__main__':
    print "Testing Bayesian preference components analysis using synthetic data..."
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
        print "Personality features of %i: %s" % (p, str(model.x[p]))
        for q in range(Npeople):
            print "Distance between personalities: %f" % np.sqrt(np.sum(model.x[p] - model.x[q])**2)**0.5
            print "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0]
            
    