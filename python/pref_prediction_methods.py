'''
Library of various methods for predicting preferences using similarities between people (collaborative filtering). 
Uses clustering, factor analysis etc. to analyse the individuals and their preferences. Then uses various methods to
uses the clusters/factors to produce predictions for held-out data.

Created on 21 Oct 2016

@author: simpson
'''
from preference_features import PreferenceComponents
from gpgrid import coord_arr_to_1d
import numpy as np
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

class PredictionTester(object):
    
    def __init__(self, datadir, k, m, nx, ny, personids, pair1coords, pair2coords, prefs, train, 
                        test, results):
        self.datadir = datadir
        self.k = k
        self.m = m
        self.nx = nx
        self.ny = ny
        self.personids = personids
        self.pair1coords = pair1coords
        self.pair2coords = pair2coords
        self.prefs = prefs
        self.trainidxs = train
        self.testidxs = test
        self.results = results 
    
        # turn the data into a big matrix
        pair1coords_1d = coord_arr_to_1d(pair1coords)
        pair2coords_1d = coord_arr_to_1d(pair2coords)
        
        flipidxs = pair1coords_1d > pair2coords_1d
        tmp = pair1coords_1d[flipidxs]
        pair1coords_1d[flipidxs] = pair2coords_1d
        pair2coords_1d[flipidxs] = tmp
        
        ucoords_1d, pairidxs_1d = np.unique([pair1coords_1d, pair2coords_1d], return_inverse=True)
        ncoords = len(ucoords_1d)
        pair1idxs = pairidxs_1d[:len(pair1coords_1d)]
        pair2idxs = pairidxs_1d[len(pair1coords_1d):]
        pairidxs_ravelled = np.ravel_multi_index((pair1idxs, pair2idxs), dims=(ncoords, ncoords))
        
        nworkers = len(np.unique(self.personids))
        self.preftable = np.zeros((ncoords, nworkers))
        self.preftable[pairidxs_ravelled, self.personids] = self.prefs        
    
    def run_affprop_avg(self):
        
        afprop = AffinityPropagation()
        afprop_labels = afprop.fit_predict(self.preftable)
        ncomponents = len(np.unique(afprop_labels))

        # cannot apply this to factor analysis -- can do another plot to show distributions. How fuzzy are memberships?        
        afprop_membership = np.zeros(ncomponents)
        membership_weights_ap = np.zeros((ncomponents, N))
        for k in range(ncomponents):
            afprop_membership[k] = np.sum(afprop_labels==k)
            membership_weights_ap[k] = (afprop_labels==k)        
    
    
    def run_gpfa(self, nfactors):
        # Task C1  ------------------------------------------------------------------------------------------------
    
        # Hypothesis: allows some personalisation but also sharing data through the means
        model_gpfa = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=False, 
                                          nfactors=nfactors)
        model_gpfa.cov_type = 'diagonal'
        model_gpfa.fit(self.personids[self.trainidxs], self.pair1coords[self.trainidxs], self.pair2coords[self.trainidxs], self.prefs[self.trainidxs])
        model_gpfa.pickle_me(self.datadir + '/c1_model_gpfa_%i.pkl' % (self.k))
          
        results_k = model_gpfa.predict(self.personids[self.testidxs], self.pair1coords[self.testidxs], self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, self.m] = results_k
        
        return results_k, model_gpfa
    
    def run_gp_combined(self):
        # Hypothesis: has benefit that there is more data to learn the GP, but no personalisation
        model_base = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=False)
        model_base.cov_type = 'diagonal'
        model_base.fit(np.zeros(len(self.personids[self.trainidxs])), self.pair1coords[self.trainidxs], self.pair2coords[self.trainidxs], self.prefs) # blank out the user ids
        model_base.pickle_me(self.datadir + '/c3_model_base_%i.pkl' % (self.k))
        
        results_k = model_base.predict(np.zeros(len(self.personids[self.testidxs])), self.pair1coords[self.testidxs], self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, self.m] = results_k
        return results_k, model_base
        
    def run_gp_separate(self):
        #run the model but without the FA part; no shared information between people. 
        #Hypothesis: splitting by person results in too little data per person
        model_gponly = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], 
                                            verbose=False, nfactors=1)
        model_gponly.cov_type = 'diagonal'
        model_gponly.max_iter = 1 # don't run VB till convergence -- gives same results as if running GPs and FA separately
        model_gponly.fit(self.personids[self.trainidxs], self.pair1coords[self.trainidxs], self.pair2coords[self.trainidxs], self.prefs[self.trainidxs])
        model_gponly.pickle_me(self.datadir + '/c1_model_gponly_%i.pkl' % (self.k))
       
        results_k = model_gponly.predict(self.personids[self.testidxs], self.pair1coords[self.testidxs], self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, self.m] = results_k
        return results_k, model_gponly