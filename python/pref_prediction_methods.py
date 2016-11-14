'''
Library of various methods for predicting preferences using similarities between people (collaborative filtering). 
Uses clustering, factor analysis etc. to analyse the individuals and their preferences. Then uses various methods to
uses the clusters/factors to produce predictions for held-out data.

Created on 21 Oct 2016

@author: simpson
'''
from preference_features import PreferenceComponents
from gpgrid import coord_arr_to_1d, coord_arr_from_1d
import numpy as np
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import BayesianGaussianMixture#DPGMM GaussianMixture
import pickle, os, logging

class PredictionTester(object):
    
    def __init__(self, datadir, k, nx, ny, personids, pair1coords, pair2coords, prefs, train, 
                        test, results):
        self.datadir = datadir
        self.k = k
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
                
        ucoords_1d, pairidxs_1d = np.unique([pair1coords_1d, pair2coords_1d], return_inverse=True)
        ncoords = len(ucoords_1d)
        self.pair1idxs = pairidxs_1d[:len(pair1coords_1d)]
        self.pair2idxs = pairidxs_1d[len(pair1coords_1d):]
        
#         # this may not be necessary as we could handle flipped comparisons in the method implementations
#         flipidxs = self.pair1idxs > self.pair2idxs
#         tmp = self.pair1idxs[flipidxs]
#         self.pair1idxs[flipidxs] = self.pair2idxs[flipidxs]
#         self.pair2idxs[flipidxs] = tmp
#         self.prefs[flipidxs] = 1 - self.prefs[flipidxs]       
        
        self.pairidxs_ravelled = np.ravel_multi_index((self.pair1idxs, self.pair2idxs), dims=(ncoords, ncoords))
        _, self.pairidxs_ravelled = np.unique(self.pairidxs_ravelled, return_inverse=True)
        npairs = np.max(self.pairidxs_ravelled) + 1
        
        nworkers = np.max(self.personids) + 1
        self.preftable = np.zeros((nworkers, npairs))
        self.preftable[:] = np.nan # + 0.5 # 0.5 is the default value
        self.preftable[self.personids, self.pairidxs_ravelled] = self.prefs
    
        self.preftable_train = np.zeros((nworkers, npairs)) + 0.5
        self.preftable_train[self.personids[self.trainidxs], self.pairidxs_ravelled[self.trainidxs]] = self.prefs[self.trainidxs]

        self.preftable_test = np.zeros((nworkers, npairs)) + 0.5
        self.preftable_test[self.personids[self.testidxs], self.pairidxs_ravelled[self.testidxs]] = self.prefs[self.testidxs]
        
        self.nworkers = nworkers
        self.A = [] # affinity matrix -- don't compute until we need it
    
    def compute_affinity_matrix(self):
        filename = self.datadir + '/affinity.pkl'
        if os.path.isfile(filename):
            with open(filename, 'r') as fh:
                self.A = pickle.load(fh)
                return
        
        # create an affinity matrix for clustering the raw data using the TRAINING DATA
        A = np.zeros((self.nworkers, self.nworkers))
        for i in range(self.nworkers):
            logging.debug('Creating affinity matrix, %i of %i rows' % (i, self.nworkers))
            agreement_i = self.preftable_train==self.preftable_train[i:i+1, :]
            A[i] = np.sum(agreement_i, axis=1) - np.sum(np.invert(agreement_i), axis=1)
        self.A = A
        with open(filename, 'w') as fh:
            pickle.dump(self.A, fh)        
    
    # Baselines
        
    def run_baseline_most_common(self, m): # stupid baseline -- assigns the same label to all data points
        most_common = 0
        if np.sum(self.prefs==0) < np.sum(self.prefs==0.5):
            most_common = 0.5
            if np.sum(self.prefs==0.5) < np.sum(self.prefs==1):
                most_common = 1
        elif np.sum(self.prefs==0) < np.sum(self.prefs==1):
            most_common = 1
        self.results[self.testidxs, m] = most_common
        
    def run_combine_avg(self, m):
        labels = np.zeros(self.nworkers) # they all belong to one cluster -- assume they are the same
        self.run_cluster_matching(labels, m)
    
    # Clustering methods with averaging of other cluster members
    
    def run_affprop_avg(self, m):
        afprop = AffinityPropagation(affinity='precomputed')
        if not len(self.A):
            self.compute_affinity_matrix()
        
        labels =  afprop.fit_predict(self.A)
        self.run_cluster_matching(labels, m)

    def run_agglomerative_avg(self, m):
        agg = AgglomerativeClustering()
        labels = agg.fit_predict(self.preftable_train.T)
        self.run_cluster_matching(labels, m)
    
    def run_raw_gmm_avg(self, m, ncomponents):
        gmm_raw = BayesianGaussianMixture(n_components=ncomponents, weight_concentration_prior=0.5, 
                                          covariance_type='diag', init_params='random') #DPGMM(nfactors)
        gmm_raw.fit(self.preftable_train)
        labels = gmm_raw.predict(self.preftable_train)
        self.run_cluster_matching(labels, m)
        
    def run_gp_affprop_avg(self, m):
        _, model = self.run_gp_separate(m)
        fbar, _ = self.gp_moments_from_model(model)
        
        afprop = AffinityPropagation()        
        labels = afprop.fit_predict(fbar)
        self.run_cluster_matching(labels, m)
          
    # gmm on the separate fbars  
    def run_gp_gmm_avg(self, m, ncomponents):
        _, model = self.run_gp_separate(m)
        fbar, _ = self.gp_moments_from_model(model)

        #gmm = GaussianMixture(n_components=ncomponents)
        gmm = BayesianGaussianMixture(n_components=ncomponents, weight_concentration_prior=0.1, 
                                      covariance_type='diag') #DPGMM(nfactors)
        gmm.fit(fbar)
        labels = gmm.predict(fbar)
        self.run_cluster_matching(labels, m)              
    
    def run_cluster_matching(self, labels, m):
       
        #get the clusters of the personids
        clusters_test = labels[self.personids[self.testidxs]]
        clusters_train = labels[self.personids[self.trainidxs]]
        
        prob_pref_test = np.zeros(self.testidxs.size) # container for the test results
        
        #get the other members of the clusters, then get their labels for the same pairs
        for i, cl in enumerate(clusters_test):
            members = clusters_train == cl #pairs from same cluster
            pair1 = self.pair1idxs[self.testidxs[i]] # id for this current pair
            pair2 = self.pair2idxs[self.testidxs[i]]
            # idxs for the matching pairs 
            matching_pair_idxs = ((self.pair1idxs[self.trainidxs]==pair1) & (self.pair2idxs[self.trainidxs]==pair2))
            # total preferences for the matching pairs 
            total_prefs_matching = np.sum((self.prefs[self.trainidxs][matching_pair_idxs & members] - 0.5) * 2)
            cluster_size = np.sum(matching_pair_idxs & members) + 1.0
    
            prob_pref_test[i] = (float(total_prefs_matching) / float(cluster_size) + 1) / 2.0
        self.results[self.testidxs, m] = prob_pref_test
    
    def run_gpfa(self, m, nfactors):
        # Task C1  ------------------------------------------------------------------------------------------------
    
        # Hypothesis: allows some personalisation but also sharing data through the means
        model_gpfa = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], 
                                          verbose=False, nfactors=nfactors)
        model_gpfa.cov_type = 'diagonal'
        model_gpfa.fit(self.personids[self.trainidxs], self.pair1coords[self.trainidxs], 
                       self.pair2coords[self.trainidxs], self.prefs[self.trainidxs])
        model_gpfa.pickle_me(self.datadir + '/c1_model_gpfa_%i.pkl' % (self.k))
          
        results_k = model_gpfa.predict(self.personids[self.testidxs], self.pair1coords[self.testidxs], 
                                       self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, m] = results_k
        
        return results_k, model_gpfa
    
    def run_gp_combined(self, m):
        # Hypothesis: has benefit that there is more data to learn the GP, but no personalisation
        model_base = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], 
                                          verbose=False)
        model_base.cov_type = 'diagonal'
        model_base.fit(np.zeros(len(self.personids[self.trainidxs])), self.pair1coords[self.trainidxs], 
                       self.pair2coords[self.trainidxs], self.prefs) # blank out the user ids
        model_base.pickle_me(self.datadir + '/c3_model_base_%i.pkl' % (self.k))
        
        results_k = model_base.predict(np.zeros(len(self.personids[self.testidxs])), self.pair1coords[self.testidxs], 
                                       self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, m] = results_k
        return results_k, model_base
        
    def run_gp_separate(self, m):
        #run the model but without the FA part; no shared information between people. 
        #Hypothesis: splitting by person results in too little data per person
        model_gponly = PreferenceComponents([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], 
                                            verbose=False, nfactors=1)
        model_gponly.cov_type = 'diagonal'
        model_gponly.max_iter = 1 # don't run VB till convergence -- gives same results as if running GPs and FA separately
        model_gponly.fit(self.personids[self.trainidxs], self.pair1coords[self.trainidxs], 
                         self.pair2coords[self.trainidxs], self.prefs[self.trainidxs])
        model_gponly.pickle_me(self.datadir + '/c1_model_gponly_%i.pkl' % (self.k))
       
        results_k = model_gponly.predict(self.personids[self.testidxs], self.pair1coords[self.testidxs], 
                                         self.pair2coords[self.testidxs])
        if self.m != None:
            self.results[self.testidxs, m] = results_k
        return results_k, model_gponly
    
    def gp_moments_from_model(self, model):
        fbar = np.zeros(model.t.shape) # posterior means
        v = np.zeros(model.t.shape) # posterior variance
        for person in model.gppref_models:
            fbar[person, :] = model.f[person][:, 0]
            v[person, :] = model.gppref_models[person].v[:, 0]
        fstd = np.sqrt(v)
        return fbar, fstd