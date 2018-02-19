'''
Personalised argumentation paper: are user features are required or latent variables sufficient/correlated with observed features?
Is prior stance a useful user feature for predicting belief change? It should be, since a user can only be convinced by 
an argument if they did not previously believe in it.
Topic-specific nature means predictions based on linguistic features are likely to be weak? 

TODO: 

Created on 19 Jun 2017

@author: simpson
'''
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
from tests import TestRunner
sys.path.append("./python/analysis/habernal_comparison")
from collab_pref_learning_vb import CollabPrefLearningVB, PreferenceComponentsFA, PreferenceComponentsSVI, PreferenceNoComponentFactors
import numpy as np

nfactors = 10

class PersonalisedTestRunner(TestRunner):

    def run_persgppl(self):
    
        common_mean = False
        
        if '_commonmean' in self.method:
            common_mean = True
            
        gp_noise = False
        if '_gpnoise' in self.method:
            gp_noise = True
    
        if '_fa' in self.method:
            model_class = PreferenceComponentsFA
        elif 'IndPrefGP' in self.method:
            model_class = PreferenceNoComponentFactors
        elif '_nofactorsvi' in self.method:
            model_class = CollabPrefLearningVB
        else:
            model_class = PreferenceComponentsSVI
    
        self.model = model_class(nitem_features=self.ndims, ls=self.ls_initial, verbose=self.verbose, 
                     nfactors=nfactors, rate_ls = 1.0 / np.mean(self.ls_initial),
                     use_common_mean_t=common_mean, uncorrelated_noise=not gp_noise, max_update_size=1000)
        self.model.max_iter = 2
        
        zero_centered_prefs = np.array(self.prefs_train, dtype=float)-1
        
        self.model.fit(self.person_train, self.a1_train, self.a2_train, self.items_feat, zero_centered_prefs, 
                  optimize=self.optimize_hyper, nrestarts=1, input_type='zero-centered')
        
        proba = self.model.predict(self.person_test, self.a1_test, self.a2_test, self.items_feat)
        if self.a_rank_test is not None:
            predicted_f = self.model.predict_f_item_person(self.person_rank_test, self.a_rank_test, self.items_feat)
    
        return proba, predicted_f, None

    def _choose_method_fun(self, feature_type):
        if 'PersPrefGP' in self.method:
            method_runner_fun = self.run_persgppl
        elif 'IndPrefGP' in self.method:
            method_runner_fun = self.run_persgppl # switches to correct class inside the method
        else:
            method_runner_fun = super(PersonalisedTestRunner, self)._choose_method_fun(feature_type)  
            
        return method_runner_fun

if __name__ == '__main__':
    dataset_increment = 0     
    datasets = ['UKPConvArgCrowdSample']
    feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both' or 'debug'

    methods = [
        'PersPrefGP_fa_noOpt', 'PersPrefGP_noOpt_gpnoise', 'PersPrefGP_noOpt', 'PersPrefGP_commonmean_noOpt', 
        'IndPrefGP_noOpt',  
               ]  
    embeddings_types = ['word_mean']#, 'skipthoughts'] # 'siamese-cbow'] 
    
    #if 'runner' not in globals():
    runner = PersonalisedTestRunner('personalised', datasets, feature_types, embeddings_types, methods, 
                                        dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=1)
    
    methods = [ 
        'PersPrefGP_noOpt_nofactorsvi', 'PersPrefGP_noOpt_gpnoise_nofactorsvi', # make sure code works without SVI for latent factors
        ]  
    
    #if 'runner' not in globals():
    runner = PersonalisedTestRunner('personalised', datasets, feature_types, embeddings_types, methods, 
                                        dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=1, npairs=1000, subsample_tr=1000)