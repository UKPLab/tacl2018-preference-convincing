'''
Incomplete. Should be based on tests.py but separated so we can provide the personalised model experiments separately 
as they will be covered in a different paper.

Created on 19 Jun 2017

@author: simpson
'''
import sys
sys.path.append("./python/analysis/habernal_comparison")
import tests
from preference_features import PreferenceComponents
import numpy as np

nfactors = 10

def run_persgppl(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, unseenids_a1, unseenids_a2, ls_initial_guess, verbose, 
             item_idx_ranktrain=None, rankscores_train=None, item_idx_ranktest=None, personIDs_train=None,
            personIDs_test=None):

    common_mean = True
    gp_noise = True
    use_fa = False
    
    if '_houlsby' in method:
        common_mean = False
        gp_noise = False
    elif '_indnoise' in method:
        gp_noise = False
    elif '_fa' in method:
        gp_noise = True
        common_mean = False
        use_fa = True

    model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=use_fa, uncorrelated_noise=not gp_noise, 
                use_common_mean_t=common_mean, max_update_size=500)
    model.max_iter = 2
    model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
              optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
    proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
    if item_idx_ranktest is not None:
        predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat) 

    return proba, predicted_f, None, model

def run_indgppl(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, unseenids_a1, unseenids_a2, ls_initial_guess, verbose, 
             item_idx_ranktrain=None, rankscores_train=None, item_idx_ranktest=None, personIDs_train=None, 
             personIDs_test=None):

    model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                    rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, no_factors=True, 
                    use_common_mean_t=False, max_update_size=200)
    model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
              optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
    proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat) 
    if item_idx_ranktest is not None:
        predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
            
    return proba, predicted_f, None, model

def pers_method_chooser(method, feature_type):
    if 'PersPrefGP' in method:
        method_runner_fun = run_persgppl
    elif 'IndPrefGP' in method:
        method_runner_fun = run_indgppl
    else:
        method_runner_fun = tests.method_chooser(method, feature_type)  
        
    return method_runner_fun

if __name__ == '__main__':
    tests.dataset_increment = 0     
    tests.datasets = ['UKPConvArgCrowdSample']
    tests.feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both' or 'debug'
    tests.methods = ['PersPrefGP_houlsby_noOpt', 'PersPrefGP_noOpt', 'PersPrefGP_indnoise_noOpt', 
                     'PersPrefGP_fa_noOpt', 'IndPrefGP_noOpt'] 
    tests.embeddings_types = ['word_mean']#, 'skipthoughts'] # 'siamese-cbow'] 
                
    default_ls_values = tests.run_test_set(test_on_train=True, choose_method=pers_method_chooser)