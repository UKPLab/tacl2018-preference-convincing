'''
Run similar tests to the habernal comparison, except:
 -- we simulate noisy pairs
 -- we allow the tests to be run with different sparsity levels
Therefore, it may be sensible to use only one setup per method to avoid the experiments taking too long.
    
Created on 7 Jun 2017

@author: edwin
'''
from tests import load_embeddings, load_train_test_data, load_ling_features, compute_lengthscale_heuristic, run_test, get_fold_data
import logging
import numpy as np

acc_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # the accuracy of the pairwise labels used to train the methods -- this is how we introduce noise
tr_pair_subsets = [0.25, 0.5, 0.75, 1.0] # fraction of the dataset we will use to train the methods
#tr_item_subset = [0.25, 0.5, 0.75, 1.0] # to be implemented later. Fix the total number of labels but vary the 
#number of items they cover -- does a densely labelled subset help? Fix no. labels by: selecting pairs randomly
# until smallest item subset size is reached; select any other pairs involving that subset; count the no. items. 

def get_noisy_fold_data(folds, fold, docids, acc, tr_pair_subset):
    trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, X, uids\
    = get_fold_data(folds, fold, docids)
    
    # now subsample the training data
    N = len(trainids_a1)
    Nsub = N * tr_pair_subset
    subidxs = np.random.choice(N, Nsub, replace=False)
    trainids_a1 = trainids_a1[subidxs]
    trainids_a2 = trainids_a2[subidxs]
    prefs_train = prefs_train[subidxs]
    personIDs_train = personIDs_train[subidxs]
    
    # now we add noise to the training data
    flip_labels = np.random.rand(Nsub) > acc
    prefs_train[flip_labels] = 2 - prefs_train[flip_labels] # labels are 0, 1 or 2
    
    return trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, X, uids
    

def run_noise_sparsity_test(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, ls_initial_guess, subsample_amount=0):
    
    for acc in acc_levels:
        for pair_subset in tr_pair_subsets:
            
            # add noise to the data in folds. 
            def get_fold_data(folds, fold, docids):
                return get_noisy_fold_data(folds, fold, docids, acc, pair_subset)
            
            run_test(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, subsample_amount, 
                        ls_initial_guess, get_fold_data=get_fold_data, expt_tag='noise%f_sparse%f' % (acc, pair_subset))

if __name__ == '__main__':
    
    datasets = ['UKPConvArgStrict']# Barney, desktop-169
    methods = ['SinglePrefGP_noOpt', 'SinglePrefGP'] # desktop-169 as well
    
    feature_types = ['both']#, 'ling', 'embeddings'] # can be 'embeddings' or 'ling' or 'both'
    embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
                      
    if 'folds' in globals() and 'dataset' in globals() and dataset == datasets[0]:
        load_data = False
    else:
        load_data = True
    
    if 'default_ls_values' not in globals():    
        default_ls_values = {}
          
    for method in methods:
        for dataset in datasets:
            if load_data:
                folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map = load_train_test_data(dataset)
                word_embeddings = load_embeddings(word_index_to_embeddings_map)
                siamese_cbow_embeddings = None#load_siamese_cbow_embeddings(word_to_indices_map)
                skipthoughts_model = None#load_skipthoughts_embeddings(word_to_indices_map)
                ling_feat_spmatrix, docids = load_ling_features(dataset)
           
            if (dataset == 'UKPConvArgMACE' or dataset == 'UKPConvArgStrict' or dataset == 'UKPConvArgAll_evalMACE') \
                and ('SinglePrefGP' not in method and 'SingleGPC' not in method and 'GP+SVM' not in method):
                
                logging.warning('Skipping method %s on dataset %s because there are no separate worker IDs.' 
                                % (method, dataset))
                continue
            
            for feature_type in feature_types:
                if feature_type == 'embeddings' or feature_type == 'both':
                    embeddings_to_use = embeddings_types
                else:
                    embeddings_to_use = ['']
                for embeddings_type in embeddings_to_use:
                    print "**** Running method %s with features %s, embeddings %s, on dataset %s ****" % (method, 
                                                    feature_type, embeddings_type, dataset)
                    if dataset in default_ls_values and feature_type in default_ls_values[dataset]:
                        default_ls_value = default_ls_values[dataset][feature_type]
                    else:
                        default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, word_embeddings,
                                                                         ling_feat_spmatrix, docids, folds)
                        if dataset not in default_ls_values:
                            default_ls_values[dataset] = {}
                        default_ls_values[dataset][feature_type] = default_ls_value
                            
                    run_noise_sparsity_test(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, subsample_amount=0, 
                        ls_initial_guess=default_ls_value)
                    
                    print "**** Completed: method %s with features %s, embeddings %s ****" % (method, feature_type, 
                                                                                           embeddings_type)