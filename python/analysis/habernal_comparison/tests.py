'''
Script for comparing our Bayesian preference learning approach with the results from Habernal 2016. 

Steps in this test:

1. Load word embeddings for the original text data that were used in the NN approach in Habernal 2016. -- done, but 
only using averages to combine them.
2. Load feature data that was used in the SVM-based approach in Habernal 2016.
3. Load the crowdsourced data. -- done. 
4. Copy a similar testing setup to Habernal 2016 (training/test split?) and run the Bayesian approach (during testing,
we can set aside some held-out data). -- done, results saved to file with no metrics computed yet except acc. 
5. Print some simple metrics that are comparable to those used in Habernal 2016. 


Thoughts:
1. NN takes into account sequence of word embeddings; here we need to use a combined embedding for whole text to avoid
a 300x300 dimensional input space.
2. So our method can only learn which elements of the embedding are important, but cannot learn from patterns in the 
sequence, unless we can find a way to encode those.
3. However, the SVM-based approach also did well. Which method is better, NN or SVM, and by how much? 
4. We should be able to improve on the SVM-based approach.
5. The advantages of our method: ranking with sparse data; personalised predictions to the individual annotators; 
uncertainty estimates for active learning and decision-making confidence thresholds. 

Created on 20 Mar 2017

@author: simpson
'''

#import logging
#logging.basicConfig(level=logging.DEBUG)

import sys
import os

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")
svm_python_path = '~/libsvm-3.22/python'

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

import pickle
import numpy as np
import time
import logging
logging.basicConfig(level=logging.DEBUG)
#from preference_features import PreferenceComponents
#from gp_pref_learning import GPPrefLearning, pref_likelihood
#from gp_classifier_svi import GPClassifierSVI
#from sklearn.svm import SVR 
#from compute_metrics import compute_metrics
from data_loading import load_train_test_data, load_embeddings, load_ling_features, data_root_dir, combine_into_libsvm_files
#import skipthoughts
#import wordEmbeddings as siamese_cbow
from joblib import Parallel, delayed
import multiprocessing
    
def _dists_f(items_feat_sample, f):
    if np.mod(f, 100) == 0:
        logging.info('computed lengthscale for feature %i' % f)
    dists = np.abs(items_feat_sample[:, np.newaxis] - items_feat_sample[np.newaxis, :])
    # we exclude the zero distances. With sparse features, these would likely downplay the lengthscale.                                
    med = np.median(dists[dists>0])
    if np.isnan(med):
        med = 1.0
    return med     
    
# Lengthscale initialisation -------------------------------------------------------------------------------------------
# use the median heuristic to find a reasonable initial length-scale. This is the median of the distances.
# First, grab a sample of points because N^2 could be too large.    
def compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings, ling_feat_spmatrix, docids, folds):
    # get the embedding values for the test data -- need to find embeddings of the whole piece of text
    if feature_type == 'both' or feature_type == 'embeddings':
        
        docidxs = []
        doc_tok_seqs = []
        for f in folds:
            doc_tok_seqs.append(folds.get(f)["test"][0])
            doc_tok_seqs.append(folds.get(f)["test"][1])
        
            testids = np.array([ids_pair.split('_') for ids_pair in folds.get(f)["test"][3]])
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 0]))
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 1])) 
        
        X, _ = get_doc_token_seqs(docidxs, doc_tok_seqs)
        
        if embeddings_type == 'word_mean':
            items_feat = get_mean_embeddings(embeddings, X)
        elif embeddings_type == 'skipthoughts':
            logging.error("not implemented yet -- lengthscale heuristic with skip thoughts")
        elif embeddings_type == 'siamese_cbow':
            logging.error("not implemented yet -- lengthscale heuristic with siamese cbow")
        else:
            logging.info("invalid embeddings type! %s" % embeddings_type)
        
    if feature_type == 'both':
        items_feat = np.concatenate((items_feat, ling_feat_spmatrix.toarray()), axis=1)
        
    if feature_type == 'ling':
        items_feat = ling_feat_spmatrix.toarray()
    
    ndims = items_feat.shape[1]
            
    starttime = time.time()
            
    N_max = 3000
    if items_feat.shape[0] > N_max:
        items_feat_sample = items_feat[np.random.choice(items_feat.shape[0], N_max, replace=False)]
    else:
        items_feat_sample = items_feat
                    
    #for f in range(items_feat.shape[1]):  
    num_jobs = multiprocessing.cpu_count()
    default_ls_value = Parallel(n_jobs=num_jobs, backend="multiprocessing")(delayed(_dists_f)(items_feat_sample[:, f], f) for f in range(ndims))
            
    ls_initial_guess = np.ones(ndims) * default_ls_value 
    logging.info('I am using a heuristic multiplier for the length-scale because median is too low to work in high-D spaces')
    #ls_initial_guess *= 1000 # this worked reasonably well but was plucked out of thin air
    ls_initial_guess *= items_feat.shape[1] # this is a heuristic, see e.g. "On the High-dimensional 
    #Power of Linear-time Kernel Two-Sample Testing under Mean-difference Alternatives" by Ramdas et al. 2014. In that 
    # paper they refer to root(no. dimensions) because they square the lengthscale in the kernel function.
            
    endtime = time.time()
    logging.info('@@@ Selected initial lengthscales in %f seconds' % (endtime - starttime))
    
    return ls_initial_guess     
    
def get_doc_token_seqs(ids, doclist):
    # X_train_a1 and trainids_a1 both have one entry per observation. We want to replace them with a list of 
    # unique arguments, and the indexes into that list. First, get the unique argument ids from trainids and testids:
    allids = np.concatenate(ids)
    uids, uidxs = np.unique(allids, return_index=True)
    # get the word index vectors corresponding to the unique arguments
    X = np.zeros(np.max(uids) + 1, dtype=object)
    start = 0
    fin = 0
    X_list = doclist
    for i in range(len(X_list)):
        fin += len(X_list[i])
        idxs = (uidxs>=start) & (uidxs<fin)
        # keep the original IDs to try to make life easier. This means the IDs become indexes into X    
        X[uids[idxs]] = np.array(X_list[i])[uidxs[idxs] - start]
        start += len(X_list[i])
        
    return X, uids    
    
def get_mean_embeddings(word_embeddings, X):
    return np.array([np.mean(word_embeddings[Xi, :], axis=0) for Xi in X])    

def get_docidxs_from_ids(all_docids, ids_to_map):
    return np.array([np.argwhere(docid==all_docids)[0][0] for docid in ids_to_map])
    
def get_fold_data(folds, fold, docids):
    #X_train_a1, X_train_a2 are lists of lists of word indexes 
    X_train_a1, X_train_a2, prefs_train, ids_train, personIDs_train = folds.get(fold)["training"]
    X_test_a1, X_test_a2, prefs_test, ids_test, personIDs_test = folds.get(fold)["test"]
    
    #trainids_a1, trainids_a2 are lists of argument ids
    trainids = np.array([ids_pair.split('_') for ids_pair in ids_train])
    if docids is None:
        docids = np.arange(np.unique(trainids).size)
    trainids_a1 = get_docidxs_from_ids(docids, trainids[:, 0])
    trainids_a2 = get_docidxs_from_ids(docids, trainids[:, 1])
    
    testids = np.array([ids_pair.split('_') for ids_pair in ids_test])
    testids_a1 = get_docidxs_from_ids(docids, testids[:, 0])
    testids_a2 = get_docidxs_from_ids(docids, testids[:, 1])
    
    X, uids = get_doc_token_seqs((trainids_a1, trainids_a2, testids_a1, testids_a2), 
                           [X_train_a1, X_train_a2, X_test_a1, X_test_a2])
        
    print("Training instances ", len(X_train_a1), " training labels ", len(prefs_train))
    print("Test instances ", len(X_test_a1), " test labels ", len(prefs_test))
    
    prefs_train = np.array(prefs_train) 
    prefs_test = np.array(prefs_test)     
    personIDs_train = np.array(personIDs_train)
    personIDs_test = np.array(personIDs_test)  
    
    personIDs = np.concatenate((personIDs_train, personIDs_test))
    _, personIdxs = np.unique(personIDs, return_inverse=True)
    personIDs_train = personIdxs[:len(personIDs_train)]
    personIDs_test = personIdxs[len(personIDs_train):]
    
    return trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, X, uids
    
def get_fold_regression_data(folds_regression, fold, docids):
    if folds_regression is not None:
        _, rankscores_train, argids_rank_train, _ = folds_regression.get(fold)["training"] # blank argument is turkIDs_rank_test
        item_idx_ranktrain = np.array([np.argwhere(trainid==docids)[0][0] for trainid in argids_rank_train])
        rankscores_train = np.array(rankscores_train)
        argids_rank_train = np.array(argids_rank_train)    
        
        _, rankscores_test, argids_rank_test, _ = folds_regression.get(fold)["test"] # blank argument is turkIDs_rank_test
        item_idx_ranktest = np.array([np.argwhere(testid==docids)[0][0] for testid in argids_rank_test])
        rankscores_test = np.array(rankscores_test)
        argids_rank_test = np.array(argids_rank_test)
    else:
        item_idx_ranktrain = None
        rankscores_train = None
        argids_rank_train = None
        item_idx_ranktest = None
        rankscores_test = None
        argids_rank_test = None    

    return item_idx_ranktrain, rankscores_train, argids_rank_train, item_idx_ranktest, rankscores_test, argids_rank_test
    
def get_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, trainids_a2, uids, embeddings=None, X=None):
    '''
    Load all the features specified by the type into an items_feat object. Remove any features where the values are all
    zeroes.
    '''
    # get the embedding values for the test data -- need to find embeddings of the whole piece of text
    if feature_type == 'both' or feature_type == 'embeddings':
        logging.info("Converting texts to mean embeddings (we could use a better sentence embedding?)...")
        if embeddings_type == 'word_mean':
            items_feat = get_mean_embeddings(embeddings, X)
#             elif embeddings_type == 'skipthoughts':
#                 items_feat = skipthoughts.encode(skipthoughts_model, X)
#             elif embeddings_type == 'siamese_cbow':
#                 items_feat = np.array([siamese_cbow_e.getAggregate(Xi) for Xi in X])
        else:
            logging.info("invalid embeddings type! %s" % embeddings_type)
        logging.info("...embeddings loaded.")
        # trim away any features not in the training data because we can't learn from them
        valid_feats = np.sum((items_feat[trainids_a1] != 0) + (items_feat[trainids_a2] != 0), axis=0) > 0
        items_feat = items_feat[:, valid_feats]
        
    elif feature_type == 'ling':
        items_feat = np.zeros((X.shape[0], 0))
        valid_feats = np.zeros(0)
        
    if feature_type == 'both' or feature_type == 'ling':
        logging.info("Obtaining linguistic features for argument texts.")
        # trim the features that are not used in training
        valid_feats_ling = np.sum( (ling_feat_spmatrix[trainids_a1, :] != 0) + 
                                   (ling_feat_spmatrix[trainids_a2, :] != 0), axis=0) > 0 
        valid_feats_ling = np.array(valid_feats_ling).flatten()
        items_feat = np.concatenate((items_feat, ling_feat_spmatrix[uids, :][:, valid_feats_ling].toarray()), axis=1)
        logging.info("...loaded all linguistic features for training and test data.")
        valid_feats = np.concatenate((valid_feats, valid_feats_ling))
        
    return items_feat, valid_feats.astype(bool)
    
def subsample_data(subsample_amount, items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,
                   testids_a1, testids_a2, prefs_test, personIDs_test, 
                   argids_rank_test=None, rankscores_test=None, item_idx_ranktest=None):
    subsample = np.arange(subsample_amount)               
            
    #personIDs_train = np.zeros(len(Xe_train1), dtype=int)[subsample, :] #
    items_feat = items_feat[subsample, :]
    
    pair_subsample_idxs = (trainids_a1<subsample_amount) & (trainids_a2<subsample_amount)
    
    trainids_a1 = trainids_a1[pair_subsample_idxs]
    trainids_a2 = trainids_a2[pair_subsample_idxs]
    prefs_train = prefs_train[pair_subsample_idxs]
    personIDs_train = personIDs_train[pair_subsample_idxs]
            
    # subsampled test data for debugging purposes only
    #personIDs_test = np.zeros(len(items_1_test), dtype=int)[subsample, :]
    pair_subsample_idxs = (testids_a1<subsample_amount) & (testids_a2<subsample_amount)
    testids_a1 = testids_a1[pair_subsample_idxs]
    testids_a2 = testids_a2[pair_subsample_idxs]
    prefs_test = prefs_test[pair_subsample_idxs]
    personIDs_test = personIDs_test[pair_subsample_idxs]
    
    if item_idx_ranktest is not None:
        argids_rank_test = argids_rank_test[item_idx_ranktest < subsample_amount]
        rankscores_test = rankscores_test[item_idx_ranktest < subsample_amount]
        item_idx_ranktest = item_idx_ranktest[item_idx_ranktest < subsample_amount]
        
    return (items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,
                   testids_a1, testids_a2, prefs_test, personIDs_test, 
                   argids_rank_test, rankscores_test, item_idx_ranktest)
    
def run_test(folds, folds_regression, dataset, method, feature_type, embeddings_type=None, embeddings=None, 
             siamese_cbow_e=None, skipthoughts_model=None, ling_feat_spmatrix=None, docids=None, subsample_amount=0, 
             default_ls=None, get_fold_data=get_fold_data, expt_tag='habernal'):
        
    # Select output paths for CSV files and final results
    output_filename_template = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/%s' % expt_tag
    output_filename_template += '_%s_%s_%s_%s' 

    resultsfile = (output_filename_template + '_test.pkl') % (dataset, method, feature_type, embeddings_type)
    modelfile = (output_filename_template + '_model') %  (dataset, method, feature_type, embeddings_type) 
    modelfile += '_fold%i.pkl'
    
    if not os.path.isdir(data_root_dir + 'outputdata'):
        os.mkdir(data_root_dir + 'outputdata')
    if not os.path.isdir(data_root_dir + 'outputdata/crowdsourcing_argumentation_expts'):
        os.mkdir(data_root_dir + 'outputdata/crowdsourcing_argumentation_expts')
                
    if not os.path.isfile(resultsfile):
        all_proba = {}
        all_predictions = {}
        all_f = {}
        
        all_target_prefs = {}
        all_target_rankscores = {}
        final_ls = {}
        times = {}

    else:
        with open(resultsfile, 'r') as fh:
            all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, _, times, final_ls = pickle.load(fh)

    all_argids_rankscores = {}
    #all_turkids_rankscores = {}

    for foldidx, fold in enumerate(folds.keys()):
        if foldidx in all_proba:
            print("Skipping fold %i, %s" % (foldidx, fold))
            continue

        # Get data for this fold --------------------------------------------------------------------------------------
        print("Fold name ", fold)
        trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test,\
                                                                        X, uids = get_fold_data(folds, fold, docids)
        
        # ranking folds
        item_idx_ranktrain, rankscores_train, _, item_idx_ranktest, rankscores_test, argids_rank_test = \
        get_fold_regression_data(folds_regression, fold, docids)
        
        items_feat, valid_feats = get_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, 
                                               trainids_a2, uids, embeddings, X)
                  
        ndims = items_feat.shape[1]

        # subsample training data for debugging purposes only ----------------------------------------------------------
        if subsample_amount > 0:
            subsample_amount, items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,\
               testids_a1, testids_a2, prefs_test, personIDs_test, argids_rank_test, rankscores_test, item_idx_ranktest\
               = subsample_data(subsample_amount, items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,
               testids_a1, testids_a2, prefs_test, personIDs_test, argids_rank_test, rankscores_test, item_idx_ranktest)
                
        # Run the chosen method ---------------------------------------------------------------------------------------
        logging.info("Starting test with method %s..." % (method))
        starttime = time.time()
        
        verbose = True
        optimize_hyper = ('noOpt' not in method)
        nfactors = 10
        
        predicted_f = None
        
        if len(default_ls) > 1:
            ls_initial_guess = default_ls[valid_feats]
        elif '_oneLS' in method:
            ls_initial_guess = np.median(default_ls)
            logging.info("Selecting a single LS for all features: %f" % ls_initial_guess)
        else:
            ls_initial_guess = default_ls          
        
        # Run the selected method
        if 'PersonalisedPrefsBayes' in method:        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, 
                                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba, predicted_f = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                        
        elif 'PersonalisedPrefsUncorrelatedNoise' in method: 
            # Note that this also does not use a common mean to match the Houlsby model.
            # TODO: suspect that with small no. factors, this may be worse, but better with large number in comparison to PersonalisedPrefsBayes with Matern noise GPs.        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                        rate_ls = 1.0 / np.mean(ls_initial_guess), 
                                        use_svi=True, use_fa=False, uncorrelated_noise=True, use_common_mean=False, 
                                        max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                            
        elif 'PersonalisedPrefsFA' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=True, 
                                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                
        elif 'PersonalisedPrefsNoFactors' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, no_factors=True, 
                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                
        elif 'PersonalisedPrefsNoCommonMean' in method:        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, use_common_mean_t=False, 
                        max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)         
                   
        elif 'IndPrefGP' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, no_factors=True, 
                            use_common_mean_t=False, max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat) 
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)                    

        elif 'SinglePrefGP' in method:
            if 'additive' in method:
                kernel_combination = '+'
            else:
                kernel_combination = '*'
            
            if 'weaksprior' in method:
                shape_s0 = 2.0
                rate_s0 = 200.0
            elif 'lowsprior':
                shape_s0 = 1.0
                rate_s0 = 1.0
            else:
                shape_s0 = 200.0
                rate_s0 = 20000.0
            
            model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial_guess, verbose=verbose, 
                        shape_s0=shape_s0, rate_s0=rate_s0,  
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, ninducing=500, max_update_size=200,
                        kernel_combination=kernel_combination)
            model.max_iter_VB = 500
            model.fit(trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, input_type='zero-centered')            
        
            proba, _ = model.predict(testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f, _ = model.predict_f(items_feat[item_idx_ranktest]) 

        elif 'GP+SVM' in method:
            model = GPPrefLearning(ninput_features=1, ls_initial=[1000], verbose=verbose, 
                        shape_s0 = 1.0, rate_s0 = 1.0,  
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=False, kernel_func='diagonal')
            model.max_iter_VB = 10
            model.fit(trainids_a1, trainids_a2, np.arange(items_feat.shape[0])[:, np.newaxis], 
                      np.array(prefs_train, dtype=float)-1, 
                      optimize=False, input_type='zero-centered') # never use optimize with diagonal kernel            

            train_idxs = np.unique([trainids_a1, trainids_a2])
            train_feats = items_feat[train_idxs]
            f, _ = model.predict_f(train_idxs[:, np.newaxis])
            svm = SVR()
            svm.fit(train_feats, f)
            test_f = svm.predict(items_feat)
            
            # apply the preference likelihood from GP method
            proba = pref_likelihood(test_f, v=testids_a1, u=testids_a2, return_g_f=False)
            if folds_regression is not None:
                predicted_f = svm.predict(items_feat[item_idx_ranktest])  
            
        elif 'SingleGPC' in method:
            if 'additive' in method:
                kernel_combination = '+'
            else:
                kernel_combination = '*'
            
            if 'weaksprior' in method:
                shape_s0 = 2.0
                rate_s0 = 200.0
            elif 'lowsprior':
                shape_s0 = 1.0
                rate_s0 = 1.0
            else:
                shape_s0 = 200.0
                rate_s0 = 20000.0
                            
            # twice as many features means the lengthscale heuristic is * 2
            model = GPClassifierSVI(ninput_features=ndims, ls_initial=np.concatenate((ls_initial_guess * 2.0, 
                                                                                      ls_initial_guess * 2.0)), 
                         verbose=verbose, shape_s0=shape_s0, rate_s0=rate_s0, rate_ls = 1.0 / np.mean(ls_initial_guess),
                         use_svi=True, ninducing=500, max_update_size=200)            
            
            # with the argument order swapped around and data replicated:

            gpc_feats = np.concatenate((np.concatenate((items_feat[trainids_a1], items_feat[trainids_a2]), axis=1),
                                    np.concatenate((items_feat[trainids_a2], items_feat[trainids_a1]), axis=1)), axis=0)
            gpc_labels = np.concatenate((np.array(prefs_train, dtype=float) * 0.5,
                                          1 - np.array(prefs_train, dtype=float) * 0.5))
 
            model.max_iter_VB = 500            
            model.fit(np.arange(len(trainids_a1)), gpc_labels, optimize=optimize_hyper, features=gpc_feats)            
        
            proba, _ = model.predict(np.concatenate((items_feat[testids_a1], items_feat[testids_a2]), axis=1))
            if folds_regression is not None:
                predicted_f = np.zeros(len(item_idx_ranktest)) # can't easily rank with this method
                
                
        elif 'SVM' in method:
            sys.path.append(os.path.expanduser(svm_python_path))
            from svmutil import svm_train, svm_predict, svm_read_problem
            svc_labels = np.concatenate((np.array(prefs_train, dtype=float) * 0.5,
                                  1 - np.array(prefs_train, dtype=float) * 0.5))
                                              
            trainfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
            trainfile = trainfile % (dataset, 'training', fold)
            if not os.path.isdir(trainfile):
                combine_into_libsvm_files(dataset, docids[trainids_a1], docids[trainids_a2], svc_labels, 
                                                       'training', fold, reverse_pairs=True)
            
            problem = svm_read_problem(trainfile) 
            model = svm_train(problem[0], problem[1], '-b 1')

            testfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
            testfile = testfile % (dataset, 'test', fold)    
            if not os.path.isdir(testfile):
                combine_into_libsvm_files(dataset, docids[testids_a1], docids[testids_a2], prefs_test * 0.5,
                                                        'test', fold)            
            problem = svm_read_problem(testfile)        
            predictions, _, proba = svm_predict(problem[0], problem[1], model, '-b 1')
            
            if folds_regression is not None:
                trainfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
                trainfile = trainfile % (dataset, 'r_training', fold)                
                if not os.path.isdir(trainfile):
                    combine_into_libsvm_files(dataset, docids[item_idx_ranktrain], None, rankscores_train, 
                                                           'r_training', fold)                
                problem = svm_read_problem(trainfile)
                rank_model = svm_train(problem[0], problem[1], '-s 4')
            
                testfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
                testfile = testfile % (dataset, 'r_test', fold)                    
                if not os.path.isdir(testfile):
                    combine_into_libsvm_files(dataset, docids[item_idx_ranktest], None, rankscores_test, 
                                                           'r_test', fold)
                
                problem = svm_read_problem(testfile)
                predicted_f, _, _ = svm_predict(problem[0], problem[1], rank_model)
        
        elif 'BI-LSTM' in method:
            if feature_type != 'embeddings':
                logging.error("BI-LSTM can only be run using embedings. Will switch to this feature type...")
            
            from keras.preprocessing import sequence
            from keras.models import Graph
            from keras.layers.core import Dense, Dropout
            from keras.layers.embeddings import Embedding
            from keras.layers.recurrent import LSTM
                        
            max_len = 300  # cut texts after this number of words (among top max_features most common words)
            batch_size = 32
            nb_epoch = 5  # 5 epochs are meaningful to prevent over-fitting...

            print len(folds.get(fold)["training"])
            X_train1, X_train2, y_train, _, _ = folds.get(fold)["training"]
            X_train = []
            for i, row1 in enumerate(X_train1):
                row1 = np.array(row1)
                np.append(row1, X_train2[i])
                X_train.append(row1)
            X_test1, X_test2, _, _, _ = folds.get(fold)["test"]
            X_test = []
            for i, row1 in enumerate(X_test1):
                row1 = np.array(row1)
                np.append(row1, X_test2[i])
                X_test.append(row1)        
            print("Pad sequences (samples x time)")
            X_train = sequence.pad_sequences(X_train, maxlen=max_len)
            X_test = sequence.pad_sequences(X_test, maxlen=max_len)
            print('X_train shape:', X_train.shape)
            print('X_test shape:', X_test.shape)
            y_train = np.array(y_train)
            
            print('Build model...')
            model = Graph()
            model.add_input(name='input', input_shape=(max_len,), dtype=int)
            # model.add_node(Embedding(max_features, 128, input_length=maxlen), name='embedding', input='input')
            model.add_node(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings]),
                           name='embedding', input='input')
            model.add_node(LSTM(64), name='forward', input='embedding')
            model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
            model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
            model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
            model.add_output(name='output', input='sigmoid')
        
            # try using different optimizers and different optimizer configs
            model.compile('adam', {'output': 'binary_crossentropy'})
        
            print('Train...')
            model.fit({'input': X_train, 'output': y_train}, batch_size=batch_size, nb_epoch=nb_epoch)
        
            print('Prediction')
            model_predict = model.predict({'input': X_test}, batch_size=batch_size)
            proba = np.array(model_predict['output'])
            
            if folds_regression is not None:
                X_train, y_train, _ = folds_regression.get(fold)["training"]
                X_test, _, _ = folds_regression.get(fold)["test"]
            
                # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
                embeddings = np.asarray([np.array(x, dtype=float) for x in word_index_to_embeddings_map.values()])
            
                print(len(X_train), 'train sequences')
                print(len(X_test), 'test sequences')
            
                print("Pad sequences (samples x time)")
                X_train = sequence.pad_sequences(X_train, maxlen=max_len)
                X_test = sequence.pad_sequences(X_test, maxlen=max_len)
                print('X_train shape:', X_train.shape)
                print('X_test shape:', X_test.shape)
                y_train = np.array(y_train)
            
                print('Build model...')
                model = Graph()
                model.add_input(name='input', input_shape=(max_len,), dtype=int)
                # model.add_node(Embedding(max_features, 128, input_length=maxlen), name='embedding', input='input')
                model.add_node(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=max_len, weights=[embeddings]),
                               name='embedding', input='input')
                model.add_node(LSTM(64), name='forward', input='embedding')
                model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
                model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
            
                # match output layer for regression better
                model.add_node(Dense(1, activation='linear', init='uniform'), name='output_layer', input='dropout')
                model.add_output(name='output', input='output_layer')
            
                # use mean absolute error loss
                model.compile('adam', {'output': 'mean_absolute_error'})
            
                print('Train...')
                model.fit({'input': X_train, 'output': y_train}, batch_size=batch_size, nb_epoch=nb_epoch)
            
                print('Prediction')
                model_predict = model.predict({'input': X_test}, batch_size=batch_size)
                # print(model_predict)
                predicted_f = np.asarray(model_predict['output']).flatten()                
        
        if hasattr(model, 'ls'):
            final_ls[foldidx] = model.ls
        else:
            final_ls[foldidx] = [0]        
        predictions = np.round(proba)
        
        endtime = time.time() 
        
        logging.info("@@@ Completed running fold %i with method %s, features %s, in %f seconds." % (foldidx, method, 
                                                                            feature_type, endtime-starttime) )
        logging.info("Accuracy for fold = %f" % (np.sum(prefs_test[prefs_test != 1] == 2 * predictions.flatten()[prefs_test != 1]) 
                                          / float(np.sum(prefs_test  != 1))) )
        endtime-starttime
        # Save the data for later analysis ----------------------------------------------------------------------------
        # Outputs from the tested method
        all_proba[foldidx] = proba
        all_predictions[foldidx] = predictions
        all_f[foldidx] = predicted_f
        
        # Save the ground truth
        all_target_prefs[foldidx] = prefs_test
        if folds_regression is not None:
            all_target_rankscores[foldidx] = rankscores_test
            all_argids_rankscores[foldidx] = argids_rank_test
            #all_turkids_rankscores[foldidx] = turkIDs_rank_test
        
        # Save the time taken
        times[foldidx] = endtime-starttime
        
        results = (all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, ls_initial_guess,
                   times, final_ls) 
        with open(resultsfile, 'w') as fh:
            pickle.dump(results, fh)
            
        #with open(modelfile % foldidx, 'w') as fh:
        #    pickle.dump(model, fh)

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
                        skipthoughts_model, ling_feat_spmatrix, docids, default_ls, subsample_amount=0):
    
    for acc in acc_levels:
        for pair_subset in tr_pair_subsets:
            
            # add noise to the data in folds. 
            def get_fold_data(folds, fold, docids):
                return get_noisy_fold_data(folds, fold, docids, acc, pair_subset)
            
            run_test(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, subsample_amount, 
                        default_ls, get_fold_data=get_fold_data, expt_tag='noise%f_sparse%f' % (acc, pair_subset))        
            
'''        
Where to run the tests:

desktop-169: all feature types, word_mean embeddings, singleprefgp + singleprefgp_onels + PersonalisedPrefsNoCommonMean
barney: all feature types, word_mean embeddings, PersonalisedPrefsBayes + PersonalisedPrefsFA
apu: all feature types, word_mean embeddings, IndPrefGP + PersonalisedPrefsNoFactors

Florence?
Google code trial server? --> all server jobs.

Run the other embeddings types on the first servers to finish.

Steps needed to run them:

1. Git clone personalised_argumentation and HeatMapBCC
2. Make a local copy of the language feature data:
3. Make a local copy of the embeddings:
4. Run!

'''
def run_test_set(run_test_fun=run_test):
    # keep these variables around in case we are restarting the script with different method settings and same data.
    global dataset
    global folds
    global folds_regression
    global word_index_to_embeddings_map
    global word_to_indices_map
    global word_embeddings
    global ling_feat_spmatrix
    global docids
    global siamese_cbow_embeddings
    global skipthoughts_model
        
    if 'dataset' not in globals():
        dataset = ''
                      
    if 'folds' in globals() and dataset == datasets[0] and 'word_embeddings' in globals():
        load_data = False
    else:
        load_data = True
    
    if 'default_ls_values' not in globals():
        global default_ls_values
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
                and ('IndPref' in method or 'Personalised' in method):
                
                logging.warning('Skipping method %s on dataset %s because there are no separate worker IDs.' 
                                % (method, dataset))
                continue
            
            for feature_type in feature_types:
                if feature_type == 'embeddings' or feature_type == 'both':
                    embeddings_to_use = embeddings_types
                else:
                    embeddings_to_use = ['']
                for embeddings_type in embeddings_to_use:
                    logging.info("**** Running method %s with features %s, embeddings %s, on dataset %s ****" % (method, 
                                                    feature_type, embeddings_type, dataset) )
                    if dataset in default_ls_values and feature_type in default_ls_values[dataset]:
                        default_ls_value = default_ls_values[dataset][feature_type]
                    else:
                        default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, word_embeddings,
                                                                         ling_feat_spmatrix, docids, folds)
                        if dataset not in default_ls_values:
                            default_ls_values[dataset] = {}
                        default_ls_values[dataset][feature_type] = default_ls_value
                            
                    run_test_fun(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, subsample_amount=0, 
                        default_ls=default_ls_value)                 
                    
                    logging.info("**** Completed: method %s with features %s, embeddings %s ****" % (method, feature_type, 
                                                                                           embeddings_type) )
    return default_ls_values
        
if __name__ == '__main__':
# # Issue #33: rerun * kernel combination with both features and weak s prior.
#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_noOpt', 'SinglePrefGP_noOpt_lowsprior', 
#                'SinglePrefGP_noOpt_additive_lowsprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#  
#      default_ls_values = run_test_set() # Run on Friday already.
#         
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='33')    

# # Issue #34 Compare kernel operators
#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#  
#     default_ls_values = run_test_set()
#     
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='34')
   
# # # Issue #35 Best setup with other datasets
#     datasets = ['UKPConvArgAll_evalMACE'] #'UKPConvArgMACE', 
#     methods = ['SinglePrefGP_noOpt_weaksprior']#, 'SinglePrefGP_noOpt_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#      
#     default_ls_values = run_test_set()
#         
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='35')
#    
#running on desktop-169 now
# # Issue #37 GP+SVR. The GPC method should be run on Barney as it needs more memory.
#     datasets = ['UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
#     methods = ['GP+SVM']
#     feature_types = ['both', 'ling'] # we run with ling as well because this is how SVM was originally run by IH.
#     embeddings_types = ['word_mean']
#         
#     default_ls_values = run_test_set()
#         
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='37')
#      
# # Issue #36 Optimize best setup
#     datasets = ['UKPConvArgStrict', 'UKPConvArgMACE']#, 'UKPConvArgAll_evalMACE']
#     methods = ['SinglePrefGP_weaksprior', 'SinglePrefGP_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#       
#     default_ls_values = run_test_set() 
#          
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='36')
#   
# # Issue #38, Run SVM on other datasets and compute missing metrics
#     datasets = ['UKPConvArgStrict', 'UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
#     methods = ['SVM']
#     feature_types = ['both', 'ling']
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#     default_ls_values = run_test_set() 
#          
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='38')

# Issue #39, Run BILSTM on other datasets and compute missing metrics
    datasets = ['UKPConvArgStrict', 'UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
    methods = ['BI-LSTM']
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
    default_ls_values = run_test_set() 
        
    compute_metrics(methods, datasets, feature_types, embeddings_types, tag='38')


# # Issue #40 Noise/sparsity tests with best setup.
#     acc_levels = [0.6, 0.7, 0.8, 0.9, 1.0] # the accuracy of the pairwise labels used to train the methods -- this is how we introduce noise
#     tr_pair_subsets = [0.25, 0.5, 0.75, 1.0] # fraction of the dataset we will use to train the methods
#     #tr_item_subset = [0.25, 0.5, 0.75, 1.0] # to be implemented later. Fix the total number of labels but vary the 
#     #number of items they cover -- does a densely labelled subset help? Fix no. labels by: selecting pairs randomly
#     # until smallest item subset size is reached; select any other pairs involving that subset; count the no. items. 
#     
#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
#  
#     default_ls_values = run_test_set(run_noise_sparsity_test)
#     
#     #TODO: the plotting and metrics for the noise/sparsity tests.
