# -- coding: utf-8 --

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

sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow"))

import pickle
import time
import logging
logging.basicConfig(level=logging.DEBUG)
from gp_pref_learning import GPPrefLearning, pref_likelihood
from gp_classifier_svi import GPClassifierSVI
from sklearn.svm import SVR 
from compute_metrics import compute_metrics
from data_loading import load_train_test_data, load_embeddings, load_ling_features, data_root_dir, \
combine_into_libsvm_files, load_siamese_cbow_embeddings, load_skipthoughts_embeddings
import skipthoughts
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
    
def _dists_f(items_feat_sample, f):
    if np.mod(f, 1000) == 0:
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
def compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings, ling_feat_spmatrix, docids, folds, 
                                  index_to_word_map):
    # get the embedding values for the test data -- need to find embeddings of the whole piece of text
    if feature_type == 'both' or feature_type == 'embeddings':
        
        docidxs = []
        doc_tok_seqs = []
        doctexts = []
        for f in folds:
            doc_tok_seqs.append(folds.get(f)["test"][0])
            doc_tok_seqs.append(folds.get(f)["test"][1])
        
            testids = np.array([ids_pair.split('_') for ids_pair in folds.get(f)["test"][3]])
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 0]))
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 1]))
            
            doctexts.append(folds.get(f)["test"][5])
            doctexts.append(folds.get(f)["test"][6])
        
        X, _, utexts = get_doc_token_seqs(docidxs, doc_tok_seqs, doctexts)
        
        if embeddings_type == 'word_mean':
            items_feat = get_mean_embeddings(embeddings, X)
        elif embeddings_type == 'skipthoughts':
            items_feat = skipthoughts.encode(embeddings, utexts)
        elif embeddings_type == 'siamese-cbow':
            items_feat = np.array([embeddings.getAggregate(index_to_word_map[Xi]) for Xi in X])
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
    
def get_doc_token_seqs(ids, doclist, texts=None):
    # X_train_a1 and trainids_a1 both have one entry per observation. We want to replace them with a list of 
    # unique arguments, and the indexes into that list. First, get the unique argument ids from trainids and testids:
    allids = np.concatenate(ids)
    uids, uidxs = np.unique(allids, return_index=True)
    # get the word index vectors corresponding to the unique arguments
    X = np.zeros(np.max(uids) + 1, dtype=object)
    
    if texts is not None:
        utexts = np.zeros(np.max(uids) + 1, dtype=object)    
    
    start = 0
    fin = 0
    X_list = doclist
    for i in range(len(X_list)):
        fin += len(X_list[i])
        idxs = (uidxs>=start) & (uidxs<fin)
        # keep the original IDs to try to make life easier. This means the IDs become indexes into X    
        X[uids[idxs]] = np.array(X_list[i])[uidxs[idxs] - start]
        
        if texts is not None:
            utexts[uids[idxs]] = np.array(texts[i])[uidxs[idxs] - start] 
        
        start += len(X_list[i])
        
    if texts is not None:
        utexts = [utext.decode('utf-8') for utext in utexts]
        return X, uids, utexts
    else:
        return X, uids    
    
def get_mean_embeddings(word_embeddings, X):
    return np.array([np.mean(word_embeddings[Xi, :], axis=0) for Xi in X])    

def get_docidxs_from_ids(all_docids, ids_to_map):
    return np.array([np.argwhere(docid==all_docids)[0][0] for docid in ids_to_map])
    
def get_fold_data(folds, fold, docids):
    #X_train_a1, X_train_a2 are lists of lists of word indexes 
    X_train_a1, X_train_a2, prefs_train, ids_train, personIDs_train, tr_a1, tr_a2 = folds.get(fold)["training"]
    X_test_a1, X_test_a2, prefs_test, ids_test, personIDs_test, test_a1, test_a2 = folds.get(fold)["test"]
    
    #trainids_a1, trainids_a2 are lists of argument ids
    trainids = np.array([ids_pair.split('_') for ids_pair in ids_train])
    if docids is None:
        docids = np.arange(np.unique(trainids).size)
    trainids_a1 = get_docidxs_from_ids(docids, trainids[:, 0])
    trainids_a2 = get_docidxs_from_ids(docids, trainids[:, 1])
    
    testids = np.array([ids_pair.split('_') for ids_pair in ids_test])
    testids_a1 = get_docidxs_from_ids(docids, testids[:, 0])
    testids_a2 = get_docidxs_from_ids(docids, testids[:, 1])
    
    X, uids, utexts = get_doc_token_seqs((trainids_a1, trainids_a2, testids_a1, testids_a2), 
                           [X_train_a1, X_train_a2, X_test_a1, X_test_a2], (tr_a1, tr_a2, test_a1, test_a2))
        
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
    
    return trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, \
        X, uids, utexts
        
def get_noisy_fold_data(folds, fold, docids, acc, tr_pair_subset=None):
    trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, X, \
    uids, utexts = get_fold_data(folds, fold, docids)
    
    # now subsample the training data
    N = len(trainids_a1)
    if tr_pair_subset is not None:
        Nsub = N * tr_pair_subset
        subidxs = np.random.choice(N, Nsub, replace=False)
        trainids_a1 = trainids_a1[subidxs]
        trainids_a2 = trainids_a2[subidxs]
        prefs_train = prefs_train[subidxs]
        personIDs_train = personIDs_train[subidxs]
    else:
        Nsub = N

    if acc != 1.0:
        # now we add noise to the training data
        flip_labels = np.random.rand(Nsub) > acc
        prefs_train[flip_labels] = 2 - prefs_train[flip_labels] # labels are 0, 1 or 2
    
    return trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, \
        X, uids, utexts   
    
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
    
def get_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, trainids_a2, uids, embeddings=None, 
                 X=None, index_to_word_map=None, utexts=None):
    '''
    Load all the features specified by the type into an items_feat object. Remove any features where the values are all
    zeroes.
    '''
    # get the embedding values for the test data -- need to find embeddings of the whole piece of text
    if feature_type == 'both' or feature_type == 'embeddings':
        logging.info("Converting texts to mean embeddings (we could use a better sentence embedding?)...")
        if embeddings_type == 'word_mean':
            items_feat = get_mean_embeddings(embeddings, X)
        elif embeddings_type == 'skipthoughts':
            items_feat = skipthoughts.encode(embeddings, utexts)
        elif embeddings_type == 'siamese-cbow':
            items_feat = np.array([embeddings.getAggregate(index_to_word_map[Xi]) for Xi in X])
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
    
# Methods for running the prediction methods --------------------------------------------------------------------------
def run_gppl(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain=None, 
             rankscores_train=None, item_idx_ranktest=None):
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
    
    if model is None:
        model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial_guess, verbose=verbose, 
                shape_s0=shape_s0, rate_s0=rate_s0,  
                rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, ninducing=500, max_update_size=200,
                kernel_combination=kernel_combination)
        model.max_iter_VB = 500
    
    model.fit(trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
              optimize=optimize_hyper, input_type='zero-centered')            

    proba, _ = model.predict(testids_a1, testids_a2, items_feat)
    if item_idx_ranktest is not None:
        predicted_f, _ = model.predict_f(items_feat[item_idx_ranktest])
    else:
        predicted_f = None
        
    return proba, predicted_f, model

def run_gpsvm(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain=None, 
             rankscores_train=None, item_idx_ranktest=None):
    if model is None:
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
    if item_idx_ranktest is not None:
        predicted_f = svm.predict(items_feat[item_idx_ranktest])
    else:
        predicted_f = None            
    return proba, predicted_f, model      
    
def run_gpc(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain=None, 
             rankscores_train=None, item_idx_ranktest=None):
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
    if model is None:
        model = GPClassifierSVI(ninput_features=ndims, ls_initial=np.concatenate((ls_initial_guess * 2.0, 
                                                                              ls_initial_guess * 2.0)), 
                 verbose=verbose, shape_s0=shape_s0, rate_s0=rate_s0, rate_ls = 1.0 / np.mean(ls_initial_guess),
                 use_svi=True, ninducing=500, max_update_size=200, kernel_combination=kernel_combination)            
        model.max_iter_VB = 500
          
    # with the argument order swapped around and data replicated:
    
    gpc_feats = np.concatenate((np.concatenate((items_feat[trainids_a1], items_feat[trainids_a2]), axis=1),
                            np.concatenate((items_feat[trainids_a2], items_feat[trainids_a1]), axis=1)), axis=0)
    gpc_labels = np.concatenate((np.array(prefs_train, dtype=float) * 0.5,
                                  1 - np.array(prefs_train, dtype=float) * 0.5))
              
    model.fit(np.arange(len(trainids_a1)), gpc_labels, optimize=optimize_hyper, features=gpc_feats)            
    
    proba, _ = model.predict(np.concatenate((items_feat[testids_a1], items_feat[testids_a2]), axis=1))
    if item_idx_ranktest is not None:
        predicted_f = np.zeros(len(item_idx_ranktest)) # can't easily rank with this method
    else:
        predicted_f = None
    return proba, predicted_f, model      

def run_svm(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain=None, 
             rankscores_train=None, item_idx_ranktest=None):
    sys.path.append(os.path.expanduser(svm_python_path))
    from svmutil import svm_train, svm_predict, svm_read_problem
    #from subprocess import call
     
    svc_labels = np.concatenate((np.array(prefs_train, dtype=float) * 0.5,
                          1 - np.array(prefs_train, dtype=float) * 0.5))
                                       
    trainfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
    trainfile = trainfile % (dataset, 'training', fold)
     
    #tempmodelfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.model'
    #tempmodelfile = tempmodelfile % (dataset, 'training', fold)
     
    #predfile = data_root_dir + '/libsvmdata/%s-%s-%s-pred.txt'
    #predfile = predfile % (dataset, 'training', fold)            
     
    nfeats = ling_feat_spmatrix.shape[1]
     
    if not os.path.isfile(trainfile):
        combine_into_libsvm_files(dataset, docids[trainids_a1], docids[trainids_a2], svc_labels, 
                                               'training', fold, nfeats, reverse_pairs=True)
     
    problem = svm_read_problem(trainfile) 
    model = svm_train(problem[0], problem[1], '-b 1')
    #model = None
    #call(['touch', tempmodelfile])
    #call([os.path.expanduser('~/libsvm-3.22/svm-train'), ' -b 1 ', trainfile, ' ', tempmodelfile], shell=True)
 
    testfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
    testfile = testfile % (dataset, 'test', fold)    
    if not os.path.isfile(testfile):
        combine_into_libsvm_files(dataset, docids[testids_a1], docids[testids_a2], np.zeros(len(testids_a1)),
                                                'test', fold, nfeats)
        
    #call(['touch', predfile]) 
    #call(['echo', '-b 1', testfile, tempmodelfile, predfile], shell=True)
    #call([os.path.expanduser('~/libsvm-3.22/svm-predict'), '-b 1', testfile, tempmodelfile, predfile], shell=True)
    #proba = np.genfromtxt(predfile, dtype=float)
       
    problem = svm_read_problem(testfile)        
    _, _, proba = svm_predict(problem[0], problem[1], model, '-b 1')
     
    if item_idx_ranktest is not None:
        trainfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
        trainfile = trainfile % (dataset, 'r_training', fold)                
        if not os.path.isfile(trainfile):
            combine_into_libsvm_files(dataset, docids[item_idx_ranktrain], None, rankscores_train, 
                                                   'r_training', fold, nfeats)                
        problem = svm_read_problem(trainfile)
        rank_model = svm_train(problem[0], problem[1], '-s 4 -h 0')
     
        testfile = data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt'
        testfile = testfile % (dataset, 'r_test', fold)                    
        if not os.path.isfile(testfile):
            combine_into_libsvm_files(dataset, docids[item_idx_ranktest], None, np.zeros(len(item_idx_ranktest)), 
                                                   'r_test', fold, nfeats)
         
        problem = svm_read_problem(testfile)
        predicted_f, _, _ = svm_predict(problem[0], problem[1], rank_model)
    else:
        predicted_f = None
    return proba, predicted_f, model 
 
def run_bilstm(fold, model, method, trainids_a1, trainids_a2, prefs_train, items_feat, embeddings, X, ndims, 
             optimize_hyper, testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain=None, 
             rankscores_train=None, item_idx_ranktest=None):     
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
        #row1 = np.array(row1)
        #np.append(row1, X_train2[i])
        row1 = row1 + X_train2[i]
        X_train.append(row1)
    X_test1, X_test2, _, _, _ = folds.get(fold)["test"]
    X_test = []
    for i, row1 in enumerate(X_test1):
        #row1 = np.array(row1)
        #np.append(row1, X_test2[i])
        row1 = row1 + X_test2[i]
        X_test.append(row1)        
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    y_train = np.array(y_train) / 2.0
    print('y_train values: ', np.unique(y_train))
 
    print('Build model...')
    if model is None:
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
     
    #proba = np.zeros(len(prefs_test))
    if item_idx_ranktest is not None:
        X_train = X[item_idx_ranktrain]
        X_test = X[item_idx_ranktest]
     
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
     
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=max_len)
        X_test = sequence.pad_sequences(X_test, maxlen=max_len)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
     
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
        model.fit({'input': X_train, 'output': rankscores_train}, batch_size=batch_size, nb_epoch=nb_epoch)
     
        print('Prediction')
        model_predict = model.predict({'input': X_test}, batch_size=batch_size)
        # print(model_predict)
        predicted_f = np.asarray(model_predict['output']).flatten()                
 
        print('Unique regression predictions: ', np.unique(predicted_f))
    else:
        predicted_f = None
    return proba, predicted_f, model         
        
def run_test(folds, folds_regression, dataset, method, feature_type, embeddings_type=None, embeddings=None, 
             ling_feat_spmatrix=None, docids=None, index_to_word_map=None, subsample_amount=0, default_ls=None, 
             expt_tag='habernal', dataset_increment=1.0, acc=1.0, initial_pair_subset=None):
    # To run the active learning tests, call this function with dataset_increment << 1.0. 
    # To add artificial noise to the data, run with acc < 1.0.
        
        
    # Select output paths for CSV files and final results
    output_filename_template = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/%s' % expt_tag
    output_filename_template += '_%s_%s_%s_%s_acc%.2f_di%.2f' 

    resultsfile = (output_filename_template + '_test.pkl') % (dataset, method, feature_type, embeddings_type, acc, 
                                                              dataset_increment)
    modelfile = (output_filename_template + '_model') %  (dataset, method, feature_type, embeddings_type, acc, 
                                                          dataset_increment)
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
                            X, uids, utexts = get_noisy_fold_data(folds, fold, docids, acc)                            
        
        # ranking folds
        item_idx_ranktrain, rankscores_train, _, item_idx_ranktest, rankscores_test, argids_rank_test = \
        get_fold_regression_data(folds_regression, fold, docids)
        
        items_feat, valid_feats = get_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, 
                                               trainids_a2, uids, embeddings, X, index_to_word_map, utexts)
                  
        ndims = items_feat.shape[1]

        # Subsample training data for active learning or just for debugging purposes------------------------------------
        if subsample_amount > 0:
            subsample_amount, items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,\
               testids_a1, testids_a2, prefs_test, personIDs_test, argids_rank_test, rankscores_test, item_idx_ranktest\
               = subsample_data(subsample_amount, items_feat, trainids_a1, trainids_a2, prefs_train, personIDs_train,
               testids_a1, testids_a2, prefs_test, personIDs_test, argids_rank_test, rankscores_test, item_idx_ranktest)
               
        npairs = len(trainids_a1)
        nseen_so_far = 0               
        if dataset_increment != 1.0:
            nnew_pairs = np.floor(dataset_increment * npairs)
            # select the initial subsample of training pairs
            if initial_pair_subset is None:
                initial_pair_subset = np.random.choice(npairs, nseen_so_far, replace=False)
                pair_subset = initial_pair_subset
        else:
            nnew_pairs = npairs
            pair_subset = np.arange(npairs)                
        
        verbose = True
        optimize_hyper = ('noOpt' not in method)
        
        model = None # initial value
        
        if len(default_ls) > 1:
            ls_initial_guess = default_ls[valid_feats]
        elif '_oneLS' in method:
            ls_initial_guess = np.median(default_ls)
            logging.info("Selecting a single LS for all features: %f" % ls_initial_guess)
        else:
            ls_initial_guess = default_ls          
        
        logging.info("Starting test with method %s..." % (method))
        starttime = time.time()        
        
        # Run the chosen method with active learning simulation if required---------------------------------------------
        while nseen_so_far < npairs:
            # run the method with the current data subset
            if 'SinglePrefGP' in method:
                method_runner_fun = run_gppl
            elif 'GP+SVM' in method:
                method_runner_fun = run_gpsvm
            elif 'SingleGPC' in method:
                method_runner_fun = run_gpc
            elif 'SVM' in method:
                method_runner_fun = run_svm
            elif 'BI-LSTM' in method:
                if feature_type != 'embeddings':
                    logging.error("BI-LSTM can only be run using embedings. Will switch to this feature type...")
                method_runner_fun = run_bilstm
                
            proba, predicted_f, model = method_runner_fun(fold, model, method, trainids_a1[pair_subset], 
                trainids_a2[pair_subset],  prefs_train[pair_subset], items_feat, embeddings, X, ndims, optimize_hyper, 
                testids_a1, testids_a2, ls_initial_guess, verbose, item_idx_ranktrain, rankscores_train, item_idx_ranktest)
                
            # get more data
            nseen_so_far += nnew_pairs
            nnew_pairs = np.floor(dataset_increment * npairs)
            if nseen_so_far > npairs:
                # the last iteration possible
                nnew_pairs = npairs - nseen_so_far
                nseen_so_far = npairs
                
            new_pair_subset = None # replace with an active learning (uncertainty sampling) step
            pair_subset = np.concatenate((pair_subset, new_pair_subset))
        
            endtime = time.time() 
            
            # make it the right shape
            if proba.ndim == 2 and proba.shape[1] > 1:
                proba = proba[:, 1:2]
            elif proba.ndim == 1:
                proba = proba[:, None]
            predictions = np.round(proba)
            
            if predicted_f == 3:
                predicted_f = predicted_f[0]
            if predicted_f.ndim == 1:
                predicted_f = predicted_f[:, None]
                
            logging.info("@@@ Completed running fold %i with method %s, features %s, %i data so far, in %f seconds." % (
                foldidx, method, feature_type, nseen_so_far, endtime-starttime) )
            logging.info("Accuracy for fold = %f" % (np.sum(prefs_test[prefs_test != 1] == 2 * predictions.flatten()[prefs_test != 1]) 
                                              / float(np.sum(prefs_test  != 1))) )            
               
            # Save the data for later analysis ----------------------------------------------------------------------------
            if hasattr(model, 'ls'):
                final_ls[foldidx] = model.ls
            else:
                final_ls[foldidx] = [0]        
            
            # Outputs from the tested method
            if foldidx not in all_proba:
                all_proba[foldidx] = proba
                all_predictions[foldidx] = predictions
                all_f[foldidx] = predicted_f
            else:
                all_proba[foldidx] = np.concatenate((all_proba[foldidx], proba), axis=1)
                all_predictions[foldidx] = np.concatenate((all_predictions[foldidx], predictions), axis=1)
                all_f[foldidx] = np.concatenate((all_f[foldidx], predicted_f), axis=1)
            
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
                
    return initial_pair_subset
            
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
    global index_to_word_map
    global ling_feat_spmatrix
    global docids
    global siamese_cbow_embeddings
    global skipthoughts_model
    global pair_subset
    global default_ls_values
        
    if 'dataset' not in globals():
        dataset = ''
                      
    if 'folds' in globals() and dataset == datasets[0] and 'word_embeddings' in globals():
        load_data = False
    else:
        load_data = True
    
    if 'default_ls_values' not in globals():
        default_ls_values = {}
        
    if 'pair_subset' not in globals():
        pair_subset = None 
          
    for method in methods:
            
        for dataset in datasets:
            if load_data:
                folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = \
                load_train_test_data(dataset)
                if 'word_mean' in embeddings_types:
                    word_embeddings = load_embeddings(word_index_to_embeddings_map)
                else:
                    word_embeddings = None
                    
                if 'skipthoughts' in embeddings_types:
                    skipthoughts_model = load_skipthoughts_embeddings(word_to_indices_map)
                else:
                    skipthoughts_model = None
                    
                if 'siamese-cbow' in embeddings_types:
                    siamese_cbow_embeddings = load_siamese_cbow_embeddings(word_to_indices_map)
                else:
                    siamese_cbow_embeddings = None
                    
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

                    if embeddings_type == 'word_mean':
                        embeddings = word_embeddings
                    elif embeddings_type == 'skipthoughts':
                        embeddings = skipthoughts_model
                    elif embeddings_type == 'siamese-cbow':
                        embeddings = siamese_cbow_embeddings
                    else:
                        embeddings = None  
                                            
                    if dataset in default_ls_values and feature_type in default_ls_values[dataset] and \
                            embeddings_type in default_ls_values[dataset][feature_type]:
                        default_ls_value = default_ls_values[dataset][feature_type][embeddings_type]
                    elif 'GP' in method:
                        default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings,
                                             ling_feat_spmatrix, docids, folds, index_to_word_map)
                        if dataset not in default_ls_values:
                            default_ls_values[dataset] = {}
                        if feature_type not in default_ls_values[dataset]:
                            default_ls_values[dataset][feature_type] = {}
                        default_ls_values[dataset][feature_type][embeddings_type] = default_ls_value
                    else:
                        default_ls_value = []
                     
                    pair_subset = run_test_fun(folds, folds_regression, dataset, method, feature_type, embeddings_type, 
                        embeddings, ling_feat_spmatrix, docids, index_to_word_map, subsample_amount=0, 
                        default_ls=default_ls_value, dataset_increment=dataset_increment, acc=acc, 
                        initial_pair_subset=pair_subset)
                    
                    logging.info("**** Completed: method %s with features %s, embeddings %s ****" % (method, feature_type, 
                                                                                           embeddings_type) )
    return default_ls_values
        
if __name__ == '__main__':
    acc = 1.0
    dataset_increment = 1.0
    
    
# # Issue #33: rerun * kernel combination with both features and weak s prior.
    datasets = ['UKPConvArgStrict']
    methods = ['SinglePrefGP_noOpt_weaksprior']
    feature_types = ['embeddings']
    #embeddings_types = ['skipthoughts']
    #default_ls_values = run_test_set()
#     methods = ['SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_noOpt', 'SinglePrefGP_noOpt_lowsprior', 
#                'SinglePrefGP_noOpt_additive_lowsprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#  
#      default_ls_values = run_test_set() # Run on Friday already.
#         
    #compute_metrics(methods, datasets, feature_types, embeddings_types, tag='33ST')    

    embeddings_types = ['word_mean']#['siamese-cbow']
    default_ls_values = run_test_set()
    compute_metrics(methods, datasets, feature_types, embeddings_types, tag='33SCB')

# # Issue #34 Compare kernel operators
#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#  
#     default_ls_values = run_test_set()
#     
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='34')
   
# # # Issue #35 Best setup with other datasets
#     datasets = ['UKPConvArgAll_evalMACE'] #'UKPConvArgMACE', 
#     methods = ['SinglePrefGP_noOpt_weaksprior']#, 'SinglePrefGP_noOpt_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
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

# #  GPC. The GPC method should be run on Barney as it needs more memory.
#     datasets = ['UKPConvArgStrict']
#     methods = ['SingleGPC']
#     feature_types = ['embeddings'] # we run with ling as well because this is how SVM was originally run by IH.
#     embeddings_types = ['word_mean']
#          
#     default_ls_values = run_test_set()
#          
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='37')

# currently running --------------
# Issue #38, Run SVM on other datasets and compute missing metrics
#     datasets = ['UKPConvArgStrict', 'UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
#     methods = ['SVM']
#     feature_types = ['both'] # it says both, but it actually only uses linguistic features
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#     default_ls_values = run_test_set() 
#             
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='38')
#   
# # Issue #36 Optimize best setup
#     datasets = ['UKPConvArgStrict', 'UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
#     methods = ['SinglePrefGP_weaksprior', 'SinglePrefGP_additive_weaksprior'] 
#     feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both'
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#          
#     default_ls_values = run_test_set() 
#             
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='36')
# ------------------------------

# # Issue #39, Run BILSTM on other datasets and compute missing metrics
#     datasets = ['UKPConvArgMACE', 'UKPConvArgAll_evalMACE']
#     methods = ['BI-LSTM']
#     feature_types = ['embeddings']
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#     default_ls_values = run_test_set() 
#          
#     compute_metrics(methods, datasets, feature_types, embeddings_types, tag='38')


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
#     embeddings_types = ['word_mean']#, 'skipthoughts', 'siamese-cbow']
#  
#     default_ls_values = run_test_set(run_noise_sparsity_test)
#     
#     #TODO: the plotting and metrics for the noise/sparsity tests.
