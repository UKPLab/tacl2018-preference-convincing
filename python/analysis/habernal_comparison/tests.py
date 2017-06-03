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

sys.path.append('../../git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python')

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")

sys.path.append(data_root_dir + '/embeddings/Siamese-CBOW/siamese-cbow')
sys.path.append(data_root_dir + '/embeddings/skip-thoughts')

import pickle
from data_loader import load_my_data_separate_args
from data_loader_regression import load_my_data as load_my_data_regression
import numpy as np
import time
from sklearn.datasets import load_svmlight_file
import logging
logging.basicConfig(level=logging.DEBUG)
from preference_features import PreferenceComponents
from gp_pref_learning import GPPrefLearning, pref_likelihood
from gp_classifier_svi import GPClassifierSVI
from sklearn.svm import SVR 
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV
#import skipthoughts
#import wordEmbeddings as siamese_cbow
from joblib import Parallel, delayed
import multiprocessing

max_len = 300  # cut texts after this number of words (among top max_features most common words)

#from keras.preprocessing import sequence#
# Copied from the above import to avoid installing additional dependencies
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def combine_lines_into_one_file(dataset, dirname=data_root_dir + '/lingdata/UKPConvArg1-Full-libsvm', 
        outputfile=data_root_dir + '/lingdata/%s-libsvm.txt'): 
    output_argid_file = outputfile % ("argids_%s" % dataset)
    outputfile = outputfile % dataset
    
    outputstr = ""
    dataids = [] # contains the argument document IDs in the same order as in the ouputfile and outputstr
    
    if os.path.isfile(outputfile):
        os.remove(outputfile)
        
    with open(outputfile, 'a') as ofh: 
        for filename in os.listdir(dirname):
            fid = filename.split('.')[0]
            print "writing from file %s with row ID %s" % (filename, fid)
            with open(dirname + "/" + filename) as fh:
                lines = fh.readlines()
            for line in lines:
                dataids.append(fid)
                outputline = line
                ofh.write(outputline)
                outputstr += outputline + '\n'
                
    if os.path.isfile(output_argid_file):
        os.remove(outputfile)
        
    dataids = np.array(dataids)[:, np.newaxis]
    np.savetxt(output_argid_file, dataids, '%s')
                
    return outputfile, outputstr, dataids   

def load_train_test_data(dataset):
    # Set experiment options and ensure CSV data is ready -------------------------------------------------------------
    
    folds_test = None # can load a separate set of data for testing into here, e.g. train on workers and test on gold standard
    
    # Select the directory containing original XML files with argument data + crowdsourced annotations.
    # See the readme in the data folder from Habernal 2016 for further explanation of folder names.    
    if dataset == 'UKPConvArgAll':
        # basic dataset for UKPConvArgAll, which requires additional steps to produce the other datasets        
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'  
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArgAllRank-CSV/'
    elif dataset == 'UKPConvArgMACE':        
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArg1-Ranking-CSV/'          
    elif dataset == 'UKPConvArgStrict':
        dirname = data_root_dir + 'argument_data/UKPConvArg1Strict-XML/'
        ranking_csvdirname = None        
    # these are not valid labels because ranking data is produced as part of other experiments        
    elif dataset == 'UKPConvArgAllR':
        dirname = None # don't need to create a new CSV file
        raise Exception('This dataset cannot be used to select an experiment. To test ranking, run with \
        dataset=UKPConvArgAll')        
    elif dataset == 'UKPConvArgRank':
        dirname = None # don't need to create a new CSV file
        raise Exception('This dataset cannot be used to select an experiment. To test ranking, run with \
        dataset=UKPConvArgMACE')
    elif dataset == 'UKPConvArgAll_evalMACE': # train on the All datasets and evaluate on the MACE dataset -- the pref
        # learning method does the combination so we can see how that compares to MACE -- can we skip those steps?
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'  
        ranking_csvdirname = None
        folds_test, folds_regression, _, _ = load_train_test_data('UKPConvArgMACE')
        dataset = 'UKPConvArgAll'
    else:
        raise Exception("Invalid dataset %s" % dataset)    
    
    print "Data directory = %s, dataset=%s" % (dirname, dataset)    
    csvdirname = data_root_dir + 'argument_data/%s-CSV/' % dataset
    # Generate the CSV files from the XML files. These are easier to work with! The CSV files from Habernal do not 
    # contain all turker info that we need, so we generate them afresh here.
    if not os.path.isdir(csvdirname):
        print("Writing CSV files...")
        os.mkdir(csvdirname)
        if dataset == 'UKPConvArgAll':
            generate_turker_CSV(dirname, csvdirname) # select all labels provided by turkers
        elif dataset == 'UKPConvArgStrict' or dataset == 'UKPConvArgMACE':
            generate_gold_CSV(dirname, csvdirname) # select only the gold labels
                
    embeddings_dir = data_root_dir + '/embeddings/'
    print "Embeddings directory: %s" % embeddings_dir
    
    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print('Loading train/test data from %s...' % csvdirname)
    folds, word_index_to_embeddings_map, word_to_indices_map = load_my_data_separate_args(csvdirname, 
                                                                                          embeddings_dir=embeddings_dir)
    if ranking_csvdirname is not None:             
        folds_regression, _ = load_my_data_regression(ranking_csvdirname, load_embeddings=False)
    else:
        folds_regression = None
        
    if folds_test is not None:
        for fold in folds:
            folds[fold]["test"] = folds_test[fold]["test"]

    return folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map
    
def load_embeddings(word_index_to_embeddings_map):
    print('Loading embeddings')
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.zeros((1 + np.max(word_index_to_embeddings_map.keys()), len(word_index_to_embeddings_map.values()[0])))
    embeddings[word_index_to_embeddings_map.keys()] = word_index_to_embeddings_map.values()
    #embeddings = np.asarray([np.array(x, dtype=np.float32) for x in word_index_to_embeddings_map.values()])
    return embeddings

# def load_siamese_cbow_embeddings(word_to_indices_map):
#     print('Loading Siamese CBOW embeddings...')
#     filename = os.path.expanduser('~/data/embeddings/Siamese-CBOW/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle')
#     return siamese_cbow.wordEmbeddings(filename)
#      
# def load_skipthoughts_embeddings(word_to_indices_map):
#     print('Loading Skip-thoughts model...')
#     model = skipthoughts.load_model()
#     return model
     
def load_ling_features(dataset):
    ling_dir = data_root_dir + 'lingdata/'
    print "Looking for linguistic features in directory %s" % ling_dir    
    print('Loading linguistic features')
    ling_file = ling_dir + "/%s-libsvm.txt" % dataset
    argids_file = ling_dir + "/%s-libsvm.txt" % ("argids_%s" % dataset)
    if not os.path.isfile(ling_file) or not os.path.isfile(argids_file):
        ling_file, _ , docids = combine_lines_into_one_file(dataset, outputfile=ling_dir+"/%s-libsvm.txt")
    else:
        docids = np.genfromtxt(argids_file, str)
        
    ling_feat_spmatrix, _ = load_svmlight_file(ling_file)
    return ling_feat_spmatrix, docids
    
def _dists_f(items_feat_sample, f):
    if np.mod(f, 100) == 0:
        print 'computed lengthscale for feature %i' % f                
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
            print "invalid embeddings type! %s" % embeddings_type
        
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
    print 'I am using a heuristic multiplier for the length-scale because median is too low to work in high-D spaces'
    #ls_initial_guess *= 1000 # this worked reasonably well but was plucked out of thin air
    ls_initial_guess *= items_feat.shape[1] # this is a heuristic, see e.g. "On the High-dimensional 
    #Power of Linear-time Kernel Two-Sample Testing under Mean-difference Alternatives" by Ramdas et al. 2014. In that 
    # paper they refer to root(no. dimensions) because they square the lengthscale in the kernel function.
            
    endtime = time.time()
    print '@@@ Selected initial lengthscales in %f seconds' % (endtime - starttime)
    
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
    
    return trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, X, uids
    
def run_test(folds, folds_regression, dataset, method, feature_type, embeddings_type=None, embeddings=None, 
             siamese_cbow_e=None, skipthoughts_model=None, ling_feat_spmatrix=None, docids=None, subsample_amount=0, 
             ls_initial_guess=None):
        
    # Select output paths for CSV files and final results
    output_filename_template = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/habernal_%s_%s_%s_%s'

    resultsfile = (output_filename_template + '_test.pkl') % (dataset, method, feature_type, embeddings_type)
    modelfile = (output_filename_template + '_model') %  (dataset, method, feature_type, embeddings_type) 
    modelfile += '_fold%i.pkl'
    
    if not os.path.isdir(data_root_dir + 'outputdata'):
        os.mkdir(data_root_dir + 'outputdata')
    if not os.path.isdir(data_root_dir + 'outputdata/crowdsourcing_argumentation_expts'):
        os.mkdir(data_root_dir + 'outputdata/crowdsourcing_argumentation_expts')
                
    all_proba = {}
    all_predictions = {}
    all_f = {}
    
    all_target_prefs = {}
    all_target_rankscores = {}
    all_argids_rankscores = {}
    all_turkids_rankscores = {}
    final_ls = {}
    
    item_ids = {}
    times = {}
    
    for foldidx, fold in enumerate(folds.keys()):
        # Get data for this fold --------------------------------------------------------------------------------------
        print("Fold name ", fold)
        trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test,\
                                                                        X, uids = get_fold_data(folds, fold, docids)
        
        # ranking folds
        if folds_regression is not None:
            _, rankscores_test, argids_rank_test, turkIDs_rank_test = folds_regression.get(fold)["test"]
            item_idx_ranktest = np.array([np.argwhere(testid==docids)[0][0] for testid in argids_rank_test])
            rankscores_test = np.array(rankscores_test)
            argids_rank_test = np.array(argids_rank_test)
        
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        if feature_type == 'both' or feature_type == 'embeddings':
            print "Converting texts to mean embeddings (we could use a better sentence embedding?)..."
            if embeddings_type == 'word_mean':
                items_feat = get_mean_embeddings(embeddings, X)
#             elif embeddings_type == 'skipthoughts':
#                 items_feat = skipthoughts.encode(skipthoughts_model, X)
#             elif embeddings_type == 'siamese_cbow':
#                 items_feat = np.array([siamese_cbow_e.getAggregate(Xi) for Xi in X])
            else:
                print "invalid embeddings type! %s" % embeddings_type
            print "...embeddings loaded."
            # trim away any features not in the training data because we can't learn from them
            valid_feats = (np.sum(items_feat[trainids_a1] != 0, axis=0)>0) & (np.sum(items_feat[trainids_a2] != 0, 
                                                                                     axis=0)>0)
            items_feat = items_feat[:, valid_feats]
            
            
        elif feature_type == 'ling':
            items_feat = np.zeros((X.shape[0], 0))
            valid_feats = np.zeros(0)
            
        if feature_type == 'both' or feature_type == 'ling':
            print "Obtaining linguistic features for argument texts."
            # trim the features that are not used in training
            valid_feats_ling = ((np.sum(ling_feat_spmatrix[trainids_a1, :] != 0, axis=0)>0) & 
                           (np.sum(ling_feat_spmatrix[trainids_a2, :] != 0, axis=0)>0)).nonzero()[1]            
            ling_feat_spmatrix = ling_feat_spmatrix[:, valid_feats_ling]
            items_feat = np.concatenate((items_feat, ling_feat_spmatrix[uids, :].toarray()), axis=1)
            print "...loaded all linguistic features for training and test data."
            
            valid_feats = np.concatenate((valid_feats, valid_feats_ling)).astype(int)
            
        ls_initial_guess = ls_initial_guess[valid_feats]
        if '_oneLS' in method:
            ls_initial_guess = np.median(ls_initial_guess)
            print "Selecting a single LS for all features: %f" % ls_initial_guess        
                
        prefs_train = np.array(prefs_train) 
        prefs_test = np.array(prefs_test)     
        personIDs_train = np.array(personIDs_train)
        personIDs_test = np.array(personIDs_test) 
          
        ndims = items_feat.shape[1]

        # subsample training data for debugging purposes only ----------------------------------------------------------
        if subsample_amount > 0:
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
            
            if folds_regression is not None:
                argids_rank_test = argids_rank_test[item_idx_ranktest < subsample_amount]
                rankscores_test = rankscores_test[item_idx_ranktest < subsample_amount]
                item_idx_ranktest = item_idx_ranktest[item_idx_ranktest < subsample_amount]
                
        # Run the chosen method ---------------------------------------------------------------------------------------
        print "Starting test with method %s..." % (method)
        starttime = time.time()
                    
        personIDs = np.concatenate((personIDs_train, personIDs_test))
        _, personIdxs = np.unique(personIDs, return_inverse=True)
        personIDs_train = personIdxs[:len(personIDs_train)]
        personIDs_test = personIdxs[len(personIDs_train):]
        
        verbose = True
        optimize_hyper = ('noOpt' not in method)
        nfactors = 10
        
        predicted_f = None
        
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
            model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial_guess, verbose=verbose, 
                        shape_s0 = 2.0, rate_s0 = 200.0,  
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, ninducing=500, max_update_size=200)
            model.max_iter_VB = 500
            model.fit(trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, input_type='zero-centered')            
        
            proba, _ = model.predict(testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f, _ = model.predict_f(items_feat[item_idx_ranktest]) 

        elif 'GP+SVM' in method:
            model = GPPrefLearning(ninput_features=1, ls_initial=ls_initial_guess, verbose=verbose, 
                        shape_s0 = 2.0, rate_s0 = 200.0,  
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=False, kernel_func='diagonal')
            model.max_iter_VB = 2
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
                predicted_f, _ = svm.predict(items_feat[item_idx_ranktest])  
            
        elif 'SingleGPC' in method:
            # twice as many features means the lengthscale heuristic is * 2
            model = GPClassifierSVI(ninput_features=ndims, ls_initial=np.concatenate((ls_initial_guess * 2.0, 
                                                                                       ls_initial_guess * 2.0)), 
                         verbose=verbose, shape_s0 = 2.0, rate_s0 = 200.0,  
                         rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, ninducing=500, max_update_size=200)            
            
            # with the argument order swapped around and data replicated:

            gpc_feats = np.concatenate((np.concatenate((items_feat[trainids_a1], items_feat[trainids_a2]), axis=1),
                                    np.concatenate((items_feat[trainids_a2], items_feat[trainids_a1]), axis=1)), axis=0)
            gpc_labels = np.concatenate((np.array(prefs_train, dtype=float) * 0.5,
                                          1 - np.array(prefs_train, dtype=float) * 0.5))
 
            # this didn't work as well on a single fold, but has not been thoroughly tested.
#             gpc_feats = np.concatenate((items_feat[trainids_a1], items_feat[trainids_a2]), axis=1)
#             gpc_labels = np.array(prefs_train, dtype=float) * 0.5
            model.max_iter_VB = 500            
            model.fit(np.arange(len(trainids_a1)), gpc_labels, optimize=optimize_hyper, features=gpc_feats)            
        
            proba, _ = model.predict(np.concatenate((items_feat[testids_a1], items_feat[testids_a2]), axis=1))
            if folds_regression is not None:
                predicted_f = np.zeros(len(item_idx_ranktest)) # can't easily rank with this method
        
        final_ls[foldidx] = model.ls    
        predictions = np.round(proba)
        
        endtime = time.time() 
        
        print "@@@ Completed running fold %i with method %s in %f seconds." % (foldidx, method, endtime-starttime)
        print "Accuracy for fold = %f" % (np.sum(prefs_test[prefs_test != 1] == 2 * predictions.flatten()[prefs_test != 1]) 
                                          / float(np.sum(prefs_test  != 1)))
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
            all_turkids_rankscores[foldidx] = turkIDs_rank_test
        
        # Save the time taken
        times[foldidx] = endtime-starttime
        
        results = (all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, ls_initial_guess,
                   times, final_ls) 
        with open(resultsfile, 'w') as fh:
            pickle.dump(results, fh)
            
        #with open(modelfile % foldidx, 'w') as fh:
        #    pickle.dump(model, fh)
            
    return model

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
        
if __name__ == '__main__':
    datasets = ['UKPConvArgStrict']#, 'UKPConvArgMACE', 'UKPConvArgAll_evalMACE'] #  this has already been run
    methods = ['SingleGPC_noOpt', 'GP+SVM_noOpt'] # Desktop-169 'SinglePrefGP_noOpt'
    #methods = ['SinglePrefGP', 'SingleGPC'] # Barney ('GP+SVM' is not possible with optimization on)
    #methods = ['GP+SVM_noOpt'] # debugging  
    
    feature_types = ['both', 'ling', 'embeddings', ] # can be 'embeddings' or 'ling' or 'both'
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
                            
                    model = run_test(folds, folds_regression, dataset, method, 
                        feature_type, embeddings_type, word_embeddings, siamese_cbow_embeddings, 
                        skipthoughts_model, ling_feat_spmatrix, docids, subsample_amount=0, 
                        ls_initial_guess=default_ls_value)
                    
                    print "**** Completed: method %s with features %s, embeddings %s ****" % (method, feature_type, 
                                                                                           embeddings_type)

# TODO: run the experiments with SinglePrefGP, GP Classifier, SinglePrefGP with no optimization. 
# Run with strict and MACE datasets to compare to Habernal + to show handling of conflicts.
# TODO: later run with All dataset to show learning from noisy data.
# TODO: later run with GP + SVM to show whether the pref learning part is helpful, even if feature regression is poor