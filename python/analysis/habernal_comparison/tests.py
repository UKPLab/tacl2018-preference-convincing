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

import pickle
from data_loader import load_my_data_separate_args
from data_loader_regression import load_my_data as load_my_data_regression
import sys, os
import numpy as np
from sklearn.metrics import accuracy_score
import time
from scipy import sparse
from sklearn.datasets import load_svmlight_file

import logging
logging.basicConfig(level=logging.DEBUG)

from preference_features import PreferenceComponents
from gp_pref_learning import GPPrefLearning
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV

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


def combine_lines_into_one_file(dataset, dirname="/home/local/UKP/simpson/data/outputdata/UKPConvArg1-Full-libsvm", 
        outputfile="/home/local/UKP/simpson/git/crowdsourcing_argumentation/data/lingdata/%s-libsvm.txt"): 
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
                
    return outputfile, outputstr, np.array(dataids)   
    
def run_test(dataset, method, feature_type, subsample_amount=0):
    
    # Set experiment options and ensure CSV data is ready -------------------------------------------------------------
    # Select the directory containing original XML files with argument data + crowdsourced annotations.
    # See the readme in the data folder from Habernal 2016 for further explanation of folder names.
         
    if dataset == 'UKPConvArgAll':
        # basic dataset for UKPConvArgAll, which requires additional steps to produce the other datasets        
        dirname = '../../git/acl2016-convincing-arguments/data/UKPConvArg1-full-XML/'  
        ranking_csvdirname = './data/UKPConvArgAllRank/'
    elif dataset == 'UKPConvArgMACE':        
        dirname = '../../git/acl2016-convincing-arguments/data/UKPConvArg1-full-XML/'
        ranking_csvdirname = '/home/local/UKP/simpson/git/acl2016-convincing-arguments/data/UKPConvArg1-Ranking-CSV/'          
    elif dataset == 'UKPConvArgStrict':
        dirname = '../../git/acl2016-convincing-arguments/data/UKPConvArg1Strict-XML/'
        ranking_csvdirname = '/home/local/UKP/simpson/git/acl2016-convincing-arguments/data/UKPConvArg1-Ranking-CSV/'        
    # these are not valid labels because ranking data is produced as part of other experiments        
    elif dataset == 'UKPConvArgAllR':
        dirname = None # don't need to create a new CSV file
        raise Exception('This dataset cannot be used to select an experiment. To test ranking, run with \
        dataset=UKPConvArgAll')        
    elif dataset == 'UKPConvArgRank':
        dirname = None # don't need to create a new CSV file
        raise Exception('This dataset cannot be used to select an experiment. To test ranking, run with \
        dataset=UKPConvArgMACE or dataset=UKPConvArgStrict')
    else:
        raise Exception("Invalid dataset %s" % dataset)    
    
    print "Data directory = %s, dataset=%s" % (dirname, dataset)
    
    # Select output paths for CSV files and final results

    resultsfile = './results/habernal_%s_%s_%s_test.pkl' % (dataset, method, feature_type) 
    csvdirname = './data/%s-CSV/' % dataset
        
    # Generate the CSV files from the XML files. These are easier to work with! The CSV files from Habernal do not 
    # contain all turker info that we need, so we generate them afresh here.
    print("Writing CSV files...")
    if not os.path.isdir(csvdirname):
        os.mkdir(csvdirname)
        if dataset == 'UKPConvArgAll':
            generate_turker_CSV(dirname, csvdirname) # select all labels provided by turkers
        elif dataset == 'UKPConvArgStrict' or dataset == 'UKPConvArgMACE':
            generate_gold_CSV(dirname, csvdirname) # select only the gold labels
        
    embeddings_dir = '../../git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python/'
    if not feature_type == 'ling':
        print "Looking for embeddings in directory %s" % embeddings_dir

    # Select linguistic features file
    ling_dir = './data/lingdata/'
    if not feature_type == 'embeddings':
        print "Looking for linguistic features in directory %s" % ling_dir    
    
    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print('Loading train/test data...')
    folds, word_index_to_embeddings_map = load_my_data_separate_args(csvdirname, embeddings_dir=embeddings_dir)             
    folds_regression, _ = load_my_data_regression(ranking_csvdirname, load_embeddings=False)
    
    if feature_type == 'both' or feature_type == 'embeddings':
        print('Loading embeddings')
        # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
        embeddings = np.asarray([np.array(x, dtype=np.float32) for x in word_index_to_embeddings_map.values()])
        
    if feature_type == 'both' or feature_type == 'ling':
        print('Loading linguistic features')
        ling_file = ling_dir + "/%s-libsvm.txt" % dataset
        if not os.path.isfile(ling_file):
            ling_file, _ , docids = combine_lines_into_one_file(dataset, outputfile=ling_dir+"/%s-libsvm.txt")
        else:
            dataids = []
            for filename in os.listdir("/home/local/UKP/simpson/data/outputdata/UKPConvArg1-Full-libsvm"):
                fid = filename.split('.')[0]
                dataids.append(fid)
            docids = np.array(dataids)
            
        ling_feat_spmatrix, _ = load_svmlight_file(ling_file)
    
    # Run test --------------------------------------------------------------------------------------------------------
    all_proba = {}
    all_predictions = {}
    all_f = {}
    
    all_target_prefs = {}
    all_target_rankscores = {}
    all_argids_rankscores = {}
    all_turkids_rankscores = {}
    length_scales = {}
    
    latent_item_features = {}
    latent_p_features = {}
    person_noise_var = {}
    item_means = {}
    people = {}
    
    item_coords = {}
    item_ids = {}
    times = {}
    
    for foldidx, fold in enumerate(folds.keys()):
        # Get data for this fold --------------------------------------------------------------------------------------
        print("Fold name ", fold)
        X_train_a1, X_train_a2, prefs_train, ids_train, personIDs_train = folds.get(fold)["training"]
        X_test_a1, X_test_a2, prefs_test, ids_test, personIDs_test = folds.get(fold)["test"]
        
        trainids = np.array([ids_pair.split('_') for ids_pair in ids_train])
        trainids_a1 = [np.argwhere(trainid==docids)[0][0] for trainid in trainids[:, 0]]
        trainids_a2 = [np.argwhere(trainid==docids)[0][0] for trainid in trainids[:, 1]]
        
        testids = np.array([ids_pair.split('_') for ids_pair in ids_test])
        testids_a1 = [np.argwhere(testid==docids)[0][0] for testid in testids[:, 0]]
        testids_a2 = [np.argwhere(testid==docids)[0][0] for testid in testids[:, 1]]        

        print("Training instances ", len(X_train_a1), " training labels ", len(prefs_train))
        print("Test instances ", len(X_test_a1), " test labels ", len(prefs_test))
        
        # ranking folds
        _, rankscores_test, argids_rank_test, turkIDs_rank_test = folds_regression.get(fold)["test"]
        
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        # TODO: allow alternatives to mean embeddings for sentences/documents
        if feature_type == 'both' or feature_type == 'embeddings':
            print "Converting texts to mean embeddings (we could use a better sentence embedding?)..."
            items_1_train = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_train_a1])
            items_2_train = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_train_a2])
            print "...training data embeddings done."
            items_1_test = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_test_a1])
            items_2_test = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_test_a2])
            print "...test data embeddings done."  
        elif feature_type == 'ling':
            # initialise the coordinates objects 
            items_1_train = np.zeros((X_train_a1.shape[0], 0))
            items_2_train = np.zeros((X_train_a2.shape[0], 0))
            items_1_test = np.zeros((X_test_a1.shape[0], 0))
            items_2_test = np.zeros((X_test_a2.shape[0], 0))
            
        if feature_type == 'both' or feature_type == 'ling':
            items_1_train = sparse.csr_matrix(items_1_train)
            items_2_train = sparse.csr_matrix(items_2_train)
            items_1_train = sparse.csr_matrix(items_1_train)
            items_2_test = sparse.csr_matrix(items_2_test)
            
            print "Obtaining linguistic features for argument texts."
            items_1_train = sparse.hstack((items_1_train, ling_feat_spmatrix[trainids_a1, :]), format='csr')
            items_2_train = sparse.hstack((items_2_train, ling_feat_spmatrix[trainids_a2, :]), format='csr')
            items_1_test = sparse.hstack((items_1_test, ling_feat_spmatrix[testids_a1, :]), format='csr')
            items_2_test = sparse.hstack((items_2_test, ling_feat_spmatrix[testids_a2, :]), format='csr')
            print "...loaded all linguistic features for training and test data."
                
        prefs_train = np.array(prefs_train) 
        prefs_test = np.array(prefs_test)     
        personIDs_train = np.array(personIDs_train)
        personIDs_test = np.array(personIDs_test) 
        
        # subsample training data for debugging purposes only
        if subsample_amount > 0:
            subsample = np.arange(subsample_amount)               
                    
            #personIDs_train = np.zeros(len(Xe_train1), dtype=int)[subsample, :] #
            items_1_train = items_1_train[subsample, :]
            items_2_train = items_2_train[subsample, :]
            prefs_train = prefs_train[subsample]
            personIDs_train = personIDs_train[subsample]
                    
            # subsampled test data for debugging purposes only
            #personIDs_test = np.zeros(len(items_1_test), dtype=int)[subsample, :]
            personIDs_test = personIDs_test[subsample]
            items_1_test = items_1_test[subsample, :]
            items_2_test = items_2_test[subsample, :]
            prefs_test = prefs_test[subsample]
        
        # Run the chosen method ---------------------------------------------------------------------------------------
        print "Starting test with method %s..." % (method)
        starttime = time.time()
        
        if sparse.issparse(items_1_train):            
            valid_feats = ((np.sum(items_1_train, axis=0)>0) & (np.sum(items_2_train, axis=0)>0)).nonzero()[1]
            items_1_train = items_1_train[:, valid_feats].toarray()
            items_2_train = items_2_train[:, valid_feats].toarray()
            items_1_test = items_1_test[:, valid_feats].toarray()
            items_2_test = items_2_test[:, valid_feats].toarray()
            
        personIDs = np.concatenate((personIDs_train, personIDs_test))
        _, personIdxs = np.unique(personIDs, return_inverse=True)
        personIDs_train = personIdxs[:len(personIDs_train)]
        personIDs_test = personIdxs[len(personIDs_train):]

        ndims = items_1_train.shape[1]
        ls_initial_guess = (np.std(items_1_train, axis=0) + np.std(items_2_train, axis=0)) / 2.0
        
        verbose = True
        optimize_hyper = False
        
        # Run the selected method
        if method == 'PersonalisedPrefsBayes':        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=10, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False)
            model.fit(personIDs_train, items_1_train, items_2_train, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba, predicted_f = model.predict(personIDs_test, items_1_test, items_2_test)
                        
        elif method == 'PersonalisedPrefsFA':
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=10, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=True)
            model.fit(personIDs_train, items_1_train, items_2_train, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba, predicted_f = model.predict(personIDs_test, items_1_test, items_2_test)
                        
        elif method == 'SinglePrefGP':
            model = GPPrefLearning(nitem_features=ndims, ls_initial=ls_initial_guess, verbose=verbose, 
                                                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True)
            model.fit(items_1_train, items_2_train, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, input_type='zero-centered')            
        
            proba = model.predict(items_1_test, items_2_test)
            predicted_f = [model.f, model.output_coords]
            
        predictions = np.round(proba * 2)
        
        endtime = time.time()
        
        print "Completed running the test with method %s in %f seconds." % (method, endtime-starttime)
        endtime-starttime
        # Save the data for later analysis ----------------------------------------------------------------------------
        # Outputs from the tested method
        all_proba[foldidx] = proba
        all_predictions[foldidx] = predictions
        all_f[foldidx] = predicted_f
        
        if method == 'PersonalisedPrefsBayes':
            length_scales[foldidx] = [model.ls, model.lsy]
            latent_item_features[foldidx] = model.w
            latent_p_features[foldidx] = model.y
            item_means[foldidx] = model.t

            person_noise_var_fold = np.zeros(model.Npeople)
            for p in model.people:
                person_noise_var_fold[p] = 1.0 / model.pref_gp[p].s
            person_noise_var[foldidx] = person_noise_var_fold
            people[foldidx] = model.people
        elif method == 'PersonalisedPrefsFA':
            length_scales[foldidx] = [model.ls, -1]

            person_noise_var_fold = np.zeros(model.Npeople)
            for p in model.people:
                person_noise_var_fold[p] = 1.0 / model.pref_gp[p].s
            person_noise_var[foldidx] = person_noise_var_fold        
            people[foldidx] = model.people
        elif method == 'SinglePrefGP':
            length_scales[foldidx] = [model.ls, -1]

        item_coords[foldidx] = model.obs_coords
        
        ids_train_sep = [[item_id.split('_')[0], item_id.split('_')[1]] for item_id in ids_train]
        item_ids[foldidx] = np.array(ids_train_sep).T.flatten()[model.obs_uidxs]
                
        # Save the ground truth
        all_target_prefs[foldidx] = prefs_test
        all_target_rankscores[foldidx] = rankscores_test
        all_argids_rankscores[foldidx] = argids_rank_test
        all_turkids_rankscores[foldidx] = turkIDs_rank_test
        
        # Save the time taken
        times[foldidx] = endtime-starttime
        
        results = (all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, length_scales, 
               latent_item_features, latent_p_features, person_noise_var, people, item_coords, item_ids, times)
        with open(resultsfile, 'w') as fh:
            pickle.dump(results, fh)
            
        # Compute metrics ---------------------------------------------------------------------------------------------
        
        prefs_test = np.array(prefs_test, dtype=float)
        acc = accuracy_score(prefs_test, predictions)
        print('Test accuracy:', acc)            
        
if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         dirname = sys.argv[1]
#     if len(sys.argv) > 2:
#         resultsfile = sys.argv[2] % ('habernal_%s_embeddings_test.pkl' % dataset)
#         csvdirname = sys.argv[2] % ('%s-CSV' % dataset)              
#     if len(sys.argv) > 3:
#         method = sys.argv[3] 
    # Select type of features to use for the test
#     if len(sys.argv) > 4:
#         feature_type = sys.argv[4]
    # Select word embeddings file
#     if len(sys.argv) > 5:
#         embeddings_dir = sys.argv[5]
#     if len(sys.argv) > 6:
#         ling_dir = sys.argv[6]

    datasets = ['UKPConvArgAll', 'UKPConvArgMACE', 'UKPConvArgStrict'] 
    methods = ['PersonalisedPrefsBayes', 'PersonalisedPrefsFA', 'SinglePrefGP']  
    feature_types = ['both', 'embeddings', 'ling'] # can be 'embeddings' or 'ling'
          
    for dataset in datasets:
        for method in methods: 
            for feature_type in feature_types:
                run_test(dataset, method, feature_type, subsample_amount=0)        