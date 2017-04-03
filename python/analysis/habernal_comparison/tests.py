'''
Script for comparing our Bayesian preference learning approach with the results from Habernal 2016. 

Tasks:

1. Load word embeddings for the original text data that were used in the NN approach in Habernal 2016.
2. Load feature data that was used in the SVM-based approach in Habernal 2016.
3. Load the crowdsourced data.
4. Copy a similar testing setup to Habernal 2016 (training/test split?) and run the Bayesian approach (during testing,
we can set aside some held-out data). 
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
import sys, os
import numpy as np
from sklearn.metrics import accuracy_score
import time

from preference_features import PreferenceComponents
from preproc_raw_data import generate_separate_CSV

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

# TODO: create function for loading embeddings data and separate function for SVM feature data

if __name__ == '__main__':
    # Word embeddings -- load from file for full dataset
    
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    else:
        dirname = '../../git/acl2016-convincing-arguments/data/UKPConvArg1-full-XML/'
    
    print "Data directory = %s" % dirname
    
    if len(sys.argv) > 2:
        embeddings_dir = sys.argv[2]
    else:
        embeddings_dir = '../../git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python/'
    print "Looking for embeddings in directory %s" % embeddings_dir
    
    if len(sys.argv) > 3:
        resultsfile = sys.argv[3] % 'test'
        csvdirname = sys.argv[3] % 'UKPConvArg1All-turkerids-CSV'
    else:
        resultsfile = './results/habernal_embeddings_test.pkl'
        csvdirname = './data/UKPConvArg1All-turkerids-CSV/'
        
    if len(sys.argv) > 4:
        feature_type = sys.argv[4]
    else:
        feature_type = 'both' # can be 'embeddings' or 'ling'
        
    # generate the CSV files
    print("Writing CSV files with turker IDs...")
    if not os.path.isdir(csvdirname):
        os.mkdir(csvdirname)
        generate_separate_CSV(dirname, csvdirname)
    
    print('Loading data...')
    if feature_type == 'both' or feature_type == 'embeddings':
        folds, word_index_to_embeddings_map = load_my_data_separate_args(csvdirname, embeddings_dir=embeddings_dir)
        # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
        embeddings = np.asarray([np.array(x, dtype=np.float32) for x in word_index_to_embeddings_map.values()])
        
    if feature_type == 'both':
        # TODO: add in the svm features
        pass
    elif feature_type == 'ling':
        # TODO: load only the svm features
        pass
    
    all_proba = np.zeros(0)
    all_predictions = np.zeros(0)
    fold_idxs = np.zeros(0)
    all_truth = np.zeros(0)
    length_scales = np.zeros(0)
    latent_item_features = []
    latent_p_features = []
    person_noise_var = []
    item_means = []
    people = []
    all_f = []
    item_coords = []
    item_ids = []
    times = []
    
    # Run the experiments!
    for foldidx, fold in enumerate(folds.keys()):
        print("Fold name ", fold)
        X_train_a1, X_train_a2, y_train, ids_train, turkerids_train = folds.get(fold)["training"]
        X_test_a1, X_test_a2, y_test, ids_test, turkerids_test = folds.get(fold)["test"]

        print("Training instances ", len(X_train_a1), " training labels ", len(y_train))
        print("Test instances ", len(X_test_a1), " test labels ", len(y_test))
        
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        # See previous emails about sentence embeddings for a better method!
        print "Converting texts to mean embeddings (we should really use a better sentence embedding)..."
        Xe_train1 = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_train_a1])
        Xe_train2 = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_train_a2])
        print "...training data embeddings done."
        Xe_test1 = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_test_a1])
        Xe_test2 = np.array([np.mean(embeddings[Xi, :], axis=0) for Xi in X_test_a2])
        print "...test data embeddings done."
        
        ndims = Xe_train1.shape[1]
        ls_initial_guess = np.std(Xe_train1, axis=0)
        
        starttime = time.time()
        
        model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=False, nfactors=10, 
                                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True)
        personIDs = np.zeros(len(Xe_train1), dtype=int) # personIDs = turkerids_train
        
        # subsampled for debugging purposes only
        items_1_coords = Xe_train1
        items_2_coords = Xe_train2
        preferences = y_train
        personIDs = personIDs
        model.fit(personIDs, items_1_coords, items_2_coords, np.array(preferences, dtype=float)-1, optimize=False, nrestarts=1, 
                  input_type='zero-centered')
        
        personIDs = np.zeros(len(Xe_test1), dtype=int)
        # personIDs = turkerids_test
        items_1_coords = Xe_test1
        items_2_coords = Xe_test2
        proba = model.predict(personIDs, items_1_coords, items_2_coords, return_f=True)
        predictions = np.round(proba * 2)
        
        endtime = time.time()
        
        y_test = np.array(y_test, dtype=float)
        acc = accuracy_score(y_test, predictions)
        print('Test accuracy:', acc)
        
        # saving the data for later analysis:
        all_proba = np.concatenate((all_proba, proba))
        all_predictions = np.concatenate((all_predictions, predictions))
        fold_idxs = np.concatenate((fold_idxs, foldidx + np.zeros(len(y_test))))
        all_truth = np.concatenate((all_truth, y_test))
        length_scales = np.concatenate((length_scales, model.ls))
        latent_item_features.append(model.w)
        latent_p_features.append(model.y)
        item_means.append(model.t)
        person_noise_var_fold = np.zeros(model.Npeople)
        for p in model.people:
            person_noise_var_fold[p] = 1.0 / model.pref_gp[p].s
        person_noise_var.append(person_noise_var_fold)
        
        people.append(model.people)
        
        all_f.append(model.f)
        item_coords.append(model.obs_coords)
        ids_train_sep = [[item_id.split('_')[0], item_id.split('_')[1]] for item_id in ids_train]
        item_ids.append(np.array(ids_train_sep).T.flatten()[model.obs_uidxs])
        
        times.append(endtime-starttime)
        
        results = (all_proba, all_predictions, all_f, fold_idxs, all_truth, length_scales, latent_item_features, latent_p_features, 
                   person_noise_var, people, item_coords, item_ids, times)
        with open(resultsfile, 'w') as fh:
            pickle.dump(results, fh)