'''
Simpler example showing how to use train and use the convincingness model for prediction.

This script trains a model on the UKPConvArgStrict dataset. So, before running this script, you need to run
"python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features from this dataset.

'''
import sys

# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")

from data_loader import load_single_file_separate_args, load_ling_features
from embeddings import load_embeddings, get_mean_embeddings
from gp_classifier_vb import compute_median_lengthscales
from gp_pref_learning import GPPrefLearning
from run_preprocessing import preprocessing_pipeline
from tests import get_docidxs_from_ids, get_doc_token_seqs
import numpy as np
import logging
import os
from os import listdir
import vocabulary_embeddings_extractor
import pickle
import pandas as pd

# set the path for the java source code here

pkl_file = './model.pkl' # location to save the trained model to
test_data_path = './data/new_test_data' # location of your test data file. MUST HAVE A .CSV SUFFIX

embeddings_dir = './data/'

training_data_path = os.path.abspath("./data/")
training_dataset = 'UKPConvArgStrict'

def load_train_dataset(dataset, embeddings):
    _, docids = load_ling_features(dataset, training_data_path)
    print('Number of documents: %i' % len(docids))

    data_root_dir = os.path.expanduser(training_data_path)
    csvdirname = os.path.join(data_root_dir, 'argument_data/%s-new-CSV/' % dataset)

    print(('Loading train/test data from %s...' % csvdirname))

    person_train = []
    a1_train = []
    a2_train = []
    ids_train = []
    prefs_train = []

    for file_name in listdir(csvdirname):
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % csvdirname + '/' + file_name)
            continue

        _, _, labels, ids, turker_ids, a1, a2 = load_single_file_separate_args(csvdirname, file_name,
                                                                                  word_to_indices_map, None)

        a1_train.extend(a1)
        a2_train.extend(a2)

        person_train.extend(turker_ids)
        prefs_train.extend(labels)
        ids_train.extend(ids)

    train_ids = np.array([ids_pair.split('_') for ids_pair in ids_train])

    print('No. documents in training set: %i' % len(np.unique([train_ids[:, 0], train_ids[:, 1]])) )

    a1_train = get_docidxs_from_ids(docids, train_ids[:, 0])
    a2_train = get_docidxs_from_ids(docids, train_ids[:, 1])

    uids, uidxs = np.unique((a1_train, a2_train), return_index=True)
    item_ids = uids[:, None] # the features are just the IDs

    return item_ids, a1_train, a2_train, prefs_train

def train_test_model(embeddings):
    # Train a model...

    # item_ids -- an Nx1 vector containing the IDs of all documents;
    # a1_train -- the ids of the first items in the pairs
    # a2_train -- the ids of the second items in the pairs
    # prefs_train -- the labels for the pairs, 1 indicates the first item is preferred, -1 indicates the second is preferred
    item_ids, a1_train, a2_train, prefs_train = load_train_dataset(training_dataset, embeddings)  # reload only if we use a new dataset

    model = GPPrefLearning(ninput_features=1, verbose=False, shape_s0=2.0, rate_s0=200.0, use_svi=True,
                           ninducing=500, max_update_size=200, kernel_func='diagonal', kernel_combination='*', delay=1)

    model.max_iter_VB = 1000
    model.fit(a1_train, a2_train, item_ids, np.array(prefs_train, dtype=float) - 1, optimize=False,
              input_type='zero-centered', use_median_ls=True)

    logging.info("**** Completed training GPPL ****")

    # Save the model in case we need to reload it
    with open(pkl_file, 'wb') as fh:
        pickle.dump(model, fh)

    print('Predicting ...')
    predicted_f, _ = model.predict_f()

    print('Results: id, score ')
    for i in range(len(predicted_f)):
        print('%s, %s' % (i, predicted_f[i]))

if __name__ == '__main__':

    print('This script trains a model on the UKPConvArgStrict dataset.')

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    embeddings = load_embeddings(word_index_to_embeddings_map)

    train_test_model(embeddings)