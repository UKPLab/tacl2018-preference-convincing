from sklearn.datasets import load_svmlight_file

from data_loader import load_single_file_separate_args, load_single_file
from data_loading import load_ling_features, load_embeddings, combine_lines_into_one_file
from gp_classifier_vb import compute_median_lengthscales
from gp_pref_learning import GPPrefLearning
from preproc_raw_data import generate_gold_CSV
from tests import get_docidxs_from_ids, get_doc_token_seqs, get_mean_embeddings
import numpy as np
import logging
import os
from os import listdir
import vocabulary_embeddings_extractor
from subprocess import call
import pickle
import pandas as pd

# set the path for the java source code here

pkl_file = './model.pkl' # location to save the trained model to
test_data_path = './data/new_test_data' # location of your test data file. MUST HAVE A .CSV SUFFIX

embeddings_dir = './data/'

java_run_path = '../acl2016-convincing-arguments/code/argumentation-convincingness-experiments-java/'
java_stanford_path = '../acl2016-convincing-arguments/code/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl/'
mvn_path = '../acl2016-convincing-arguments/code/'
classpath = "./target/argumentation-convincingness-experiments-java-1.0-SNAPSHOT.jar:target/lib/*"
stanford_classpath = "./target/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl-1.7.0.jar:" \
                     "./target/lib/*"

def load_dataset(dataset):
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    dirname = data_root_dir + 'argument_data/UKPConvArg1Strict-XML/'
    csvdirname = data_root_dir + 'argument_data/%s-new-CSV/' % dataset

    generate_gold_CSV(dirname, csvdirname)  # select only the gold labels

    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print(('Loading train/test data from %s...' % csvdirname))

    files = listdir(csvdirname)

    for file_name in files:
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % csvdirname + '/' + file_name)
            files.remove(file_name)

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')

    person_train = []
    a1_train = []
    a2_train = []
    ids_train = []
    prefs_train = []
    X_a1 = []
    X_a2 = []

    for file_name in files:
        Xa1, Xa2, labels, ids, turker_ids, a1, a2 = load_single_file_separate_args(csvdirname, file_name,
                                                                                  word_to_indices_map, None)

        X_a1.extend(Xa1)
        X_a2.extend(Xa2)

        a1_train.extend(a1)
        a2_train.extend(a2)

        person_train.extend(turker_ids)
        prefs_train.extend(labels)
        ids_train.extend(ids)

    train_ids = np.array([ids_pair.split('_') for ids_pair in ids_train])

    docids = np.unique(train_ids)

    a1_train = get_docidxs_from_ids(docids, train_ids[:, 0])
    a2_train = get_docidxs_from_ids(docids, train_ids[:, 1])

    X, u_ids = get_doc_token_seqs((a1_train, a2_train), [X_a1, X_a2])

    ling_feat_spmatrix, docids = load_ling_features(dataset)

    logging.info("Converting texts to mean embeddings (we could use a better sentence embedding?)...")
    embeddings = load_embeddings(word_index_to_embeddings_map)
    items_feat = get_mean_embeddings(embeddings, X)
    logging.info("...embeddings loaded.")

    # trim away any features not in the training data because we can't learn from them
    valid_feats = np.sum((items_feat[a1_train] != 0) + (items_feat[a2_train] != 0), axis=0) > 0
    items_feat = items_feat[:, valid_feats]

    logging.info("Obtaining linguistic features for argument texts.")
    # trim the features that are not used in training
    ling_items_feat = ling_feat_spmatrix[u_ids, :].toarray()
    items_feat = np.concatenate((items_feat, ling_items_feat), axis=1)
    logging.info("...loaded all linguistic features for training and test data.")

    print('Found %i features.' % items_feat.shape[1])

    ndims = items_feat.shape[1]

    ls_initial = compute_median_lengthscales(items_feat)

    return items_feat, ling_feat_spmatrix.shape[1], embeddings, word_to_indices_map, a1_train, \
           a2_train, prefs_train, ls_initial, ndims

if __name__ == '__main__':

    # acc = 1.0
    # dataset_increment = 0
    #
    # # Train a model on the UKPConvArgStrict data
    #
    # dataset = 'UKPConvArgStrict'
    # items_feat, n_ling_feats, embeddings, word_to_indices_map, valid_feats_ling, a1_train, a2_train, prefs_train, ls_initial, ndims \
    #     = load_dataset(dataset)  # reload only if we use a new dataset
    #
    # model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial, verbose=False,
    #                             shape_s0=2.0, rate_s0=200.0, rate_ls=1.0 / np.mean(ls_initial),
    #                             use_svi=True, ninducing=500, max_update_size=200, kernel_combination='*',
    #                             forgetting_rate=0.7, delay=1.0)
    #
    # model.max_iter_VB = 2000
    #
    # print("no. features: %i" % items_feat.shape[1])
    #
    # model.fit(a1_train, a2_train, items_feat, np.array(prefs_train, dtype=float) - 1, optimize=False,
    #           input_type='zero-centered')
    #
    # logging.info("**** Completed training GPPL ****")
    #
    # # Save the model in case we need to reload it
    #
    # with open(pkl_file, 'wb') as fh:
    #     pickle.dump(model, fh)

    # Load the model and the embeddings from file
    with open(pkl_file, 'rb') as fh:
        model = pickle.load(fh)
    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    embeddings = load_embeddings(word_index_to_embeddings_map)

    # Now load some test documents for RANKING and extract their features

    # From Ivan Habernal's preprocessing pipeline, first, compile it
    call(['mvn', 'package'], cwd=mvn_path)

    # step 1, convert to CSV:

    input = os.path.abspath(test_data_path)
    tmp = os.path.abspath('./data/new_ranking1')
    script = 'PipelineSeparateArguments'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.preprocessing'

    call(['java', '-cp', classpath,
          package + '.' + script,
          input, tmp], cwd=java_run_path)
    print('Completed step 1')

    # step 2, sentiment analysis
    tmp2 = os.path.abspath('./data/new_ranking2')
    script = 'StanfordSentimentAnnotator'
    package = 'de.tudarmstadt.ukp.dkpro.core.stanfordsentiment'

    # call(['java', '-cp', stanford_classpath,
    #       package + '.' + script,
    #       tmp, tmp2], cwd=java_stanford_path)
    # print('Completed step 2')

    # step 3, extract features
    tmp3 = os.path.abspath('./data/new_ranking3')
    script = 'ExtractFeaturesPipeline'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.features'
    arg = 'false' # not using argument pairs here

    # call(['java', '-cp', classpath,
    #       package + '.' + script,
    #       tmp2, tmp3, arg], cwd=java_run_path)
    # print('Completed step 3')

    # step 4, export to SVMLib format
    output = os.path.abspath('./data/new_ranking_libsvm')
    script = 'SVMLibEverythingExporter'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.svmlib'

    # call(['java', '-cp', classpath,
    #       package + '.' + script, tmp3, output], cwd=java_run_path)
    # print('Completed step 4')

    # Load the linguistic features
    ling_dir = output
    print(("Loading linguistic features from %s" % ling_dir))
    ling_file, _, docids = combine_lines_into_one_file('new_test_data',
                                                       dirname=ling_dir,
                                                       outputfile=ling_dir + "/%s-libsvm.txt")
    print('Completed combining libSVM files.')

    ling_feat_spmatrix, _ = load_svmlight_file(ling_file, n_features=model.features.shape[1] - len(embeddings[0]))
    print('Loaded libSVM data')

    X = []
    test_ids = []
    a = []

    for file_name in listdir(input):
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % input + '/' + file_name)
            continue

        data = pd.read_csv(os.path.join(input, file_name), delimiter='\t', na_values=[])
        data = data.fillna('N/A')

        ids = [file_lineid in data['#id'].values]
        a1 = data['argument'].values

        a1_tokens = [vocabulary_embeddings_extractor.tokenize(a1_line) for a1_line in a1]
        a1_indices = [[word_to_indices_map.get(word, 2) for word in a1_tokens_line] for a1_tokens_line in a1_tokens]
        Xa1 = [[1] + a1_indices_line for a1_indices_line in a1_indices]

        valid_args = ids in docids
        a1 = a1[valid_args]
        Xa1 = Xa1[valid_args]
        ids = ids[valid_args]

        a.extend(a1)
        X.extend(Xa1)
        test_ids.extend(ids)

    # load the embeddings
    X_test, _ = get_doc_token_seqs((test_ids), [X])
    emb_feat = get_mean_embeddings(embeddings, X_test)

    ling_items_feat = ling_feat_spmatrix.toarray()
    test_items_feat = np.concatenate((emb_feat, ling_items_feat), axis=1)

    print('Predicting ...')
    predicted_f, _ = model.predict_f(None, test_ids)

    print('Results: ')
    print(predicted_f)