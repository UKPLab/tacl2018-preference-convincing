from sklearn.datasets import load_svmlight_file

from data_loader import load_single_file_separate_args
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

training_data_path = "~/data/personalised_argumentation/"

java_run_path = '../acl2016-convincing-arguments/code/argumentation-convincingness-experiments-java/'
java_stanford_path = '../acl2016-convincing-arguments/code/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl/'
mvn_path = '../acl2016-convincing-arguments/code/'
classpath = "./target/argumentation-convincingness-experiments-java-1.0-SNAPSHOT.jar:target/lib/*"
stanford_classpath = "./target/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl-1.7.0.jar:" \
                     "./target/lib/*"

training_dataset = 'UKPConvArgStrict'

def load_dataset(dataset, embeddings):
    ling_feat_spmatrix, docids = load_ling_features(dataset, training_data_path)

    data_root_dir = os.path.expanduser(training_data_path)
    dirname = data_root_dir + 'argument_data/UKPConvArg1Strict-XML/'
    csvdirname = data_root_dir + 'argument_data/%s-new-CSV/' % dataset

    generate_gold_CSV(dirname, csvdirname)  # select only the gold labels

    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print(('Loading train/test data from %s...' % csvdirname))

    person_train = []
    a1_train = []
    a2_train = []
    ids_train = []
    prefs_train = []
    X_a1 = []
    X_a2 = []

    for file_name in listdir(csvdirname):
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % csvdirname + '/' + file_name)
            continue

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

    a1_train = get_docidxs_from_ids(docids, train_ids[:, 0])
    a2_train = get_docidxs_from_ids(docids, train_ids[:, 1])

    X, u_ids = get_doc_token_seqs((a1_train, a2_train), [X_a1, X_a2])
    items_feat = get_mean_embeddings(embeddings, X)

    ling_items_feat = ling_feat_spmatrix[u_ids, :].toarray()
    items_feat = np.concatenate((items_feat, ling_items_feat), axis=1)

    print('Found %i features.' % items_feat.shape[1])

    ndims = items_feat.shape[1]

    return items_feat, ling_feat_spmatrix.shape[1], word_to_indices_map, a1_train, \
           a2_train, prefs_train, ndims

def train_model(embeddings):
    # Train a model...
    items_feat, n_ling_feats, word_to_indices_map, valid_feats_ling, a1_train, a2_train, prefs_train, ndims \
        = load_dataset(training_dataset, embeddings)  # reload only if we use a new dataset

    ls_initial = compute_median_lengthscales(items_feat)

    model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial, verbose=False,
                                shape_s0=2.0, rate_s0=200.0, rate_ls=1.0 / np.mean(ls_initial),
                                use_svi=True, ninducing=500, max_update_size=200, kernel_combination='*',
                                forgetting_rate=0.7, delay=1.0)

    model.max_iter_VB = 2000

    print("no. features: %i" % items_feat.shape[1])

    model.fit(a1_train, a2_train, items_feat, np.array(prefs_train, dtype=float) - 1, optimize=False,
              input_type='zero-centered')

    logging.info("**** Completed training GPPL ****")

    # Save the model in case we need to reload it

    with open(pkl_file, 'wb') as fh:
        pickle.dump(model, fh)

def preprocessing_pipeline(input, output):
    # From Ivan Habernal's preprocessing pipeline, first, compile it
    call(['mvn', 'package'], cwd=mvn_path)

    # step 0, remove any '_' tokens as these will break the method
    tmp0 = os.path.abspath('./data/new_test_data0')
    if not os.path.exists(tmp0):
        os.mkdir(tmp0)

    for input_file in os.listdir(input):
        text_data = pd.read_csv(os.path.join(input, input_file), sep='\t', keep_default_na=False, index_col=0)
        text_data.replace('_', ' ', regex=True, inplace=True, )
        text_data.replace('\t', ' ', regex=True, inplace=True, )

        text_data.to_csv(os.path.join(tmp0, input_file), sep='\t')


    # step 1, convert to CSV:
    tmp = os.path.abspath('./data/new_ranking1')
    script = 'PipelineSeparateArguments'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.preprocessing'

    call(['java', '-cp', classpath,
          package + '.' + script,
          tmp0, tmp], cwd=java_run_path)
    print('Completed step 1')

    # step 2, sentiment analysis
    tmp2 = os.path.abspath('./data/new_ranking2')
    script = 'StanfordSentimentAnnotator'
    package = 'de.tudarmstadt.ukp.dkpro.core.stanfordsentiment'

    call(['java', '-cp', stanford_classpath,
          package + '.' + script,
          tmp, tmp2], cwd=java_stanford_path)
    print('Completed step 2')

    # step 3, extract features
    tmp3 = os.path.abspath('./data/new_ranking3')
    script = 'ExtractFeaturesPipeline'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.features'
    arg = 'false' # not using argument pairs here

    call(['java', '-cp', classpath,
          package + '.' + script,
          tmp2, tmp3, arg], cwd=java_run_path)
    print('Completed step 3')

    # step 4, export to SVMLib format
    feature_dir = os.path.join(os.path.expanduser(training_data_path), 'tempdata/all3')# use this directory to get a mapping from features to integers that matches the training set
    script = 'SVMLibEverythingExporter'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.svmlib'

    call(['java', '-cp', classpath,
          package + '.' + script, tmp3, output, feature_dir], cwd=java_run_path)
    print('Completed step 4')

def load_test_dataset(output):
    # Load the linguistic features
    print(("Loading linguistic features from %s" % output))
    ling_feat_spmatrix, docids = load_ling_features('new_test_data',
                       output,
                       '',
                       output,
                       model.features.shape[1] - len(embeddings[0])
                       )

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

        ids = data['#id'].values
        a1 = data['argument'].values

        a1_tokens = [vocabulary_embeddings_extractor.tokenize(a1_line) for a1_line in a1]
        a1_indices = [[word_to_indices_map.get(word, 2) for word in a1_tokens_line] for a1_tokens_line in a1_tokens]
        Xa1 = np.array([[1] + a1_indices_line for a1_indices_line in a1_indices])

        valid_args = np.in1d(ids, docids)
        a1 = a1[valid_args]
        Xa1 = Xa1[valid_args]
        ids = ids[valid_args]

        a.extend(a1)
        X.extend(Xa1)
        test_ids.extend(ids)

    # load the embeddings
    X_test, uids = get_doc_token_seqs((test_ids), [X]) # X_test is in the order of sorted test_ids
    emb_feat = get_mean_embeddings(embeddings, X_test)

    # ling_feat_spmatrix is in the order of lines in ling_file, so map back to order of test_ids
    docid_to_idx_map = np.argsort(docids).flatten()
    ling_items_feat = ling_feat_spmatrix.toarray()[docid_to_idx_map, :]
    test_items_feat = np.concatenate((emb_feat, ling_items_feat), axis=1)

    return test_items_feat, uids

if __name__ == '__main__':

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    embeddings = load_embeddings(word_index_to_embeddings_map)

    train_model(embeddings)

    # Load the model and the embeddings from file
    with open(pkl_file, 'rb') as fh:
        model = pickle.load(fh)

    # Now load some test documents for RANKING and extract their features
    input = os.path.abspath(test_data_path)
    output = os.path.abspath('./data/new_ranking_libsvm')
    preprocessing_pipeline(input, output)

    test_items_feat, text_ids = load_test_dataset(output)

    print('Predicting ...')
    predicted_f, _ = model.predict_f(out_feats=test_items_feat)

    print('Results: id, score ')
    for i in range(len(text_ids)):
        print('%s, %s' % (text_ids[i], predicted_f[i]))