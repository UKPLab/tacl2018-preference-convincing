# -- coding: utf-8 --

from __future__ import absolute_import
from __future__ import print_function

import os
from copy import copy

import numpy as np
from os import listdir

from sklearn.datasets import load_svmlight_file

import vocabulary_embeddings_extractor
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV


def load_single_file(directory, file_name, word_to_indices_map, nb_words=None):
    """
    Loads a single file and returns a tuple of x vectors and y labels
    :param directory: dir
    :param file_name: file name
    :param word_to_indices_map: words to their indices
    :param nb_words: maximum word index to be kept; otherwise treated as OOV
    :return: tuple of lists of integers
    """
    f = open(os.path.join(directory, file_name), 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    x_vectors = []
    y_labels = []
    id_vector = []

    x_vectors_reversed = []
    y_labels_reversed = []

    for line in lines:
        # print line
        toks = line.split('\t')
        arg_id = toks[0]
        label = toks[1]
        a1 = toks[-2]
        a2 = toks[-1]
        # print(arg_id, label, a1, a2)

        id_vector.append(arg_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        a2_tokens = vocabulary_embeddings_extractor.tokenize(a2)

        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        a2_indices = [word_to_indices_map.get(word, 2) for word in a2_tokens]

        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x = [1] + a1_indices + [1] + a2_indices
        # print(x)

        # let's do the oversampling trick :)
        x_reverse = [1] + a2_indices + [1] + a1_indices

        # map class to vector
        if 'a1' == label:
            y = 0
            y_reverse = 2
        elif 'a2' == label:
            y = 2
            y_reverse = 0
        else:
            y = 1
            y_reverse = 1

        x_vectors.append(x)
        y_labels.append(y)

        x_vectors_reversed.append(x_reverse)
        y_labels_reversed.append(y_reverse)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors]
        x_vectors_reversed = [[2 if word_index >= nb_words else word_index for word_index in x] for x in
                              x_vectors_reversed]

    train_instances = x_vectors
    train_labels = y_labels

    return train_instances, train_labels, id_vector, x_vectors_reversed, y_labels_reversed

def load_single_file_separate_args(directory, file_name, word_to_indices_map, nb_words=None):
    """
    Loads a single file and returns a tuple of x vectors and y labels
    :param directory: dir
    :param file_name: file name
    :param word_to_indices_map: words to their indices
    :param nb_words: maximum word index to be kept; otherwise treated as OOV
    :return: tuple of lists of integers
    """
    f = open(directory + file_name, 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    x_vectors_a1 = []
    x_vectors_a2 = []
    train_a1 = []
    train_a2 = []
    y_labels = []
    id_vector = []
    turkerids = []
    
    for line in lines:
        # print line
        toks = line.split('\t')
        if len(toks) != 5:
            raise Exception
        arg_id, turker_id, label, a1, a2 = toks 
        # print(arg_id, label, a1, a2)

        id_vector.append(arg_id)
        turkerids.append(turker_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        a2_tokens = vocabulary_embeddings_extractor.tokenize(a2)

        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        a2_indices = [word_to_indices_map.get(word, 2) for word in a2_tokens]
        
        train_a1.append(a1)
        train_a2.append(a2)
        
        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x1 = [1] + a1_indices 
        x2 = [1] + a2_indices
        # print(x)

        # map class to vector
        if 'a1' == label:
            y = 2
        elif 'a2' == label:
            y = 0
        else:
            y = 1

        x_vectors_a1.append(x1)
        x_vectors_a2.append(x2)
        y_labels.append(y)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors_a1 = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors_a1]
        x_vectors_a2 = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors_a2]

    train_instances_a1 = x_vectors_a1
    train_instances_a2 = x_vectors_a2
    train_labels = y_labels

    return train_instances_a1, train_instances_a2, train_labels, id_vector, turkerids, train_a1, train_a2

def load_my_data(directory, test_split=0.2, nb_words=None, add_reversed_training_data=False, embeddings_dir=''):
    # directory = '/home/habi/research/data/convincingness/step5-gold-data/'
    # directory = '/home/user-ukp/data2/convincingness/step7-learning-11-no-eq/'
    files = listdir(directory)
    # print(files)

    for file_name in files:
        if file_name.split('.')[-1] != 'csv':                
            print("Skipping files without .csv suffix: %s" % directory + '/' + file_name)
            files.remove(file_name)

    # folds
    folds = dict()
    for file_name in files:
        training_file_names = copy(files)
        # remove current file
        training_file_names.remove(file_name)
        folds[file_name] = {"training": training_file_names, "test": file_name}

    # print(folds)

    word_to_indices_map, word_index_to_embeddings_map = vocabulary_embeddings_extractor.load_all(embeddings_dir + 
                                                                                'vocabulary.embeddings.all.pkl.bz2')

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        # print(file_name)
        test_instances, test_labels, ids, x_vectors_reversed, y_labels_reversed = load_single_file(directory, file_name,
                                                                                                   word_to_indices_map,
                                                                                                   nb_words)
        all_loaded_files[file_name] = test_instances, test_labels, ids, x_vectors_reversed, y_labels_reversed
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        # print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances, test_labels, ids, test_x_vectors_reversed, test_y_labels_reversed = all_loaded_files.get(
            file_name)

        # add tuple
        current_fold["test"] = test_instances, test_labels, ids

        # now collect all training instances
        all_training_instances = []
        all_training_labels = []
        all_training_ids = []
        for training_file_name in folds.get(file_name)["training"]:
            training_instances, training_labels, ids, x_vectors_reversed, y_labels_reversed = all_loaded_files.get(
                training_file_name)
            all_training_instances.extend(training_instances)
            all_training_labels.extend(training_labels)
            all_training_ids.extend(ids)

            if add_reversed_training_data:
                all_training_instances.extend(x_vectors_reversed)
                all_training_labels.extend(y_labels_reversed)
                all_training_ids.extend(ids)

        current_fold["training"] = all_training_instances, all_training_labels, all_training_ids

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map

def load_my_data_separate_args(directory, test_split=0.2, nb_words=None, add_reversed_training_data=False,
                               embeddings_dir=''):
    # directory = '/home/habi/research/data/convincingness/step5-gold-data/'
    # directory = '/home/user-ukp/data2/convincingness/step7-learning-11-no-eq/'
    files = listdir(directory)
    
    for file_name in files:
        if file_name.split('.')[-1] != 'csv':                
            print("Skipping files without .csv suffix: %s" % directory + '/' + file_name)
            files.remove(file_name)
    
    # print(files)

    # folds
    folds = dict()
    for file_name in files:
        training_file_names = copy(files)
        # remove current file
        training_file_names.remove(file_name)
        folds[file_name] = {"training": training_file_names, "test": file_name}

    # print(folds)

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        #print(file_name)
        test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2 = \
                                    load_single_file_separate_args(directory, file_name, word_to_indices_map, nb_words)
        all_loaded_files[file_name] = test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        #print("Test fold: ")
        #print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2 = all_loaded_files.get(file_name)

        # add tuple
        current_fold["test"] = test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2

        # now collect all training instances
        all_tr_instances_a1 = []
        all_tr_instances_a2 = []
        all_tr_labels = []
        all_tr_ids = []
        all_tr_turker_ids = []
        all_tr_a1 = []
        all_tr_a2 = []
        for training_file_name in folds.get(file_name)["training"]:
            tr_instances_a1, tr_instances_a2, training_labels, ids, turker_ids, tr_a1, tr_a2 = \
                                                                            all_loaded_files.get(training_file_name)
            #print("Training file: ")
            #print(training_file_name)
            all_tr_instances_a1.extend(tr_instances_a1)
            all_tr_instances_a2.extend(tr_instances_a2)
            all_tr_labels.extend(training_labels)
            all_tr_ids.extend(ids)
            all_tr_turker_ids.extend(turker_ids)
            all_tr_a1.extend(tr_a1)
            all_tr_a2.extend(tr_a2)

        current_fold["training"] = all_tr_instances_a1, all_tr_instances_a2, all_tr_labels, all_tr_ids, \
                all_tr_turker_ids, all_tr_a1, all_tr_a2

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map

def __main__():
    np.random.seed(1337)  # for reproducibility

    # todo try with 1000 and fix functionality
    max_words = 1000
    batch_size = 32
    nb_epoch = 10

    print('Loading data...')
    folds, word_index_to_embeddings_map = load_my_data("/home/user-ukp/data2/convincingness/ConvArgStrict/")

    # print statistics
    for fold in folds.keys():
        print("Fold name ", fold)
        training_instances, training_labels = folds.get(fold)["training"]
        test_instances, test_labels = folds.get(fold)["test"]

        print("Training instances ", len(training_instances), " training labels ", len(training_labels))
        print("Test instances ", len(test_instances), " test labels ", len(test_labels))

# __main__()
data_root_dir = os.path.abspath("./data/")


def combine_lines_into_one_file(dataset_name, dirname=os.path.join(data_root_dir, 'lingdata/UKPConvArg1-Full-libsvm'),
                                outputfile=os.path.join(data_root_dir, 'lingdata/%s-libsvm.txt')):
    output_argid_file = outputfile % ("argids_%s" % dataset_name)
    outputfile = outputfile % dataset_name

    outputstr = ""
    dataids = [] # contains the argument document IDs in the same order as in the ouputfile and outputstr

    if os.path.isfile(outputfile):
        os.remove(outputfile)
    if os.path.isfile(output_argid_file):
        os.remove(output_argid_file)

    with open(outputfile, 'a') as ofh:
        for filename in os.listdir(dirname):

            if os.path.samefile(outputfile, os.path.join(dirname, filename)):
                continue

            if filename.split('.')[-1] != 'txt':
                continue

            fid = filename.split('.')[0]
            print(("writing from file %s with row ID %s" % (filename, fid)))
            with open(dirname + "/" + filename) as fh:
                lines = fh.readlines()
            for line in lines:
                dataids.append(fid)
                outputline = line.split('#')[0]
                if outputline[-1] != '\n':
                    outputline += '\n'

                ofh.write(outputline)
                outputstr += outputline + '\n'

    dataids = np.array(dataids)[:, np.newaxis]
    np.savetxt(output_argid_file, dataids, '%s')

    return outputfile, outputstr, dataids


def load_train_test_data(dataset):
    # Set experiment options and ensure CSV data is ready -------------------------------------------------------------

    folds_regression = None # test data for regression (use the folds object for training)
    folds_test = None # can load a separate set of data for testing classifications, e.g. train on workers and test on gold standard
    folds_regression_test = None # for when the test for the ranking is different to the training data

    # Select the directory containing original XML files with argument data + crowdsourced annotations.
    # See the readme in the data folder from Habernal 2016 for further explanation of folder names.


    norankidx = dataset.find('_noranking')
    if norankidx > -1:
        dataset = dataset[:norankidx]

    if dataset == 'UKPConvArgCrowd':
        # basic dataset, requires additional steps to produce the other datasets
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArgAllRank-CSV/')

    elif dataset == 'UKPConvArgCrowdSample':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/')

    elif dataset == 'UKPConvArgMACE' or dataset == 'UKPConvArgAll':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-Ranking-CSV/')

    elif dataset == 'UKPConvArgStrict':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1Strict-XML/')
        ranking_csvdirname = None

    elif dataset == 'UKPConvArgCrowd_evalMACE': # train on the crowd dataset and evaluate on the MACE dataset
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArgAllRank-CSV/')
        folds_test, _, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowd'

    elif dataset == 'UKPConvArgCrowdSample_evalMACE':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/')
        folds_test, _, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowdSample'

    else:
        raise Exception("Invalid dataset %s" % dataset)

    if norankidx > -1:
        ranking_csvdirname = None
        folds_regression_test = None

    print(("Data directory = %s, dataset=%s" % (dirname, dataset)))
    csvdirname = os.path.join(data_root_dir, 'argument_data/%s-new-CSV/' % dataset)
    # Generate the CSV files from the XML files. These are easier to work with! The CSV files from Habernal do not
    # contain all turker info that we need, so we generate them afresh here.
    if not os.path.isdir(csvdirname):
        print("Writing CSV files...")
        os.mkdir(csvdirname)
        if 'UKPConvArgCrowd' in dataset: #dataset == 'UKPConvArgCrowd': # not for CrowdSample -- why not? Should be possible.
            generate_turker_CSV(dirname, csvdirname) # select all labels provided by turkers
        else: #if 'UKPConvArgStrict' in dataset or 'UKPConvArgAll' in dataset or dataset == 'UKPConvArgCrowdSample':
            generate_gold_CSV(dirname, csvdirname) # select only the gold labels

    embeddings_dir = './data/'
    print(("Embeddings directory: %s" % embeddings_dir))

    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print(('Loading train/test data from %s...' % csvdirname))
    folds, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = load_my_data_separate_args(csvdirname,
                                                                                          embeddings_dir=embeddings_dir)
    # print(list(folds.keys())[0])
    # print(folds[list(folds.keys())[0]]["training"][0][:20][:10])
    # print(folds[list(folds.keys())[0]]["training"][1][:20][:10])
    # print(folds[list(folds.keys())[0]]["training"][2][:10])
    # print(folds[list(folds.keys())[0]]["training"][3][:20])
    if ranking_csvdirname is not None:
        folds_regression, _ = load_my_data_regression(ranking_csvdirname, embeddings_dir=embeddings_dir,
                                                      load_embeddings=True)



    if folds_test is None:
        folds_test = folds
        #for fold in folds:
        #    folds[fold]["test"] = folds_test[fold]["test"]

    if folds_regression_test is not None:
        for fold in folds_regression:
            folds_regression[fold]["test"] = folds_regression_test[fold]["test"]

    return folds, folds_test, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map


def load_single_file_regression(directory, file_name, word_to_indices_map, nb_words=None):
    """
    Loads a single file and returns a tuple of x vectors and y labels
    :param directory: dir
    :param file_name: file name
    :param word_to_indices_map: words to their indices
    :param nb_words: maximum word index to be kept; otherwise treated as OOV
    :return: tuple of lists of integers
    """
    f = open(directory + file_name, 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    x_vectors = []
    y_labels = []
    id_vector = []
    turkIDs = []
    train_a1 = []

    for line in lines:
        # print line
        toks = line.split('\t')
        if len(toks) == 3:
            arg_id, score, a1 = toks
        elif len(toks) == 4:
            arg_id, score, a1, turkID = toks
            turkIDs.append(turkID)
        else:
            print("Bad line in file %s: %s" % (directory + file_name, line))
            continue
            #(arg_id, label, a1, a2)

        id_vector.append(arg_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        if word_to_indices_map is not None:
            a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        else:
            a1_indices = []

        train_a1.append(a1)

        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x = [1] + a1_indices + [1]
        # print(x)

        # convert score to float. The scores seem to be negated.
        y = -float(score)

        x_vectors.append(x)
        y_labels.append(y)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors]

    train_instances = x_vectors
    train_labels = y_labels

    return train_instances, train_labels, id_vector, turkIDs, train_a1


def load_my_data_regression(directory, test_split=0.2, nb_words=None, embeddings_dir='', load_embeddings=True):
    # directory = '/home/habi/research/data/convincingness/step5-gold-data/'
    # directory = '/home/user-ukp/data2/convincingness/step7-learning-11-no-eq/'
    files = listdir(directory)
    # print(files)

    for file_name in files:
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % directory + '/' + file_name)
            files.remove(file_name)

    # folds
    folds = dict()
    for file_name in files:
        training_file_names = copy(files)
        # remove current file
        training_file_names.remove(file_name)
        folds[file_name] = {"training": training_file_names, "test": file_name}

    # print(folds)
    if load_embeddings:
        word_to_indices_map, word_index_to_embeddings_map, _ = vocabulary_embeddings_extractor.load_all(
            embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    else:
        word_to_indices_map = None
        word_index_to_embeddings_map = None

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        # print(file_name)
        test_instances, test_labels, ids, turkIDs, argtexts = load_single_file_regression(directory, file_name, word_to_indices_map, nb_words)
        all_loaded_files[file_name] = test_instances, test_labels, ids, turkIDs, argtexts
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        # print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances, test_labels, ids, turkIDs, argtexts = all_loaded_files.get(file_name)

        # add tuple
        current_fold["test"] = test_instances, test_labels, ids, turkIDs, argtexts

        # now collect all training instances
        all_training_instances = []
        all_training_labels = []
        all_training_ids = []
        all_training_turkIDs = []
        all_training_texts = []
        for training_file_name in folds.get(file_name)["training"]:
            training_instances, training_labels, ids, turkIDs, texts = all_loaded_files.get(training_file_name)
            all_training_instances.extend(training_instances)
            all_training_labels.extend(training_labels)
            all_training_ids.extend(ids)
            all_training_turkIDs.extend(turkIDs)
            all_training_texts.extend(texts)

        current_fold["training"] = all_training_instances, all_training_labels, all_training_ids, all_training_turkIDs, \
                                    all_training_texts

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map


def load_ling_features(dataset,
                       root_dir=data_root_dir,
                       ling_subdir='lingdata/',
                       input_dir=os.path.join(data_root_dir, 'lingdata/UKPConvArg1-Full-libsvm'),
                       max_n_features=None):

    ling_dir = os.path.join(root_dir, ling_subdir)
    if not os.path.exists(ling_dir):
        os.mkdir(ling_dir)

    print(("Looking for linguistic features in directory %s" % ling_dir))
    print('Loading linguistic features')
    ling_file = ling_dir + "/%s-libsvm.txt" % dataset
    argids_file = ling_dir + "/%s-libsvm.txt" % ("argids_%s" % dataset)
    if not os.path.isfile(ling_file) or not os.path.isfile(argids_file):
        ling_file, _ , docids = combine_lines_into_one_file(dataset,
                                                            dirname=input_dir,
                                                            outputfile=os.path.join(ling_dir, "%s-libsvm.txt")
                                                            )
    else:
        docids = np.genfromtxt(argids_file, str)
        print('Reloaded %i docids from file. ' % len(docids))

    ling_feat_spmatrix, _ = load_svmlight_file(ling_file, n_features=max_n_features)
    return ling_feat_spmatrix, docids