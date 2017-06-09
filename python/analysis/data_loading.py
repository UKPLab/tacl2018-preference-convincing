'''
Helper functions for loading the data to run tests using the dataset from Ivan Habernal, 2016, ACL.

Created on 10 Jun 2017

@author: edwin
'''
import os, sys

data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")

sys.path.append('../../git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python')
sys.path.append(data_root_dir + '/embeddings/Siamese-CBOW/siamese-cbow')
sys.path.append(data_root_dir + '/embeddings/skip-thoughts')

import numpy as np
from data_loader import load_my_data_separate_args
from data_loader_regression import load_my_data as load_my_data_regression
from sklearn.datasets import load_svmlight_file
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV

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
    
    folds_regression = None # test data for regression (use the folds object for training)
    folds_test = None # can load a separate set of data for testing classifications, e.g. train on workers and test on gold standard
    
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