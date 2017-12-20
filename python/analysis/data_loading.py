# -- coding: utf-8 --

'''
Helper functions for loading the data to run tests using the dataset from Ivan Habernal, 2016, ACL.

Created on 10 Jun 2017

@author: edwin
'''
import os, sys

data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")

sys.path.append('../../git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python')
sys.path.append(os.path.expanduser('~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow'))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))

from data_loader import load_my_data_separate_args
from data_loader_regression import load_my_data as load_my_data_regression
from sklearn.datasets import load_svmlight_file
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV
import numpy as np

def combine_into_libsvm_files(dataset, ids1, ids2, labels, dataset_type, fold, nfeats,
        dirname=data_root_dir + '/lingdata/UKPConvArg1-Full-libsvm', 
        outputfile=data_root_dir + '/libsvmdata/%s-%s-%s-libsvm.txt', reverse_pairs=False, 
        embeddings=None, a1=None, a2=None): 
    outputfile = outputfile % (dataset, dataset_type, fold)
    
    outputstr = ""
    dataids = [] # contains the argument document IDs in the same order as in the ouputfile and outputstr
    
    if os.path.isfile(outputfile):
        os.remove(outputfile)
    
    with open(outputfile, 'a') as ofh:    
        for row in range(len(ids1)):
            # each file should contain only one line
            fname1 = '%s.libsvm.txt' % ids1[row]
            with open(dirname + "/" + fname1) as fh:
                lines = fh.readlines()

            comment_split_line = lines[0][1:].split('#')                
            outputline = str(float(labels[row])) + comment_split_line[0] 
            
            if embeddings is not None:
                first_embedding_feature_id = nfeats
                if ids2 is not None:
                    first_embedding_feature_id += nfeats
                
                docvec = embeddings[a1[row], :]
                for i, v in enumerate(docvec):
                    outputline += str(int(i) + first_embedding_feature_id)
                    outputline += ':' + str(v) + '\t'
                  
            if ids2 is not None:
                fname2 = ids2[row] + '.libsvm.txt'
                with open(dirname + "/" + fname2) as fh:
                    lines2 = fh.readlines()

                # move comments at end of first line to end of complete joint line
                comment_split_line2 = lines2[0][1:].split('#')
                for feat in comment_split_line2[0].split('\t'):
                    if not len(feat):
                        continue
                    outputline += str(int(feat.split(':')[0]) + nfeats)
                    outputline += ':' + feat.split(':')[1] + '\t'
                # we could re-add the comments back in, but this seems to be problematic for libsvm, not sure why?
                #if len(comment_split_line) > 1:
                #    outputline += '\t#' + comment_split_line[1] + '_' + comment_split_line_complete[1]  
        
                if embeddings is not None:
                    first_embedding_feature_id = nfeats * 2 + embeddings.shape[1]
                    
                    docvec = embeddings[a2[row], :]
                    for i, v in enumerate(docvec):
                        outputline += str(int(i) + first_embedding_feature_id)
                        outputline += ':' + str(v) + '\t'        
        
            outputline += '\n'
        
            ofh.write(outputline)
                
            if reverse_pairs and ids2 is not None:
                outputline = str(float(1 - labels[row])) + comment_split_line2[0] 

                if embeddings is not None:
                    first_embedding_feature_id = nfeats
                    if ids2 is not None:
                        first_embedding_feature_id += nfeats
                
                    docvec = embeddings[a2[row], :]
                    for i, v in enumerate(docvec):
                        outputline += str(int(i) + first_embedding_feature_id)
                        outputline += ':' + str(v) + '\t'
                
                largestsofar = nfeats
                
                for feat in comment_split_line[0].split('\t'):
                    if not len(feat):
                        continue
                    outputline += str(int(feat.split(':')[0]) + nfeats)
                    outputline += ':' + feat.split(':')[1] + '\t'
                    if int(feat.split(':')[0]) + nfeats < largestsofar:
                        print 'Parsing error...'
                    largestsofar = int(feat.split(':')[0]) + nfeats
                    
                if embeddings is not None:
                    first_embedding_feature_id = nfeats * 2 + embeddings.shape[1]
                    
                    docvec = embeddings[a1[row], :]
                    for i, v in enumerate(docvec):
                        outputline += str(int(i) + first_embedding_feature_id)
                        outputline += ':' + str(v) + '\t'    
                    
                outputline += '\n'
                ofh.write(outputline)
                
    return outputfile, outputstr, dataids

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
            print("writing from file %s with row ID %s" % (filename, fid))
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
    folds_regression_test = None # for when the test for the ranking is different to the training data
    
    # Select the directory containing original XML files with argument data + crowdsourced annotations.
    # See the readme in the data folder from Habernal 2016 for further explanation of folder names.  
    
    
    norankidx = dataset.find('_noranking')
    if norankidx > -1:
        dataset = dataset[:norankidx]
      
    if dataset == 'UKPConvArgCrowd':
        # basic dataset, requires additional steps to produce the other datasets        
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'  
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArgAllRank-CSV/'
    elif dataset == 'UKPConvArgCrowdSample':
        dirname = data_root_dir + 'argument_data/UKPConvArg1-crowdsample-XML/'  
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/'
    elif dataset == 'UKPConvArgMACE' or dataset == 'UKPConvArgAll':   
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArg1-Ranking-CSV/'          
    elif dataset == 'UKPConvArgStrict':
        dirname = data_root_dir + 'argument_data/UKPConvArg1Strict-XML/'
        ranking_csvdirname = None        
    elif dataset == 'UKPConvArgCrowd_evalMACE': # train on the crowd dataset and evaluate on the MACE dataset
        dirname = data_root_dir + 'argument_data/UKPConvArg1-full-XML/'  
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArgAllRank-CSV/'
        folds_test, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowd'
    elif dataset == 'UKPConvArgCrowdSample_evalMACE':
        dirname = data_root_dir + 'argument_data/UKPConvArg1-crowdsample-XML/'  
        ranking_csvdirname = data_root_dir + 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/'
        folds_test, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowdSample'
    else:
        raise Exception("Invalid dataset %s" % dataset)

    if norankidx > -1:
        ranking_csvdirname = None
        folds_regression_test = None
    
    print("Data directory = %s, dataset=%s" % (dirname, dataset))
    csvdirname = data_root_dir + 'argument_data/%s-new-CSV/' % dataset
    # Generate the CSV files from the XML files. These are easier to work with! The CSV files from Habernal do not 
    # contain all turker info that we need, so we generate them afresh here.
    if not os.path.isdir(csvdirname):
        print("Writing CSV files...")
        os.mkdir(csvdirname)
        if 'UKPConvArgCrowd' in dataset: #dataset == 'UKPConvArgCrowd': # not for CrowdSample -- why not? Should be possible.
            generate_turker_CSV(dirname, csvdirname) # select all labels provided by turkers
        else: #if 'UKPConvArgStrict' in dataset or 'UKPConvArgAll' in dataset or dataset == 'UKPConvArgCrowdSample':
            generate_gold_CSV(dirname, csvdirname) # select only the gold labels
                
    embeddings_dir = data_root_dir + '/embeddings/'
    print("Embeddings directory: %s" % embeddings_dir)
    
    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print('Loading train/test data from %s...' % csvdirname)
    folds, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = load_my_data_separate_args(csvdirname, 
                                                                                          embeddings_dir=embeddings_dir)
    print folds.keys()[0]
    print folds[folds.keys()[0]]["test"][0]
    print folds[folds.keys()[0]]["test"][1]
    print folds[folds.keys()[0]]["test"][2]
    print folds[folds.keys()[0]]["test"][3]
    if ranking_csvdirname is not None:             
        folds_regression, _ = load_my_data_regression(ranking_csvdirname, embeddings_dir=embeddings_dir, 
                                                      load_embeddings=True)
        
    if folds_test is not None:
        for fold in folds:
            folds[fold]["test"] = folds_test[fold]["test"]
    if folds_regression_test is not None:
        for fold in folds_regression:
            folds_regression[fold]["test"] = folds_regression_test[fold]["test"] 

    return folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map
    
def load_embeddings(word_index_to_embeddings_map):
    print('Loading embeddings')
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.zeros((1 + np.max(word_index_to_embeddings_map.keys()), len(word_index_to_embeddings_map.values()[0])))
    embeddings[word_index_to_embeddings_map.keys()] = word_index_to_embeddings_map.values()
    #embeddings = np.asarray([np.array(x, dtype=np.float32) for x in word_index_to_embeddings_map.values()])
    return embeddings

def load_skipthoughts_embeddings(word_to_indices_map):
    print('Loading Skip-thoughts model...')
    global skipthoughts
    import skipthoughts
    model = skipthoughts.load_model()
    return model

def load_siamese_cbow_embeddings(word_to_indices_map):
    print('Loading Siamese CBOW embeddings...')
    filename = os.path.expanduser('~/data/personalised_argumentation/embeddings/Siamese-CBOW/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle')
    import wordEmbeddings as siamese_cbow
    return siamese_cbow.wordEmbeddings(filename)
     
def load_ling_features(dataset):
    ling_dir = data_root_dir + 'lingdata/'
    print("Looking for linguistic features in directory %s" % ling_dir) 
    print('Loading linguistic features')
    ling_file = ling_dir + "/%s-libsvm.txt" % dataset
    argids_file = ling_dir + "/%s-libsvm.txt" % ("argids_%s" % dataset)
    if not os.path.isfile(ling_file) or not os.path.isfile(argids_file):
        ling_file, _ , docids = combine_lines_into_one_file(dataset, outputfile=ling_dir+"/%s-libsvm.txt")
    else:
        docids = np.genfromtxt(argids_file, str)
        
    ling_feat_spmatrix, _ = load_svmlight_file(ling_file)
    return ling_feat_spmatrix, docids