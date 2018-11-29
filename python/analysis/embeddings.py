# -- coding: utf-8 --

'''
Helper functions for loading the data to run tests using the dataset from Ivan Habernal, 2016, ACL.

Created on 10 Jun 2017

@author: edwin
'''
import os, sys

sys.path.append(os.path.abspath('./git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python'))
sys.path.append(os.path.expanduser('~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow'))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))

import numpy as np


def load_embeddings(word_index_to_embeddings_map):
    print('Loading embeddings')
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.zeros((1 + np.max(list(word_index_to_embeddings_map.keys())), len(list(word_index_to_embeddings_map.values())[0])))
    embeddings[list(word_index_to_embeddings_map.keys())] = list(word_index_to_embeddings_map.values())
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


def get_mean_embeddings(word_embeddings, X):
    return np.array([np.mean(word_embeddings[Xi, :], axis=0) for Xi in X])