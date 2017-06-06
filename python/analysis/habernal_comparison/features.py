'''
Created on 1 Jun 2017

Load a set of feature lengthscales from a good run with 'both' types of features. 
Sort them by lengthscale.
Plot the distribution.

Identify which type of feature they are: add colours or markers to the plot.

Provide a zoomed-in variant for the best 25 features.

@author: simpson
'''

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from tests import load_train_test_data, load_embeddings, load_ling_features, get_fold_data, get_mean_embeddings
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
    
    dataset = 'UKPConvArgStrict'#'UKPConvArgAll_evalMACE'#
    method = 'SinglePrefGP'
    feature_type = 'both'
    embeddings_type = 'word_mean'
      
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/' + \
                    'habernal_%s_%s_%s_%s_test.pkl' % (dataset, method, feature_type, embeddings_type)
  
    with open(resultsfile, 'r') as fh:
        data = pickle.load(fh)
          
    nFolds = len(data[0])
      
    # Sort the features by their ID. 
    # If we have discarded some features that were all zeros, the current index will not be the original feature idx.
    # How to map them back? Reload the original data and find out which features were discarded.
      
    folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map = load_train_test_data(dataset)
    word_embeddings = load_embeddings(word_index_to_embeddings_map)
    ling_feat_spmatrix, docids = load_ling_features(dataset)
      
    for foldidx, fold in enumerate(folds.keys()):
        if foldidx >= nFolds:
            print "data incomplete -- foldidx %i has not been run" % foldidx
            continue
        
        trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test,\
                                                                        X, uids = get_fold_data(folds, fold, docids)
                                                                                  
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        if feature_type == 'both' or feature_type == 'embeddings':
            if embeddings_type == 'word_mean':
                items_feat = get_mean_embeddings(word_embeddings, X)
                  
            nfeats = items_feat.shape[1]
            # trim away any features not in the training data because we can't learn from them
            valid_feats = np.argwhere((np.sum(items_feat[trainids_a1] != 0, axis=0)>0) 
                                        & (np.sum(items_feat[trainids_a2] != 0, axis=0)>0)).T[0]
              
        elif feature_type == 'ling':
            items_feat = np.zeros((X.shape[0], 0))
            valid_feats = np.zeros(0)
            nfeats = 0
              
        if feature_type == 'both' or feature_type == 'ling':
            nfeats += ling_feat_spmatrix.shape[1]
            print "Obtaining linguistic features for argument texts."
            # trim the features that are not used in training
            valid_feats_ling = ((np.sum(ling_feat_spmatrix[trainids_a1, :] != 0, axis=0)>0) & 
                           (np.sum(ling_feat_spmatrix[trainids_a2, :] != 0, axis=0)>0)).nonzero()[1]
            valid_feats_ling += items_feat.shape[1]
            print "...loaded all linguistic features for training and test data."
              
            valid_feats = np.concatenate((valid_feats, valid_feats_ling)).astype(int)
              
        # take the mean ls for each feature across the folds
        if foldidx == 0:
            mean_ls = np.zeros(nfeats) 
            totals = np.zeros(nfeats, dtype=int)
              
        #mean_ls = data[7][foldidx]
        #print "Warning: not computing means."
        mean_ls[valid_feats] += data[7][foldidx]
        totals[valid_feats] += 1
         
    #mean_ls[totals != 0] = mean_ls[totals != 0] / totals[totals != 0]
    
    # assign category labels to each feature
    feat_cats = np.empty(nfeats, dtype=object)
    nembeddings = items_feat.shape[1]
    feat_cats[:nembeddings] = "embeddings"
    
    catnames = np.array(['embeddings', '_pos_ngram', 'ProductionRule', #'AdjectiveRate', 'AdverbRate', 
         'Rate', 'Ratio', 'DependencyTreeDepth', 'Modal', 'flesch', 'coleman', 'ari', 'sentiment', 'spell_skill', 
         '_length', '_'])
    marks = np.array(['2', 'p', '^', 'H', 'x', ',', 'D', '<', '>', 'v', ',', '8', '1', 'o', '*'])
    col = np.array(['r', 'lightgreen', 'b', 'y', 'purple', 'black', 'darkgoldenrod', 'magenta', 'darkgreen', 'darkblue',
                    'brown', 'darkgray', 'orange', 'dodgerblue', 'lightgray', 'cyan', ])
       
    with open("/home/local/UKP/simpson/data/personalised_argumentation/tempdata/feature_names_all3.txt", 'r') as fh:
        lines = fh.readlines()
    
    featnames = lines[0].strip()
    featidxs = lines[1].strip()
    
    if featnames[-1] == ']':
        featnames = featnames[:-1]
    if featnames[0] == '[':
        featnames = featnames[1:]
        
    featidxs = np.fromstring(featidxs, dtype=int, sep=',') + nembeddings
    featnames = np.array(featnames.split(', '), dtype=str)
    
    for f, fname in enumerate(featnames):
        featnames[f] = featnames[f][2:] # skip the a1 bit at the start
        for catname in catnames:
            if catname in fname:
                print "%i, Recognised %s as type %s" % (f, fname, catname)
                feat_cats[nembeddings + f] = catname
                break
        if not feat_cats[nembeddings + f]:
            print "%i, Unrecognised language feature: %s" % (f, fname)
            feat_cats[nembeddings + f] = 'unigram/other'
    
    feat_cats[feat_cats=='_'] = 'bigram'
    feat_cats[feat_cats=='ari'] = 'readability'
    feat_cats[feat_cats=='coleman'] = 'readability'
    feat_cats[feat_cats=='flesch'] = 'readability'
    feat_cats[feat_cats=='_pos_ngram'] = 'POS'
    feat_cats[feat_cats=='_length'] = 'sentence_length_etc'
    feat_cats[feat_cats=='Modal'] = 'modal_verb'
    for f in range(len(feat_cats)):
        feat_cats[f] = feat_cats[f].lower()

    # sort by length scale
    sorted_idxs = np.argsort(mean_ls)
    sorted_vals = mean_ls[sorted_idxs]
    
    # ignore those that were not valid
    sorted_vals = sorted_vals[totals[sorted_idxs]>0]
    sorted_idxs = sorted_idxs[totals[sorted_idxs]>0]

    sorted_cats = feat_cats[sorted_idxs]
    sorted_cats = sorted_cats[totals[sorted_idxs]>0]
    
    embeddingnames = np.empty(nembeddings, dtype=object)
    for e in range(nembeddings):
        embeddingnames[e] = 'Emb_dimension_%i' % e
        
    featnames = np.concatenate((embeddingnames, featnames))
    sorted_featnames = featnames[sorted_idxs]
    sorted_featnames = sorted_featnames[totals[sorted_idxs]>0]
    
    plt.figure()
    
    # Try a histogram instead? For each length-scale band, how many features of each type are there?    
    
    cat_arr = []
    labels = []
    for c, cat in enumerate(np.unique(feat_cats)):
        clengthscales = sorted_vals[sorted_cats == cat]
        cat_arr.append(clengthscales)
        labels.append(cat)

    plt.hist(cat_arr, label=labels, color=col[:len(labels)], histtype='bar')
    plt.xlabel('length-scale')
    plt.ylabel('no. features')
    plt.legend(loc='best')
    
    plt.figure()

    rowsize = 5
        
    for c, cat in enumerate(np.unique(feat_cats)):
        clengthscales = sorted_vals[sorted_cats == cat]
        #plt.scatter(clengthscales, np.zeros(len(clengthscales)) + (1+c)*1000, marker=marks[c], color=col[c])
        ax = plt.subplot(len(labels)/rowsize + 1, rowsize, c+1)
        plt.plot(clengthscales, color=col[c], label=cat, marker=marks[c], linewidth=0)
        plt.title(cat)
        plt.ylim(np.min(sorted_vals), np.max(sorted_vals))
        
        frame1 = plt.gca()
        if np.mod(c, rowsize):
            frame1.axes.get_yaxis().set_ticks([])
        else:
            plt.ylabel('length-scale')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

    plt.xlabel('features')
    plt.show()
    
    output = np.concatenate((sorted_cats[:, None], featnames[sorted_idxs][:, None], sorted_vals[:, None]), axis=1)
    np.savetxt("./results/features.tsv", output, fmt='%s\t%s\t%s', delimiter='\t', header='category, feature_name, length-scale')