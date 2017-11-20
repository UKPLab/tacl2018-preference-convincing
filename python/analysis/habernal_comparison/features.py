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
from tests import load_train_test_data, load_embeddings, load_ling_features, get_fold_data, load_features
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
    
    dataset = 'UKPConvArgStrict'#'UKPConvArgAll_evalMACE'#
    method = 'SinglePrefGP_weaksprior'
    feature_type = 'both'
    embeddings_type = 'word_mean'
    di = 0.00
      
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsfile = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/' + \
        resultsfile_template % (dataset, method, 
        feature_type, embeddings_type, 1.0, di) + '_test.pkl'
                    
    resultsdir = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/' + \
        resultsfile_template % (dataset, method, 
        feature_type, embeddings_type, 1.0, di)      

    foldorderfile = None
    if foldorderfile is not None:
        fold_order = np.genfromtxt(os.path.expanduser(foldorderfile), 
                                                    dtype=str)
    elif os.path.isfile(resultsdir + '/foldorder.txt'):
        fold_order = np.genfromtxt(os.path.expanduser(resultsdir + '/foldorder.txt'), 
                                                    dtype=str)
    else:
        fold_order = None 
    nFolds = 1
    start_fold = 12
    end_fold = 12
    if os.path.isfile(resultsfile): 
        
        with open(resultsfile, 'r') as fh:
            data = pickle.load(fh)
                
        if nFolds < 1:
            nFolds = len(data[0])
    else:
        data = None                        

    min_folds = 0
      
    # Sort the features by their ID. 
    # If we have discarded some features that were all zeros, the current index will not be the original feature idx.
    # How to map them back? Reload the original data and find out which features were discarded.
      
    folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = load_train_test_data(dataset)
    word_embeddings = load_embeddings(word_index_to_embeddings_map)
    ling_feat_spmatrix, docids = load_ling_features(dataset)
      
    #default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, word_embeddings,
    #                                         ling_feat_spmatrix, docids, folds, index_to_word_map)      
      
    mean_ls = None
    for foldidx, fold in enumerate(folds.keys()):
        if foldidx < start_fold or foldidx > end_fold:
            continue
        
        if fold_order is None: # fall back to the order on the current machine
            fold = folds.keys()[foldidx]
        else:
            fold = fold_order[foldidx] 
            if fold[-2] == "'" and fold[0] == "'":
                fold = fold[1:-2]
            elif fold[-1] == "'" and fold[0] == "'":
                fold = fold[1:-1]  
            fold_order[foldidx] = fold        
        
        # look for new-style data in separate files for each fold. Prefer new-style if both are found.
        foldfile = resultsdir + '/fold%i.pkl' % foldidx
        if os.path.isfile(foldfile):
            with open(foldfile, 'r') as fh:
                data_f = pickle.load(fh)
        else: # convert the old stuff to new stuff
            if data is None:
                min_folds = foldidx+1
                print 'Skipping fold with no data %i' % foldidx
                print "Skipping results for %s, %s, %s, %s" % (method, 
                                                               dataset, 
                                                               feature_type, 
                                                               embeddings_type)
                print "Skipped filename was: %s, old-style results file would be %s" % (foldfile, 
                                                                                        resultsfile)
                continue        
        
            if not os.path.isdir(resultsdir):
                os.mkdir(resultsdir)
            data_f = []
            for thing in data:
                if foldidx in thing:
                    data_f.append(thing[foldidx])
                else:
                    data_f.append(thing)
            with open(foldfile, 'w') as fh:
                pickle.dump(data_f, fh)        
        
        trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, \
                                                                    X, uids, utexts = get_fold_data(folds, fold, docids)
                                                                                  
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        items_feat, valid_feats = load_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, trainids_a2, uids, 
                                                word_embeddings, X)
              
        nfeats = len(valid_feats)
        # take the mean ls for each feature across the folds
        if foldidx == 0 or mean_ls is None:
            mean_ls = np.zeros(nfeats, dtype=float) 
            totals = np.zeros(nfeats, dtype=int)
              
        #print "Warning: not computing means."
        mean_ls[valid_feats] += data_f[7] / data_f[5]
        print "Max normed l: %f" % np.max(data_f[7] / data_f[5])
        totals[valid_feats] += 1
         
    mean_ls = mean_ls[valid_feats]
    totals = totals[valid_feats]
    mean_ls[totals != 0] = mean_ls[totals != 0] / totals[totals != 0]
    
    if feature_type == 'debug':
        feat_cats = np.array(['one', 'two', 'three'])
        featnames = feat_cats
        col = np.array(['r', 'lightgreen', 'b'])
        marks = np.array(['2', 'p', '^'])
        nembeddings = 3
    else:
        # assign category labels to each feature
        feat_cats = np.empty(nfeats, dtype=object)
        nembeddings = word_embeddings.shape[1]
        feat_cats[:nembeddings] = "embeddings"
        
        catnames = np.array(['embeddings', '_pos_ngram', 'ProductionRule', #'AdjectiveRate', 'AdverbRate', 
             'Rate', 'Ratio', 'DependencyTreeDepth', 'Modal', 'flesch', 'coleman', 'ari', 'sentiment', 'spell_skill', 
             '_length', '_'])
        marks = np.array(['2', 'p', '^', 'H', 'x', ',', 'D', '<', '>', 'v', ',', '8', '1', 'o', '*'])
        col = np.array(['r', 'lightgreen', 'b', 'y', 'purple', 'black', 'darkgoldenrod', 'magenta', 'darkgreen', 'darkblue',
                        'brown', 'darkgray', 'orange', 'dodgerblue', 'lightgray', 'cyan', ])
           
        with open(data_root_dir + "/tempdata/feature_names_all3.txt", 'r') as fh:
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
                feat_cats[nembeddings + f] = 'unigram'
        
        feat_cats[feat_cats=='_'] = 'bigram'
        feat_cats[feat_cats=='ari'] = 'readability'
        feat_cats[feat_cats=='coleman'] = 'readability'
        feat_cats[feat_cats=='flesch'] = 'readability'
        feat_cats[feat_cats=='spell_skill'] = 'readability'
        feat_cats[feat_cats=='Rate'] = 'grammar'
        feat_cats[feat_cats=='Ratio'] = 'grammar'
        feat_cats[feat_cats=='_pos_ngram'] = 'POS'
        feat_cats[feat_cats=='_length'] = 'readability'
        feat_cats[feat_cats=='Modal'] = 'grammar'
        feat_cats[feat_cats=='DependencyTreeDepth'] = 'dep. tree'
        feat_cats[feat_cats=='ProductionRule'] = 'dep. tree'
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
    
    '''
    An alternative to plotting the distributions would be to list the top ten most important and least important features.
    '''
    figure_path = os.path.expanduser('./documents/pref_learning_for_convincingness/figures/features/')
    
    np.savetxt(figure_path + '/feature_table.tex', np.concatenate((sorted_featnames[:, None], sorted_vals[:, None]), 
                                                                  axis=1), fmt='%s & %.5f \\nonumber\\\\')
    
    # Try a histogram instead? For each length-scale band, how many features of each type are there?    
    plt.figure()
    
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
   
    plt.savefig(figure_path + 'hist.pdf') 
    
    # produce content for a latex table
    
    plt.figure(figsize=(10,3))
    
    meds = []
    low = []
    high = []
    mins = []
    maxs = []
    vals = []
    for c, cat in enumerate(np.unique(feat_cats)):
        clengthscales = sorted_vals[sorted_cats == cat]
        #print '%s & %s & %s' & (cat, np.median(clengthscales), np.percentile(clengthscales, 25), np.percentile(clengthscales, 75))
        #meds.append(np.median(clengthscales))
        #low.append(np.percentile(clengthscales, 25))
        #high.append(np.percentile(clengthscales, 75))
        #mins.append(np.min(clengthscales))
        #maxs.append(np.max(clengthscales))
        vals.append(clengthscales)
        
    bp = plt.boxplot(vals, labels=labels, notch=0, whiskerprops={'linestyle':'solid'}, 
                     patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')    
    for patch in bp['boxes']:
        patch.set_facecolor('tan')
    plt.ylabel('Mean normalised length-scale')
    plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.gca().set_axisbelow(True)
    
    plt.ylim(0,3)
    
    plt.savefig(figure_path + 'boxplot.pdf')
    
    ############
    
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