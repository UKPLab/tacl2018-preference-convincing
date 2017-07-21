'''
Show the effect of cycles and undecided labels in preference pairs in simple training datasets on:
- GPPL
- SVC
- Ranking using PageRank

We use these three because they have different ways of modelling preference pairs: as noisy observations at both points;
as classifications; as graphs.  

Created on 20 Jul 2017

@author: simpson
'''
import numpy as np
import matplotlib.pyplot as plt
from tests import run_gppl, load_ling_features, get_features, get_noisy_fold_data, load_embeddings, \
                        compute_lengthscale_heuristic
from data_loading import load_train_test_data
import networkx as nx
from sklearn.svm import SVC
import os

def run_pagerank(trainids_a1, trainids_a2, prefs_train):
    G = nx.DiGraph()
    for i in range(len(trainids_a1)):
        if prefs_train[i] == 2:
            G.add_edge(trainids_a1[i], trainids_a2[i])
        elif prefs_train[i] == 0:
            G.add_edge(trainids_a2[i], trainids_a1[i])
            
    rank_dict = nx.pagerank_numpy(G, alpha=0.85)
    rankscores = np.zeros(len(rank_dict))
    rankscores[rank_dict.keys()] = rank_dict.values()
    return rankscores
    
def run_svm(trainids_a1, trainids_a2, prefs_train, items_feat, testids_a1, testids_a2):    
    svc = SVC(probability=True)
    svc.fit(np.concatenate((np.concatenate((items_feat[trainids_a1, :], items_feat[trainids_a2, :]), axis=1), 
            np.concatenate((items_feat[trainids_a2, :], items_feat[trainids_a1, :]), axis=1)), axis=0), 
            np.concatenate((np.array(prefs_train) / 2.0, 1 - np.array(prefs_train) / 2.0)) )
    #results['SVM'] = svc.decision_function(targets_single_arr)
    proba = svc.predict_proba(np.concatenate((items_feat[testids_a1, :], items_feat[testids_a2, :]), axis=1))
    return proba[:, 0]

def plot_probas(total_p, label, outputdir):
    mean_p = total_p / float(nrepeats)

    # Plot classifications of all pairs as a coloured 3x3 table
    plt.figure()
    data = mean_p.reshape(N, N) # do 1 - to get the preference for the argument along x axis over arg along y axis 
    im = plt.imshow(data, interpolation='nearest')
    plt.grid('on')
    plt.title('%s: Predicted Preferences: p(arg_x > arg_y)' % label)
    plt.xlabel('Argument ID')
    plt.ylabel('Argument ID')
    plt.colorbar(im)  
    
    plt.savefig(outputdir + '/' + label + '_probas.pdf')  
    
def plot_scores(total_f, var_f, label, outputdir):
    mean_f = total_f / float(nrepeats)
        
    # Plot the latent function values for a, b, and c as a bar chart
    plt.figure()
    if var_f is not None:
        var_f /= float(nrepeats)**2
        plt.bar(sample_objs - 0.45, mean_f, 0.9, color='lightblue', yerr=np.sqrt(var_f))
    else:
        plt.bar(sample_objs - 0.45, mean_f, 0.9, color='lightblue')
        
    plt.gca().set_xticks(sample_objs)
    plt.gca().set_xticklabels(obj_labels)
    plt.gca().spines['bottom'].set_position('zero')
    
    if var_f is None:
        plt.title('%s: Estimated latent function values' % label)
    else:
        plt.title('%s: Mean latent function values with STD error bars' % label)
    plt.grid('on')
    plt.xlim([-1, len(sample_objs)])
    
    plt.savefig(outputdir + '/' + label + '_scores.pdf')
    
if __name__ == '__main__':
    # start by loading some realistic feature data. We don't need the preference pairs -- we'll make them up!    
    feature_type = 'embeddings'
    embeddings_type = 'word_mean'
    dataset = 'UKPConvArgStrict'
    method = 'SinglePrefGP_weaksprior_noOpt'
    
    if 'folds' not in globals():
        # load some example data.
        folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = \
                                                                                    load_train_test_data(dataset)    
    
    ling_feat_spmatrix, docids = load_ling_features(dataset)
    embeddings = load_embeddings(word_index_to_embeddings_map)

    
    if 'default_ls_values' not in globals():
        default_ls_values = {}
        
    if dataset in default_ls_values and feature_type in default_ls_values[dataset] and \
            embeddings_type in default_ls_values[dataset][feature_type]:
        default_ls_value = default_ls_values[dataset][feature_type][embeddings_type]
    elif 'GP' in method:
        default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings,
                             ling_feat_spmatrix, docids, folds, index_to_word_map)
        if dataset not in default_ls_values:
            default_ls_values[dataset] = {}
        if feature_type not in default_ls_values[dataset]:
            default_ls_values[dataset][feature_type] = {}
        default_ls_values[dataset][feature_type][embeddings_type] = default_ls_value
    else:
        default_ls_value = []

    fold = folds.keys()[0]
    print("Fold name ", fold)
    trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test,\
                        X, uids, utexts = get_noisy_fold_data(folds, fold, docids, 1.0)                            
        
    items_feat, valid_feats = get_features(feature_type, ling_feat_spmatrix, embeddings_type, trainids_a1, 
                                           trainids_a2, uids, embeddings, X, index_to_word_map, utexts)
    ndims = items_feat.shape[1]
    # Generate simple training data containing a->b, b->c, c->a cycle.
    nrepeats = 25

    total_f_gppl = 0
    total_p_gppl = 0
    total_v_gppl = 0
    total_p_svm = 0    
    total_f_pagerank = 0
    
    for r in range(nrepeats):
        print('*** Running repeat %i ***' % r)
        real_doc_idxs = np.random.choice(items_feat.shape[0], size=3, replace=False)
    
        trainids_a1 = [0, 1, 2, 3, 4]#[real_doc_idxs[0], real_doc_idxs[1], real_doc_idxs[2]]
        trainids_a2 = [1, 2, 0, 4, 3]#[real_doc_idxs[1], real_doc_idxs[2], real_doc_idxs[0]]
        prefs_train = [0, 0, 0, 0, 2]
                
        sample_objs = np.sort(np.unique(np.concatenate((trainids_a1, trainids_a2))))
        sample_objs_ycoords = np.mod(sample_objs, 1+len(sample_objs)/4)#np.random.randint(len(sample_objs)-1, size=len(sample_objs))#[0, 1, 0] # just for plotting nicely
        obj_labels = []
        for arg in sample_objs:
            obj_labels.append('arg%i' % arg)
                
        items_feat_r = items_feat[sample_objs, :]
    
        # test all possible pairs!
        N = len(sample_objs)
        testids_a1 = np.tile(sample_objs[:, None], (1, N)).flatten()
        testids_a2 = np.tile(sample_objs[None, :], (N, 1)).flatten()
    
        if r == 0:
            # Plot the training data graph. Only do this once as plots will look the same -- only arg features change.
            fig = plt.figure(figsize=(4,3))
            
            for p in range(len(prefs_train)):  
                if prefs_train[p] == 0:
                    a1 = trainids_a1[p]
                    a2 = trainids_a2[p]
                    headwidth = 0.1
                elif prefs_train[p] == 2:
                    a1 = trainids_a2[p]
                    a2 = trainids_a1[p]
                    headwidth = 0.1
                else:
                    headwidth = 0
                                                                    
                plt.arrow(sample_objs[a1], sample_objs_ycoords[a1], 
                          (sample_objs[a2] - sample_objs[a1]) / 2.0, 
                          (sample_objs_ycoords[a2] - sample_objs_ycoords[a1]) / 2.0,
                          color='black', head_width=headwidth)            
                plt.arrow(sample_objs[a1] + (sample_objs[a2]-sample_objs[a1]) / 2.0, 
                          sample_objs_ycoords[a1] + (sample_objs_ycoords[a2] -
                                                                 sample_objs_ycoords[a1]) / 2.0, 
                          (sample_objs[a2]-sample_objs[a1]) / 2.0, 
                          (sample_objs_ycoords[a2] - sample_objs_ycoords[a1]) / 2.0,
                          color='black')

            
            plt.scatter(sample_objs, sample_objs_ycoords, marker='o', s=400, color='black')
            
            for obj in range(len(sample_objs)):
                plt.text(sample_objs[obj]+0.18, sample_objs_ycoords[obj]+0.08, obj_labels[obj])
                        
            plt.xlim([-1.0, len(sample_objs)])
            plt.ylim([-0.5, np.max(sample_objs_ycoords) + 0.5])        
            plt.axis('off')
            plt.title('Argument Preference Graph')
    
        # Run GPPL
        proba, predicted_f, _, model = run_gppl(fold, None, method, trainids_a1, trainids_a2, prefs_train, items_feat_r, 
                 embeddings, X, ndims, False, testids_a1, testids_a2, None, None, 
                 default_ls_value, verbose=False, item_idx_ranktrain=None, rankscores_train=None, 
                 item_idx_ranktest=sample_objs)      
        _, f_var = model.predict_f(sample_objs, use_training_items=True)
        
        total_p_gppl += proba
        total_v_gppl += f_var
        total_f_gppl += predicted_f    
        
        # Run SVC
        proba = run_svm(trainids_a1, trainids_a2, prefs_train, items_feat, testids_a1, testids_a2)
        total_p_svm += proba
        
        # Run PageRank        
        predicted_f = run_pagerank(trainids_a1, trainids_a2, prefs_train)      
        total_f_pagerank += predicted_f           
    
    output_dir = os.path.expanduser(
        '~/git/crowdsourcing_argumentation/documents/pref_learning_for_convincingness/figures/cycles_demo/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plot_probas(total_p_gppl, 'GPPL', output_dir)
    plot_scores(total_f_gppl, total_v_gppl, 'GPPL', output_dir)
    plot_probas(total_p_svm, 'SVM', output_dir)
    plot_scores(total_f_pagerank, None, 'PageRank', output_dir)
    
    plt.show()
    