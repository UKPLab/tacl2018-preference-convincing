'''
Load a results file generated by "tests.py". Compute the key metrics for pairwise preference labeling:

- F1 score
- AUC (can also be computed from precision and recall, the total numbers of each class in the gold standard and the 
total number of instances marked as positive, if the method gives only discrete labels). 
- Cross entropy error

Compute key ranking metrics:

- Kendall's tau
- What was used in Habernal paper?

Plot results? Could show a bar chart to compare performance when we have a lot of methods. Mainly, we can use tables
with median and quartiles (check against habernal paper to ensure it is comparable).

Created on 18 May 2017

@author: edwin
'''

import numpy as np
import pandas as pd
import os 
import pickle
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from scipy.stats import pearsonr, spearmanr, kendalltau
from tests import load_train_test_data

def get_fold_data(data, f):
    # discrete labels are 0, 1 or 2
    gold_disc = np.array(data[3][f])
    pred_disc = np.array(data[1][f]) * 2
    # probabilities
    gold_prob = gold_disc / 2.0
    pred_prob = np.array(data[0][f])
    
    gold_disc = gold_disc[np.abs(pred_prob - 0.5) > 0.3]
    pred_disc = pred_disc[np.abs(pred_prob - 0.5) > 0.3] 
    
    gold_prob = gold_prob[np.abs(pred_prob - 0.5) > 0.3] 
    pred_prob = pred_prob[np.abs(pred_prob - 0.5) > 0.3] 
    
    # scores used to rank
    gold_rank = np.array(data[4][f])
    pred_rank = np.array(data[2][f])
    
    if pred_rank.ndim == 3:
        pred_rank = pred_rank[0]
    pred_rank = pred_rank.flatten()

    return gold_disc, pred_disc, gold_prob, pred_prob, gold_rank, pred_rank

if __name__ == '__main__':
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")

    datasets = ['UKPConvArgAll', 'UKPConvArgMACE', 'UKPConvArgStrict']
    
    methods = ['SinglePrefGP', 'SinglePrefGP_oneLS']
    #methods = ['PersonalisedPrefsBayes', 'PersonalisedPrefsFA']
    #methods = ['IndPrefGP', 'PersonalisedPrefsNoFactors'] # IndPrefGP means separate preference GPs for each worker 
    
    feature_types = ['both', 'embeddings', 'ling'] # can be 'embeddings' or 'ling' or 'both'
    embeddings_to_use = ['word_mean']#, 'skipthoughts', 'siamese_cbow']
    
    folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map = load_train_test_data(datasets[0])
    
    results_f1      = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))
    results_acc     = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))
    results_logloss = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))
    results_auc     = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))

    results_pearson  = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))
    results_spearman = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))
    results_kendall  = np.zeros((len(methods) * len(datasets), len(feature_types) * len(embeddings_to_use), len(folds)))

    row_index = np.zeros(len(methods) * len(datasets), dtype=str)
    columns = np.zeros(len(feature_types) * len(embeddings_to_use), dtype=str)
    
    row = 0
    
    for method in methods:
        for dataset in datasets:
            
            folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map = load_train_test_data(dataset)
            
            row_index[row] = method + ', ' + dataset
            col = 0
            
            for feature_type in feature_types:
                for embeddings_type in embeddings_to_use:    
                    resultsfile = data_root_dir + 'outputdata/crowdsourcing_argumentation_expts/' + \
                    'habernal_%s_%s_%s_%s_test.pkl' % (dataset, method, feature_type, embeddings_type)
                    
                    if os.path.isfile(resultsfile): 
                        
                        with open(resultsfile, 'r') as fh:
                            data = pickle.load(fh)
                                
                        nFolds = len(data[0])

                        for f in range(nFolds):
                            gold_disc, pred_disc, gold_prob, pred_prob, gold_rank, pred_rank = get_fold_data(data, f)
                        
                            results_f1[row, col, f]      = f1_score(gold_disc[gold_disc!=1], pred_disc[gold_disc!=1], 
                                                                    average='macro')
                            #skip the don't knows
                            results_acc[row, col, f]     = accuracy_score(gold_disc[gold_disc!=1], 
                                                                          pred_disc[gold_disc!=1]) 
                            
                            results_logloss[row, col, f] = log_loss(gold_prob[gold_disc!=1], pred_prob[gold_disc!=1])
                            results_auc[row, col, f]     = roc_auc_score(gold_prob[gold_disc!=1], pred_prob[gold_disc!=1]) # macro
    
                            results_pearson[row, col, f]  = pearsonr(gold_rank, pred_rank)[0]
                            results_spearman[row, col, f] = spearmanr(gold_rank, pred_rank)[0]
                            results_kendall[row, col, f]  = kendalltau(gold_rank, pred_rank)[0]
                          
                        results_f1[row, col, -1] = np.mean(results_f1[row, col, :-1])
                        results_acc[row, col, -1] = np.mean(results_acc[row, col, :-1])
                        results_logloss[row, col, -1] = np.mean(results_logloss[row, col, :-1])
                        results_auc[row, col, -1] = np.mean(results_auc[row, col, :-1])
                        
                        results_pearson[row, col, -1] = np.mean(results_pearson[row, col, :-1])
                        results_spearman[row, col, -1] = np.mean(results_spearman[row, col, :-1])
                        results_kendall[row, col, -1] = np.mean(results_kendall[row, col, :-1])
                            
                    else:
                        print "Skipping results for %s, %s, %s, %s" % (method, dataset, feature_type, embeddings_type)
                        print "Skipped filename was: %s" % resultsfile
                    
                    if row == 1: # set the column headers    
                        columns[col] = feature_type + ', ' + embeddings_type
                    
                    col += 1
                    
            row += 1

    mean_results_f1 = pd.DataFrame(results_f1[:, :, -1], columns=columns, index=row_index)
    print "Macro-F1 scores: "
    print mean_results_f1
        
    mean_results_acc = pd.DataFrame(results_acc[:, :, -1], columns=columns, index=row_index)
    print "Accuracy (for UKPConvArgAll and UKPConvArgMACE we now exclude don't knows; for UKPConvArgStrict they are\
    already ommitted):"
    print mean_results_acc
    
    mean_results_auc = pd.DataFrame(results_auc[:, :, -1], columns=columns, index=row_index)
    print "AUC ROC (if AUC is higher than accuracy and F1 score, it suggests that decision boundary is not calibrated\
    or that accuracy may improve if we exclude data points close to the decision boundary): "
    print mean_results_auc
    
    mean_results_logloss = pd.DataFrame(results_logloss[:, :, -1], columns=columns, index=row_index)
    print "Cross Entropy classification error (quality of the probability labels is taken into account)"
    print mean_results_logloss
    
    mean_results_pearson = pd.DataFrame(results_pearson[:, :, -1], columns=columns, index=row_index)
    print "Pearson's r:"
    print mean_results_pearson
    
    mean_results_spearman = pd.DataFrame(results_spearman[:, :, -1], columns=columns, index=row_index)
    print "Spearman's rho:"
    print mean_results_spearman
    
    mean_results_kendall = pd.DataFrame(results_kendall[:, :, -1], columns=columns, index=row_index)
    print "Kendall's tau:"
    print mean_results_kendall
    
    # TODO: Show how the method resolves cycles
    
    # TODO: Show the features that were chosen
    
    # TODO: Correlations between reasons and features?
    
    # TODO: Correlations between reasons and latent argument features found using preference components?
                    