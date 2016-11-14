'''
Compute classification metrics for the preference learning models. Plot the predictions.

Created on 21 Oct 2016

@author: simpson
'''
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, log_loss, accuracy_score

def compute_metrics(nmethods, gold_prefs, predictions):
    
    # Task C2, C4: Compute accuracy metrics ---------------------------------------------------------------------------
    logging.info('Task C2/C4, accuracy metrics')
    metrics = {}
    metrics['acc'] = np.zeros(nmethods)
    metrics['f1'] = np.zeros(nmethods)
    metrics['auc_roc'] = np.zeros(nmethods)
    metrics['log_loss'] = np.zeros(nmethods)
     
    # Not sure how to deal with preference labels where the true label is 0.5 with f1 score. For ROC curve we can 
    # combine two AUCs for negative class and positive class.  
     
    for i in range(nmethods):        
        ind_array = np.concatenate( ( predictions[:, i:i+1] < 1.0/3.0, 
                                      (predictions[:, i:i+1] >= 1.0/3.0) & (predictions[:, i:i+1] < 2.0/3.0),
                                      predictions[:, i:i+1] > 2.0/3.0 ), axis=1)
        ind_array_gold = np.concatenate( ( gold_prefs[:, np.newaxis] == 0, 
                                           gold_prefs[:, np.newaxis] == 0.5,
                                           gold_prefs[:, np.newaxis] == 1 ), axis=1)
        
        mistakes = np.round(ind_array) != ind_array_gold
        print ind_array[np.sum(mistakes, axis=1), :]
        print ind_array_gold[np.sum(mistakes, axis=1), :]
        
        metrics['acc'][i] = accuracy_score(ind_array_gold, ind_array)
        metrics['f1'][i] = f1_score(ind_array_gold, ind_array, average='weighted')
         
        auc_a_less_b = roc_auc_score(gold_prefs==0, 1 - predictions[:, i])
        frac_a_less_b = np.sum(gold_prefs==0) / float(len(gold_prefs))
          
        auc_a_more_b = roc_auc_score(gold_prefs==1, predictions[:, i])
        frac_a_more_b = np.sum(gold_prefs==1) / float(len(gold_prefs))
          
        auc_a_equal_b = roc_auc_score(gold_prefs==0.5, 2 * (1 - np.abs(predictions[:, i] - 0.5)))
        frac_a_equal_b = np.sum(gold_prefs==0.5) / float(len(gold_prefs))
          
        metrics['auc_roc'][i] = auc_a_less_b * frac_a_less_b + auc_a_more_b * frac_a_more_b + auc_a_equal_b * frac_a_equal_b 
          
        predictions_safe = predictions[:, i].copy()
        predictions_safe[predictions[:, i]<1e-7] = 1e-7
        predictions_safe[predictions[:, i]>(1-1e-7)] = 1 - 1e-7
        metrics['log_loss'][i] = -np.mean(gold_prefs * np.log(predictions_safe) + (1 - gold_prefs) * np.log(1 - predictions_safe))
    
    return metrics

def plot_metrics(plotdir, metrics, nmethods, method_labels, nfolds, nanno, nanno_is_min=False):
    
    # Task C9/C10: Plotting metrics -----------------------------------------------------------------------------------
    logging.info('Task C9/10, plotting accuracy metrics')
    _, ax = plt.subplots()
    if nanno_is_min:
        ax.set_title('F1 Scores with %i-fold Cross Validation (data points with at least %i annotators)' % (nfolds, nanno))
    else:
        ax.set_title('F1 Scores with %i-fold Cross Validation (data points with %i annotators)' % (nfolds, nanno))
    ind = np.arange(nmethods)
    width = 0.6
    ax.bar(ind, metrics['f1'], width=width)
    ax.set_xlabel('Method')
    ax.set_ylabel('F1 Score')
    ax.set_xticks(ind + (width/2.0))
    ax.set_xticklabels(method_labels)
     
    plt.savefig(plotdir + '/f1scores.eps') 
     
    _, ax = plt.subplots()
    ax.set_title('AUC of ROC Curve with %i-fold Cross Validation' % nfolds)
    ax.bar(ind, metrics['auc_roc'], width=width)
    ax.set_xlabel('Method')
    ax.set_ylabel('AUC')
    ax.set_xticks(ind + (width/2.0))
    ax.set_xticklabels(method_labels)
         
    plt.savefig(plotdir + '/auc_roc.eps')
     
    _, ax = plt.subplots()
    ax.set_title('Cross Entropy Error with %i-fold Cross Validation' % nfolds)
    plt.bar(ind, metrics['log_loss'], width=width)
    ax.set_xlabel('Method')
    ax.set_ylabel('Cross Entropy')
    ax.set_xticks(ind + (width/2.0))
    ax.set_xticklabels(method_labels)
         
    plt.savefig(plotdir + '/cross_entropy.eps')
    
    _, ax = plt.subplots()
    ax.set_title('Accuracy with %i-fold Cross Validation' % nfolds)
    plt.bar(ind, metrics['acc'], width=width)
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ind + (width/2.0))
    ax.set_xticklabels(method_labels)
         
    plt.savefig(plotdir + '/accuracy.eps')                    