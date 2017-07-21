'''
Created on 20 Jul 2017

@author: simpson
'''
import os
#import numpy as np
import matplotlib.pyplot as plt

import compute_metrics

def plot_active_learning_results(results, ylabel, title=None, ax=None):
    ax = results.plot(kind='line', ax=ax, title=title, legend=True)
    plt.ylabel(ylabel)
    plt.xlabel('No. labels')
    return ax

if __name__ == '__main__':
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None 
        expt_settings['foldorderfile'] = None
      
    compute_metrics.data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    compute_metrics.foldorderfile = None
    compute_metrics.resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'
        
    ax1 = None
    ax2 = None
    ax3 = None
    ax4 = None
    ax5 = None
    ax6 = None
    ax7 = None
    ax8 = None
    ax9 = None
      
    npairs = 0
    di = 0
  
#     # Active Learning experiments
    methods = ['BI-LSTM']#['SinglePrefGP_weaksprior']#, 'SinglePrefGP_noOpt_weaksprior'] # 'SVM',
    # shouldn't need this any more because we look for the file automatically 
    #expt_settings['foldorderfile'] = "~/Dropbox/titanx_foldorder.txt" # change this depending on where we ran the tests... None if no file available.
    datasets = ['UKPConvArgCrowdSample_evalMACE']#['UKPConvArgStrict']
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']#'skipthoughts']
    npairs = 200#11126
    di = 2#1000
    max_no_folds = 32
       
    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics.compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs, 
                      max_no_folds=max_no_folds)
       
    ax1 = plot_active_learning_results(mean_results[0], 'F1 score', 'Mean over test topics')
    ax2 = plot_active_learning_results(mean_results[7], 'F1 score', 'Mean over training topics')
  
    ax8 = plot_active_learning_results(mean_results[1], 'Accuracy', 'Mean over test topics')
    ax9 = plot_active_learning_results(mean_results[8], 'Accuracy', 'Mean over training topics')  
  
    ax3 = plot_active_learning_results(mean_results[2], 'AUC', 'Mean over test topics')
    ax4 = plot_active_learning_results(mean_results[9], 'AUC', 'Mean over training topics')
  
    ax5 = plot_active_learning_results(mean_results[4], 'Pearson correlation', 'Mean over test topics')
    ax6 = plot_active_learning_results(mean_results[5], 'Spearman correlation', 'Mean over test topics')
    ax7 = plot_active_learning_results(mean_results[6], "Kendall's Tau", 'Mean over test topics')
  
    expt_settings['foldorderfile'] = None
    expt_settings['fold_order'] = None
  
    methods = ['SinglePrefGP_noOpt_weaksprior']#['SinglePrefGP_weaksprior']#, 'SinglePrefGP_noOpt_weaksprior'] # 'SVM',
    datasets = ['UKPConvArgCrowdSample_evalMACE']#['UKPConvArgStrict']
    feature_types = ['both']
    embeddings_types = ['word_mean']#'skipthoughts']
    npairs = 200#11126
    di = 2#1000
    max_no_folds = 32
      
    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics.compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs, 
                      max_no_folds=max_no_folds)
      
    ax1 = plot_active_learning_results(mean_results[0], 'F1 score', 'Mean over test topics', ax1)
    ax2 = plot_active_learning_results(mean_results[7], 'F1 score', 'Mean over training topics', ax2)
     
    ax8 = plot_active_learning_results(mean_results[1], 'Accuracy', 'Mean over test topics', ax8)
    ax9 = plot_active_learning_results(mean_results[8], 'Accuracy', 'Mean over training topics', ax9)     
     
    ax3 = plot_active_learning_results(mean_results[2], 'AUC', 'Mean over test topics', ax3)
    ax4 = plot_active_learning_results(mean_results[9], 'AUC', 'Mean over training topics', ax4)
     
    ax5 = plot_active_learning_results(mean_results[4], 'Pearson correlation', 'Mean over test topics', ax5)
    ax6 = plot_active_learning_results(mean_results[5], 'Spearman correlation', 'Mean over test topics', ax6)
    ax7 = plot_active_learning_results(mean_results[6], "Kendall's Tau", 'Mean over test topics', ax7)
    
    methods = ['SVM']#['SinglePrefGP_weaksprior']#, 'SinglePrefGP_noOpt_weaksprior'] # 'SVM',
    datasets = ['UKPConvArgCrowdSample_evalMACE']#['UKPConvArgStrict']
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']#'skipthoughts']
    npairs = 200#11126
    di = 2#1000
    max_no_folds = 32
    
    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics.compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs, 
                      max_no_folds=max_no_folds)
    
    ax1 = plot_active_learning_results(mean_results[0], 'F1 score', 'Mean over test topics', ax1)
    ax2 = plot_active_learning_results(mean_results[7], 'F1 score', 'Mean over training topics', ax2)    

    ax8 = plot_active_learning_results(mean_results[1], 'Accuracy', 'Mean over test topics', ax8)
    ax9 = plot_active_learning_results(mean_results[8], 'Accuracy', 'Mean over training topics', ax9)
    
    ax3 = plot_active_learning_results(mean_results[2], 'AUC', 'Mean over test topics', ax3)
    ax4 = plot_active_learning_results(mean_results[9], 'AUC', 'Mean over training topics', ax4)    
    
    ax5 = plot_active_learning_results(mean_results[4], 'Pearson correlation', 'Mean over test topics', ax5)
    ax6 = plot_active_learning_results(mean_results[5], 'Spearman correlation', 'Mean over test topics', ax6)
    ax7 = plot_active_learning_results(mean_results[6], "Kendall's Tau", 'Mean over test topics', ax7)    
    
    figure_save_path = '/home/local/UKP/simpson/git/crowdsourcing_argumentation/documents/pref_learning_for_convincingness/figures/active_learning'
    
    plt.figure(ax1.figure.number)
    plt.grid()
    ax1.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_f1')
    
    plt.figure(ax2.figure.number)
    plt.grid()
    ax2.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/train_f1')
    
    plt.figure(ax8.figure.number)
    plt.grid()
    ax8.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_acc')
    
    plt.figure(ax9.figure.number)
    plt.grid()
    ax9.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/train_acc')    
        
    plt.figure(ax3.figure.number)
    plt.grid()
    ax3.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_auc')
    
    plt.figure(ax4.figure.number)
    plt.grid()
    ax4.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/train_auc')
        
    plt.figure(ax5.figure.number)
    plt.grid()
    ax5.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_pearson')
        
    plt.figure(ax6.figure.number)
    plt.grid()
    ax6.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_spearman')
        
    plt.figure(ax7.figure.number)
    plt.grid()
    ax7.figure.set_size_inches(6, 4)
    plt.tight_layout()
    plt.legend(labels=['Bi-LSTM', 'GPPL', 'SVM'], loc='best')
    plt.savefig(figure_save_path + '/test_kendall')    