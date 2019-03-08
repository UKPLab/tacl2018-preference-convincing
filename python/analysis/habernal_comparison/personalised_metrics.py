import os
import sys

from compute_metrics import compute_metrics

if __name__ == '__main__':
    
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None

    expt_settings['foldorderfile'] = None

    expt_root_dir = 'personalised_Qfix3'  #13_bigger_t'#10_from_cluster'

    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'
    min_fold_no = 0
    max_fold_no = 32

    npairs = 0
    acc = 1.0
    di = 0

    test_to_run = int(sys.argv[1])

    if test_to_run == 6:
        # Personalised predictions -- single-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['SinglePrefGP_noOpt_weaksprior']
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True)
        print("Completed compute metrics")

    elif test_to_run == 7:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_noOpt_weaksprior']
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          max_fold_no=max_fold_no, foldername=expt_root_dir)
        print("Completed compute metrics")

    if test_to_run == 4:
        # Personalised predictions -- single-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['SinglePrefGP_weaksprior'] # 'GP+SVM','SinglePrefGP_noOpt_weaksprior', 'SingleGPC_noOpt_weaksprior',
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True)
        print("Completed compute metrics")

    elif test_to_run == 5:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_weaksprior'] # 'GP+SVM','SinglePrefGP_noOpt_weaksprior', 'SingleGPC_noOpt_weaksprior',
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              max_fold_no=max_fold_no, foldername=expt_root_dir)
        print("Completed compute metrics")

    elif test_to_run == 0:
        # Personalised predictions -- multi-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['PersPrefGP_commonmean_noOpt_weakersprior'] # 'SinglePrefGP_noOpt_weaksprior',
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True)

        print("Completed compute metrics")

    elif test_to_run == 1:
        # Consensus predictions -- multi-user methods (single-user methods were already included in the TACL paper so
        # can be copied from there).
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['PersConsensusPrefGP_commonmean_noOpt_weaksprior'] # 'SinglePrefGP_noOpt_weaksprior',
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          max_fold_no=max_fold_no, foldername=expt_root_dir)

        print("Completed compute metrics")

    elif test_to_run == 2:
        # Personalised predictions -- multi-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['SinglePrefGP_weaksprior', 'PersPrefGP_commonmean_weaksprior']
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True)

        print("Completed compute metrics")

    elif test_to_run == 3:
        # Consensus predictions -- multi-user methods (single-user methods were already included in the TACL paper so
        # can be copied from there).
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_weaksprior', 'PersConsensusPrefGP_commonmean_weaksprior']
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              max_fold_no=max_fold_no, foldername=expt_root_dir)

        print("Completed compute metrics")
