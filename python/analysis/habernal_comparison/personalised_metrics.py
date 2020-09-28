import os
import sys

from compute_metrics import compute_metrics

if __name__ == '__main__':
    
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None

    expt_settings['foldorderfile'] = None

    if len(sys.argv) > 2:
        npairs = int(sys.argv[2])
    else:
        npairs = 0 #5000

    if len(sys.argv) > 3:
        lsm = int(sys.argv[3])
    else:
        lsm = 1

    if len(sys.argv) > 4:
        turker_filter = int(sys.argv[4])
    else:
        turker_filter = 0

    expt_root_dir = 'D05-%i_P%i' % (lsm, npairs)

    print('Loading data from %s' % expt_root_dir)

    min_fold_no = 0
    max_fold_no = 32

    npairs = 0
    acc = 1.0
    di = 0

    test_to_run = int(sys.argv[1])

    eval_training_set = True

    if test_to_run == 6:
        # Personalised predictions -- single-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['SinglePrefGP_noOpt_weaksprior']
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                          split_by_person=True, compute_tr_performance=eval_training_set, turker_filter=turker_filter)
        print("Completed compute metrics")

    elif test_to_run == 7:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_noOpt_weaksprior']
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                          compute_tr_performance=eval_training_set, turker_filter=turker_filter)
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
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)
        print("Completed compute metrics")

    elif test_to_run == 5:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_weaksprior'] # 'GP+SVM','SinglePrefGP_noOpt_weaksprior', 'SingleGPC_noOpt_weaksprior',
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)
        print("Completed compute metrics")

    elif test_to_run == 0:
        # Personalised predictions -- multi-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['PersPrefGP_commonmean_noOpt_weaksprior'] # 'SinglePrefGP_noOpt_weaksprior',
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True,
                          compute_tr_performance=eval_training_set, turker_filter=turker_filter)

        print("Completed compute metrics")

    elif test_to_run == 13:
        # Personalised predictions -- multi-user methods

        datasets = ['UKPConvArgCrowdSample']
        methods = ['PersPrefGP_noOpt_weaksprior'] # 'SinglePrefGP_noOpt_weaksprior',
        feature_types = ['both'] # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
        = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True,
                          compute_tr_performance=eval_training_set, turker_filter=turker_filter)

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
                          min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                          compute_tr_performance=eval_training_set, turker_filter=turker_filter)

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
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir, split_by_person=True,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)

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
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)

        print("Completed compute metrics")

    elif test_to_run == 8:
        # Consensus predictions -- multi-user methods (single-user methods were already included in the TACL paper so
        # can be copied from there).
        datasets = ['UKPConvArgCrowdSample']
        methods = [
            #'crowdBT',
            'cBT_GP',
        ]
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                              split_by_person=True, compute_tr_performance=eval_training_set, turker_filter=turker_filter)

        print("Completed compute metrics")

    elif test_to_run == 9:
        # Consensus predictions -- multi-user methods (single-user methods were already included in the TACL paper so
        # can be copied from there).
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = [
            #'crowdBT',
            'cBT_GP',
        ]
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)

        print("Completed compute metrics")

    elif test_to_run == 11:
        # Consensus predictions -- multi-user methods (single-user methods were already included in the TACL paper so
        # can be copied from there).
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = [
            'crowdBT',
        ]
        feature_types = ['both']  # 'both'
        embeddings_types = ['word_mean']

        results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
        tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
            = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                              min_fold_no=min_fold_no, max_fold_no=max_fold_no, foldername=expt_root_dir,
                              compute_tr_performance=eval_training_set, turker_filter=turker_filter)

        print("Completed compute metrics")