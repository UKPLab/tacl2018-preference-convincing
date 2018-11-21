from tests import TestRunner

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    # For plots 1 and 2 ------------------------------------------------------------------------------------------------
    datasets = ['UKPConvArgCrowdSample_evalMACE']
    methods = ['SinglePrefGP_noOpt_weaksprior_M2',
               'SinglePrefGP_noOpt_weaksprior_M10',
               'SinglePrefGP_noOpt_weaksprior_M100',
               'SinglePrefGP_noOpt_weaksprior_M200',
               'SinglePrefGP_noOpt_weaksprior_M300',
               'SinglePrefGP_noOpt_weaksprior_M400',
               'SinglePrefGP_noOpt_weaksprior_M500',
               'PersPrefGP_commonmean_noOpt_weaksprior_M2',
               'PersPrefGP_commonmean_noOpt_weaksprior_M10',
               'PersPrefGP_commonmean_noOpt_weaksprior_M100',
               'PersPrefGP_commonmean_noOpt_weaksprior_M200',
               'PersPrefGP_commonmean_noOpt_weaksprior_M300',
               'PersPrefGP_commonmean_noOpt_weaksprior_M400',
               'PersPrefGP_commonmean_noOpt_weaksprior_M500',
            ]
    feature_types = ['both', 'embeddings']
    embeddings_types = ['word_mean']

    runner = TestRunner('personalised', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)

    # For Plot 3: Scaling with N_tr ------------------------------------------------------------------------------------
    datasets = ['UKPConvArgCrowdSampl_evalMACE']
    methods = ['SinglePrefGP_noOpt_weaksprior_M0', 'SinglePrefGP_noOpt_weaksprior_M100',
               'PersPrefGP_noOpt_weaksprior_M0', 'PersPrefGP_noOpt_weaksprior_M100',
               'SVM', 'BI-LSTM'
        ] # M0 will mean no SVI
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']

    runner = TestRunner('personalised_50', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=50)

    runner = TestRunner('personalised_100', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=100)

    runner = TestRunner('personalised_200', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=200)

    runner = TestRunner('personalised_300', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=300)

    runner = TestRunner('personalised_400', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=400)

    runner = TestRunner('personalised_500', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=500)

    # For plot 4: no. features versus runtime --------------------------------------------------------------------------

    datasets = ['UKPConvArgCrowdSample_evalMACE']
    methods = ['PersPrefGP_noOpt_weaksprior_M500', 'SinglePrefGP_noOpt_weaksprior_M500', 'SVM', 'BI-LSTM']
    feature_types = ['debug']
    ndebug_features = 30
    embeddings_types = ['word_mean']

    if not 'runner' in globals():
        runner = TestRunner('personalised_30feats', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)

    datasets = ['UKPConvArgCrowdSample_evalMACE']
    methods = ['SinglePrefGP_noOpt_weaksprior_M500', 'PersPrefGP_noOpt_weaksprior_M500', 'SVM', 'BI-LSTM']
    feature_types = ['debug']
    ndebug_features = 3000
    embeddings_types = ['word_mean']

    if not 'runner' in globals():
        runner = TestRunner('personalised_3000feats', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)
