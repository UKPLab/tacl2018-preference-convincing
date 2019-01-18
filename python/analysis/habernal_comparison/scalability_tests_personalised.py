import sys

from personalised_tests import PersonalisedTestRunner as TestRunner

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    max_no_folds = 10

    if len(sys.argv) > 1:
        test_to_run = int(sys.argv[1])
    else:
        test_to_run = 0


    # For plots 0 and 1 ------------------------------------------------------------------------------------------------
    if test_to_run == 0:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = [
                   'SinglePrefGP_noOpt_weaksprior_M2',
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
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0)

    # For plot 2: scaling with no. pairs -------------------------------------------------------------------------------
    if test_to_run == 2:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = [#'SinglePrefGP_noOpt_weaksprior_M0',
                   #'SinglePrefGP_noOpt_weaksprior_M100',
                   #'PersPrefGP_noOpt_weaksprior_commonmean_M0',
                   'PersPrefGP_noOpt_weaksprior_commonmean_M100_F5',
                   #'BI-LSTM'#'SVM', 'BI-LSTM'
            ] # M0 will mean no SVI
        feature_types = ['embeddings']
        embeddings_types = ['word_mean']

        runner = TestRunner('personalised_P1000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=1000)

        runner = TestRunner('personalised_P2000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=2000)

        runner = TestRunner('personalised_P4000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=4000)

        runner = TestRunner('personalised_P6000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=6000)

        runner = TestRunner('personalised_P8000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=8000)

        runner = TestRunner('personalised_P1000', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=1000)

    # The tests below are less important and may not make it to the new paper.
    # For Plot 3: Scaling with N_tr ------------------------------------------------------------------------------------
    if test_to_run == 3:
        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = [
            #'SinglePrefGP_noOpt_weaksprior_M0',
            'SinglePrefGP_noOpt_weaksprior_M100',
            #'PersPrefGP_noOpt_weaksprior_commonmean_M0',
            'PersPrefGP_noOpt_weaksprior_commonmean_M100_F5',
                   #'BI-LSTM'#'SVM', 'BI-LSTM'
            ] # M0 will mean no SVI
        feature_types = ['embeddings']
        embeddings_types = ['word_mean']

        runner = TestRunner('personalised_50', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=50)

        runner = TestRunner('personalised_100', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=100)

        runner = TestRunner('personalised_200', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=200)

        runner = TestRunner('personalised_300', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=300)

        runner = TestRunner('personalised_400', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=400)

        runner = TestRunner('personalised_500', datasets, feature_types, embeddings_types, methods,
                                dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0, subsample_tr=500)

    # For plot 4: no. features versus runtime --------------------------------------------------------------------------

    if test_to_run == 4:

        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['PersPrefGP_noOpt_weaksprior_commonmean_M500', 'SinglePrefGP_noOpt_weaksprior_M500']#, 'SVM', 'BI-LSTM']
        feature_types = ['debug']
        ndebug_features = 30
        embeddings_types = ['word_mean']

        if not 'runner' in globals():
            runner = TestRunner('personalised_30feats', datasets, feature_types, embeddings_types, methods,
                                    dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0)

        datasets = ['UKPConvArgCrowdSample_evalMACE']
        methods = ['SinglePrefGP_noOpt_weaksprior_M500', 'PersPrefGP_noOpt_weaksprior_commonmean_M500']#, 'SVM', 'BI-LSTM']
        feature_types = ['debug']
        ndebug_features = 3000
        embeddings_types = ['word_mean']

        if not 'runner' in globals():
            runner = TestRunner('personalised_3000feats', datasets, feature_types, embeddings_types, methods,
                                    dataset_increment)
        runner.run_test_set(min_no_folds=0, max_no_folds=max_no_folds, npairs=0)
