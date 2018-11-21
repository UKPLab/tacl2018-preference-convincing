from tests import TestRunner

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    # Classifications task
    datasets = ['UKPConvArgCrowdSample_evalMACE']

    # Create a plot for the runtime/accuracy against M + include other methods with ling + Glove features
    methods = ['SinglePrefGP_noOpt_weaksprior', 'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior', ]
    feature_types = ['debug'] # 'both'
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    #     datasets = ['UKPConvArgStrict'] # 'UKPConvArgCrowdSample_evalMACE',
    # #     #methods = ['BI-LSTM']
    #     methods = ['SVM']
    # #     methods = [#'SinglePrefGP_noOpt_weaksprior_M2', 'SinglePrefGP_noOpt_weaksprior_M10',
    # #                #'SinglePrefGP_noOpt_weaksprior_M100', 'SinglePrefGP_noOpt_weaksprior_M300',
    # #                #'SinglePrefGP_noOpt_weaksprior_M600', # leave out 'SinglePrefGP_noOpt_weakersprior_M500' because we've already done it
    # #                'SinglePrefGP_noOpt_weaksprior_M400', 'SinglePrefGP_noOpt_weaksprior_M500', 'SinglePrefGP_noOpt_weaksprior_M700'
    # #                ]
    #     feature_types = ['embeddings'] # 'both',
    #     embeddings_types = ['word_mean']
    #
    #     runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
    #                             dataset_increment)
    #     runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)

    #     methods = ['SinglePrefGP_noOpt_weaksprior_M200']
    #     feature_types = ['embeddings', 'both']
    #     embeddings_types = ['word_mean']
    #
    #     runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
    #                             dataset_increment)
    #     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0)

    #     datasets = ['UKPConvArgStrict']
    #     methods = ['SVM'] # 'BI-LSTM']
    #     feature_types = ['debug']
    #     ndebug_features = 30
    #     embeddings_types = ['word_mean']
    #
    #     #if not 'runner' in globals():
    #     runner = TestRunner('crowdsourcing_argumentation_expts_30feats', datasets, feature_types, embeddings_types, methods,
    #                             dataset_increment)
    #     runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)
    #
    datasets = ['UKPConvArgStrict']
    methods = ['SinglePrefGP_noOpt_weaksprior_M500', 'SVM']  # 'BI-LSTM', , ] ,
    feature_types = ['embeddings']
    ndebug_features = 3000
    embeddings_types = ['siamese-cbow']

    # if not 'runner' in globals():
    runner = TestRunner('crowdsourcing_argumentation_expts_3000feats', datasets, feature_types, embeddings_types,
                        methods,
                        dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)

#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_weaksprior_M500']#'SVM'] # 'BI-LSTM']#, ,
#     feature_types = ['debug']
#     ndebug_features = 20000
#     embeddings_types = ['word_mean']
#
#     #if not 'runner' in globals():
#     runner = TestRunner('crowdsourcing_argumentation_expts_10000feats', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=15, npairs=0)

#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_weaksprior_M500'] # 'BI-LSTM']
#     feature_types = ['debug']
#     ndebug_features = 30
#     embeddings_types = ['word_mean']
#
#     #if not 'runner' in globals():
#     runner = TestRunner('crowdsourcing_argumentation_expts_30feats', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0)

#     # scaling with N_tr
#     datasets = ['UKPConvArgStrict']
#     methods = ['SinglePrefGP_noOpt_weaksprior_M0', 'SinglePrefGP_noOpt_weaksprior_M100'] # M0 will mean no SVI
#     feature_types = ['embeddings']
#     embeddings_types = ['word_mean']
#
# #     runner = TestRunner('crowdsourcing_argumentation_expts_50', datasets, feature_types, embeddings_types, methods,
# #                             dataset_increment)
# #     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=50)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_100', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=100)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_200', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=200)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_300', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=300)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_400', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=400)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_500', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=500)

#     datasets = ['UKPConvArgStrict']
#     methods = [ 'SVM_small']#'BI-LSTM'] # M0 will mean no SVI #,
#     feature_types = ['embeddings']
#     embeddings_types = ['word_mean']
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_50', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=50)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_100', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=100)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_200', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=200)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_300', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=300)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_400', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=400)
#
#     runner = TestRunner('crowdsourcing_argumentation_expts_500', datasets, feature_types, embeddings_types, methods,
#                             dataset_increment)
#     runner.run_test_set(min_no_folds=0, max_no_folds=32, npairs=0, subsample_tr=500)
