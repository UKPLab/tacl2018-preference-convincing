from tests import TestRunner
import tests

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    # Classifications task
    datasets = ['UKPConvArgStrict']

    # Create a plot for the runtime/accuracy against M + include other methods with ling + Glove features
    methods = ['SinglePrefGP_noOpt_weaksprior_M2', 'SinglePrefGP_noOpt_weaksprior_M10',
               'SinglePrefGP_noOpt_weaksprior_M100', 'SinglePrefGP_noOpt_weaksprior_M200',
               'SinglePrefGP_noOpt_weaksprior_M300', 'SinglePrefGP_noOpt_weaksprior_M400',
               'SinglePrefGP_noOpt_weaksprior_M500', 'SinglePrefGP_noOpt_weaksprior_M600',
               'SinglePrefGP_noOpt_weaksprior_M700',
               'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior', ]
    feature_types = ['both', 'embeddings']
    embeddings_types = ['word_mean']

    # runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
    #                     dataset_increment)
    # runner.run_test_set()
    #
    # # Varying number of arguments, N
    # feature_types = ['embeddings']
    # methods = ['SinglePrefGP_noOpt_weaksprior_M100', 'SinglePrefGP_noOpt_weaksprior_M0',
    #            'SVM_small', 'BI-LSTM']
    #
    # Nvals = [50, 100, 200, 300, 400, 500]
    #
    # for N in Nvals:
    #     runner = TestRunner('crowdsourcing_argumentation_expts_%i' % N, datasets, feature_types, embeddings_types,
    #                         methods, dataset_increment)
    #     runner.run_test_set(subsample_tr=N)

    # Varying number of features
    methods = ['SinglePrefGP_noOpt_weaksprior_M500', 'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior']
    feature_types = ['debug']
    tests.ndebug_features = 30
    runner = tests.TestRunner('crowdsourcing_argumentation_expts_30feats', datasets, feature_types, embeddings_types, methods,
                        dataset_increment, )
    runner.run_test_set()

    feature_types = ['debug']
    ndebug_features = 3000
    runner = tests.TestRunner('crowdsourcing_argumentation_expts_3000feats', datasets, feature_types, embeddings_types, methods,
                        dataset_increment, )
    runner.run_test_set()

