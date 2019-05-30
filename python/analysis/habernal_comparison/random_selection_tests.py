# -- coding: utf-8 --

'''
Run a test on the argument convincingness dataset, but use something like ten-fold cross validation,
rather than splitting the data by topic.
Unlike standard cross validation, we use only 1/10th of the data in each fold as training data, and test on prediction
for all items. This means that we use roughly half of the unique pairs, with only one annotator per pair.

Background:
- the gold standard was defined by MACE
- with several annotators per pair, and many pairs from the training topics, the individual biases cancel out
- this means there is not much benefit to learning the model of the consensus function from using crowdGPPL in the
cross topic setup
- predicting the personalised preferences is also hard because preferences on each topic can be very different from one
another

Hypothesis:
- if we have few data points, and only one label per pair from one worker, worker biases may be important
- if we have a small set of data from the test topics, biases in that data may also be important for both inferring
consensus and personal preferences
- personal predictions may be less reliant on the consensus when we have some pairs for the test topics, because we
can then infer how the workers deviate from the consensus in each topic -- when predicting cross-topic preferences,
any learned personal biases may not be accurate.
'''

import logging
from scipy.stats.stats import kendalltau
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from personalised_tests import PersonalisedTestRunner
from tests import get_docidxs_from_ids, get_doc_token_seqs

logging.basicConfig(level=logging.DEBUG)

import sys
import os

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")
svm_python_path = '~/libsvm-3.22/python'
sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow"))
sys.path.append(os.path.expanduser(svm_python_path))

import time
import pandas as pd
import numpy as np


class RandomSelectionTestRunner(PersonalisedTestRunner):
    
    def __init__(self, current_expt_output_dir):
        self.folds = None
        self.initial_pair_subset = {}
        self.default_ls_values = {}
        
        self.expt_output_dir = current_expt_output_dir

        self.vscales = []  # record the latent factor scales


    def _choose_method_fun(self, feature_type):
        if 'crowdBT' in self.method:
            method_runner_fun = self.run_crowd_bt
        elif 'cBT_GP' in self.method:
            method_runner_fun = self.run_crowd_bt_gppl
        else:
            method_runner_fun = super(RandomSelectionTestRunner, self)._choose_method_fun(feature_type)

        return method_runner_fun


    def _set_resultsfile(self, dataset, method):
        # To run the active learning tests, call this function with dataset_increment << 1.0.
        # To add artificial noise to the data, run with acc < 1.0.
        output_data_dir = os.path.join(data_root_dir, 'outputdata/')
        if not os.path.isdir(output_data_dir):
            os.mkdir(output_data_dir)

        output_data_dir = os.path.join(output_data_dir, self.expt_output_dir)
        if not os.path.isdir(output_data_dir):
            os.mkdir(output_data_dir)

            # Select output paths for CSV files and final results
        output_filename_template = os.path.join(output_data_dir, '%s_%s')
        results_stem = output_filename_template % (dataset, method)

        if not os.path.isdir(results_stem):
            os.mkdir(results_stem)

        pair_pred_file = os.path.join(results_stem, 'pair_pred.csv')
        pair_prob_file = os.path.join(results_stem, 'pair_prob.csv')
        pair_gold_file = os.path.join(results_stem, 'pair_gold.csv')
        ratings_file = os.path.join(results_stem, 'ratings.csv')
        results_file = os.path.join(results_stem, 'metrics.csv')

        return results_stem, pair_pred_file, pair_prob_file, pair_gold_file, ratings_file, results_file

    # Methods for running the prediction methods --------------------------------------------------------------------------
    def run_crowd_bt(self):

        nitems = self.items_feat.shape[0]
        workers = np.unique(self.person_train)
        nworkers = np.max(workers) + 1

        # initialise variational parameters
        Es = np.zeros(nitems)
        Eeta = np.ones(nworkers) * 0.9
        sigma = 0.01 * np.ones(nitems)
        alpha = np.ones(nworkers) * 9
        beta = np.ones(nworkers)

        for pair_idx in range(len(self.a1_train)):

            # get the indices
            a1 = self.a1_train[pair_idx]
            a2 = self.a2_train[pair_idx]

            if self.prefs_train[pair_idx] == 2: # swap so a1 is the preferred one
                tmp = a1
                a1 = a2
                a2 = tmp

            k = self.person_train[pair_idx]

            # update the means
            prob_incr = (alpha[k] * np.exp(Es[a1]) / (alpha[k] * np.exp(Es[a1]) \
                       + beta[k] * np.exp(Es[a2])) - np.exp(Es[a1]) / (np.exp(Es[a1]) + np.exp(Es[a2])) )
            Es[a1] = Es[a1] + sigma[a1]**2 * prob_incr
            Es[a2] = Es[a2] - sigma[a2]**2 * prob_incr

            var_diff = alpha[k] * np.exp(Es[a1]) * beta[k] * np.exp(Es[a2]) / (alpha[k]*np.exp(Es[a1]) +
                        beta[k]*np.exp(Es[a2])) - np.exp(Es[a1]) * np.exp(Es[a2]) / (np.exp(Es[a1]) + np.exp(Es[a2]))
            sigma[a1] = np.sqrt(sigma[a1]**2 * np.max([1 + sigma[a1]**2 * (var_diff), 10e-4]))
            sigma[a2] = np.sqrt(sigma[a2]**2 * np.max([1 + sigma[a2]**2 * (var_diff), 10e-4]))

            C1 = np.exp(Es[a1]) / (np.exp(Es[a1] + np.exp(Es[a2]))) + 0.5 * (sigma[a1]**2 + sigma[a2]**2) \
             * np.exp(Es[a1]) * np.exp(Es[a2]) * (np.exp(Es[a2]) - np.exp(Es[a1])) / (np.exp(Es[a1]) + np.exp(Es[a2]))**3
            C2 = 1 - C1

            C = (C1 * alpha[k] + C2 * beta[k]) / (alpha[k] + beta[k]) # normalisation constant for p( 1 > 2 | worker k)

            Eeta[k] = (C1 * (alpha[k] + 1) * alpha[k] + C2 * alpha[k] * beta[k]) / (C * (alpha[k] + beta[k] + 1) * (alpha[k] + beta[k]))
            Eeta_sq_k = (C1 * (alpha[k] + 2) * (alpha[k] + 1) * alpha[k] + C2 * (alpha[k] + 1) * alpha[k] * beta[k]) / \
                        (C * (alpha[k] + beta[k] + 2) * (alpha[k] + beta[k] + 1) * (alpha[k] + beta[k]))

            alpha[k] = (Eeta[k] - Eeta_sq_k) * Eeta[k] / (Eeta_sq_k - Eeta[k]**2)
            beta[k] = (Eeta[k] - Eeta_sq_k) * (1 - Eeta[k]) / (Eeta_sq_k - Eeta[k]**2)

            # if np.mod(pair_idx, 100) == 0:
            print('Learning crowdBT, iteration %i' % pair_idx)

        print('Completed online learning of crowd BT')

        balance = 0.000001
        proba = np.exp(Es[self.a1_test]) / (np.exp(Es[self.a1_test]) + np.exp(Es[self.a2_test]) + balance)

        self.crowdBT_sigma = sigma
        self.crowdBT_s = Es

        scores = Es[self.a_rank_test]

        return proba, scores, None


    def run_crowd_bt_gppl(self):

        # we first train crowd_bt as above. Then, we use the scores for items that were compared in training
        # to train a GP regression model. The GP then predicts the scores of all items. This means we can generalise
        # from the training items to all items, plus the GP will do some smoothing over the training items in case they
        # had sparse noisy data.

        self.run_crowd_bt()
        gpr = GaussianProcessRegressor(kernel=Matern(), alpha=self.crowdBT_sigma**2)
        gpr.fit(self.items_feat, self.crowdBT_s)

        predicted_f = gpr.predict(self.items_feat)

        balance = 0.000001
        proba = np.exp(predicted_f[self.a1_test]) / (np.exp(predicted_f[self.a1_test]) + np.exp(predicted_f[self.a2_test]) + balance)

        predicted_f = predicted_f[self.a_rank_test]

        return proba, predicted_f, None


    def run_test_set(self, no_folds, dataset, method):

        self.method = method

        if self.folds is None or self.dataset != dataset:
            self._load_dataset(dataset)  # reload only if we use a new dataset

        if (dataset == 'UKPConvArgAll' or dataset == 'UKPConvArgStrict' or dataset == 'UKPConvArgCrowd_evalAll') \
                and ('IndPref' in method or 'Personalised' in method):
            logging.warning(
                'Skipping method %s on dataset %s because there are no separate worker IDs.' % (method, dataset))
            return

        logging.info("**** Running method %s on dataset %s ****" % (method, dataset) )

        feature_type = 'both'  # can be 'embeddings' or 'ling' or 'both' or 'debug'
        embeddings_type = 'word_mean'
        self._set_embeddings(embeddings_type) 
        self._init_ls(feature_type, embeddings_type)
        results_stem, pair_pred_file, pair_prob_file, pair_gold_file, ratings_file, results_file = self._set_resultsfile(dataset, method)

        np.random.seed(121) # allows us to get the same initialisation for all methods/feature types/embeddings

        # performance metrics are saved in a CSV file, with rows for each fold, and columns for each data type
        # predictions are saved in a CSV file, columns correspond to data points.
        # For ratings, the first column is gold, but for pairs the gold is in a separate file (because the pairs are different in each fold)
        try:
            pair_pred = pd.read_csv(pair_pred_file).values.tolist()
        except:
            pair_pred = []

        try:
            pair_prob = pd.read_csv(pair_prob_file).values.tolist()
        except:
            pair_prob = []

        try:
            rating_pred = pd.read_csv(ratings_file).values.tolist()
        except:
            rating_pred = []

        try:
            metrics = pd.read_csv(results_file).values.tolist()
        except:
            metrics = []

        pair_gold_by_fold = []

        a1 = []
        a2 = []
        pair_gold = []
        pair_person = []

        rating_a = []
        rating_gold = []
        rating_person = []

        X_a1 = []
        X_a2 = []
        text_a1 = []
        text_a2 = []

        # load the data from all topics
        for topic in self.folds:
            # X_train_a1, X_train_a2 are lists of lists of word indexes
            X_topic_a1, X_topic_a2, prefs_topic, ids_topic, person_topic, text_topic_a1, text_topic_a2 = self.folds.get(topic)["test"]

            testids = np.array([ids_pair.split('_') for ids_pair in ids_topic])
            a1_topic = get_docidxs_from_ids(self.docids, testids[:, 0], )
            a2_topic = get_docidxs_from_ids(self.docids, testids[:, 1])

            a1 = a1 + a1_topic.tolist()
            a2 = a2 + a2_topic.tolist()

            X_a1 = X_a1 + X_topic_a1
            X_a2 = X_a2 + X_topic_a2

            text_a1 = text_a1 + text_topic_a1
            text_a2 = text_a2 + text_topic_a2

            print(("Topic instances ", len(X_topic_a1), " test labels ", len(prefs_topic)))

            pair_gold = pair_gold + prefs_topic
            pair_person = pair_person + person_topic

            _, ratings_topic, argids_topic, person_rank_topic, _ = self.folds_r.get(topic)["test"]
            item_idx_topic = [np.argwhere(itemid == self.docids)[0][0] for itemid in argids_topic]
            rating_a = rating_a + item_idx_topic
            rating_gold = rating_gold + ratings_topic
            rating_person = rating_person + person_rank_topic

        # map all the person IDs to consecutive indexes
        upersonIDs, pair_person = np.unique(pair_person, return_inverse=True)
        rating_person = np.array([np.argwhere(upersonIDs == p.strip())[0][0] if p.strip() in upersonIDs else -1
                                      for p in rating_person])

        X, uids, utexts = get_doc_token_seqs((a1, a2), [X_a1, X_a2], (text_a1, text_a2))
        self.X = X

        if len(rating_pred) is 0:
            rating_pred = [rating_gold] # first row is gold
            pd.DataFrame(rating_pred).to_csv(ratings_file)


        pairs_non_neutral = np.array(pair_gold) != 1

        a1 = np.array(a1)[pairs_non_neutral]
        a2 = np.array(a2)[pairs_non_neutral]
        pair_gold = np.array(pair_gold)[pairs_non_neutral]
        pair_person = np.array(pair_person)[pairs_non_neutral]

        rating_a = np.array(rating_a)
        rating_gold = np.array(rating_gold)
        rating_person = np.array(rating_person)

        print('Total number of items = %i' % len(rating_a))

        if subset > 0:
            a1 = a1[:subset]
            a2 = a2[:subset]
            pair_gold = pair_gold[:subset]
            pair_person = pair_person[:subset]

            subidxs = np.unique((a1, a2))
            subidxs = np.in1d(rating_a, subidxs)
            rating_a = rating_a[subidxs]
            rating_gold = rating_gold[subidxs]

            if len(rating_person):
                rating_person = rating_person[subidxs]

        print('Testing with %i items' % len(rating_a))

        kfolder = KFold(n_splits=no_folds)

        # we switch the training and test sets because we actually want to train on a small subset
        for foldidx, (tr_pair_idxs, test_pair_idxs) in enumerate(kfolder.split(pair_gold)):

            if foldidx >= max_no_folds:
                break

            self.model = None # initial value

            if len(pair_pred) > foldidx:
                print("Skipping fold %i" % (foldidx))
                continue

            # Get data for this fold --------------------------------------------------------------------------------------
            print(("Fold %i", foldidx))

            # split the pairwise data
            self.a1_train = a1[tr_pair_idxs]
            self.a2_train = a2[tr_pair_idxs]
            self.prefs_train = pair_gold[tr_pair_idxs]
            self.person_train = pair_person[tr_pair_idxs]

            self.a1_test = a1[test_pair_idxs]
            self.a2_test = a2[test_pair_idxs]
            prefs_test = pair_gold[test_pair_idxs] # gold for evaluation
            self.person_test = pair_person[test_pair_idxs]

            self.a_rank_test = rating_a
            self.person_rank_test = rating_person

            self.a1_unseen = None # don't try to predict on this

            self.load_features(feature_type, embeddings_type, self.a1_train, self.a2_train, uids, utexts)
            #items_feat = items_feat[:, :ndebug_features]     
    
            self.verbose = verbose
            self.optimize_hyper = ('noOpt' not in method)
                        
            if len(self.default_ls) > 1:
                self.ls_initial = self.default_ls[self.valid_feats]
            else:
                self.ls_initial = self.default_ls

            if '_oneLS' in method:
                self.ls_initial = np.median(self.ls_initial)
                logging.info("Selecting a single LS for all features: %f" % self.ls_initial)

            logging.info("Starting test with method %s..." % (method))
            starttime = time.time()        
            
            logging.info('****** Fitting model with %i pairs in fold %i ******' % (len(self.prefs_train), foldidx))


            # run the method with the current data subset
            method_runner_fun = self._choose_method_fun(feature_type)
            proba, predicted_f, _ = method_runner_fun()

            endtime = time.time()

            # make it the right shape
            proba = np.array(proba)
            if proba.ndim == 2 and proba.shape[1] > 1:
                proba = proba[:, 1:2]
            elif proba.ndim == 1:
                proba = proba[:, None]
            predictions = np.round(proba).astype(int)

            if predicted_f is not None:
                predicted_f = np.array(predicted_f)
                if predicted_f.ndim == 3:
                    predicted_f = predicted_f[0]
                if predicted_f.ndim == 1:
                    predicted_f = predicted_f[:, None]

            logging.info("@@@ Completed fold %i with method %s in %f seconds." % (foldidx, method, endtime-starttime))

            #compute all metrics here
            acc = accuracy_score(prefs_test == 2, predictions)
            CEE = log_loss(prefs_test == 2, proba)
            tau, _ = kendalltau(rating_gold, predicted_f)
            runtime = endtime-starttime

            print('Results: acc = %f, CEE = %f, tau = %f, runtime = %f' % (acc, CEE, tau, runtime))

            metrics.append([acc, CEE, tau, runtime])
            pd.DataFrame(metrics).to_csv(results_file)

            # Save the data for later analysis ----------------------------------------------------------------------------
            pair_pred.append(predictions.flatten().tolist())
            pair_prob.append(proba.flatten().tolist())
            rating_pred.append(predicted_f.flatten().tolist())
            pair_gold_by_fold.append(prefs_test.flatten().tolist())

            pd.DataFrame(pair_pred).to_csv(pair_pred_file)
            pd.DataFrame(pair_prob).to_csv(pair_prob_file)
            pd.DataFrame(rating_pred).to_csv(ratings_file)
            pd.DataFrame(pair_gold_by_fold).to_csv()


        logging.info("**** Completed: method %s ****" % (method) )

        metrics.append(np.mean(metrics, axis=0).tolist())
        pd.DataFrame(metrics).to_csv(results_file)

        print('Wrote output files to %s' % results_stem)

if __name__ == '__main__':

    test_to_run = int(sys.argv[1])

    test_dir = 'randsel4'

    # UKPConvArgCrowdSample tests prediction of personal data.
    # UKPConvArgCrowdSample_evalMACE uses the personal data as input, but predicts the global labels/rankings.

    nfolds = 32
    max_no_folds = 1 # subset for debugging
    subset = 0
    verbose = False

    data_root_dir = os.path.abspath("./data/")

    if 'runner' not in globals():
        runner = RandomSelectionTestRunner(test_dir)

    # PERSONALISED PREDICTION
    if test_to_run == 0:
        dataset = 'UKPConvArgCrowdSample'
        method = 'PersPrefGP_commonmean_noOpt_weaksprior'
        runner.run_test_set(nfolds, dataset, method)

    # CONSENSUS PREDICTION
    elif test_to_run == 1:
        dataset = 'UKPConvArgCrowdSample_evalMACE'
        method = 'PersConsensusPrefGP_commonmean_noOpt_weaksprior'
        runner.run_test_set(nfolds, dataset, method)

    # # PERSONALISED WITH ARD
    # elif test_to_run == 2:
    #     dataset = 'UKPConvArgCrowdSample'
    #     method = 'PersPrefGP_commonmean_weaksprior'
    #     runner.run_test_set(nfolds, dataset, method)
    #
    # # CONSENSUS WITH ARD
    # elif test_to_run == 3:
    #     dataset = 'UKPConvArgCrowdSample_evalMACE'
    #     method = 'PersConsensusPrefGP_commonmean_weaksprior'
    #     runner.run_test_set(nfolds, dataset, method)

    # PERSONALISED PREDICTION for single GP -----------------------------------------------------------------
    # elif test_to_run == 4:
    #     method = 'SinglePrefGP_weaksprior'
    #     dataset = 'UKPConvArgCrowdSample'
    #     runner.run_test_set(nfolds, dataset, method)
    #
    # elif test_to_run == 5:
    #     method = 'SinglePrefGP_weaksprior'
    #     dataset = 'UKPConvArgCrowdSample_evalMACE'
    #     runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 6:
        method = 'SinglePrefGP_noOpt_weaksprior'
        dataset = 'UKPConvArgCrowdSample'
        runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 7:
        method = 'SinglePrefGP_noOpt_weaksprior'
        dataset = 'UKPConvArgCrowdSample_evalMACE'
        runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 8:
        method = 'crowdBT'
        dataset = 'UKPConvArgCrowdSample'
        runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 9:
        method = 'crowdBT'
        dataset = 'UKPConvArgCrowdSample_evalMACE'
        runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 10:
        method = 'cBT_GP'
        dataset = 'UKPConvArgCrowdSample'
        runner.run_test_set(nfolds, dataset, method)

    elif test_to_run == 11:
        method = 'cBT_GP'
        dataset = 'UKPConvArgCrowdSample_evalMACE'
        runner.run_test_set(nfolds, dataset, method)

    # Plot the scales of the latent factors ----------------------------------------------------------------------
    if test_to_run < 4:
        vscales = np.mean(runner.vscales, axis=0)

        logging.getLogger().setLevel(logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 4))

        markers = ['x', 'o', '+', '>', '<', '*']

        plt.plot(np.arange(vscales.shape[0]), vscales, marker=markers[0], label='UKPConvArgCrowdSample',
                 linewidth=2, markersize=8)

        plt.ylabel('Inverse scale 1/s')
        plt.xlabel('Factor ID')

        plt.grid('on', axis='y')
        plt.legend(loc='best')
        plt.tight_layout()

        if not os.path.exists('./results'):
            os.mkdir('./results')

        figure_root_path = './results/conv_factors'
        if not os.path.exists(figure_root_path):
            os.mkdir(figure_root_path)

        plt.savefig(figure_root_path + '/UKPConvArgCrowdSample_factor_scales.pdf')

        np.savetxt(figure_root_path + '/UKPConvArgCrowdSample_factor_scales.csv', vscales, delimiter=',', fmt='%f')

        logging.getLogger().setLevel(logging.DEBUG)