'''
Personalised argumentation paper: are user features are required or latent variables sufficient/correlated with observed features?
Is prior stance a useful user feature for predicting belief change? It should be, since a user can only be convinced by
an argument if they did not previously believe in it.
Topic-specific nature means predictions based on linguistic features are likely to be weak?

TODO:

Created on 19 Jun 2017

@author: simpson
'''
import os
import pickle
import sys
import logging
import matplotlib.pyplot as plt # do this here so we don't get the debugging crap later from the logger

from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

logging.basicConfig(level=logging.DEBUG)
from tests import TestRunner
sys.path.append("./python/analysis/habernal_comparison")
from collab_pref_learning_svi import CollabPrefLearningSVI
import numpy as np

nfactors = 50
max_Kw_size = 2000

rate_s = 20000

class PersonalisedTestRunner(TestRunner):

    def run_crowd_bt(self):

        nitems = self.items_feat.shape[0]
        workers = np.unique(self.person_train)
        nworkers = np.max(workers) + 1

        # initialise variational parameters
        Es = np.zeros(nitems)
        Eeta = np.ones(nworkers) * 0.9
        sigma = np.ones(nitems)
        alpha = np.ones(nworkers) * 9
        beta = np.ones(nworkers)

        balance = 0.000001 # tiny amount to ensure numerical stability

        for pair_idx in range(len(self.a1_train)):

            # get the indices
            a1 = self.a1_train[pair_idx]
            a2 = self.a2_train[pair_idx]

            if self.prefs_train[pair_idx] != 2:  # swap so a1 is the preferred one
                tmp = a1
                a1 = a2
                a2 = tmp

            k = self.person_train[pair_idx]

            # update the means
            prob_incr = alpha[k] * np.exp(Es[a1]) / (alpha[k] * np.exp(Es[a1]) + beta[k] * np.exp(Es[a2]) + balance) \
                         - np.exp(Es[a1]) / (np.exp(Es[a1]) + np.exp(Es[a2]) + balance)
            Es[a1] = Es[a1] + sigma[a1] ** 2 * prob_incr
            Es[a2] = Es[a2] - sigma[a2] ** 2 * prob_incr

            var_diff = alpha[k]  *np.exp(Es[a1]) * beta[k] * np.exp(Es[a2]) / \
                       ((alpha[k] * np.exp(Es[a1]) + beta[k] * np.exp(Es[a2]))**2 + balance) \
                       - np.exp(Es[a1]) * np.exp(Es[a2]) / ((np.exp(Es[a1]) + np.exp(Es[a2]))**2 + balance)
            sigma[a1] = np.sqrt(sigma[a1] ** 2 * np.max([1 + sigma[a1] ** 2 * (var_diff), 10e-4]))
            sigma[a2] = np.sqrt(sigma[a2] ** 2 * np.max([1 + sigma[a2] ** 2 * (var_diff), 10e-4]))

            C1 = np.exp(Es[a1]) / (np.exp(Es[a1] + np.exp(Es[a2])) + balance) \
                 + 0.5 * (sigma[a1] ** 2 + sigma[a2] ** 2) \
                 * np.exp(Es[a1]) * np.exp(Es[a2]) * (np.exp(Es[a2]) - np.exp(Es[a1])) \
                 / (np.exp(Es[a1]) + np.exp(Es[a2]) + balance) ** 3

            C2 = 1 - C1

            C = (C1 * alpha[k] + C2 * beta[k]) / (alpha[k] + beta[k] + balance)  # normalisation constant for p( 1 > 2 | worker k)

            Eeta[k] = (C1 * (alpha[k] + 1) * alpha[k] + C2 * alpha[k] * beta[k]) / (
                        C * (alpha[k] + beta[k] + 1) * (alpha[k] + beta[k]) + balance)
            Eeta_sq_k = (C1 * (alpha[k] + 2) * (alpha[k] + 1) * alpha[k] + C2 * (alpha[k] + 1) * alpha[k] * beta[k]) / \
                        (C * (alpha[k] + beta[k] + 2) * (alpha[k] + beta[k] + 1) * (alpha[k] + beta[k]) + balance)

            alpha[k] = (Eeta[k] - Eeta_sq_k) * Eeta[k] / (Eeta_sq_k - Eeta[k] ** 2 + balance)
            beta[k] = (Eeta[k] - Eeta_sq_k) * (1 - Eeta[k]) / (Eeta_sq_k - Eeta[k] ** 2 + balance)

            if np.mod(pair_idx, 100) == 0:
                print('Learning crowdBT, iteration %i' % pair_idx)

        print('Completed online learning of crowd BT')

        proba = np.exp(Es[self.a1_test]) / (np.exp(Es[self.a1_test]) + np.exp(Es[self.a2_test]) + balance)

        self.crowdBT_sigma = sigma
        self.crowdBT_s = Es

        scores = Es[self.a_rank_test]

        return proba, scores, None


    def run_crowd_bt_gpr(self):

        # we first train crowd_bt as above. Then, we use the scores for items that were compared in training
        # to train a GP regression model. The GP then predicts the scores of all items. This means we can generalise
        # from the training items to all items, plus the GP will do some smoothing over the training items in case they
        # had sparse noisy data.

        self.run_crowd_bt()
        # estimate the output scale of the GP using same prior as for GPPL
        function_var = (np.var(self.crowdBT_s) * len(self.crowdBT_s) + 200.0) / (len(self.crowdBT_s) + 200.0)
        gpr = GaussianProcessRegressor(kernel=Matern(self.ls_initial) * function_var, alpha=self.crowdBT_sigma ** 2)
        gpr.fit(self.items_feat, self.crowdBT_s)

        predicted_f = gpr.predict(self.items_feat)

        balance = 0.000001
        proba = np.exp(predicted_f[self.a1_test]) / (
                    np.exp(predicted_f[self.a1_test]) + np.exp(predicted_f[self.a2_test]) + balance)

        predicted_f = predicted_f[self.a_rank_test]

        return proba, predicted_f, None


    def _train_persgppl(self):
        common_mean = False

        if '_commonmean' in self.method:
            common_mean = True

        if 'weaksprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = rate_s #200.0
        elif 'lowsprior' in self.method:
            shape_s0 = 1.0
            rate_s0 = 1.0
        elif 'weakersprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = 2000.0
        else:
            shape_s0 = 200.0
            rate_s0 = 20000.0

        if '_M' in self.method:
            validx = self.method.find('_M') + 2
            M = int(self.method[validx:].split('_')[0])
        else:
            M = 500

        if M == 0:
            M = self.items_feat.shape[0]

        if '_F' in self.method:
            valididx = self.method.find('_F') + 2
            F = int(self.method[valididx:].split('_')[0])
        else:
            F = nfactors

        if '_SS' in self.method:
            validx = self.method.find('_SS') + 3
            SS = int(self.method[validx:])
        else:
            SS = 200

        self.model = CollabPrefLearningSVI(nitem_features=self.ndims, ls=self.ls_initial, verbose=self.verbose,
                                           nfactors=F, rate_ls=1.0 / np.mean(self.ls_initial),
                                           use_common_mean_t=common_mean, max_update_size=SS, use_lb=True,
                                           shape_s0=shape_s0, rate_s0=rate_s0,
                                           shape_sy0=1e10, rate_sy0=1e10,
                                           ninducing=M, forgetting_rate=0.9,#0.7,
                                           delay=10.0,
                                           exhaustive_train_count=3)#1.0)

        self.model.max_iter = 500 # same as for single user GPPL
        self.model.max_Kw_size = max_Kw_size

        zero_centered_prefs = np.array(self.prefs_train, dtype=float) - 1

        # subsample for debugging!!!
        # self.chosen_people = np.unique(self.person_test)[:50]
        # tridxs = np.in1d(self.person_train, self.chosen_people)

        #self.model.uselowerbound = False
        self.model.use_local_obs_posterior_y = False

        self.model.fit(self.person_train, self.a1_train, self.a2_train, self.items_feat, zero_centered_prefs,
                       optimize=self.optimize_hyper, nrestarts=1, input_type='zero-centered')


    def run_persgppl(self):
        '''
        Make personalised predictions
        :return:
        '''

        self._train_persgppl()

        if self.vscales is not None:
            self.vscales.append(np.sort((self.model.rate_sw / self.model.shape_sw) *
                                        (self.model.rate_sw / self.model.shape_sw))[::-1])

        proba = self.model.predict(self.person_test, self.a1_test, self.a2_test)

        # what did we change?
        # - more iterations ( 200 --> 500 )
        # - smaller rates ( 200 --> 20 ) because there are multiple factors whose scales all add up
        # next: try increasing delay so that y doesn't disappear to zero so easily when the person is not seen until
        # later batch of training data
        # Also: consider that when users are independent, then the stochastic updates are also updating only some of
        # the variables. But the others are getting set back to zero... this is only a problem in this model because
        # if they are initialised to zero, they don't move much because w and y are scaled by each other's current
        # estimates. It might be okay if variance of w and y is large because this is added to the scale factor

        if self.a_rank_test is not None:
            predicted_f = self.model.predict_f_item_person(self.a_rank_test, self.person_rank_test)

        return proba, predicted_f, None

    def run_persgppl_consensus(self):
        '''
        Predict the consensus from multiple people's opinions.
        '''

        # look for a file that was trained on the same data but with the personalised predictions instead of MACE consensus.
        # pretrainedmodelfile = self.modelfile.replace('_evalMACE', '')
        # pretrainedmodelfile = pretrainedmodelfile.replace('Consensus', '')
        #
        # logging.info('Looking for a pretrained model at %s' % pretrainedmodelfile)
        #
        # if os.path.exists(pretrainedmodelfile):
        #     with open(pretrainedmodelfile, 'rb') as fh:
        #         self.model = pickle.load(fh)
        #         logging.info('Reloaded a pretrained model :)')
        # else:
        #     logging.info('I didnae find any pretrained model :(')
        #     self._train_persgppl()

        self._train_persgppl()

        if self.vscales is not None:
            self.vscales.append(np.sort(self.model.rate_sw / self.model.shape_sw)[::-1])

        proba = self.model.predict_common(None, self.a1_test, self.a2_test)
        if self.a_rank_test is not None:
            predicted_f = self.model.predict_t()[self.a_rank_test]

        print('Max probability = %f, min = %f' % (np.max(proba), np.min(proba)))

        return proba, predicted_f, None

    def _choose_method_fun(self, feature_type):
        if 'PersPrefGP' in self.method:
            method_runner_fun = self.run_persgppl
        elif 'PersConsensusPrefGP' in self.method:
            method_runner_fun = self.run_persgppl_consensus
        elif 'IndPrefGP' in self.method:
            method_runner_fun = self.run_persgppl # switches to correct class inside the method
        elif 'crowdBT' in self.method:
            method_runner_fun = self.run_crowd_bt
        elif 'cBT_GP' in self.method:
            method_runner_fun = self.run_crowd_bt_gpr
        else:
            method_runner_fun = super(PersonalisedTestRunner, self)._choose_method_fun(feature_type)

        return method_runner_fun

if __name__ == '__main__':

    test_to_run = int(sys.argv[1])

    test_dir = 'rate_s_tests_single'

    methods = ['SinglePrefGP_noOpt_weaksprior']
    datasets = ['UKPConvArgCrowdSample_evalMACE']
    dataset_increment = 0
    # UKPConvArgCrowdSample tests prediction of personal data.
    # UKPConvArgCrowdSample_evalMACE uses the personal data as input, but predicts the global labels/rankings.
    feature_types = ['both']  # can be 'embeddings' or 'ling' or 'both' or 'debug'
    embeddings_types = ['word_mean']

    if test_to_run == -1:
        # use this setting for debugging

        # runner = PersonalisedTestRunner(test_dir, datasets, feature_types, embeddings_types, methods,
        #                                 dataset_increment)
        # runner.run_test_set(min_no_folds=0, max_no_folds=5)

        rate_s_vals = [20000]#[2*1e5, 2*1e6]#[20000] #2, 20, 200, 2000]

        for rate_s in rate_s_vals:
            test_dir = 'rate_s_%i_sy10_extr3' % rate_s

            methods = ['PersConsensusPrefGP_commonmean_noOpt_weaksprior']
            runner = PersonalisedTestRunner(test_dir, datasets, feature_types, embeddings_types, methods,
                                            dataset_increment)
            runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
            runner.methods = ['PersConsensusPrefGP_commonmean_noOpt_weaksprior']
            runner.run_test_set(min_no_folds=0, max_no_folds=5)

    if len(sys.argv) > 2:
        test_dir = sys.argv[2]

    datasets = ['UKPConvArgCrowdSample']
    methods = ['PersPrefGP_commonmean_noOpt_weaksprior']

    runner = PersonalisedTestRunner(test_dir, datasets, feature_types, embeddings_types, methods,
                                    dataset_increment)

    # PERSONALISED PREDICTION
    if test_to_run == 0:
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    # CONSENSUS PREDICTION
    elif test_to_run == 1:
        runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        runner.methods = ['PersConsensusPrefGP_commonmean_noOpt_weaksprior']
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

        # runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        # runner.methods = ['PersConsensusPrefGP_commonmean_noOpt_weakersprior']
        # runner.run_test_set(min_no_folds=0, max_no_folds=32)
        #
        # runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        # runner.methods = ['PersConsensusPrefGP_commonmean_noOpt_lowsprior']
        # runner.run_test_set(min_no_folds=0, max_no_folds=32)

    # PERSONALISED WITH ARD
    elif test_to_run == 2:
        runner.datasets = ['UKPConvArgCrowdSample']
        runner.methods = ['PersPrefGP_commonmean_weaksprior']
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    # CONSENSUS WITH ARD
    elif test_to_run == 3:
        runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        runner.methods = ['PersConsensusPrefGP_commonmean_weaksprior']
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

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

        figure_root_path = './results/conv_factors'
        if not os.path.exists(figure_root_path):
            os.mkdir(figure_root_path)

        plt.savefig(figure_root_path + '/UKPConvArgCrowdSample_factor_scales.pdf')

        np.savetxt(figure_root_path + '/UKPConvArgCrowdSample_factor_scales.csv', vscales, delimiter=',', fmt='%f')

        logging.getLogger().setLevel(logging.DEBUG)

    # PERSONALISED PREDICTION for other methods -----------------------------------------------------------------
    if test_to_run == 4:
        methods = [
               # 'SVM', 'GP+SVM', 'Bi-LSTM' # forget these methods as the other paper showed they were worse already, and the SVM
               # does not scale either -- it's worse than the GP.
               'SinglePrefGP_weaksprior' # 'SinglePrefGP_noOpt_weaksprior',
            ]
        runner.datasets = ['UKPConvArgCrowdSample']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    elif test_to_run == 5:
        methods = ['SinglePrefGP_weaksprior']
        runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    elif test_to_run == 6:
        methods = [
               # 'SVM', 'GP+SVM', 'Bi-LSTM' # forget these methods as the other paper showed they were worse already, and the SVM
               # does not scale either -- it's worse than the GP.
               'SinglePrefGP_noOpt_weaksprior' # 'SinglePrefGP_noOpt_weaksprior',
            ]
        runner.datasets = ['UKPConvArgCrowdSample']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    elif test_to_run == 7:
        methods = [
               # 'SVM', 'GP+SVM', 'Bi-LSTM' # forget these methods as the other paper showed they were worse already, and the SVM
               # does not scale either -- it's worse than the GP.
               'SinglePrefGP_noOpt_weaksprior' # 'SinglePrefGP_noOpt_weaksprior',
            ]
        runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    elif test_to_run == 8:
        methods = [
               #'crowdBT', # no point running this because it cannot predict on the test instances, for aggregation only
               'cBT_GP',
            ]
        runner.datasets = ['UKPConvArgCrowdSample']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)

    #elif test_to_run == 9: # commented so we run both tests with cBT
        methods = [
               #'crowdBT', # no point running this because it cannot predict on the test instances, for aggregation only
               'cBT_GP',
        ]
        runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
        runner.methods = methods
        runner.run_test_set(min_no_folds=0, max_no_folds=32)