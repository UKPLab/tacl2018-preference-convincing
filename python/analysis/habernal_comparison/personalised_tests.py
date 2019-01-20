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
logging.basicConfig(level=logging.DEBUG)
from tests import TestRunner
sys.path.append("./python/analysis/habernal_comparison")
from collab_pref_learning_svi import CollabPrefLearningSVI
import numpy as np

nfactors = 50
max_Kw_size = 2000

class PersonalisedTestRunner(TestRunner):

    def _train_persgppl(self):
        common_mean = False

        if '_commonmean' in self.method:
            common_mean = True

        if 'weaksprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = 200.0
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

        self.model = CollabPrefLearningSVI(nitem_features=self.ndims, ls=self.ls_initial, verbose=self.verbose,
                                           nfactors=F, rate_ls=1.0 / np.mean(self.ls_initial),
                                           use_common_mean_t=common_mean, max_update_size=1000, use_lb=True,
                                           shape_s0=shape_s0, rate_s0=rate_s0, ninducing=M, delay=2)

        self.model.max_iter = 200 # same as for single user GPPL
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

        # people = np.unique(self.person_test)
        #
        # text_out = ""
        #
        # for p in people:
        #     text_out += '%i & ' % p
        #     testidxs = self.person_test == p
        #
        #     trsize = np.sum(self.person_train == p)
        #     logging.info("No. training egs: %i" % trsize)
        #     text_out += '%i & ' % trsize
        #
        #     tesize = np.sum(self.person_test == p)
        #     logging.info("No. test egs: %i" % tesize)
        #     text_out += '%i & ' % tesize
        #
        #     proba = self.model.predict(self.person_test[testidxs], self.a1_test[testidxs],
        #                            self.a2_test[testidxs])  # , self.items_feat)
        #     common_proba = self.model.predict_common(None, self.a1_test[testidxs], self.a2_test[testidxs])
        #
        #     prefs_test = self.prefs_test[testidxs]
        #
        #     per_acc = np.sum(prefs_test[prefs_test != 1] == 2 * np.round(proba).flatten()[prefs_test != 1]
        #                ) / float(np.sum(prefs_test != 1))
        #     con_acc = np.sum(prefs_test[prefs_test != 1] == 2 * np.round(common_proba).flatten()[prefs_test != 1]
        #                ) / float(np.sum(prefs_test != 1))
        #
        #     logging.info("Test personal accuracy = %f" % per_acc)
        #
        #     logging.info("Test consensus-to-personal accuracy = %f" % con_acc)
        #
        #     text_out += '%f & %f \\\\\n' % (per_acc, con_acc)
        #
        # with open(self.results_stem + '/personal_results_%i.tex' % self.foldidx, 'w') as fh:
        #     fh.writelines(text_out)

        # subsample for debugging!!!
        # testidxs = np.in1d(self.person_test, self.chosen_people)



        # print('Fraction of differences between personal and consensus pairwise predctions: %f' %
        #       (np.sum(np.round(proba.flatten()) != np.round(self.common_proba.flatten())) / float(len(proba.flatten())) ) )
        #
        #
        #

        #
        #
        # tridxs = np.in1d(self.person_train, self.chosen_people)
        # prefs_train = self.prefs_train[tridxs]
        #
        # trproba = self.model.predict(self.person_train[tridxs], self.a1_train[tridxs], self.a2_train[tridxs])
        # trcommon_proba = self.model.predict_common(None, self.a1_train[tridxs], self.a2_train[tridxs])
        #
        # logging.info("Train personal accuracy = %f" % (
        #         np.sum(prefs_train[prefs_train != 1] == 2 * np.round(trproba).flatten()[prefs_train != 1]
        #                ) / float(np.sum(prefs_train != 1))))
        #
        # logging.info("Train consensus-to-personal accuracy = %f" % (
        #         np.sum(prefs_train[prefs_train != 1] == 2 * np.round(trcommon_proba).flatten()[prefs_train != 1]
        #                ) / float(np.sum(prefs_train != 1))))
        #
        # print(np.any(np.isnan(proba)))
        # print(np.any(np.isinf(proba)))
        # for p in proba:
        #     print(p)
        #
        # print('common proba: ')
        #
        # for p in common_proba:
        #     print(p)

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

        return proba, predicted_f, None

    def _choose_method_fun(self, feature_type):
        if 'PersPrefGP' in self.method:
            method_runner_fun = self.run_persgppl
        elif 'PersConsensusPrefGP' in self.method:
            method_runner_fun = self.run_persgppl_consensus
        elif 'IndPrefGP' in self.method:
            method_runner_fun = self.run_persgppl # switches to correct class inside the method
        else:
            method_runner_fun = super(PersonalisedTestRunner, self)._choose_method_fun(feature_type)  
            
        return method_runner_fun

if __name__ == '__main__':

    test_to_run = int(sys.argv[1])

    test_dir = 'personalised_17'

    dataset_increment = 0     
    # UKPConvArgCrowdSample tests prediction of personal data.
    # UKPConvArgCrowdSample_evalMACE uses the personal data as input, but predicts the global labels/rankings.
    feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both' or 'debug'
    embeddings_types = ['word_mean']

    datasets = ['UKPConvArgCrowdSample']
    methods = ['PersPrefGP_commonmean_noOpt_weakersprior']

    if 'runner' not in globals():
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
