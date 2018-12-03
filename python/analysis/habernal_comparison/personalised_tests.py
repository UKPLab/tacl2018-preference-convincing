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
            M = int(self.method[validx:])
        else:
            M = 500

        self.model = CollabPrefLearningSVI(nitem_features=self.ndims, ls=self.ls_initial, verbose=self.verbose,
                                           nfactors=nfactors, rate_ls=1.0 / np.mean(self.ls_initial),
                                           use_common_mean_t=common_mean, max_update_size=1000, use_lb=True,
                                           shape_s0=shape_s0, rate_s0=rate_s0, ninducing=M)

        self.model.max_iter = 200

        zero_centered_prefs = np.array(self.prefs_train, dtype=float) - 1

        self.model.fit(self.person_train, self.a1_train, self.a2_train, self.items_feat, zero_centered_prefs,
                       optimize=self.optimize_hyper, nrestarts=1, input_type='zero-centered')

    def run_persgppl(self):
        '''
        Make personalised predictions
        :return:
        '''

        self._train_persgppl()

        if self.vscales is not None:
            self.vscales.append(np.sort(self.model.rate_sw / self.model.shape_sw)[::-1])

        proba = self.model.predict(self.person_test, self.a1_test, self.a2_test, self.items_feat)

        print(np.any(np.isnan(proba)))
        print(np.any(np.isinf(proba)))
        for p in proba:
            print(p)

        if self.a_rank_test is not None:
            predicted_f = self.model.predict_f_item_person(self.a_rank_test, self.person_rank_test, self.items_feat)
    
        return proba, predicted_f, None

    def run_persgppl_consensus(self):
        '''
        Predict the consensus from multiple people's opinions.
        '''

        # look for a file that was trained on the same data but with the personalised predictions instead of MACE consensus.
        pretrainedmodelfile = self.modelfile.replace('_evalMACE', '')
        pretrainedmodelfile = pretrainedmodelfile.replace('Consensus', '')

        logging.info('Looking for a pretrained model at %s' % pretrainedmodelfile)

        if os.path.exists(pretrainedmodelfile):
            with open(pretrainedmodelfile, 'rb') as fh:
                self.model = pickle.load(fh)
                logging.info('Reloaded a pretrained model :)')
        else:
            logging.info('I didnae find any pretrained model :(')
            self._train_persgppl()

        if self.vscales is not None:
            self.vscales.append(np.sort(self.model.rate_sw / self.model.shape_sw)[::-1])

        proba = self.model.predict_common(None, self.a1_test, self.a2_test)
        if self.a_rank_test is not None:
            predicted_f = self.model.predict_t(self.model.obs_coords[self.a_rank_test])

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
    dataset_increment = 0     
    # UKPConvArgCrowdSample tests prediction of personal data.
    # UKPConvArgCrowdSample_evalMACE uses the personal data as input, but predicts the global labels/rankings.
    feature_types = ['both'] # can be 'embeddings' or 'ling' or 'both' or 'debug'
    embeddings_types = ['word_mean']

    datasets = ['UKPConvArgCrowdSample']
    methods = ['PersPrefGP_commonmean_noOpt_weaksprior']

    if 'runner' not in globals():
        runner = PersonalisedTestRunner('personalised_2', datasets, feature_types, embeddings_types, methods,
                                        dataset_increment)
        runner.save_collab_model = True

    # PERSONALISED PREDICTION
    # runner.run_test_set(min_no_folds=0, max_no_folds=1)

    # CONSENSUS PREDICTION
    runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
    runner.methods = ['PersConsensusPrefGP_commonmean_noOpt_weaksprior']
    runner.run_test_set(min_no_folds=0, max_no_folds=1)

    # PERSONALISED WITH ARD
    runner.datasets = ['UKPConvArgCrowdSample']
    runner.methods = ['PersPrefGP_commonmean_weaksprior']
    runner.run_test_set(min_no_folds=0, max_no_folds=1)

    # CONSENSUS WITH ARD
    runner.datasets = ['UKPConvArgCrowdSample_evalMACE']
    runner.methods = ['PersConsensusPrefGP_commonmean_weaksprior']
    runner.run_test_set(min_no_folds=0, max_no_folds=1)

    # Plot the scales of the latent factors ----------------------------------------------------------------------
    vscales = np.mean(runner.vscales, axis=0)

    logging.getLogger().setLevel(logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))

    markers = ['o', 'x', '+', '>', '<', '*']

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

    # # PERSONALISED PREDICTION for other methods -----------------------------------------------------------------
    # methods = [
    #            # 'SVM', 'GP+SVM', # forget these methods as the other paper showed they were worse already, and the SVM
    #            # does not scale either -- it's worse than the GP.
    #            'SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_weaksprior', 'Bi-LSTM',
    #         ]
    # embeddings_types = ['word_mean']
    #
    # if 'runner' not in globals():
    #     runner = PersonalisedTestRunner('personalised', datasets, feature_types, embeddings_types, methods,
    #                                     dataset_increment)
    #     runner.save_collab_model = True
    #
    # runner.run_test_set(min_no_folds=0, max_no_folds=32)
