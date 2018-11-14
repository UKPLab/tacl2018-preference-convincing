'''
Run a series of tests on synthetic data to show the effect of noise on recovering the latent preference functions.
'''
import sys

# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/test")

import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from collab_pref_learning_svi import CollabPrefLearningSVI
from gp_pref_learning import GPPrefLearning
from gp_pref_learning_test import gen_synthetic_prefs as gen_single, split_dataset, evaluate_models
from collab_pref_learning_test import gen_synthetic_personal_prefs as gen_multi, \
    split_dataset as split_multiuser_dataset, evaluate_models_common_mean as evaluate_multiuser_models
from per_user_pref_learning import GPPrefPerUser

markers = ['o', 'x', '+', '>', '<', '*']

verbose = True

def plot_result(idx, label, ylabel, method, fig=None, lineidx=0):

    if verbose:
        logging.basicConfig(level=logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

    if fig is None:
        fig = plt.figure(figsize=(5, 4))
    else:
        plt.figure(fig.number)

    # ax1.semilogx(inverse_scales
    plt.plot(mean_results[:, 0], mean_results[:, idx],
              marker=markers[lineidx], label=method, linewidth=2, markersize=8)

    plt.ylabel(ylabel)
    # plt.xlabel('inverse function scale, s')
    plt.xlabel('noise rate in pairwise training labels')
    plt.grid('on', axis='y')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(figure_save_path + '/%s.pdf' % label)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)  # switch back to the debug output

    return fig

if __name__ == '__main__':
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    fix_seeds = True

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)

    # SETTINGS FOR ALL THE NOISE TESTS
    nreps = 25
    nx = 10
    ny = 10
    N = 100
    P = 5000

    # the independent variable that we adjust:
    inverse_scales = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 10, 100]

    noise_plots = [None, None, None, None, None, None, None]

    figure_root_path = './results/synth_sandbox'
    if not os.path.exists(figure_root_path):
        os.mkdir(figure_root_path)

    figure_save_path = figure_root_path + '/single_user/'
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    # NOISE TEST, SINGLE USER DATA, SINGLE USER MODEL

    # mean_results = []
    # std_results = []
    #
    # for s in inverse_scales:
    #
    #     results_s = []
    #
    #     for rep in range(nreps):
    #
    #         ls = [np.random.rand() * 20, np.random.rand() * 20]
    #
    #         prefs, item_features, pair1idxs, pair2idxs, f = gen_single(
    #             nx=nx,
    #             ny=ny,
    #             N=N,
    #             P=P,
    #             ls=ls,
    #             s=s
    #         )
    #
    #         ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points, \
    #         ftest, pair1idxs_test, pair2idxs_test, prefs_test, test_points = \
    #             split_dataset(N, f, pair1idxs, pair2idxs, prefs)
    #
    #         # # Create a GPPrefLearning model
    #         model = GPPrefLearning(2, mu0=0, shape_s0=2, rate_s0=2, ls_initial=None, use_svi=True, ninducing=50,
    #                                max_update_size=100, forgetting_rate=0.9, verbose=True)
    #
    #         print(("--- Repeating single user test, rep %i ---" % rep))
    #         results_s.append(evaluate_models(
    #             model, item_features, f,
    #             ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points,
    #             ftest, pair1idxs_test, pair2idxs_test, test_points
    #         ))
    #
    #     print('Single user test: all reps completed for inverse scale %f. Mean and stds of the metrics:' % s)
    #
    #     mean_results_s = np.mean(results_s, axis=0)
    #     std_results_s = np.std(results_s, axis=0)
    #
    #     print('noise rate in training data: %f, %f' % (mean_results_s[0], std_results_s[0]))
    #     print('tau_obs: %f, %f' % (mean_results_s[1], std_results_s[1]))
    #     print('tau_test: %f, %f' % (mean_results_s[2], std_results_s[2]))
    #     print('brier: %f, %f' % (mean_results_s[3], std_results_s[3]))
    #     print('cee: %f, %f' % (mean_results_s[4], std_results_s[4]))
    #     print('f1: %f, %f' % (mean_results_s[5], std_results_s[5]))
    #     print('acc: %f, %f' % (mean_results_s[6], std_results_s[6]))
    #     print('roc: %f, %f' % (mean_results_s[7], std_results_s[7]))
    #
    #     mean_results.append(mean_results_s)
    #     std_results.append(std_results_s)
    #
    # # let's plot our results.
    # mean_results = np.array(mean_results)
    # std_results = np.array(std_results)
    #
    # noise_plots[0] = plot_result(1, 'tau_obs', 'tau (training)', method='SU, single user data')
    # noise_plots[1] = plot_result(2, 'tau_test', 'tau (test)', method='SU, single user data')
    # noise_plots[2] = plot_result(3, 'brier', 'brier score', method='SU, single user data')
    # noise_plots[3] = plot_result(4, 'cee', 'cross entropy error (nats)', method='SU, single user data')
    # noise_plots[4] = plot_result(5, 'f1', 'F1 score', method='SU, single user data')
    # noise_plots[5] = plot_result(6, 'acc', 'accuracy', method='SU, single user data')
    # noise_plots[6] = plot_result(7, 'roc', 'area under ROC curve', method='SU, single user data')

    # NOISE TEST, MULTI USER DATA, MODELS: SU, POOL, MU ---------------------------------------------------------------

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)

    Nfactors = 3
    Npeople = 25

    mean_results = []
    std_results = []

    mean_results_pool = []
    std_results_pool = []

    mean_results_m = []
    std_results_m = []

    figure_save_path = figure_root_path + '/multi_user_consensus/'
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    for s in inverse_scales:

        results_s = []
        results_s_pool = []
        results_s_multi = []

        for rep in range(nreps):

            ls = [np.random.rand() * 20, np.random.rand() * 20]
            lsy = [np.random.rand() * 20, np.random.rand() * 20]

            # relative importance of the latent factors is determined by ratio of sigma to s. Larger s relative to
            # sigma means common mean is more important. The noise of the pairwise labels depends on 1/s + 1/sigma,
            prefs, item_features, person_features, pair1idxs, pair2idxs, personidxs, F, w, f, y = gen_multi(
                Nfactors=Nfactors,
                nx=nx,
                ny=ny,
                N=N,
                Npeople=Npeople,
                P=P,
                ls=ls,
                sigma=s,
                s=0.2,
                lsy=lsy,
                Npeoplefeatures=2
            )

            Ftrain, pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points, \
            Ftest, pair1idxs_test, pair2idxs_test, personidxs_test, prefs_test, test_points = \
                split_multiuser_dataset(N, F, pair1idxs, pair2idxs, personidxs, prefs)

            ninducing = 50

            # Create a GPPrefLearning model
            model = CollabPrefLearningSVI(2, 2, mu0=0, shape_s0=0.1, rate_s0=0.1, ls=None, nfactors=Npeople, ninducing=ninducing,
                                   max_update_size=1000, forgetting_rate=0.9, verbose=True, use_lb=True)

            print(("--- Repeating multi user test, rep %i ---" % rep))
            results_s_multi.append(evaluate_multiuser_models(
                model, item_features, person_features, f,
                pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points,
                pair1idxs_test, pair2idxs_test, test_points
            ))

            # Create a GPPrefLearning model
            model = GPPrefLearning(2, mu0=0, shape_s0=0.1, rate_s0=0.1, ls_initial=None, use_svi=True, ninducing=ninducing,
                                   max_update_size=1000, forgetting_rate=0.9, verbose=True)

            print(("--- Repeating pooled test, rep %i ---" % rep))
            results_s_pool.append(evaluate_models(
                model, item_features, f,
                f[train_points], pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points,
                f[test_points], pair1idxs_test, pair2idxs_test, test_points
            ))

            # Create a GPPrefLearning model per person
            model = GPPrefPerUser(Npeople, max_update_size=1000, shape_s0=0.1, rate_s0=0.1)

            print(("--- Repeating separate user test, rep %i ---" % rep))
            results_s.append(evaluate_multiuser_models(
                model, item_features, person_features, f,
                pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points,
                pair1idxs_test, pair2idxs_test, test_points
            ))


        print('Per-User Model: all reps completed for inverse scale %f. Mean and stds of the metrics:' % s)

        mean_results_s = np.mean(results_s, axis=0)
        std_results_s = np.std(results_s, axis=0)

        print('noise rate in training data: %f, %f' % (mean_results_s[0], std_results_s[0]))
        print('tau_obs: %f, %f' % (mean_results_s[1], std_results_s[1]))
        print('tau_test: %f, %f' % (mean_results_s[2], std_results_s[2]))
        print('brier: %f, %f' % (mean_results_s[3], std_results_s[3]))
        print('cee: %f, %f' % (mean_results_s[4], std_results_s[4]))
        print('f1: %f, %f' % (mean_results_s[5], std_results_s[5]))
        print('acc: %f, %f' % (mean_results_s[6], std_results_s[6]))
        print('roc: %f, %f' % (mean_results_s[7], std_results_s[7]))

        mean_results.append(mean_results_s)
        std_results.append(std_results_s)

        mean_results_s_pool = np.mean(results_s_pool, axis=0)
        std_results_s_pool = np.std(results_s_pool, axis=0)

        print('Pooled Model: all reps completed for inverse scale %f. Mean and stds of the metrics:' % s)

        print('noise rate in training data: %f, %f' % (mean_results_s_pool[0], std_results_s_pool[0]))
        print('tau_obs: %f, %f' % (mean_results_s_pool[1], std_results_s_pool[1]))
        print('tau_test: %f, %f' % (mean_results_s_pool[2], std_results_s_pool[2]))
        print('brier: %f, %f' % (mean_results_s_pool[3], std_results_s_pool[3]))
        print('cee: %f, %f' % (mean_results_s_pool[4], std_results_s_pool[4]))
        print('f1: %f, %f' % (mean_results_s_pool[5], std_results_s_pool[5]))
        print('acc: %f, %f' % (mean_results_s_pool[6], std_results_s_pool[6]))
        print('roc: %f, %f' % (mean_results_s_pool[7], std_results_s_pool[7]))

        mean_results_pool.append(mean_results_s_pool)
        std_results_pool.append(std_results_s_pool)

        mean_results_s_m = np.mean(results_s_multi, axis=0)
        std_results_s_m = np.std(results_s_multi, axis=0)

        print('Multi-User Model: all reps completed for inverse scale %f. Mean and stds of the metrics:' % s)

        print('noise rate in training data: %f, %f' % (mean_results_s_m[0], std_results_s_m[0]))
        print('tau_obs: %f, %f' % (mean_results_s_m[1], std_results_s_m[1]))
        print('tau_test: %f, %f' % (mean_results_s_m[2], std_results_s_m[2]))
        print('brier: %f, %f' % (mean_results_s_m[3], std_results_s_m[3]))
        print('cee: %f, %f' % (mean_results_s_m[4], std_results_s_m[4]))
        print('f1: %f, %f' % (mean_results_s_m[5], std_results_s_m[5]))
        print('acc: %f, %f' % (mean_results_s_m[6], std_results_s_m[6]))
        print('roc: %f, %f' % (mean_results_s_m[7], std_results_s_m[7]))

        mean_results_m.append(mean_results_s_m)
        std_results_m.append(std_results_s_m)

    mean_results = np.array(mean_results)
    std_results = np.array(std_results)

    plot_result(1, 'tau_obs', 'tau (training)', 'SU, multi user data', noise_plots[0])
    plot_result(2, 'tau_test', 'tau (test)', 'SU, multi user data', noise_plots[1])
    plot_result(3, 'brier', 'brier score', 'SU, multi user data', noise_plots[2])
    plot_result(4, 'cee', 'cross entropy error (nats)', 'SU, multi user data', noise_plots[3])
    plot_result(5, 'f1', 'F1 score', 'SU, multi user data', noise_plots[4])
    plot_result(6, 'acc', 'accuracy', 'SU, multi user data', noise_plots[5])
    plot_result(7, 'roc', 'area under ROC curve', 'SU, multi user data', noise_plots[6])

    mean_results = np.array(mean_results_pool)
    std_results = np.array(std_results_pool)

    plot_result(1, 'tau_obs', 'tau (training)', 'Pooled, multi user data', noise_plots[0])
    plot_result(2, 'tau_test', 'tau (test)', 'Pooled, multi user data', noise_plots[1])
    plot_result(3, 'brier', 'brier score', 'Pooled, multi user data', noise_plots[2])
    plot_result(4, 'cee', 'cross entropy error (nats)', 'Pooled, multi user data', noise_plots[3])
    plot_result(5, 'f1', 'F1 score', 'Pooled, multi user data', noise_plots[4])
    plot_result(6, 'acc', 'accuracy', 'Pooled, multi user data', noise_plots[5])
    plot_result(7, 'roc', 'area under ROC curve', 'Pooled, multi user data', noise_plots[6])

    mean_results = np.array(mean_results_m)
    std_results = np.array(std_results_m)

    plot_result(1, 'tau_obs', 'tau (training)', 'MU, multi user data', noise_plots[0])
    plot_result(2, 'tau_test', 'tau (test)', 'MU, multi user data', noise_plots[1])
    plot_result(3, 'brier', 'brier score', 'MU, multi user data', noise_plots[2])
    plot_result(4, 'cee', 'cross entropy error (nats)', 'MU, multi user data', noise_plots[3])
    plot_result(5, 'f1', 'F1 score', 'MU, multi user data', noise_plots[4])
    plot_result(6, 'acc', 'accuracy', 'MU, multi user data', noise_plots[5])
    plot_result(7, 'roc', 'area under ROC curve', 'MU, multi user data', noise_plots[6])