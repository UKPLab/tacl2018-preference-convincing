'''
Run a series of tests on synthetic data to show the effect of noise on recovering the latent preference functions.
'''
import sys

# include the paths for the other directories
from scipy.stats.stats import pearsonr, kendalltau

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/test")

import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collab_pref_learning_svi import CollabPrefLearningSVI
from collab_pref_learning_test import gen_synthetic_personal_prefs as gen_multi, \
    split_dataset as split_multiuser_dataset

markers = ['o', 'x', '+', '>', '<', '*']

verbose = True

def plot_result(idx, filename, xlabel, ylabel, linelabel, fig=None, lineidx=0):

    if verbose:
        logging.basicConfig(level=logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

    if fig is None:
        fig = plt.figure(figsize=(5, 4))
    else:
        plt.figure(fig.number)

    # ax1.semilogx(inverse_scales
    plt.plot(mean_results[:, 0], mean_results[:, idx],
             marker=markers[lineidx], label=linelabel, linewidth=2, markersize=8)

    plt.ylabel(ylabel)
    # plt.xlabel('inverse function scale, s')
    plt.xlabel(xlabel)
    plt.grid('on', axis='y')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(figure_save_path + '/%s.pdf' % filename)

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
    nx = 20
    ny = 20
    N = nx * ny

    # the independent variable that we adjust:
    inverse_scales = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 10, 100]

    noise_plots = [None, None, None, None, None, None, None]

    figure_root_path = './results/synth'
    if not os.path.exists(figure_root_path):
        os.mkdir(figure_root_path)

    # DATATSET SIZE TEST -----------------------------------------------------------------------------------------------
    # MULTI USER OBSERVATIONS, MEASURING CORRELATION BETWEEN DISCOVERED AND TRUE LATENT FACTORS, MODEL: MU

    s = 1
    P_values = [40, 80, 160, 320, 640, 1280, 2560]

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)

    Nfactor_values = [1, 3, 10, 20]  # repeat with different no.s of factors

    figure_save_path = figure_root_path + '/multi_factor_correlations_P/'
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    for Nfactors in Nfactor_values:

        Npeople = 20

        mean_results = []
        std_results = []

        mean_results_pool = []
        std_results_pool = []

        mean_results_m = []
        std_results_m = []

        for P in P_values:

            results_s_multi = []

            for rep in range(nreps):

                ls = [np.random.rand() * 40, np.random.rand() * 40]
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
                model = CollabPrefLearningSVI(2, 2, mu0=0, shape_s0=0.1, rate_s0=0.1, ls=None, nfactors=Npeople,
                                              ninducing=ninducing,
                                              max_update_size=1000, forgetting_rate=0.9, verbose=True, use_lb=True)

                print(("--- Repeating multi user test, rep %i ---" % rep))

                model.fit(
                    personidxs_tr,
                    pair1idxs_tr,
                    pair2idxs_tr,
                    item_features,
                    prefs_tr,
                    person_features,
                    optimize=False,
                    use_median_ls=True
                )

                print(("Final lower bound: %f" % model.lowerbound()))

                # Predict at all locations
                ypred = model.y

                ordering = np.zeros(Nfactors)  # assign the predicted factors to their closest matches by correlation

                unassigned_f = np.arange(
                    Nfactors)  # list of true factors that have not yet been matched to a predicted factor
                unassigned_fpred = np.arange(Nfactors)

                total_r = 0

                for ass in range(Nfactors):

                    r = np.zeros((Nfactors, Nfactors))

                    for f in unassigned_f:
                        for fpred in unassigned_fpred:
                            r[f, fpred] = np.abs(pearsonr(y[f], ypred[fpred])[0])

                    maxidx = np.argmax(r)
                    max_f, max_fpred = np.unravel_index(maxidx, (Nfactors, Nfactors))

                    total_r += r[max_f, max_fpred]

                    unassigned_f = unassigned_f[unassigned_f != max_f]
                    unassigned_fpred = unassigned_fpred[unassigned_fpred != max_fpred]

                mean_r = total_r / float(Nfactors)
                print("Mean factor correlation (Pearson's r): %.3f" % mean_r)

                # noise rate in the pairwise data -- how many of the training pairs conflict with the ordering suggested by f?
                prefs_tr_noisefree = (F[pair1idxs_tr, personidxs_tr] > F[pair2idxs_tr, personidxs_tr]).astype(float)

                noise_rate = 1.0 - np.mean(prefs_tr == prefs_tr_noisefree)
                print('Noise rate in the pairwise training labels: %f' % noise_rate)

                Fpred = model.predict_f(item_features, person_features)

                tau_test = kendalltau(F[test_points], Fpred[test_points])[0]
                print("Kendall's tau on the test data: %f" % tau_test)

                results_s_multi.append([P, mean_r, tau_test, noise_rate])

            mean_results_s_m = np.mean(results_s_multi, axis=0)
            std_results_s_m = np.std(results_s_multi, axis=0)

            print('Multi-User Model: all reps completed for inverse scale %f. Mean and stds of the metrics:' % s)

            print('noise rate in training data: %f, %f' % (mean_results_s_m[0], std_results_s_m[0]))
            print("Mean factor correlation (Pearson's r): %f, %f" % (mean_results_s_m[1], std_results_s_m[1]))

            mean_results_m.append(mean_results_s_m)
            std_results_m.append(std_results_s_m)

        mean_results = np.array(mean_results_m)
        std_results = np.array(std_results_m)

        noise_plots[1] = plot_result(1, "num_pairs_r", 'number of pairwise training labels',
                            "Mean factor correlation (Pearson's r)", 'num_factors=%i' % Nfactors, noise_plots[1])

        noise_plots[2] = plot_result(2, "num_pairs_tau_test", 'number of pairwise training labels',
                            "tau (test data)", 'num_factors=%i' % Nfactors, noise_plots[2])
