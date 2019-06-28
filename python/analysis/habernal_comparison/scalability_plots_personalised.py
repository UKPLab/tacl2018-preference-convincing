'''
Created on Jan 8, 2018

@author: simpson
'''

import os
import sys

import numpy as np
from compute_metrics import load_results_data, get_fold_data
import pickle
from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt
import compute_metrics

figure_save_path = './results/scalability'
if not os.path.isdir(figure_save_path):
    os.mkdir(figure_save_path)

if __name__ == '__main__':

    # The first part loads runtimes and accuracies with varying no. inducing points
    if len(sys.argv) > 1:
        test_to_run = int(sys.argv[1])
    else:
        test_to_run = 0

    expt_settings = {}
    expt_settings['folds'] = None
    expt_settings['foldorderfile'] = None
    data_root_dir = "./data/"
    resultsfile_template = 'habernal_%s_%s_%s_%s_di%.2f'

    expt_settings['dataset'] = 'UKPConvArgCrowdSample_evalMACE'

    expt_settings['acc'] = 1.0
    expt_settings['di'] = 0.0
    max_no_folds = 32
    expt_settings['embeddings_type'] = 'word_mean'

    tag = 'p4'

    if test_to_run == 0:
        foldername = tag

        # Create a plot for the runtime/accuracy against M + include other methods with ling + Glove features
        methods =  ['SinglePrefGP_noOpt_weaksprior_M2',
                    'SinglePrefGP_noOpt_weaksprior_M10',
                    'SinglePrefGP_noOpt_weaksprior_M100',
                    'SinglePrefGP_noOpt_weaksprior_M200',
                    'SinglePrefGP_noOpt_weaksprior_M300',
                    'SinglePrefGP_noOpt_weaksprior_M400',
                    'SinglePrefGP_noOpt_weaksprior_M500',
                    #'SinglePrefGP_noOpt_weaksprior_M600',
                    #'SinglePrefGP_noOpt_weaksprior_M700',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M2',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M10',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M100',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M200',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M300',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M400',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M500',
                    #'PersPrefGP_commonmean_noOpt_weaksprior_F5_M600',
                    #'PersPrefGP_commonmean_noOpt_weaksprior_F5_M700',
                    ]

        xstring = 'Inducing Points, M'
        filename = 'num_inducing_32310_features'
        which_leg = 'accuracy'

    elif test_to_run == 5:
        foldername = tag

        # Create a plot for the runtime/accuracy against M + include other methods with ling + Glove features
        methods =  ['SinglePrefGP_noOpt_weaksprior_M2',
                    'SinglePrefGP_noOpt_weaksprior_M10',
                    'SinglePrefGP_noOpt_weaksprior_M100',
                    'SinglePrefGP_noOpt_weaksprior_M200',
                    'SinglePrefGP_noOpt_weaksprior_M300',
                    'SinglePrefGP_noOpt_weaksprior_M400',
                    'SinglePrefGP_noOpt_weaksprior_M500',
                    #'SinglePrefGP_noOpt_weaksprior_M600',
                    #'SinglePrefGP_noOpt_weaksprior_M700',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M2',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M10',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M100',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M200',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M300',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M400',
                    'PersPrefGP_commonmean_noOpt_weaksprior_F5_M500',
                    #'PersPrefGP_commonmean_noOpt_weaksprior_F5_M600',
                    #'PersPrefGP_commonmean_noOpt_weaksprior_F5_M700',
                    ]

        xstring = 'pairs per iteration, P_i'
        filename = 'P_i_32310_features'
        which_leg = 'runtimes'

    if test_to_run == 0 or test_to_run == 5:

        # Load the results for accuracy and runtime vs. no. inducing points with both feature sets
        expt_settings['feature_type'] = 'both'

        docids = None

        dims_methods = np.array(['SinglePrefGP_noOpt_weaksprior_M500', 'PersPrefGP_commonmean_noOpt_M500', 'SVM', 'BI-LSTM'])
        #, 'SinglePrefGP_weaksprior'])
        runtimes_dims = np.zeros((len(dims_methods), 4))

        runtimes_both = np.zeros(len(methods))
        runtimes_both_lq = np.zeros(len(methods))
        runtimes_both_uq = np.zeros(len(methods))

        acc_both = np.zeros(len(methods))

        for m, expt_settings['method'] in enumerate(methods):
            print("Processing method %s" % expt_settings['method'])

            data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, resultsfile_template,
                                                      expt_settings, max_no_folds=max_no_folds, foldername=foldername)

            acc_m = np.zeros(nFolds)
            runtimes_m = np.zeros(nFolds)

            for f in range(nFolds):
                print("Processing fold %i" % f)
                if expt_settings['fold_order'] is None: # fall back to the order on the current machine
                    if expt_settings['folds'] is None:
                        continue
                    fold = list(expt_settings['folds'].keys())[f]
                else:
                    fold = expt_settings['fold_order'][f]
                    if fold[-2] == "'" and fold[0] == "'":
                        fold = fold[1:-2]
                    elif fold[-1] == "'" and fold[0] == "'":
                        fold = fold[1:-1]
                    expt_settings['fold_order'][f] = fold

                # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                foldfile = resultsdir + '/fold%i.pkl' % f
                if os.path.isfile(foldfile):
                    with open(foldfile, 'rb') as fh:
                        data_f = pickle.load(fh, encoding='latin1')
                else: # convert the old stuff to new stuff
                    min_folds = f+1
                    print('Skipping fold with no data %i' % f)
                    print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'],
                                                                   expt_settings['dataset'],
                                                                   expt_settings['feature_type'],
                                                                   expt_settings['embeddings_type']))
                    print("Skipped filename was: %s, old-style results file would be %s" % (foldfile,
                                                                                            resultsfile))
                    continue

                gold_disc, pred_disc, gold_prob, pred_prob, gold_rank, pred_rank, pred_tr_disc, \
                            pred_tr_prob, pred_tr_rank, postprocced = get_fold_data(data_f, f, expt_settings)

                acc_m[f] = accuracy_score(gold_disc[gold_disc!=1], pred_disc[gold_disc!=1])
                runtimes_m[f] = data_f[6]

            acc_m = acc_m[acc_m>0]
            runtimes_m = runtimes_m[runtimes_m>0]

            if len(acc_m):
                acc_both[m] = np.mean(acc_m)
                runtimes_both[m] = np.mean(runtimes_m)
                runtimes_both_lq[m] = np.mean(runtimes_m) + np.std(runtimes_m)#np.percentile(runtimes_m, 25)
                runtimes_both_uq[m] = np.mean(runtimes_m) - np.std(runtimes_m)#np.percentile(runtimes_m, 75)

                if expt_settings['method'] in dims_methods:
                    m_dims = dims_methods == expt_settings['method']
                    runtimes_dims[m_dims, 3] = runtimes_both[m]


        fig1, ax1 = plt.subplots(figsize=(5,4))
        x_gppl = np.array([2, 10, 100, 200, 300, 400, 500])#, 600, 700])
        h1, = ax1.plot(x_gppl, runtimes_both[0:7], color='blue', marker='o', label='runtime, GPPL',
                       linewidth=2, markersize=8)

        ax1.fill_between(x_gppl, runtimes_both_lq[0:7], runtimes_both_uq[0:7], alpha=0.2, edgecolor='blue',
                         facecolor='blue')


        h2, = ax1.plot(x_gppl, runtimes_both[7:], marker='<', color='green', linestyle='-.', label='runtime, crowdGPPL',
                       linewidth=2, markersize=8)

        ax1.fill_between(x_gppl, runtimes_both_lq[7:], runtimes_both_uq[7:], alpha=0.2, edgecolor='green',
                         facecolor='green')

        ax1.set_ylabel('Runtime (s)')
        plt.xlabel('No. %s' % xstring)
        ax1.grid('on', axis='y')
        ax1.spines['left'].set_color('blue')
        ax1.tick_params('y', colors='blue')
        ax1.yaxis.label.set_color('blue')

        if which_leg == 'runtimes':
            leg = plt.legend(loc='best')

        ax1_2 = ax1.twinx()
        h3, = ax1_2.plot(x_gppl, acc_both[0:7], color='black', marker='x', label='acc., GPPL',
                         linewidth=2, markersize=8)
        h4, = ax1_2.plot(x_gppl, acc_both[7:], color='black', linestyle='--', marker='>', label='acc., crowdGPPL',
                         linewidth=2, markersize=8)

        ax1_2.set_ylabel('Accuracy')
        plt.yticks([0.6, 0.65, 0.70, 0.75])

        if which_leg == 'accuracy':
            leg = plt.legend(loc='best')#handles=[h1, h2], loc='lower right')

        plt.tight_layout()
        plt.savefig(figure_save_path + '/%s.pdf' % filename)


    # Third plot: number of pairs, P, versus runtime (with Glove features) ---------------------------------------------
    if test_to_run == 2:
        expt_settings['feature_type'] = 'embeddings'
        methods = ['SinglePrefGP_noOpt_weaksprior_M100', #'SinglePrefGP_noOpt_weaksprior_M0',
                   'PersPrefGP_noOpt_weaksprior_commonmean_M100_F5'] #'PersPrefGP_commonmean_noOpt_weaksprior_M1052',
                   #'SVM', 'BI-LSTM']

        Nvals = [500, 1000, 2000, 4000, 7000, 10000]
        runtimes_N = np.zeros((len(methods), len(Nvals)))
        runtimes_N_lq = np.zeros((len(methods), len(Nvals)))
        runtimes_N_uq = np.zeros((len(methods), len(Nvals)))

        for n, N in enumerate(Nvals):
            foldername = tag + '_P%i/' % N

            for m, expt_settings['method'] in enumerate(methods):
                print("Processing method %s" % expt_settings['method'])

                data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir,
                                      resultsfile_template, expt_settings, max_no_folds=max_no_folds, foldername=foldername)

                runtimes_m = np.zeros(nFolds)

                for f in range(nFolds):
                    print("Processing fold %i" % f)
                    fold = expt_settings['fold_order'][f]
                    if fold[-2] == "'" and fold[0] == "'":
                        fold = fold[1:-2]
                    elif fold[-1] == "'" and fold[0] == "'":
                        fold = fold[1:-1]
                    expt_settings['fold_order'][f] = fold

                    # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                    foldfile = resultsdir + '/fold%i.pkl' % f
                    if os.path.isfile(foldfile):
                        with open(foldfile, 'rb') as fh:
                            data_f = pickle.load(fh, encoding='latin1')
                    else:  # convert the old stuff to new stuff
                        if data is None:
                            min_folds = f + 1
                            print('Skipping fold with no data %i' % f)
                            print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'],
                                                                           expt_settings['dataset'],
                                                                           expt_settings['feature_type'],
                                                                           expt_settings['embeddings_type']))
                            print("Skipped filename was: %s, old-style results file would be %s" % (foldfile,
                                                                                                    resultsfile))
                            continue

                        if not os.path.isdir(resultsdir):
                            os.mkdir(resultsdir)
                        data_f = []
                        for thing in data:
                            if f in thing:
                                data_f.append(thing[f])
                            else:
                                data_f.append(thing)
                        with open(foldfile, 'wb') as fh:
                            pickle.dump(data_f, fh)

                    runtimes_m[f] = data_f[6]

                runtimes_N[m, n] = np.mean(runtimes_m)
                runtimes_N_lq[m, n] = np.mean(runtimes_m) - np.std(runtimes_m)  #  np.percentile(runtimes_m, 25)
                runtimes_N_uq[m, n] = np.mean(runtimes_m) + np.std(runtimes_m)  #  np.percentile(runtimes_m, 75)

        fig3, ax3 = plt.subplots(figsize=(3, 2.6))

        Nvals = np.array(Nvals) / 1000

        ax3.plot(Nvals, runtimes_N[0], label='GPPL M=100', marker='o', color='blue', linewidth=2, linestyle='-',
                 markersize=8)

        ax3.fill_between(Nvals, runtimes_N_lq[0], runtimes_N_uq[0], alpha=0.2, edgecolor='blue',
                         facecolor='blue')

        ax3.plot(Nvals, runtimes_N[1], label='crowdGPPL M=100', marker='<', color='green', linestyle='-.',
                 linewidth=2, markersize=8)

        ax3.fill_between(Nvals, runtimes_N_lq[1], runtimes_N_uq[1], alpha=0.2, edgecolor='green',
                         facecolor='green')

        ax3.set_xlabel('1000 pairwise training labels')
        ax3.set_ylabel('Runtime (s)')
        ax3.yaxis.grid('on')
        ax3.set_ylim(-5, 200)
        #plt.legend(loc='best')

        plt.tight_layout()
        plt.savefig(figure_save_path + '/num_pairs.pdf')

    # Fourth plot: training set size N versus runtime (with Glove features) ---------------------------------------------
    if test_to_run == 3:
        expt_settings['feature_type'] = 'embeddings'
        methods = ['SinglePrefGP_noOpt_weaksprior_M100',  # 'SinglePrefGP_noOpt_weaksprior_M0',
                   'PersPrefGP_noOpt_weaksprior_commonmean_M100_F5']  # 'PersPrefGP_commonmean_noOpt_weaksprior_M1052',
        # 'SVM', 'BI-LSTM']

        Nvals = [50, 100, 200, 300, 400, 500]
        runtimes_N = np.zeros((len(methods), len(Nvals)))
        runtimes_N_lq = np.zeros((len(methods), len(Nvals)))
        runtimes_N_uq = np.zeros((len(methods), len(Nvals)))

        for n, N in enumerate(Nvals):
            foldername = tag + '_%i/' % N

            for m, expt_settings['method'] in enumerate(methods):
                print("Processing method %s" % expt_settings['method'])

                data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir,
                                                                          resultsfile_template, expt_settings,
                                                                          max_no_folds=max_no_folds,
                                                                          foldername=foldername)

                runtimes_m = np.zeros(nFolds)

                for f in range(nFolds):
                    print("Processing fold %i" % f)
                    fold = expt_settings['fold_order'][f]
                    if fold[-2] == "'" and fold[0] == "'":
                        fold = fold[1:-2]
                    elif fold[-1] == "'" and fold[0] == "'":
                        fold = fold[1:-1]
                    expt_settings['fold_order'][f] = fold

                    # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                    foldfile = resultsdir + '/fold%i.pkl' % f
                    if os.path.isfile(foldfile):
                        with open(foldfile, 'rb') as fh:
                            data_f = pickle.load(fh, encoding='latin1')
                    else:  # convert the old stuff to new stuff
                        if data is None:
                            min_folds = f + 1
                            print('Skipping fold with no data %i' % f)
                            print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'],
                                                                           expt_settings['dataset'],
                                                                           expt_settings['feature_type'],
                                                                           expt_settings['embeddings_type']))
                            print("Skipped filename was: %s, old-style results file would be %s" % (foldfile,
                                                                                                    resultsfile))
                            continue

                        if not os.path.isdir(resultsdir):
                            os.mkdir(resultsdir)
                        data_f = []
                        for thing in data:
                            if f in thing:
                                data_f.append(thing[f])
                            else:
                                data_f.append(thing)
                        with open(foldfile, 'wb') as fh:
                            pickle.dump(data_f, fh)

                    runtimes_m[f] = data_f[6]

                runtimes_N[m, n] = np.mean(runtimes_m)
                runtimes_N_lq[m, n] = np.mean(runtimes_m) - np.std(runtimes_m)  #  np.percentile(runtimes_m, 25)
                runtimes_N_uq[m, n] = np.mean(runtimes_m) + np.std(runtimes_m)  #  np.percentile(runtimes_m, 75)

        fig3, ax3 = plt.subplots(figsize=(3, 2.6))

        ax3.plot(Nvals, runtimes_N[0], label='GPPL M=100', marker='o', color='blue', linewidth=2, linestyle='-',
                 markersize=8)

        ax3.fill_between(Nvals, runtimes_N_lq[0], runtimes_N_uq[0], alpha=0.2, edgecolor='blue',
                         facecolor='blue')

        ax3.plot(Nvals, runtimes_N[1], label='crowdGPPL M=100', marker='<', color='green', linestyle='-.',
                 linewidth=2, markersize=8)

        ax3.fill_between(Nvals, runtimes_N_lq[1], runtimes_N_uq[1], alpha=0.2, edgecolor='green',
                         facecolor='green')

        ax3.set_xlabel('No. arguments in training set')
        ax3.set_ylabel('Runtime (s)')
        ax3.yaxis.grid('on')
        ax3.set_ylim(-5, 200)
        plt.legend(loc='best')

        plt.tight_layout()
        plt.savefig(figure_save_path + '/num_arguments.pdf')

    # Fifth plot: no. features versus runtime -------------------------------------------------------------------------
    if test_to_run == 4:
        expt_settings['feature_type'] = 'debug'
        for n, dim in enumerate(['30feats', '', '3000feats']):
            foldername = 'personalised_%s/' % dim
            print("Processing %s" % dim)
            for m, expt_settings['method'] in enumerate(dims_methods):
                print("Processing method %s" % expt_settings['method'])

                if not len(dim):
                    continue
                data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, resultsfile_template,
                                                          expt_settings, max_no_folds=max_no_folds, foldername=foldername)
                expt_settings_master = expt_settings
                runtimes_m = np.zeros(nFolds)
                print(resultsdir)
                for f in range(nFolds):
                    print("Processing fold %i" % f)
                    if expt_settings['fold_order'] is None:  # fall back to the order on the current machine
                        if expt_settings['folds'] is None:
                            print("Skipping fold %i because no fold order file" % f)
                            continue
                        fold = list(expt_settings['folds'].keys())[f]
                    else:
                        fold = expt_settings['fold_order'][f]
                        if fold[-2] == "'" and fold[0] == "'":
                            fold = fold[1:-2]
                        elif fold[-1] == "'" and fold[0] == "'":
                            fold = fold[1:-1]
                        expt_settings['fold_order'][f] = fold

                    # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                    foldfile = resultsdir + '/fold%i.pkl' % f
                    if os.path.isfile(foldfile):
                        with open(foldfile, 'rb') as fh:
                            data_f = pickle.load(fh, encoding='latin1')
                    else:  # convert the old stuff to new stuff
                        if data is None:
                            min_folds = f + 1
                            print('Skipping fold with no data %i' % f)
                            print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'],
                                                                           expt_settings['dataset'],
                                                                           expt_settings['feature_type'],
                                                                           expt_settings['embeddings_type']))
                            print("Skipped filename was: %s, old-style results file would be %s" % (foldfile,
                                                                                                    resultsfile))
                            continue

                        if not os.path.isdir(resultsdir):
                            os.mkdir(resultsdir)
                        data_f = []
                        for thing in data:
                            if f in thing:
                                data_f.append(thing[f])
                            else:
                                data_f.append(thing)
                        with open(foldfile, 'wb') as fh:
                            pickle.dump(data_f, fh)

                    runtimes_m[f] = data_f[6]

                if np.sum(runtimes_m > 0):
                    runtimes_dims[m, n] = np.mean(runtimes_m[runtimes_m > 0])

                expt_settings = expt_settings_master

        fig4, ax4 = plt.subplots(figsize=(5, 4))

        x_dims = [1, 2, 3, 4.0322]  # 3.5228353136605302,
        # x_labels = [30, 300, 3000, 32310]
        x_ticklocs = [1, 2, 3, 4]  # 3.5228353136605302,
        x_labels = ['3e1', '3e2', '3e3', '3e4']  # '10e3',
        plt.xticks(x_ticklocs, x_labels)

        h1, = ax4.plot(x_dims, runtimes_dims[2], label='SVM', marker='>', color='black',
                       clip_on=False, linewidth=2, markersize=8)
        h2, = ax4.plot(x_dims, runtimes_dims[3], label='BiLSTM', marker='^', color='red',
                       clip_on=False, linewidth=2, markersize=8)
        h3, = plt.plot(x_dims, runtimes_dims[0], label='GPPL', marker='o', color='blue',
                       clip_on=False, linewidth=2, markersize=8)
        h4, = plt.plot(x_dims, runtimes_dims[1], label='crowdGPPL', marker='<', color='green',
                       clip_on=False, linewidth=2, markersize=8)

        plt.xlim(0.9, 4.1)
        plt.xlabel('No. Features')
        ax4.legend(handles=[h3, h4, h1, h2], labels=['GPPL', 'crowdGPPL', 'SVM', 'BiLSTM'],
                   loc=(0.15, 0.6))
        ax4.yaxis.grid('on')

        plt.legend(loc='best')
        plt.ylabel('Runtime (s)')
        # plt.ylabel('Runtime (s) for GPPL, M=500, medi.')
        plt.xticks(x_ticklocs, x_labels)

        plt.tight_layout()
        plt.savefig(figure_save_path + '/num_features.pdf')