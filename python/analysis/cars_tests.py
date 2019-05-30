'''
Abbasnejad et al. cars dataset
'''
import datetime
import os
import sys

from sklearn.model_selection._split import KFold

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/test")

import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

# include the paths for the other directories
import time
from scipy.optimize._minimize import minimize
from scipy.stats.stats import kendalltau
from collab_pref_learning_fitc import CollabPrefLearningFITC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from collab_pref_learning_svi import CollabPrefLearningSVI
# from collab_pref_learning_svi_old import CollabPrefLearningSVI
from gp_pref_learning import GPPrefLearning
from per_user_pref_learning import GPPrefPerUser


verbose = False


def convert_discrete_to_continuous(features, cols_to_convert):

    new_features = None

    for col in np.arange(features.shape[1]):

        if col not in cols_to_convert:
            if new_features is None:
                new_features = features[:, col:col+1]
            else:
                new_features = np.concatenate((new_features, features[:, col:col+1]), axis=1)
            continue

        maxval = np.max(features[:, col])
        minval = np.min(features[:, col])

        nvals = maxval - minval + 1
        vals = np.arange(nvals) + minval

        disc_vecs = None
        for val in vals:
            if disc_vecs is None:
                disc_vecs = (features[:, col] == val)[:, None]
            else:
                disc_vecs = np.concatenate((disc_vecs, (features[:, col]==val)[:, None]), axis=1)

        if new_features is None:
            new_features = disc_vecs
        else:
            new_features = np.concatenate((new_features, disc_vecs), axis=1)

    return new_features


def run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr,
                   u_test=None, i1_test=None, i2_test=None,
                   ninducing=None, use_common_mean=True):

    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    if ninducing is None:
        ninducing = np.max([ifeats.shape[0], ufeats.shape[0]])

    # TODO check whether this setup still works for Sushi-B tests. Then run on conv tests -- probably need to tune sy hyperparameters
    # TODO test with the original selection of user inducing points again.

    model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_sy0=1e6 if sushiB else 1e6, rate_sy0=1e6 if sushiB else 1e6, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                  use_common_mean_t=use_common_mean, delay=delay)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 200
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=True)
    # model.use_local_obs_posterior_y = False

    if vscales is not None:
        vscales.append(np.sort((model.rate_sw / model.shape_sw) * (model.rate_sy / model.shape_sy))[::-1])

    if u_test is None:
        return model

    # fpred = model.predict_f(ifeats[active_items], ufeats)
    # rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)
    fpred = model.predict_f()
    rho_pred = model.predict(u_test, i1_test, i2_test)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_pooled(_, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, __, i1_test, i2_test):
    # we can use more inducing points because we don't have to compute GPs for the users and items separately,
    # so indcrease the number to make comparison fair.
    pool_ninducing = int(ninducing * 2**(1/3.0))

    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=pool_ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                   verbose=verbose)

    model.max_iter_VB = 500
    model.fit(i1_tr, i2_tr, ifeats, prefs_tr, optimize=optimize, use_median_ls=True)

    fpred, _ = np.tile(model.predict_f(), (1, ufeats.shape[0]))
    rho_pred, _ = model.predict(None, i1_test, i2_test)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test):

    # we can use more inducing points because we don't have to compute GPs for the users and items separately,
    # so indcrease the number to make comparison fair.
    joint_ninducing = int(ninducing * 2**(1/3.0))

    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=joint_ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate, verbose=verbose)

    model.max_iter_VB = 500

    # we need to use only the features for the subset of users in the training set!
    # if user features are not very informative, then the inducing points may be fairly useless.
    # this might explain why performance is low for joint model and crowd-GPPL.
    # However, BMF is and GPPL\u is still too low?

    joint_ifeats = np.tile(ifeats, (ufeats.shape[0], 1))
    joint_ufeats = np.tile(ufeats, (1, ifeats.shape[0])).reshape((ufeats.shape[0]*ifeats.shape[0], ufeats.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)

    i1_tr = i1_tr + (ifeats.shape[0] * u_tr)
    i2_tr = i2_tr + (ifeats.shape[0] * u_tr)

    model.fit(i1_tr, i2_tr, joint_feats, prefs_tr, optimize=optimize, use_median_ls=True)


    # need to split this up to compute because predict needs pairwise covariance terms and ends up computing full covariance
    batchsize = 100
    nbatches = int(np.ceil(np.unique(u_test).shape[0] / float(batchsize)))

    rho_pred = []
    for batch in range(nbatches):
        # all of the pairs and features that relate to a batch of users
        idxs = (u_test >= (batch) * batchsize) & (u_test < (batch+1) * batchsize)
        u_test_b = u_test[idxs]
        i1_test_b = i1_test[idxs]
        i2_test_b = i2_test[idxs]

        joint_feats_idxs_b, pairs_b = np.unique([i1_test_b + (ifeats.shape[0] * u_test_b),
                                                 i2_test_b + (ifeats.shape[0] * u_test_b)],
                                                return_inverse=True)
        pairs_b = pairs_b.reshape(2, i1_test_b.shape[0])

        rho_pred_b, _ = model.predict(joint_feats[joint_feats_idxs_b], pairs_b[0], pairs_b[1])
        rho_pred = np.append(rho_pred, rho_pred_b)

    joint_ifeats = np.tile(ifeats, (ufeats.shape[0], 1))
    joint_ufeats = np.tile(ufeats, (1, ifeats.shape[0])).reshape((ufeats.shape[0]*ifeats.shape[0],
                                                                       ufeats.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)
    fpred, _ = model.predict_f(joint_feats)
    fpred = fpred.reshape(ufeats.shape[0], ifeats.shape[0]).T

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test):
    model = GPPrefPerUser(ufeats.shape[0], max_update_size, shape_s0, rate_s0, ifeats.shape[1], ninducing)
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, personids=None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test):

    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], 0, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_sy0=1e6 if sushiB else 1e6, rate_sy0=1e6 if sushiB else 1e6, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                  use_common_mean_t=True, delay=delay)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 500
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test):
    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningSVI(1, 1, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_sy0=1e6 if sushiB else 1e6, rate_sy0=1e6 if sushiB else 1e6, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True, kernel_func='diagonal',
                                  delay=delay)
    model.max_Kw_size = max_Kw_size
    model.max_iter = 500
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, use_common_mean=False):
    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningFITC(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                   shape_sy0=1e6 if sushiB else 1e6, rate_sy0=1e6 if sushiB else 1e6, ls=None,
                                   nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                   forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                   use_common_mean_t=use_common_mean, delay=delay,
                                   exhaustive_train_count=0)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 500
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def train_test(method_name, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test):

    if method_name == 'crowd-GPPL':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ninducing=ninducing)
    elif method_name == 'crowd-GPPL-noConsensus':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ninducing=ninducing, use_common_mean=False)
    elif method_name == 'crowd-GPPL-noInduc':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ninducing=None)
    elif method_name == 'GPPL-pooled':
        return run_GPPL_pooled(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'GPPL-joint':
        return run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'GPPL-per-user':
        return run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'crowd-GPPL\\u':
        return run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'crowd-BMF':
        return run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'crowd-GPPL-FITC\\u-noConsensus': # No common mean, i.e. like Houlsby but SVI
        return run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test)
    elif method_name == 'crowd-GPPL-FITC\\u':
        return run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, use_common_mean=True)


def run_sushi_expt(methods, expt_name):
    # predictions from all reps and methods
    fpred_all = []
    rho_pred_all = []

    # metrics from all reps and methods
    acc_all = []
    logloss_all = []
    times_all = []

    # for repeatability
    np.random.seed(30)

    results_path = './results/' + expt_name
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    kfolder = KFold(n_splits=no_folds)

    # we switch the training and test sets because we actually want to train on a small subset
    for foldidx, (tr_pair_idxs, test_pair_idxs) in enumerate(kfolder.split(prefs)):

        # Get training and test data
        u_tr = userids[tr_pair_idxs]
        i1_tr = items1[tr_pair_idxs]
        i2_tr = items2[tr_pair_idxs]
        prefs_tr = prefs[tr_pair_idxs]

        u_test = userids[test_pair_idxs]
        i1_test = items1[test_pair_idxs]
        i2_test = items2[test_pair_idxs]
        prefs_test = prefs[test_pair_idxs]

        print(u_tr)
        print(i1_tr)
        print(i2_tr)
        print(prefs_tr)

        fpred_r = []
        rho_pred_r = []

        acc_r = []
        logloss_r = []
        times_r = []

        for m in methods:
            # Train and Predict
            logging.info("Starting test with method %s..." % (m))
            starttime = time.time()

            fpred, rho_pred, fpred_unseen, rho_pred_unseen = train_test(m, u_tr, i1_tr, i2_tr, item_features,
                                        user_features, prefs_tr, u_test, i1_test, i2_test)

            endtime = time.time()
            times_r.append(endtime - starttime)

            # Save predictions
            fpred_r.append(fpred.flatten())
            rho_pred_r.append(rho_pred.flatten())

            # Compute metrics
            acc_m = accuracy_score(prefs_test, np.round(rho_pred))
            logloss_m = log_loss(prefs_test.flatten(), rho_pred.flatten())


            # Save metrics
            acc_r.append(acc_m)
            logloss_r.append(logloss_m)

            print('Results for %s at rep %i: acc=%.2f, CEE=%.2f'
                  % (m, foldidx, acc_m, logloss_m))

        fpred_all.append(fpred_r)
        rho_pred_all.append(rho_pred_r)

        acc_all.append(acc_r)
        logloss_all.append(logloss_r)
        times_all.append(times_r)

        # save predictions to file
        np.savetxt(results_path + '/fpred_rep%i.csv' % foldidx, fpred_r, delimiter=',', fmt='%f')
        np.savetxt(results_path + '/rho_pred_rep%i.csv' % foldidx, rho_pred_r, delimiter=',', fmt='%f')

        # Compute means and stds of metrics (or medians/quartiles for plots?)
        acc_mean = np.mean(np.array(acc_all), axis=0)
        logloss_mean = np.mean(np.array(logloss_all), axis=0)
        times_mean = np.mean(np.array(times_all), axis=0)

        acc_std = np.std(np.array(acc_all), axis=0)
        logloss_std = np.std(np.array(logloss_all), axis=0)
        times_std = np.std(np.array(times_all), axis=0)

        # Print means and stds of metrics in Latex format ready for copying into a table
        print('Table of results:')

        line = 'Method & Acc. & CEE & tau & runtime (s) & Acc (unseen users) & CEE & tau \\\\ \n'
        print(line)
        lines = [line]

        for m, method in enumerate(methods):
            line = method + ' & '
            line += '%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\\\\n'\
                    % (acc_mean[m], acc_std[m], logloss_mean[m], logloss_std[m], times_mean[m], times_std[m])
            print(line)
            lines.append(line)

        with open(results_path + '/results.tex', 'w') as fh:
            fh.writelines(lines)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        test_to_run = int(sys.argv[1])
    else:
        test_to_run = 0

    # Experiment name tag
    tag = datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')

    no_folds = 10

    item_feat_file = './data/abbasnejad_cars/items1.csv'
    user_feat_file = './data/abbasnejad_cars/users1.csv'

    # Load feature data ----------------------------------------------------------------------------------------------------
    item_data = pd.read_csv(item_feat_file, index_col=0)
    item_features = item_data.values.astype(float)
    item_features = convert_discrete_to_continuous(item_features, cols_to_convert=[0, 1, 2, 3])

    user_data = pd.read_csv(user_feat_file, index_col=0)
    user_features = user_data.values.astype(float)
    user_features = convert_discrete_to_continuous(user_features, cols_to_convert=[0, 1, 2, 3, 4])


    prefs_file = './data/abbasnejad_cars/prefs1.csv'
    pref_data = pd.read_csv(prefs_file).values
    # remove control questions
    pref_data = pref_data[pref_data[:, 3]==0]

    userids = pref_data[:, 0]
    items1 = pref_data[:, 1]
    items2 = pref_data[:, 2]
    prefs = np.ones(len(items1)) # first item always preferred

    nusers = user_features.shape[0]
    nitems = item_features.shape[0]

    print('Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
    print('Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
    print('User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))

    # Hyperparameters common to most models --------------------------------------------------------------------------------
    max_facs = 20
    shape_s0 = 1.0
    rate_s0 = 100.0
    max_update_size = 200 # there are 20 x 100 = 2000 pairs in total. After 10 iterations, all pairs are seen.
    delay = 5
    ninducing = 25
    forgetting_rate = 0.9
    max_Kw_size = 5000

    methods = [
       'crowd-GPPL',
       'crowd-GPPL-noInduc',
       'crowd-GPPL\\u',
       'crowd-BMF',
       'crowd-GPPL-FITC\\u-noConsensus', # Like Houlsby CP (without user features)
       'GPPL-pooled',
       'GPPL-per-user',
    ]

    run_sushi_expt(methods, 'sushi_10small' + tag)