'''
Test on the smaller Sushi dataset with 10 items. This is for comparison with Houlsby et al in terms of classification
error.
'''
import os
import sys

# include the paths for the other directories
import time

from scipy.optimize._minimize import minimize

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/test")

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

from collab_pref_learning_svi import CollabPrefLearningSVI
from gp_pref_learning import GPPrefLearning
from per_user_pref_learning import GPPrefPerUser


def extract_pairs_from_ranking(ranked_items):

    item1 = ranked_items[:, 0:1]
    items2 = ranked_items[:, 1:] # the first item is preferred to all these

    npairs = items2.shape
    items1 = np.zeros(npairs, dtype=int) + item1
    prefs = np.ones(npairs)
    userids = np.tile(np.arange(ranked_items.shape[0])[:, None], (1, npairs[1]))

    userids = userids.flatten()
    items1 = items1.flatten()
    items2 = items2.flatten()
    prefs = prefs.flatten()

    if npairs[1] > 1:
        userids_next, items1_next, items2_next, prefs_next = extract_pairs_from_ranking(ranked_items[:, 1:])
        userids = np.concatenate((userids, userids_next))
        items1 = np.concatenate((items1, items1_next))
        items2 = np.concatenate((items2, items2_next))
        prefs = np.concatenate((prefs, prefs_next))

    return userids, items1, items2, prefs


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
                   u_test=None, i1_test=None, i2_test=None, ufeats_test=None):

    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True)

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=True)

    if vscales is not None:
        vscales.append(np.sort(model.rate_sw / model.shape_sw)[::-1])

    if u_test is None:
        return model

    fpred = model.predict_f(ifeats[active_items], ufeats_test)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_pooled(_, i1_tr, i2_tr, ifeats, __, prefs_tr, ___, i1_test, i2_test, ufeats_test):
    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate, verbose=True)

    model.fit(i1_tr, i2_tr, ifeats, prefs_tr, optimize=optimize, use_median_ls=True)

    fpred, _ = np.tile(model.predict_f(ifeats[active_items]), (1, ufeats_test.shape[0]))
    rho_pred, _ = model.predict(ifeats, i1_test, i2_test)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):
    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate, verbose=True)

    model.uselowerbound = False

    joint_ifeats = np.tile(ifeats, (ufeats.shape[0], 1))
    joint_ufeats = np.tile(ufeats, (1, ifeats.shape[0])).reshape((ufeats.shape[0]*ifeats.shape[0], ufeats.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)

    i1_tr = i1_tr + (ifeats.shape[0] * u_tr)
    i2_tr = i2_tr + (ifeats.shape[0] * u_tr)

    model.fit(i1_tr, i2_tr, joint_feats, prefs_tr, optimize=optimize, use_median_ls=True)

    # need to split this up to compute because predict needs pairwise covariance terms and ends up computing full covariance
    batchsize = 100
    nbatches = int(np.ceil(u_test.shape[0] / float(batchsize)))

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

    joint_ifeats = np.tile(ifeats[active_items], (ufeats_test.shape[0], 1))
    joint_ufeats = np.tile(ufeats_test, (1, active_items.shape[0])).reshape((ufeats_test.shape[0]*active_items.shape[0],
                                                                       ufeats_test.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)
    fpred, _ = model.predict_f(joint_feats)
    fpred = fpred.reshape(active_items.shape[0], ufeats_test.shape[0])

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users):
    model = GPPrefPerUser(ufeats.shape[0], max_update_size, shape_s0, rate_s0, ifeats.shape[1], ninducing)
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(ifeats[active_items], chosen_users)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users):

    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], 0, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True)

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(ifeats[active_items], person_features=None)
    fpred = fpred[:, chosen_users]
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users):
    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(1, 1, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True, kernel_func='diagonal')

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(ifeats[active_items], person_features=None)
    fpred = fpred[:, chosen_users]
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


def run_collab_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):
    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True, use_common_mean_t=False)

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=True)

    fpred = model.predict_f(ifeats[active_items], ufeats_test)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred


# def run_GPPL_separate_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):
#     Nfactors = ufeats.shape[0]
#     if Nfactors > 50:
#         Nfactors = 50 # this is the maximum
#
#     model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
#                                   ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
#                                   verbose=True, use_lb=True, kernel_func='diagonal')
#
#     model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=not optimize)
#
#     fpred = model.predict_f(ifeats, ufeats_test)
#     rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)
#
#     # return predictions of preference scores for training users, new testing users, and pairwise testing labels
#     return fpred, rho_pred


def opt_scale_crowd_GPPL(shape_s0, rate_s0, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr):
    '''
    Optimize the function scale to select values of shape_s0 and rate_s0 using Bayesian model selection.

    :return: optimal values of shape_s0 and rate_s0
    '''

    def run_crowd_GPPL_wrapper(loghypers, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr):
        global shape_s0
        global rate_s0
        global optimize

        shape_s0 = np.exp(loghypers[0])
        rate_s0 = np.exp(loghypers[1])
        optimize = True # ensures we use the optimal length-scales when finding the optimal scale hyperparameters

        print('Running with shape_s0 = %f and rate_s0 = %f' % (shape_s0, rate_s0))
        model = run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr)
        lb = model.lowerbound()
        print('Obtained lower bound %f with shape_s0 = %f and rate_s0 = %f' % (lb, shape_s0, rate_s0))
        return -lb

    # initialguess = np.log([shape_s0, rate_s0])
    # args = (u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr)
    # res = minimize(run_crowd_GPPL_wrapper, initialguess, args=args,
    #                method='Nelder-Mead', options={'maxiter': 100, 'fatol': 1e-3, 'gatol': 1e10})
    # opt_hyperparams = res['x']
    # shape_s0 = np.exp(opt_hyperparams[0])
    # rate_s0 = np.exp(opt_hyperparams[1])

    sh_vals = [0.1, 1, 10, 100, 1000]
    r_vals = [0.1, 1, 10, 100, 1000]

    minval = np.inf
    min_sh_idx = -1
    min_r_idx = -1

    for sh, shape_s0 in enumerate(sh_vals):
        for r, rate_s0 in enumerate(r_vals):
            lb = run_crowd_GPPL_wrapper([np.log(shape_s0), np.log(rate_s0)], u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr)
            if lb < minval:
                minval = lb
                min_sh_idx = sh
                min_r_idx = r
                print('New best value: %f, with hypers %f and %f' % (-lb, sh, r))

    shape_s0 = sh_vals[min_sh_idx]
    rate_s0 = r_vals[min_r_idx]

    print('Final best value: %f, with hypers %f and %f' % (-minval, shape_s0, rate_s0))

    return shape_s0, rate_s0


def train_test(method_name, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users):

    ufeats_test = ufeats[chosen_users] # the features of the users whose scores we should predict

    if method_name == 'crowd-GPPL':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    elif method_name == 'GPPL-pooled':
        return run_GPPL_pooled(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    elif method_name == 'GPPL-joint':
        return run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    elif method_name == 'GPPL-per-user':
        return run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users)
    elif method_name == 'crowd-GPPL\\u':
        return run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users)
    elif method_name == 'crowd-BMF':
        return run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users)
    elif method_name == 'collab-GPPL': # No common mean, i.e. like Houlsby but SVI
        return run_collab_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    # elif method_name == 'GPPL+BMF': # like Khan. Not implemented yet -- get results from Khan paper where possible.
    #     return run_GPPL_separate_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)


def subsample_data():

    if debug_small:
        nusers_tr = 3
        npairs_tr = 4
        npairs_test = 1
    elif sushiB:
        nusers_tr = 5000
        npairs_tr = 10
        npairs_test = 1
    else:
        nusers_tr = 1000
        npairs_tr = 15
        npairs_test = 5

    # select 1000 random users
    chosen_users = np.random.choice(nusers, nusers_tr, replace=False)

    user_pairidxs_tr = np.random.choice(int(prefs.shape[0] / nusers), size=npairs_tr, replace=False)
    user_pairidxs_test = np.random.choice(int(prefs.shape[0] / nusers), size=npairs_test, replace=False)

    pairidxs_tr = np.empty(0, dtype=int)
    pairidxs_test = np.empty(0, dtype=int)

    for u in chosen_users:
        uidxs = np.argwhere(userids == u).flatten()

        pairidxs_tr = np.append(pairidxs_tr, uidxs[user_pairidxs_tr])
        pairidxs_test = np.append(pairidxs_test, uidxs[user_pairidxs_test])

    u_tr = userids[pairidxs_tr]
    i1_tr = items1[pairidxs_tr]
    i2_tr = items2[pairidxs_tr]
    prefs_tr = prefs[pairidxs_tr]

    u_test = userids[pairidxs_test]
    i1_test = items1[pairidxs_test]
    i2_test = items2[pairidxs_test]
    prefs_test = prefs[pairidxs_test]

    ranked_lists = ranking_data.values[chosen_users, 2:]
    scores = np.zeros((nitems, nusers_tr)) - 1
    scores[ranked_lists, np.arange(nusers_tr)[:, None]] = ranked_lists.shape[1] - np.arange(ranked_lists.shape[1])[None, :]

    # some of the metrics fail if all the labels are ones, and other methods could cheat, so flip half at random
    idxs_to_flip = np.random.choice(len(pairidxs_test), int(0.5 * len(pairidxs_test)), replace=False)
    tmp = i1_test[idxs_to_flip]
    i1_test[idxs_to_flip] = i2_test[idxs_to_flip]
    i2_test[idxs_to_flip] = tmp
    prefs_test[idxs_to_flip] = 1 - prefs_test[idxs_to_flip]

    return u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, scores, chosen_users


def run_sushi_expt(methods, expt_name):
    nreps = 1

    # predictions from all reps and methods
    fpred_all = []
    rho_pred_all = []

    # metrics from all reps and methods
    acc_all = []
    logloss_all = []
    spearman_all = []
    times_all = []

    # for repeatability
    np.random.seed(209)

    results_path = './results/' + expt_name
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for rep in range(nreps):
        # Get training and test data
        u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, scores, chosen_users = subsample_data()

        fpred_r = []
        rho_pred_r = []

        acc_r = []
        logloss_r = []
        spearman_r = []
        times_r = []

        for m in methods:
            # Train and Predict
            logging.info("Starting test with method %s..." % (m))
            starttime = time.time()

            fpred, rho_pred = train_test(m, u_tr, i1_tr, i2_tr, item_features, user_features, prefs_tr,
                                         u_test, i1_test, i2_test, chosen_users)

            endtime = time.time()
            times_r.append(endtime - starttime)

            # Save predictions
            fpred_r.append(fpred.flatten())
            rho_pred_r.append(rho_pred.flatten())

            # Compute metrics
            acc_m = accuracy_score(prefs_test, np.round(rho_pred))
            logloss_m = log_loss(prefs_test.flatten(), rho_pred.flatten())
            spearman_m = spearmanr(scores[scores > -1].flatten(), fpred[scores > -1].flatten())[0]

            # Save metrics
            acc_r.append(acc_m)
            logloss_r.append(logloss_m)
            spearman_r.append(spearman_m)

            print('Results for %s at rep %i: acc=%.2f, CEE=%.2f, r=%.2f' % (m, rep, acc_m, logloss_m, spearman_m))

        fpred_all.append(fpred_r)
        rho_pred_all.append(rho_pred_r)

        acc_all.append(acc_r)
        logloss_all.append(logloss_r)
        spearman_all.append(spearman_r)
        times_all.append(times_r)

        # save predictions to file
        np.savetxt(results_path + '/fpred_rep%i.csv' % rep, fpred_r, delimiter=',', fmt='%f')
        np.savetxt(results_path + '/rho_pred_rep%i.csv' % rep, rho_pred_r, delimiter=',', fmt='%f')

    # Compute means and stds of metrics (or medians/quartiles for plots?)
    acc_mean = np.mean(np.array(acc_all), axis=0)
    logloss_mean = np.mean(np.array(logloss_all), axis=0)
    spearman_mean = np.mean(np.array(spearman_all), axis=0)
    times_mean = np.mean(np.array(times_all), axis=0)

    acc_std = np.std(np.array(acc_all), axis=0)
    logloss_std = np.std(np.array(logloss_all), axis=0)
    spearman_std = np.std(np.array(spearman_all), axis=0)
    times_std = np.std(np.array(times_all), axis=0)

    # Print means and stds of metrics in Latex format ready for copying into a table
    print('Table of results:')

    line = 'Method & Acc. & CEE & r & runtime (s)\\\\ \n'
    print(line)
    lines = [line]

    for m, method in enumerate(methods):
        line = method + ' & '
        line += '%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\\\ \n' % (acc_mean[m], acc_std[m], logloss_mean[m],
                                        logloss_std[m], spearman_mean[m], spearman_std[m], times_mean[m], times_std[m])
        print(line)
        lines.append(line)

    with open(results_path + '/results.tex', 'w') as fh:
        fh.writelines(lines)

# Load feature data ----------------------------------------------------------------------------

item_feat_file = './data/sushi3-2016/sushi3.idata'
user_feat_file = './data/sushi3-2016/sushi3.udata'

item_data = pd.read_csv(item_feat_file, sep='\t', index_col=0, header=None)
item_features = item_data.values[:, 1:].astype(float)
item_features = convert_discrete_to_continuous(item_features, cols_to_convert=[2])

user_data = pd.read_csv(user_feat_file, sep='\t', index_col=0, header=None)
user_features = user_data.values.astype(float)
user_features = convert_discrete_to_continuous(user_features, cols_to_convert=[0, 3, 4, 6, 7])


# Load SUSHI-A dataset -------------------------------------------------------------------------------------------------

sushi_prefs_file = './data/sushi3-2016/sushi3a.5000.10.order'
ranking_data = pd.read_csv(sushi_prefs_file, sep=' ', skiprows=1, header=None)

userids, items1, items2, prefs = extract_pairs_from_ranking(ranking_data.values[:, 2:].astype(int))

nusers = len(np.unique(userids))
active_items = np.unique(np.array([items1, items2]))
item_features = item_features[:np.max(active_items)+1, :]
nitems = len(active_items)
print('Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
print('Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
print('User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))


# for debugging --------------------------------------------------------------------------------------------------------

debug_small = False

if debug_small:
    ndebug = 50
    userids = userids[:ndebug]
    items1 = items1[:ndebug]
    items2 = items2[:ndebug]
    prefs = prefs[:ndebug]

    nusers = len(np.unique(userids))
    nitems = len(np.unique(np.array([items1, items2])))
    item_features = item_features[:20, :] # np.unique(np.array([items1, items2]))]
    user_features = user_features[np.unique(userids)]

    print('Debug: Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
    print('Debug: Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
    print('Debug: User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))



# Hyperparameters common to most models --------------------------------------------------------------------------------

shape_s0 = 0.1
rate_s0 = 0.1
max_update_size = 1000
ninducing = 500
forgetting_rate = 0.9

sushiB = False

# OPTIMISE THE FUNcTION SCALE FIRST ON ONE FOLD of Sushi A, NO DEV DATA NEEDED -----------------------------------------

print('Optimizing function scales ...')
np.random.seed(2309234)
u_tr, i1_tr, i2_tr, prefs_tr, _, _, _, _, _, _ = subsample_data()
shape_s0, rate_s0 = opt_scale_crowd_GPPL(shape_s0, rate_s0, u_tr, i1_tr, i2_tr,
                                         item_features, user_features, prefs_tr)
print('Found scale hyperparameters: %f, %f' % (shape_s0, rate_s0))

# Experiment name tag
tag = '_3'

np.savetxt('./results/' + 'scale_hypers' + tag + '.csv', [shape_s0, rate_s0], fmt='%f', delimiter=',')

# Run Test NO LENGTHSCALE OPTIMIZATION ---------------------------------------------------------------------------------

vscales = None # don't record the v scale factors

# Repeat 25 times... Run each method and compute its metrics.
methods = [
           'crowd-GPPL',
           'GPPL-pooled',
           'GPPL-joint',
           'GPPL-per-user',
           'crowd-GPPL\\u',
           'crowd-BMF',
           'collab-GPPL', # Houlsby
           # 'GPPL+BMF' # khan -- excluded from this experiment
           ]

optimize = False
sushiB = False
run_sushi_expt(methods, 'sushi_10' + tag)

# OPTIMIZE ARD ---------------------------------------------------------------------------------------------------------

vscales = []

# Repeat 25 times... Run each method and compute its metrics.
methods = [
           'crowd-GPPL',
           'GPPL-pooled',
           'GPPL-joint',
           # 'GPPL-per-user',
           'crowd-GPPL\\u',
           # 'crowd-BMF',
           # 'collab-GPPL', # Houlsby
           # 'GPPL+BMF' # khan -- excluded from this experiment
           ]

# hyperparameters common to most models
optimize = True
sushiB = False
run_sushi_expt(methods, 'sushi_10_opt' + tag)

vscales_A = vscales
vscales = None

# Load SUSHI-B dataset -------------------------------------------------------------------------------------------------

sushi_prefs_file = './data/sushi3-2016/sushi3b.5000.10.order'
ranking_data = pd.read_csv(sushi_prefs_file, sep=' ', skiprows=1, header=None)

userids, items1, items2, prefs = extract_pairs_from_ranking(ranking_data.values[:, 2:].astype(int))

nusers = len(np.unique(userids))
active_items = np.unique(np.array([items1, items2]))
item_features = item_features[:np.max(active_items)+1, :]
nitems = len(active_items)
print('Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
print('Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
print('User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))

# for debugging --------------------------------------------------------------------------------------------------------

debug_small = False

if debug_small:
    ndebug = 50
    userids = userids[:ndebug]
    items1 = items1[:ndebug]
    items2 = items2[:ndebug]
    prefs = prefs[:ndebug]

    nusers = len(np.unique(userids))
    nitems_debug = len(np.unique(np.array([items1, items2])))
    #item_features = item_features[:20, :] # np.unique(np.array([items1, items2]))]
    user_features = user_features[np.unique(userids)]

    print('Debug: Found %i users, %i items, and %i pairs per user.' % (nusers, nitems_debug, prefs.shape[0]/nusers))
    print('Debug: Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
    print('Debug: User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))

# SUSHI B dataset, no opt. ---------------------------------------------------------------------------------------------

vscales = None # don't record the v factor scale factors

# Repeat 25 times... Run each method and compute its metrics.
methods = [
           'crowd-GPPL',
           'GPPL-pooled',
           'GPPL-joint',
           'GPPL-per-user',
           'crowd-GPPL\\u',
           'crowd-BMF',
           'collab-GPPL', # Houlsby
           # 'GPPL+BMF' # khan -- excluded from this experiment
           ]

# hyperparameters common to most models
optimize = False
sushiB = True
run_sushi_expt(methods, 'sushi_100' + tag)

# SUSHI B dataset, ARD -------------------------------------------------------------------------------------------------

vscales = []

# Repeat 25 times... Run each method and compute its metrics.
methods = [
           'crowd-GPPL',
           'GPPL-pooled',
           'GPPL-joint',
           # 'GPPL-per-user',
           'crowd-GPPL\\u',
           # 'crowd-BMF',
           # 'collab-GPPL', # Houlsby
           # 'GPPL+BMF' # khan -- excluded from this experiment
           ]

# hyperparameters common to most models
optimize = True
sushiB = True
run_sushi_expt(methods, 'sushi_100_opt' + tag)

vscales_B = vscales

# Plot the latent factor scales ----------------------------------------------------------------------------------------

vscales_A = np.mean(vscales_A, axis=0)
vscales_B = np.mean(vscales_B, axis=0)

logging.basicConfig(level=logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

fig = plt.figure(figsize=(5, 4))

markers = ['o', 'x', '+', '>', '<', '*']

plt.plot(np.arange(vscales_A.shape[0]), vscales_A, marker=markers[0], label='Sushi A', linewidth=2, markersize=8)
plt.plot(np.arange(vscales_B.shape[0]), vscales_B, marker=markers[1], label='Sushi B', linewidth=2, markersize=8)

plt.ylabel('Inverse scale 1/s')
plt.xlabel('Factor ID')

plt.grid('on', axis='y')
plt.legend(loc='best')
plt.tight_layout()

figure_root_path = './results/sushi_factors'
if not os.path.exists(figure_root_path):
    os.mkdir(figure_root_path)

plt.savefig(figure_root_path + '/sushi_factor_scales.pdf')

np.savetxt(figure_root_path + '/sushi_A_factor_scales.csv', vscales_A, delimiter=',', fmt='%f')
np.savetxt(figure_root_path + '/sushi_B_factor_scales.csv', vscales_B, delimiter=',', fmt='%f')

logging.basicConfig(level=logging.DEBUG)  # switch back to the debug output
