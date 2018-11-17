'''
Test on the smaller Sushi dataset with 10 items. This is for comparison with Houlsby et al in terms of classification
error.
'''
import os
import sys

# include the paths for the other directories
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

from collab_pref_learning_svi import CollabPrefLearningSVI
from gp_pref_learning import GPPrefLearning
from per_user_pref_learning import GPPrefPerUser

sushi_prefs_file = './data/sushi3-2016/sushi3a.5000.10.order'
item_feat_file = './data/sushi3-2016/sushi3.idata'
user_feat_file = './data/sushi3-2016/sushi3.udata'

item_data = pd.read_csv(item_feat_file, sep='\t', index_col=0, header=None)
item_features = item_data.values[:, 1:].astype(float)

user_data = pd.read_csv(user_feat_file, sep='\t', index_col=0, header=None)
user_features = user_data.values.astype(float)

ranking_data = pd.read_csv(sushi_prefs_file, sep=' ', skiprows=1, header=None)

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

    # some of the metrics fail if all the labels are ones, and other methods could cheat, so flip half at random
    idxs_to_flip = np.random.choice(npairs[1], int(0.5 * npairs[1]), replace=False)

    tmp = items1[idxs_to_flip]
    items1[idxs_to_flip] = items2[idxs_to_flip]
    items2[idxs_to_flip] = tmp
    prefs[idxs_to_flip] = 1 - prefs[idxs_to_flip]

    return userids, items1, items2, prefs

userids, items1, items2, prefs = extract_pairs_from_ranking(ranking_data.values[:, 2:].astype(int))

nusers = len(np.unique(userids))
active_items = np.unique(np.array([items1, items2]))
item_features = item_features[:np.max(active_items)+1, :]
nitems = len(active_items)
print('Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))

print('Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
print('User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))

# for debugging --------------------------------------------------------------------------------------------------------
# ndebug = 50
# userids = userids[:ndebug]
# items1 = items1[:ndebug]
# items2 = items2[:ndebug]
# prefs = prefs[:ndebug]
#
# # need to do this in case the sample only contains ones or zeros in the test set.
# idxs_to_flip = np.random.choice(ndebug, int(0.5 * ndebug), replace=False)
# tmp = items1[idxs_to_flip]
# items1[idxs_to_flip] = items2[idxs_to_flip]
# items2[idxs_to_flip] = tmp
# prefs[idxs_to_flip] = 1 - prefs[idxs_to_flip]
#
# nusers = len(np.unique(userids))
# nitems = len(np.unique(np.array([items1, items2])))
# item_features = item_features[:20, :] # np.unique(np.array([items1, items2]))]
# user_features = user_features[np.unique(userids)]
#
# print('Debug: Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
# print('Debug: Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
# print('Debug: User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))

# ----------------------------------------------------------------------------------------------------------------------

# Get the test values for the sushi according to the users' rankings.
ranks_test = ranking_data.values[:, 2:] # use this to compute spearman

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
shape_s0 = 0.1
rate_s0 = 0.1
max_update_size = 1000
ninducing = 50
forgetting_rate = 0.9
optimize = False

def run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):

    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True)

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=not optimize)

    fpred = model.predict_f(ifeats[active_items], ufeats_test)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred

def run_GPPL_pooled(_, i1_tr, i2_tr, ifeats, __, prefs_tr, ___, i1_test, i2_test, ufeats_test):
    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate, verbose=True)

    model.fit(i1_tr, i2_tr, ifeats, prefs_tr, optimize=optimize, use_median_ls=not optimize)

    fpred, _ = np.tile(model.predict_f(ifeats[active_items]), (1, ufeats_test.shape[0]))
    rho_pred, _ = model.predict(ifeats, i1_test, i2_test)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred

def run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):
    model = GPPrefLearning(ifeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls_initial=None, use_svi=True,
                   ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate, verbose=True)

    joint_ifeats = np.tile(ifeats, (ufeats.shape[0], 1))
    joint_ufeats = np.tile(ufeats, (1, ifeats.shape[0])).reshape((ufeats.shape[0]*ifeats.shape[0], ufeats.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)

    i1_tr = i1_tr + (ifeats.shape[0] * u_tr)
    i2_tr = i2_tr + (ifeats.shape[0] * u_tr)

    model.fit(i1_tr, i2_tr, joint_feats, prefs_tr, optimize=optimize, use_median_ls=not optimize)

    rho_pred, _ = model.predict(joint_feats, i1_test, i2_test)

    joint_ifeats = np.tile(ifeats[active_items], (ufeats_test.shape[0], 1))
    joint_ufeats = np.tile(ufeats_test, (1, active_items.shape[0])).reshape((ufeats_test.shape[0]*active_items.shape[0],
                                                                       ufeats_test.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)
    fpred, _ = model.predict_f(joint_feats)
    fpred = fpred.reshape(active_items.shape[0], ufeats_test.shape[0])

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred

def run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, chosen_users):
    model = GPPrefPerUser(ufeats.shape[0], max_update_size, shape_s0, rate_s0, ifeats.shape[1])
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=not optimize)

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

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=not optimize)

    fpred = model.predict_f(ifeats[active_items], person_features=None, personids=chosen_users)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred

def run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test):
    Nfactors = ufeats.shape[0]
    if Nfactors > 50:
        Nfactors = 50 # this is the maximum

    model = CollabPrefLearningSVI(1, 1, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0, ls=None, nfactors=Nfactors,
                                  ninducing=ninducing, max_update_size=max_update_size, forgetting_rate=forgetting_rate,
                                  verbose=True, use_lb=True, kernel_func='diagonal')

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=not optimize)

    fpred = model.predict_f(ifeats[active_items], person_features=None, personids=chosen_users)
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

    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=not optimize)

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
        return run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    elif method_name == 'collab-GPPL': # No common mean, i.e. like Houlsby but SVI
        return run_collab_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)
    # elif method_name == 'GPPL+BMF': # like Khan. Not implemented yet -- get results from Khan paper where possible.
    #     return run_GPPL_separate_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, ufeats_test)

def subsample_data():

    nusers_tr = 1000

    npairs_tr = 15
    npairs_test = 5

    # select 1000 random users
    chosen_users = np.random.choice(nusers, size=nusers_tr, replace=False)

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
    scores[ranked_lists, np.arange(nusers_tr)[:, None]] = nitems - np.arange(nitems)[None, :]

    # some of the metrics fail if all the labels are ones, and other methods could cheat, so flip half at random
    idxs_to_flip = np.random.choice(len(pairidxs_test), int(0.5 * len(pairidxs_test)), replace=False)
    tmp = i1_test[idxs_to_flip]
    i1_test[idxs_to_flip] = i2_test[idxs_to_flip]
    i2_test[idxs_to_flip] = tmp
    prefs_test[idxs_to_flip] = 1 - prefs_test[idxs_to_flip]

    return u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, scores, chosen_users

nreps = 25
nmethods = len(methods)

# predictions from all reps and methods
fpred_all = []
rho_pred_all = []

# metrics from all reps and methods
acc_all = []
logloss_all = []
spearman_all = []

# for repeatability
np.random.seed(209)

results_path = './results/sushi_10'
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

    for m in methods:
        # Train and Predict
        fpred, rho_pred = train_test(m, u_tr, i1_tr, i2_tr, item_features, user_features, prefs_tr,
                                     u_test, i1_test, i2_test, chosen_users)

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

    # save predictions to file
    pd.DataFrame(fpred_all).to_csv(results_path + '/fpred_rep%i.csv' % rep, sep=',', header=False)
    pd.DataFrame(rho_pred_all).to_csv(results_path + '/rho_pred_rep%i.csv' % rep, sep=',')

# Compute means and stds of metrics (or medians/quartiles for plots?)
acc_mean = np.mean(np.array(acc_all), axis=0)
logloss_mean = np.mean(np.array(logloss_all), axis=0)
spearman_mean = np.mean(np.array(spearman_all), axis=0)

acc_std = np.std(np.array(acc_all), axis=0)
logloss_std = np.std(np.array(logloss_all), axis=0)
spearman_std = np.std(np.array(spearman_all), axis=0)

# Print means and stds of metrics in Latex format ready for copying into a table
print('Table of results:')

line = 'Method & Acc. & CEE & r \\'
print(line)
lines = [line]

for m, method in enumerate(methods):
    line = method + ' & '
    line += '%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\' % (acc_mean[m], acc_std[m], logloss_mean[m], logloss_std[m],
                                                            spearman_mean[m], spearman_std[m])
    print(line)
    lines.append(line)

with open(results_path + '/results.tex', 'w') as fh:
    fh.writelines(lines)