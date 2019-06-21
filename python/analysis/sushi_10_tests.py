'''
Test on the smaller Sushi dataset with 10 items. This is for comparison with Houlsby et al in terms of classification
error.

Current status:
- learning sy seems to help a lot on Sushi and possibly on some convincingness consensus results.
- however, the corrected Q computations have not been computed with sy learning turned on
- so we need to test all to ensure that the current setup works
- since Q has changed, it may mean that shape_s and rate_s values also need to change
- it may be better to tune hypers for sw, sy, st separately to improve performance on convincingness dataset

TODO Run Sushi A, 25 reps, crowdGPPL with ***no SVI*** and same subsamples as in the paper. Can we reproduce the results with current code?
TODO Run Sushi A crowdGPPL with SVI, with/without user features, same settings as the paper. Does stochastic sampling cause a problem?
TODO Run Sushi B as described in the paper.
TODO Run convincingness consensus. shape/rate may need tuning separately for sw, sy, st.
TODO Run convincingness personalised. shape/rate may need tuning separately for sw, sy, st.
'''
import datetime
import os
import sys

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


verbose = True

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
                   u_test=None, i1_test=None, i2_test=None, u_un=None, i1_un=None, i2_un=None, ufeats_un=None,
                   ninducing=None, use_common_mean=True):

    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    if ninducing is None:
        ninducing = np.max([ifeats.shape[0], ufeats.shape[0]])

    model = CollabPrefLearningSVI(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_sy0=1e2, rate_sy0=1e2, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                  use_common_mean_t=use_common_mean, delay=delay)

    model.factors_with_features = np.arange(Nfactors/2)
    model.use_local_obs_posterior_y = False
    model.max_Kw_size = max_Kw_size
    model.max_iter = 200
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, ufeats, optimize, use_median_ls=True)

    if vscales is not None:
        vscales.append(np.sort((model.rate_sw / model.shape_sw) * (model.rate_sy / model.shape_sy))[::-1])

    if u_test is None:
        return model

    # fpred = model.predict_f(ifeats[active_items], ufeats)
    # rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)
    fpred = model.predict_f()
    rho_pred = model.predict(u_test, i1_test, i2_test)

    # fpred_un = model.predict_f(None, ufeats_un)
    # rho_pred_un = model.predict(u_un, i1_un, i2_un, None, ufeats_un)
    # fpred_un = model.predict_f(ifeats[active_items], ufeats_un)
    # rho_pred_un = model.predict(u_un, i1_un, i2_un, ifeats, ufeats)
    fpred_un = None
    rho_pred_un = None

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un


def run_GPPL_pooled(_, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, __, i1_test, i2_test,
                    ___, i1_un, i2_un, ufeats_un):
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

    if len(i1_un):
        rho_pred_un, _ = model.predict(ifeats, i1_un, i2_un)
    else:
        rho_pred_un = []

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, np.tile(fpred[:, 0:1], (1, ufeats_un.shape[0])), rho_pred_un


def run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                   u_un=None, i1_un=None, i2_un=None, ufeats_un=None):

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


    batchsize = 100
    nbatches = int(np.ceil(np.unique(u_test).shape[0] / float(batchsize)))

    rho_pred_un = []
    for batch in range(nbatches):
        # all of the pairs and features that relate to a batch of users
        idxs = (u_un >= (batch) * batchsize) & (u_un < (batch+1) * batchsize)
        u_test_b = u_un[idxs]
        i1_test_b = i1_un[idxs]
        i2_test_b = i2_un[idxs]

        joint_feats_idxs_b, pairs_b = np.unique([i1_test_b + (ifeats.shape[0] * u_test_b),
                                                 i2_test_b + (ifeats.shape[0] * u_test_b)],
                                                return_inverse=True)
        pairs_b = pairs_b.reshape(2, i1_test_b.shape[0])

        rho_pred_b, _ = model.predict(joint_feats[joint_feats_idxs_b], pairs_b[0], pairs_b[1])
        rho_pred_un = np.append(rho_pred_un, rho_pred_b)

    joint_ifeats = np.tile(ifeats, (ufeats_un.shape[0], 1))
    joint_ufeats = np.tile(ufeats_un, (1, ifeats.shape[0])).reshape((ufeats_un.shape[0]*ifeats.shape[0],
                                                                       ufeats_un.shape[1]))
    joint_feats = np.concatenate((joint_ifeats, joint_ufeats), axis=1)
    fpred_un, _ = model.predict_f(joint_feats)
    fpred_un = fpred_un.reshape(ufeats_un.shape[0], ifeats.shape[0]).T

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un


def run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, _, __, ufeats_un):
    model = GPPrefPerUser(ufeats.shape[0], max_update_size, shape_s0, rate_s0, ifeats.shape[1], ninducing)
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, personids=None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, np.zeros((ifeats.shape[0], ufeats_un.shape[0])), 0.5 * np.ones(len(u_un))


def run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                             u_un, i1_un, i2_un, ufeats_un):

    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], 0, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_st0=shape_s0, rate_st0=rate_s0,
                                  shape_sy0=1e2, rate_sy0=1e2, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                  use_common_mean_t=True, delay=delay)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 200
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    fpred_un = model.predict_f(None, personids=np.arange(ufeats_un.shape[0]) + ufeats.shape[0])
    rho_pred_un = model.predict(ufeats.shape[0] + np.zeros(len(u_un)), i1_un, i2_un, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un


def run_khan(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                             u_un, i1_un, i2_un, ufeats_un):

    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningSVI(ifeats.shape[1], 0, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_st0=1, rate_st0=1e10, # khan model has no prior -- we mimic this with a very vague prior
                                  shape_sy0=1e10, rate_sy0=1e10, ls=None,
                                  nfactors=Nfactors, ninducing=np.max([ifeats.shape[0], ufeats.shape[0]]),
                                  max_update_size=50000, # use all pairs
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True, kernel_func='diagonal',
                                  use_common_mean_t=True, delay=delay, personal_component=True)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 200
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    fpred_un = model.predict_f(None, personids=np.arange(ufeats_un.shape[0]) + ufeats.shape[0])
    rho_pred_un = model.predict(ufeats.shape[0] + np.zeros(len(u_un)), i1_un, i2_un, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un

def run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un, ufeats_un):
    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningSVI(1, 1, mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                  shape_sy0=1e10, rate_sy0=1e10, ls=None,
                                  nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                  forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True, kernel_func='diagonal',
                                  delay=delay)
    model.max_Kw_size = max_Kw_size
    model.max_iter = 500
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, None)

    fpred_un = model.predict_f(None, personids=np.arange(ufeats_un.shape[0]) + ufeats.shape[0])
    rho_pred_un = model.predict(ufeats.shape[0] + np.zeros(len(u_un)), i1_un, i2_un, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un


def run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                              u_un, i1_un, i2_un, ufeats_un, use_common_mean=False):
    Nfactors = ufeats.shape[0]
    if Nfactors > max_facs:
        Nfactors = max_facs # this is the maximum

    model = CollabPrefLearningFITC(ifeats.shape[1], ufeats.shape[1], mu0=0, shape_s0=shape_s0, rate_s0=rate_s0,
                                   shape_sy0=1e10, rate_sy0=1e10, ls=None,
                                   nfactors=Nfactors, ninducing=ninducing, max_update_size=max_update_size,
                                   forgetting_rate=forgetting_rate, verbose=verbose, use_lb=True,
                                   use_common_mean_t=use_common_mean, delay=delay,
                                   exhaustive_train_count=0)

    model.max_Kw_size = max_Kw_size
    model.max_iter = 200
    model.fit(u_tr, i1_tr, i2_tr, ifeats, prefs_tr, None, optimize, use_median_ls=True)

    fpred = model.predict_f(None, None)
    rho_pred = model.predict(u_test, i1_test, i2_test, None, None)

    fpred_un = model.predict_f(None, personids=np.arange(ufeats_un.shape[0]) + ufeats.shape[0])  # ufeats_un)
    rho_pred_un = model.predict(ufeats.shape[0] + np.zeros(len(u_un)), i1_un, i2_un, ifeats, None)

    # return predictions of preference scores for training users, new testing users, and pairwise testing labels
    return fpred, rho_pred, fpred_un, rho_pred_un


def opt_scale_crowd_GPPL(shape_s0, rate_s0, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr,
                         u_test, i1_test, i2_test, prefs_test, chosen_users):
    '''
    Optimize the function scale to select values of shape_s0 and rate_s0 using Bayesian model selection.

    :return: optimal values of shape_s0 and rate_s0
    '''

    def run_crowd_GPPL_wrapper(loghypers, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr,
                               u_test, i1_test, i2_test, prefs_test):
        global shape_s0
        global rate_s0
        global optimize

        shape_s0 = np.exp(loghypers[0])
        rate_s0 = np.exp(loghypers[1])
        optimize = False # optimize = True ensures we use the optimal length-scales when finding the optimal scale
        # hyperparameters but takes much longer to run

        print('Running with shape_s0 = %f and rate_s0 = %f' % (shape_s0, rate_s0))
        model = run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, ninducing=25)
        #lb = model.lowerbound()
        #print('Obtained lower bound %f with shape_s0 = %f and rate_s0 = %f' % (lb, shape_s0, rate_s0))
        #return -lb
        rho_pred = model.predict(u_test, i1_test, i2_test, ifeats, ufeats)
        acc_m = accuracy_score(prefs_test.astype(int), np.round(rho_pred).astype(int))

        print('Accuracy of %f with shape = %f and rate = %f' % (acc_m, shape_s0, rate_s0))

        return -acc_m

    # initialguess = np.log([shape_s0, rate_s0])
    # args = (u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr)
    # res = minimize(run_crowd_GPPL_wrapper, initialguess, args=args,
    #                method='Nelder-Mead', options={'maxiter': 100, 'fatol': 1e-3, 'gatol': 1e10})
    # opt_hyperparams = res['x']
    # shape_s0 = np.exp(opt_hyperparams[0])
    # rate_s0 = np.exp(opt_hyperparams[1])

    sh_vals = [1]#[0.1, 1, 10, 100]
    r_vals = [0.1, 1, 10, 100, 1000, 10000]

    minval = np.inf
    min_sh_idx = -1
    min_r_idx = -1

    u_tr = np.array([np.argwhere(chosen_users == u).flatten()[0] for u in u_tr])
    u_test = np.array([np.argwhere(chosen_users == u).flatten()[0] for u in u_test])

    for sh, shape_s0 in enumerate(sh_vals):
        for r, rate_s0 in enumerate(r_vals):
            lb = run_crowd_GPPL_wrapper([np.log(shape_s0), np.log(rate_s0)], u_tr, i1_tr, i2_tr, ifeats,
                                        ufeats[chosen_users], prefs_tr, u_test, i1_test, i2_test, prefs_test)
            if lb < minval:
                minval = lb
                min_sh_idx = sh
                min_r_idx = r
                print('New best value: %f, with hypers %f and %f' % (-lb, shape_s0, rate_s0))

    shape_s0 = sh_vals[min_sh_idx]
    rate_s0 = r_vals[min_r_idx]

    print('Final best value: %f, with hypers %f and %f' % (-minval, shape_s0, rate_s0))

    return shape_s0, rate_s0


def train_test(method_name, u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
               ufeats_un):

    if method_name == 'crowd-GPPL':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un, ninducing=ninducing)
    elif method_name == 'crowd-GPPL-noConsensus':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un, ninducing=ninducing, use_common_mean=False)
    elif method_name == 'crowd-GPPL-noInduc':
        return run_crowd_GPPL(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un, ninducing=None)
    elif method_name == 'GPPL-pooled':
        return run_GPPL_pooled(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un)
    elif method_name == 'GPPL-joint':
        return run_GPPL_joint(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un)
    elif method_name == 'GPPL-per-user':
        return run_GPPL_per_user(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un)
    elif method_name == 'crowd-GPPL\\u':
        return run_crowd_GPPL_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                                        u_un, i1_un, i2_un, ufeats_un)
    elif method_name == 'khan':
        return run_khan(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                                        u_un, i1_un, i2_un, ufeats_un)
    elif method_name == 'crowd-BMF':
        return run_crowd_BMF(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test, u_un, i1_un, i2_un,
                              ufeats_un)
    elif method_name == 'crowd-GPPL-FITC\\u-noConsensus': # No common mean, i.e. like Houlsby but SVI
        return run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                                         u_un, i1_un, i2_un, ufeats_un)
    elif method_name == 'crowd-GPPL-FITC\\u':
        return run_collab_FITC_without_u(u_tr, i1_tr, i2_tr, ifeats, ufeats, prefs_tr, u_test, i1_test, i2_test,
                                         u_un, i1_un, i2_un, ufeats_un, use_common_mean=True)

def subsample_data(test_number):

    if debug_small:
        nusers_tr = 5
        npairs_tr = 4
        npairs_test = 1
        nusers_unseen = 2
    elif test_number == -1: # optimization on dev set. Use lots of data for evaluation and training.
        nusers_tr = 100
        npairs_tr = 25
        npairs_test = 20
        nusers_unseen = 0
    elif test_number == 0 or test_number == 1: # sushi-A-small
        nusers_tr = 100
        npairs_tr = 5
        npairs_test = 25
        nusers_unseen = 0
    elif test_number == 2 or test_number == 3: # sushi-A
        nusers_tr = 100
        npairs_tr = 20
        npairs_test = 25
        nusers_unseen = 0
    elif test_number == 4 or test_number == 5: # sushiB
        nusers_tr = 5000
        npairs_tr = 10
        npairs_test = 1
        nusers_unseen = 0

    # make sure we don't try to select more users than there really are
    if nusers_tr + nusers_unseen > nusers:
        nusers_tr = nusers - nusers_unseen

    # select 1000 random users # select the first N users
    chosen_users = np.random.choice(nusers, nusers_tr + nusers_unseen, replace=False)  #np.arange(nusers_tr + nusers_unseen)
    chosen_users_unseen = chosen_users[nusers_tr:]
    chosen_users = chosen_users[:nusers_tr]

    pairidxs_tr = np.empty(0, dtype=int)
    pairidxs_test = np.empty(0, dtype=int)

    for u in chosen_users:
        uidxs = np.argwhere(userids == u).flatten()

        npairs_tr_u = npairs_tr
        npairs_test_u = npairs_test
        if npairs_tr + npairs_test > len(uidxs):
            if len(uidxs) > 1:
                npairs_tr_u = len(uidxs) - 1
                npairs_test_u = 1
            else:
                npairs_tr_u = len(uidxs)
                npairs_test_u = 0

        user_pairidxs = np.random.choice(len(uidxs), size=npairs_tr_u + npairs_test_u, replace=False)
        user_pairidxs_tr = user_pairidxs[:npairs_tr_u]
        user_pairidxs_test = user_pairidxs[npairs_tr_u:]

        pairidxs_tr = np.append(pairidxs_tr, uidxs[user_pairidxs_tr])
        pairidxs_test = np.append(pairidxs_test, uidxs[user_pairidxs_test])

    pairidxs_unseen = np.empty(0, dtype=int)

    for u in chosen_users_unseen:
        uidxs = np.argwhere(userids == u).flatten()

        user_pairidxs = np.random.choice(len(uidxs), size=npairs_test, replace=False)

        pairidxs_unseen = np.append(pairidxs_unseen, uidxs[user_pairidxs])

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

    ranked_lists = ranking_data.values[chosen_users_unseen, 2:]
    scores_unseen = np.zeros((nitems, nusers_unseen)) - 1
    scores_unseen[ranked_lists, np.arange(nusers_unseen)[:, None]] = ranked_lists.shape[1] - np.arange(ranked_lists.shape[1])[None, :]

    # some of the metrics fail if all the labels are ones, and other methods could cheat, so flip half at random
    idxs_to_flip = np.random.choice(len(pairidxs_test), int(0.5 * len(pairidxs_test)), replace=False)
    tmp = i1_test[idxs_to_flip]
    i1_test[idxs_to_flip] = i2_test[idxs_to_flip]
    i2_test[idxs_to_flip] = tmp
    prefs_test[idxs_to_flip] = 1 - prefs_test[idxs_to_flip]

    i1_unseen = items1[pairidxs_unseen]
    i2_unseen = items2[pairidxs_unseen]
    u_unseen = userids[pairidxs_unseen]
    prefs_unseen = prefs[pairidxs_unseen]

    if len(pairidxs_unseen):
        idxs_to_flip = np.random.choice(len(pairidxs_unseen), int(0.5 * len(pairidxs_unseen)), replace=False)
        tmp = i1_unseen[idxs_to_flip]
        i1_unseen[idxs_to_flip] = i2_unseen[idxs_to_flip]
        i2_unseen[idxs_to_flip] = tmp
        prefs_unseen[idxs_to_flip] = 1 - prefs_unseen[idxs_to_flip]

    return u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, scores, chosen_users, u_unseen, \
           i1_unseen, i2_unseen, prefs_unseen, scores_unseen, chosen_users_unseen


def run_sushi_expt(methods, expt_name, test_to_run):
    # predictions from all reps and methods
    fpred_all = []
    rho_pred_all = []

    # metrics from all reps and methods
    acc_all = []
    logloss_all = []
    tau_all = []
    times_all = []

    acc_unseen_all = []
    logloss_unseen_all = []
    tau_unseen_all = []

    # for repeatability
    np.random.seed(30)

    seeds = np.random.randint(1, 1e6, nreps)

    results_path = './results/' + expt_name
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for rep in range(nreps):

        np.random.seed(seeds[rep])

        # Get training and test data
        u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, scores, chosen_users, \
            u_unseen, i1_unseen, i2_unseen, prefs_unseen, scores_unseen, chosen_users_unseen \
            = subsample_data(test_to_run)
        u_tr = np.array([np.argwhere(chosen_users == u).flatten()[0] for u in u_tr])
        u_test = np.array([np.argwhere(chosen_users == u).flatten()[0] for u in u_test])
        u_unseen = np.array([np.argwhere(chosen_users_unseen == u).flatten()[0] for u in u_unseen])

        print(u_tr)
        print(i1_tr)
        print(i2_tr)
        print(prefs_tr)

        fpred_r = []
        rho_pred_r = []

        acc_r = []
        logloss_r = []
        tau_r = []
        times_r = []
        acc_unseen_r = []
        logloss_unseen_r = []
        tau_unseen_r = []

        for m in methods:
            # Train and Predict
            logging.info("Starting test with method %s..." % (m))
            starttime = time.time()

            fpred, rho_pred, fpred_unseen, rho_pred_unseen = train_test(m, u_tr, i1_tr, i2_tr, item_features,
                                        user_features[chosen_users], prefs_tr,
                                        u_test, i1_test, i2_test, u_unseen, i1_unseen, i2_unseen,
                                        user_features[chosen_users_unseen])

            endtime = time.time()
            times_r.append(endtime - starttime)

            # Save predictions
            fpred_r.append(fpred.flatten())
            rho_pred_r.append(rho_pred.flatten())

            # Compute metrics
            acc_m = accuracy_score(prefs_test, np.round(rho_pred))
            logloss_m = log_loss(prefs_test.flatten(), rho_pred.flatten())
            kendall_tau = kendalltau(scores[scores > -1].flatten(), fpred[scores > -1].flatten())[0]

            acc_un_m = accuracy_score(prefs_unseen, np.round(rho_pred_unseen)) if len(prefs_unseen) > 1 else 0.5
            logloss_un_m = log_loss(prefs_unseen.flatten(), rho_pred_unseen.flatten()) if len(prefs_unseen) > 1 else 0
            kendall_un_tau = kendalltau(scores_unseen[scores_unseen > -1].flatten(),
                                        fpred_unseen[scores_unseen > -1].flatten())[0] if np.sum(scores_unseen) > 1 else 0

            # Save metrics
            acc_r.append(acc_m)
            logloss_r.append(logloss_m)
            tau_r.append(kendall_tau)
            acc_unseen_r.append(acc_un_m)
            logloss_unseen_r.append(logloss_un_m)
            tau_unseen_r.append(kendall_un_tau)

            print('Results for %s at rep %i: acc=%.2f, CEE=%.2f, tau=%.2f, unseen users: acc=%.2f, CEE=%.2f, tau=%.2f'
                  % (m, rep, acc_m, logloss_m, kendall_tau, acc_un_m, logloss_un_m, kendall_un_tau))

        fpred_all.append(fpred_r)
        rho_pred_all.append(rho_pred_r)

        acc_all.append(acc_r)
        logloss_all.append(logloss_r)
        tau_all.append(tau_r)
        times_all.append(times_r)

        acc_unseen_all.append(acc_unseen_r)
        logloss_unseen_all.append(logloss_unseen_r)
        tau_unseen_all.append(tau_unseen_r)

        # save predictions to file
        np.savetxt(results_path + '/fpred_rep%i.csv' % rep, fpred_r, delimiter=',', fmt='%f')
        np.savetxt(results_path + '/rho_pred_rep%i.csv' % rep, rho_pred_r, delimiter=',', fmt='%f')

        # Compute means and stds of metrics (or medians/quartiles for plots?)
        acc_mean = np.mean(np.array(acc_all), axis=0)
        logloss_mean = np.mean(np.array(logloss_all), axis=0)
        tau_mean = np.mean(np.array(tau_all), axis=0)
        times_mean = np.mean(np.array(times_all), axis=0)

        acc_un_mean = np.mean(np.array(acc_unseen_all), axis=0)
        logloss_un_mean = np.mean(np.array(logloss_unseen_all), axis=0)
        tau_un_mean = np.mean(np.array(tau_unseen_all), axis=0)

        acc_std = np.std(np.array(acc_all), axis=0)
        logloss_std = np.std(np.array(logloss_all), axis=0)
        tau_std = np.std(np.array(tau_all), axis=0)
        times_std = np.std(np.array(times_all), axis=0)

        acc_un_std = np.std(np.array(acc_unseen_all), axis=0)
        logloss_un_std = np.std(np.array(logloss_unseen_all), axis=0)
        tau_un_std = np.std(np.array(tau_unseen_all), axis=0)

        # Print means and stds of metrics in Latex format ready for copying into a table
        print('Table of results:')

        line = 'Method & Acc. & CEE & tau & runtime (s) & Acc (unseen users) & CEE & tau \\\\ \n'
        print(line)
        lines = [line]

        for m, method in enumerate(methods):
            line = method + ' & '
            line += '%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f)\\\\\n'\
                    % (acc_mean[m], acc_std[m], logloss_mean[m], logloss_std[m], tau_mean[m], tau_std[m],
                       times_mean[m], times_std[m], acc_un_mean[m], acc_un_std[m], logloss_un_mean[m], logloss_un_std[m],
                       tau_un_mean[m], tau_un_std[m])
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

    vscales = None  # don't record the v factor scale factors
    vscales_A = None
    vscales_B = None

    nreps = 25

    debug_small = False # set to true to use a small subset of data

    item_feat_file = './data/sushi3-2016/sushi3.idata'
    user_feat_file = './data/sushi3-2016/sushi3.udata'

    if test_to_run == -1 or test_to_run == 0 or test_to_run == 1 or test_to_run == 2 or test_to_run == 3:

        # Initialise output paths ----------------------------------------------------------------------------------------------

        figure_root_path = './results/sushi_factors'
        if not os.path.exists(figure_root_path):
            os.mkdir(figure_root_path)


        # Load feature data ----------------------------------------------------------------------------------------------------
        item_data = pd.read_csv(item_feat_file, sep='\t', index_col=0, header=None)
        item_features = item_data.values[:, 1:].astype(float)
        item_features = convert_discrete_to_continuous(item_features, cols_to_convert=[2])

        user_data = pd.read_csv(user_feat_file, sep='\t', index_col=0, header=None, usecols=[0,3,4,5,6,10])
        user_features = user_data.values.astype(float)
        user_features = convert_discrete_to_continuous(user_features, cols_to_convert=[1,2,3])#0, 3, 4, 6, 7])


        # Load SUSHI-A dataset -------------------------------------------------------------------------------------------------

        sushi_prefs_file = './data/sushi3-2016/sushi3a.5000.10.order'
        ranking_data = pd.read_csv(sushi_prefs_file, sep=' ', skiprows=1, header=None)

        userids, items1, items2, prefs = extract_pairs_from_ranking(ranking_data.values[:, 2:].astype(int))

        nusers = len(np.unique(userids))
        active_items, items_contiguous = np.unique(np.array([items1, items2]), return_inverse=True)
        item_features = item_features[active_items]
        items_contiguous = items_contiguous.reshape(2, len(items1))
        items1 = items_contiguous[0]
        items2 = items_contiguous[1]
        nitems = len(active_items)
        print('Found %i users, %i items, and %i pairs per user.' % (nusers, nitems, prefs.shape[0]/nusers))
        print('Item features: %i items, %i features.' % (item_features.shape[0], item_features.shape[1]))
        print('User features: %i users, %i features.'% (user_features.shape[0], user_features.shape[1]))


        # for debugging --------------------------------------------------------------------------------------------------------
        if debug_small:
            ndebug = 200
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
        max_facs = 20
        shape_s0 = 1.0
        rate_s0 = 100.0
        max_update_size = 200 # there are 20 x 100 = 2000 pairs in total. After 10 iterations, all pairs are seen.
        delay = 5
        ninducing = 25
        forgetting_rate = 0.9
        max_Kw_size = 5000

        sushiB = False
        vscales = None


    # OPTIMISE THE FUNcTION SCALE FIRST ON ONE FOLD of Sushi A, NO DEV DATA NEEDED -----------------------------------------
    if test_to_run < 4: # gest the dev data from sushi A
        np.random.seed(2309234)
        u_tr, i1_tr, i2_tr, prefs_tr, u_test, i1_test, i2_test, prefs_test, _, chosen_users, \
                _, _, _, _, _, _ = subsample_data(-1)

    if test_to_run == -1:
        print('Optimizing function scales ...')
        sushiA_small = False
        shape_s0, rate_s0 = opt_scale_crowd_GPPL(shape_s0, rate_s0, u_tr, i1_tr, i2_tr,
                                                 item_features, user_features, prefs_tr,
                                                 u_test, i1_test, i2_test, prefs_test, chosen_users)

        print('Found scale hyperparameters: %f, %f' % (shape_s0, rate_s0))
        np.savetxt('./results/' + 'scale_hypers' + tag + '.csv', [shape_s0, rate_s0], fmt='%f', delimiter=',')


    if test_to_run < 4:
        # remove the dev set users so we don't see them in testing
        not_chosen_users = np.ones(user_data.shape[0], dtype=bool)
        not_chosen_users[chosen_users] = False
        user_data = user_data[not_chosen_users]

    if test_to_run == 0:

        # SMALL data test to show benefits of user features --------------------------------------------------------------------

        # Repeat 25 times... Run each method and compute its metrics.
        methods = [
                   #'khan',
                   'crowd-GPPL',
                   # 'crowd-GPPL-noInduc',
                   # 'crowd-GPPL\\u',
                   # 'crowd-BMF',
                   # 'crowd-GPPL-FITC\\u-noConsensus', # Like Houlsby CP (without user features)
                   # 'GPPL-pooled',
                   # 'GPPL-per-user',
                   ]

        optimize = False
        sushiB = False
        sushiA_small = True
        run_sushi_expt(methods, 'sushi_10small' + tag, test_to_run)

    if test_to_run == 1:

        # SMALL-A with OPTIMIZATION --------------------------------------------------------------------------------------------

        methods = [
                   'crowd-GPPL',
                  ]

        optimize = True
        sushiB = False
        sushiA_small = True
        run_sushi_expt(methods, 'sushi_10small' + tag, test_to_run)

        methods = [
                   'crowd-GPPL\\u',
                   ]

        optimize = True
        sushiB = False
        sushiA_small = True
        run_sushi_expt(methods, 'sushi_10small' + tag)

    if test_to_run == 2:
        # Run Test NO LENGTHSCALE OPTIMIZATION ---------------------------------------------------------------------------------

        # Repeat 25 times... Run each method and compute its metrics.
        methods = [
                   'khan',
                   #'crowd-GPPL',
                   #'crowd-GPPL-noInduc',
                   # 'crowd-GPPL\\u',
                   #'crowd-BMF',
                   #'crowd-GPPL-FITC\\u-noConsensus', # Like Houlsby CP (without user features)
                   #'GPPL-pooled',
                   #'GPPL-per-user',
                   ]

        optimize = False
        sushiB = False
        sushiA_small = False
        run_sushi_expt(methods, 'sushi_10' + tag, test_to_run)

    if test_to_run == 3:

        # OPTIMIZE ARD ---------------------------------------------------------------------------------------------------------

        vscales = []

        # Repeat 25 times... Run each method and compute its metrics.
        methods = [
                   'crowd-GPPL',
                   'crowd-GPPL\\u',
                   'crowd-GPPL-FITC\\u-noConsensus', # Houlsby -- included to show that the LB found using crowd method is more useful for optimisation
                   ]

        # hyperparameters common to most models
        optimize = True
        sushiB = False
        sushiA_small = False
        run_sushi_expt(methods, 'sushi_10_opt' + tag, test_to_run)

        vscales_A = vscales
        vscales_A = np.mean(vscales_A, axis=0)
        if vscales_A.ndim > 0:
            np.savetxt(figure_root_path + '/sushi_A_factor_scales.csv', vscales_A, delimiter=',', fmt='%f')

        vscales = None


    if test_to_run == 4 or test_to_run == 5:
        # Reload the full sets of features (previously, we subsampled on the relevant items for Sushi-A ------------------------

        item_data = pd.read_csv(item_feat_file, sep='\t', index_col=0, header=None)
        item_features = item_data.values[:, 1:].astype(float)
        item_features = convert_discrete_to_continuous(item_features, cols_to_convert=[2])

        user_data = pd.read_csv(user_feat_file, sep='\t', index_col=0, header=None)
        user_features = user_data.values.astype(float)
        user_features = convert_discrete_to_continuous(user_features, cols_to_convert=[0, 3, 4, 6, 7])

        # Load SUSHI-B dataset -------------------------------------------------------------------------------------------------

        sushi_prefs_file = './data/sushi3-2016/sushi3b.5000.10.order'
        ranking_data = pd.read_csv(sushi_prefs_file, sep=' ', skiprows=1, header=None)

        userids, items1, items2, prefs = extract_pairs_from_ranking(ranking_data.values[:, 2:].astype(int))

        nusers = len(np.unique(userids))
        active_items, items_contiguous = np.unique(np.array([items1, items2]), return_inverse=True)
        item_features = item_features[active_items]
        items_contiguous = items_contiguous.reshape(2, len(items1))
        items1 = items_contiguous[0]
        items2 = items_contiguous[1]
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

        # SUSHI B, global parameters ------------------------------------------------------------------------------------

        max_facs = 20
        shape_s0 = 1.0
        rate_s0 = 100.0  #0.1
        forgetting_rate = 0.9
        max_Kw_size = 5000
        max_update_size = 2000
        delay = 5
        ninducing = 500 # allow us to handle more users.

    if test_to_run == 4:
        # SUSHI B dataset, no opt. ---------------------------------------------------------------------------------------------

        # Repeat 25 times... Run each method and compute its metrics.
        methods = [
                   'khan',
                   #'crowd-GPPL',
                   # 'crowd-GPPL\\u',
                   #'crowd-BMF',
                   #'crowd-GPPL-FITC\\u-noConsensus', # Like Houlsby CP (without user features)
                   #'GPPL-pooled',
                   #'GPPL-per-user',
        ]

        # hyperparameters common to most models
        optimize = False
        sushiB = True
        sushiA_small = False
        run_sushi_expt(methods, 'sushi_100' + tag, test_to_run)

    if test_to_run == 5:
        # SUSHI B dataset, ARD -------------------------------------------------------------------------------------------------

        vscales = []

        # Repeat 25 times... Run each method and compute its metrics.
        methods = [
                   'crowd-GPPL',
                   'crowd-GPPL\\u',
                   'crowd-GPPL-FITC\\u-noConsensus', # Houlsby
                   ]

        # hyperparameters common to most models
        optimize = True
        sushiB = True
        sushiA_small = False
        run_sushi_expt(methods, 'sushi_100_opt' + tag, test_to_run)

        vscales_B = vscales

# Plot the latent factor scales ----------------------------------------------------------------------------------------

if vscales_A is not None or vscales_B is not None:

    logging.basicConfig(level=logging.WARNING) # matplotlib prints loads of crap to the debug and info outputs

    fig = plt.figure(figsize=(5, 4))

    markers = ['x', 'o', '+', '>', '<', '*']

    if vscales_A is not None:
        plt.plot(np.arange(vscales_A.shape[0]), vscales_A, marker=markers[0], label='Sushi A', linewidth=2, markersize=8)

    if vscales_B is not None:
        vscales_B = np.mean(vscales_B, axis=0)
        plt.plot(np.arange(vscales_B.shape[0]), vscales_B, marker=markers[1], label='Sushi B', linewidth=2, markersize=8)

    plt.ylabel('Inverse scale 1/s')
    plt.xlabel('Factor ID')

    plt.grid('on', axis='y')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(figure_root_path + '/sushi_factor_scales.pdf')

    np.savetxt(figure_root_path + '/sushi_B_factor_scales.csv', vscales_B, delimiter=',', fmt='%f')

    logging.basicConfig(level=logging.DEBUG)  # switch back to the debug output
