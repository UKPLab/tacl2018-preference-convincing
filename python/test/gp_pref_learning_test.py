'''
Simple synthetic data tests for the GP preference learning module.
Created on 3 Mar 2017

@author: simpson
'''
import os
import pickle

from scipy.stats import multivariate_normal as mvn, kendalltau, norm, bernoulli
import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals
from gp_pref_learning import GPPrefLearning
import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.metrics import f1_score, roc_auc_score

def gen_synthetic_prefs(nx=10, ny=10, N=100, P=5000, ls=[1, 40], s=100, cov_type='matern'):
    # f_prior_mean should contain the means for all the grid squares
    # P is number of pairs for training
    # s is inverse precision scale for the latent function.
    logging.info('Generating synthetic data with lengthscales: %s' % str(ls))

    if cov_type == 'matern':

        if N > nx * ny:
            N = nx * ny # can't have more locations than there are grid squares (only using discrete values here)

        # Some random feature values
        xvals = np.random.choice(nx, N, replace=True)[:, np.newaxis]
        yvals = np.random.choice(ny, N, replace=True)[:, np.newaxis]

        # remove repeated coordinates
        for coord in range(N):
            while np.sum((xvals==xvals[coord]) & (yvals==yvals[coord])) > 1:
                xvals[coord] = np.random.choice(nx, 1)
                yvals[coord] = np.random.choice(ny, 1)

        K = matern_3_2_from_raw_vals(np.concatenate((xvals.astype(float), yvals.astype(float)), axis=1), ls)
        f = mvn.rvs(cov=K/s) # zero mean. # generate the function values for the pairs
    else:
        f = np.random.normal(scale=1/s**0.5, size=(N))

    # generate pairs indices
    pair1idxs = np.random.choice(N, P, replace=True)
    pair2idxs = np.random.choice(N, P, replace=True)
    
    # remove indexes of pairs that compare the same data points -- the correct answer is trivial
    while(np.sum(pair1idxs==pair2idxs)):
        matchingidxs = pair1idxs==pair2idxs
        pair2idxs[matchingidxs] = np.random.choice(N, np.sum(matchingidxs), replace=True)
      
    # generate the discrete labels from the noisy preferences
    g_f = (f[pair1idxs] - f[pair2idxs]) / np.sqrt(2)
    phi = norm.cdf(g_f)
    prefs = bernoulli.rvs(phi)

    if np.unique(prefs).shape[0] == 1:
        idxs_to_flip = np.random.choice(len(pair1idxs), int(0.5 * len(pair1idxs)), replace=False)
        tmp = pair1idxs[idxs_to_flip]
        pair1idxs[idxs_to_flip] = pair2idxs[idxs_to_flip]
        pair2idxs[idxs_to_flip] = tmp
        prefs[idxs_to_flip] = 1 - prefs[idxs_to_flip]

    if cov_type == 'matern':
        item_features = np.concatenate((xvals, yvals), axis=1)
    else:
        item_features = None

    return prefs, item_features, pair1idxs, pair2idxs, f

def split_dataset(N, f, pair1idxs, pair2idxs, prefs):
    # test set size
    test_size = 0.2

    P = len(prefs)

    # select some data points as test only
    Ntest = int(test_size * N)
    test_points = np.random.choice(N, Ntest, replace=False)
    test_points = np.in1d(np.arange(N), test_points)
    train_points = np.invert(test_points)

    ftrain = f[train_points]
    ftest = f[test_points]

    train_pairs = train_points[pair1idxs] & train_points[pair2idxs]
    Ptrain = np.sum(train_pairs)
    pair1idxs_tr = pair1idxs[train_pairs]
    pair2idxs_tr = pair2idxs[train_pairs]
    prefs_tr = prefs[train_pairs]

    test_pairs = test_points[pair1idxs] & test_points[pair2idxs]
    Ptest = np.sum(test_pairs)
    pair1idxs_test = pair1idxs[test_pairs]
    pair2idxs_test = pair2idxs[test_pairs]
    prefs_test = prefs[test_pairs]

    # some pairs with one train and one test item will be discarded
    print("No. training pairs: %i" % Ptrain)
    print("No. test pairs: %i" % Ptest)

    return ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points, ftest, \
           pair1idxs_test, pair2idxs_test, prefs_test, test_points

def evaluate_models(model, item_features, f,
                    ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points,
                    ftest, pair1idxs_test, pair2idxs_test, test_points, personidxs_tr=None, personidxs_test=None):

    model.fit(
        pair1idxs_tr,
        pair2idxs_tr,
        item_features,
        prefs_tr,
        optimize=False,
        use_median_ls=True
    )

    # print(("Final lower bound: %f" % model.lowerbound()))

    # Predict at all locations
    fpred, vpred = model.predict_f(item_features)

    fpred_train = fpred[train_points] if item_features is not None else fpred

    if ftrain.ndim == 2:
        # Ftrain contains values for multiple functions (columns), but the predictions are only one column
        tau_obs = kendalltau(ftrain, np.tile(fpred_train, (1, ftrain.shape[1]) ))[0]
    else:
        tau_obs = kendalltau(ftrain, fpred_train)[0]
    print("Kendall's tau (observations): %.3f" % tau_obs)

    if item_features is not None:

        # Evaluate the accuracy of the predictions
        # print("RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
        # print("NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
        if ftest.ndim == 2:
            tau_test = kendalltau(ftest, np.tile(fpred[test_points], (1,ftest.shape[1]) ))[0]
        else:
            tau_test = kendalltau(ftest, fpred[test_points])[0]
        print("Kendall's tau (test): %.3f" % tau_test)

    # noise rate in the pairwise data -- how many of the training pairs conflict with the ordering suggested by f?
    if personidxs_tr is None:
        prefs_tr_noisefree = (f[pair1idxs_tr] > f[pair2idxs_tr]).astype(float)
    else:
        prefs_tr_noisefree = (f[pair1idxs_tr, personidxs_tr] > f[pair2idxs_tr, personidxs_tr]).astype(float)

    noise_rate = 1.0 - np.mean(prefs_tr == prefs_tr_noisefree)
    print('Noise rate in the pairwise training labels: %f' % noise_rate)

    if item_features is not None:

        if personidxs_test is None:
            t = (f[pair1idxs_test] > f[pair2idxs_test]).astype(int)
        else:
            t = (f[pair1idxs_test, personidxs_test] > f[pair2idxs_test, personidxs_test]).astype(int)

        if np.unique(t).shape[0] == 1:
            idxs_to_flip = np.random.choice(len(pair1idxs_test), int(0.5 * len(pair1idxs_test)), replace=False)
            tmp = pair1idxs_test[idxs_to_flip]
            pair1idxs_test[idxs_to_flip] = pair2idxs_test[idxs_to_flip]
            pair2idxs_test[idxs_to_flip] = tmp
            t[idxs_to_flip] = 1 - t[idxs_to_flip]

        rho_pred, var_rho_pred = model.predict(item_features, pair1idxs_test, pair2idxs_test)
        rho_pred = rho_pred.flatten()
        t_pred = np.round(rho_pred)

        brier = np.sqrt(np.mean((t - rho_pred) ** 2))
        print("Brier score of %.3f" % brier)
        cee = -np.sum(t * np.log(rho_pred) + (1 - t) * np.log(1 - rho_pred))
        print("Cross entropy error of %.3f" % cee)

        f1 = f1_score(t, t_pred)
        print("F1 score of %.3f" % f1)
        acc = np.mean(t == t_pred)
        print("Accuracy of %.3f" % acc)
        roc = roc_auc_score(t, rho_pred)
        print("ROC of %.3f" % roc)
    else:
        tau_test = 0.5
        brier = 1.0
        cee = 1.0
        f1 = 0
        acc = 0.5
        roc = 0.5

    return noise_rate, tau_obs, tau_test, brier, cee, f1, acc, roc

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    
        
    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)

    ls = [np.random.rand() * 20, np.random.rand() * 20]

    N = 200

    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    pklfile = './tmp/synthdata.pkl'
    if os.path.exists(pklfile):
        with open(pklfile, 'rb') as fh:
            prefs, item_features, pair1idxs, pair2idxs, f = pickle.load(fh)
    else:

        prefs, item_features, pair1idxs, pair2idxs, f = gen_synthetic_prefs(
            nx=100,
            ny=100,
            N=N,
            P=5000,
            ls=ls,
            s=0.1,
            cov_type='matern'
        )

        with open(pklfile, 'wb') as fh:
            pickle.dump((prefs, item_features, pair1idxs, pair2idxs, f), fh)

    ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points, \
    ftest, pair1idxs_test, pair2idxs_test, prefs_test, test_points = \
        split_dataset(N, f, pair1idxs, pair2idxs, prefs)

    models = {}

    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results

    # # Create a GPPrefLearning model
    model = GPPrefLearning(2, mu0=0, shape_s0=100, rate_s0=100, ls_initial=None, use_svi=True, ninducing=50,
                           max_update_size=100, forgetting_rate=0.9, verbose=True)
    models['SVI'] = model
    
    # Create a GPPrefLearning model
    model = GPPrefLearning(2, mu0=0, shape_s0=100, rate_s0=100, ls_initial=None, use_svi=False)
    model.verbose = True

    #models['VB'] = model

    model = GPPrefLearning(2, mu0=0, shape_s0=100, rate_s0=100, ls_initial=None, use_svi=True, kernel_func='diagonal')
    model.verbose = True

    #models['diag'] = model

    for modelkey in models:
        model = models[modelkey]
        print(("--- Running model %s ---" % modelkey))
        evaluate_models(model, item_features, f, ftrain, pair1idxs_tr, pair2idxs_tr, prefs_tr, train_points,
                        ftest, pair1idxs_test, pair2idxs_test, test_points)