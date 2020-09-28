'''
Test the preference_features module with some simple synthetic data test

Created on 3 Mar 2017

@author: edwin
'''
import logging
import os
import sys

from gp_pref_learning import GPPrefLearning

logging.basicConfig(level=logging.DEBUG)

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/lukin_comparison")

import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals
from scipy.stats import multivariate_normal as mvn, norm, bernoulli, kendalltau
from scipy.linalg import block_diag
from collab_pref_learning_vb import CollabPrefLearningVB
from collab_pref_learning_svi import CollabPrefLearningSVI
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_models_personal(model, item_features, person_features, F,
                             pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points,
                             pair1idxs_test, pair2idxs_test, personidxs_test, test_points):
    '''
    Test performance in predicting the ground truth or common mean preference function
    from multi-user labels.
    '''

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

    #print(("Final lower bound: %f" % model.lowerbound()))

    # Predict at all locations
    Fpred = model.predict_f(item_features, person_features)

    tau_obs = kendalltau(F[train_points], Fpred[train_points])[0]
    print("Kendall's tau (observations): %.3f" % tau_obs)

    # Evaluate the accuracy of the predictions
    # print("RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
    # print("NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
    tau_test = kendalltau(F[test_points], Fpred[test_points])[0]
    print("Kendall's tau (test): %.3f" % tau_test)

    # noise rate in the pairwise data -- how many of the training pairs conflict with the ordering suggested by f?
    prefs_tr_noisefree = (F[pair1idxs_tr, personidxs_tr] > F[pair2idxs_tr, personidxs_tr]).astype(float)
    noise_rate = 1.0 - np.mean(prefs_tr == prefs_tr_noisefree)
    print('Noise rate in the pairwise training labels: %f' % noise_rate)

    t = (F[pair1idxs_test, personidxs_test] > F[pair2idxs_test, personidxs_test]).astype(int)

    if np.unique(t).shape[0] == 1:
        idxs_to_flip = np.random.choice(len(pair1idxs_test), int(0.5 * len(pair1idxs_test)), replace=False)
        tmp = pair1idxs_test[idxs_to_flip]
        pair1idxs_test[idxs_to_flip] = pair2idxs_test[idxs_to_flip]
        pair2idxs_test[idxs_to_flip] = tmp
        t[idxs_to_flip] = 1 - t[idxs_to_flip]

    rho_pred = model.predict(personidxs_test, pair1idxs_test, pair2idxs_test, item_features, person_features)
    rho_pred = rho_pred.flatten()
    t_pred = np.round(rho_pred)

    brier = np.sqrt(np.mean((t - rho_pred) ** 2))
    print("Brier score of %.3f" % brier)
    rho_pred[rho_pred < 1e-5] = 1e-5
    rho_pred[rho_pred > 1-1e-5] = 1-1e-5
    cee = -np.mean(t * np.log(rho_pred) + (1 - t) * np.log(1 - rho_pred))
    print("Cross entropy error of %.3f" % cee)

    f1 = f1_score(t, t_pred)
    print("F1 score of %.3f" % f1)
    acc = np.mean(t == t_pred)
    print("Accuracy of %.3f" % acc)
    roc = roc_auc_score(t, rho_pred)
    print("ROC of %.3f" % roc)

    return noise_rate, tau_obs, tau_test, brier, cee, f1, acc, roc

def evaluate_models_common_mean(model, item_features, person_features, f,
                    pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points,
                    pair1idxs_test, pair2idxs_test, test_points):
    '''
    Test performance in predicting the ground truth or common mean preference function
    from multi-user labels.
    '''

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

    #print(("Final lower bound: %f" % model.lowerbound()))

    # Predict at all locations
    fpred = model.predict_t(item_features)

    tau_obs = kendalltau(f[train_points], fpred[train_points])[0]
    print("Kendall's tau (observations): %.3f" % tau_obs)

    # Evaluate the accuracy of the predictions
    # print("RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
    # print("NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
    tau_test = kendalltau(f[test_points], fpred[test_points])[0]
    print("Kendall's tau (test): %.3f" % tau_test)

    # noise rate in the pairwise data -- how many of the training pairs conflict with the ordering suggested by f?
    prefs_tr_noisefree = (f[pair1idxs_tr] > f[pair2idxs_tr]).astype(float)
    noise_rate = 1.0 - np.mean(prefs_tr == prefs_tr_noisefree)
    print('Noise rate in the pairwise training labels: %f' % noise_rate)

    t = (f[pair1idxs_test] > f[pair2idxs_test]).astype(int)
    if np.unique(t).shape[0] == 1:
        idxs_to_flip = np.random.choice(len(pair1idxs_test), int(0.5 * len(pair1idxs_test)), replace=False)
        tmp = pair1idxs_test[idxs_to_flip]
        pair1idxs_test[idxs_to_flip] = pair2idxs_test[idxs_to_flip]
        pair2idxs_test[idxs_to_flip] = tmp
        t[idxs_to_flip] = 1 - t[idxs_to_flip]

    rho_pred = model.predict_common(item_features, pair1idxs_test, pair2idxs_test)
    rho_pred = rho_pred.flatten()
    t_pred = np.round(rho_pred)

    brier = np.sqrt(np.mean((t - rho_pred) ** 2))
    print("Brier score of %.3f" % brier)
    rho_pred[rho_pred < 1e-5] = 1e-5
    rho_pred[rho_pred > 1-1e-5] = 1-1e-5
    cee = -np.mean(t * np.log(rho_pred) + (1 - t) * np.log(1 - rho_pred))
    print("Cross entropy error of %.3f" % cee)

    f1 = f1_score(t, t_pred)
    print("F1 score of %.3f" % f1)
    acc = np.mean(t == t_pred)
    print("Accuracy of %.3f" % acc)
    roc = roc_auc_score(t, rho_pred)
    print("ROC of %.3f" % roc)

    return noise_rate, tau_obs, tau_test, brier, cee, f1, acc, roc


def split_dataset(N, F, pair1idxs, pair2idxs, personidxs, prefs):
    # test set size
    test_size = 0.5

    # select some data points as test only
    Ntest = int(np.ceil(test_size * N))
    if Ntest < 2: Ntest = 2 # need to have at least one pair!

    test_points = np.random.choice(N, Ntest, replace=False)
    test_points = np.in1d(np.arange(N), test_points)
    train_points = np.invert(test_points)

    Ftrain = F[train_points]
    Ftest = F[test_points]

    train_pairs = train_points[pair1idxs] & train_points[pair2idxs]
    Ptrain = np.sum(train_pairs)
    pair1idxs_tr = pair1idxs[train_pairs]
    pair2idxs_tr = pair2idxs[train_pairs]
    prefs_tr = prefs[train_pairs]
    personidxs_tr = personidxs[train_pairs]

    test_pairs = test_points[pair1idxs] & test_points[pair2idxs]
    Ptest = np.sum(test_pairs)
    pair1idxs_test = pair1idxs[test_pairs]
    pair2idxs_test = pair2idxs[test_pairs]
    prefs_test = prefs[test_pairs]
    personidxs_test = personidxs[test_pairs]

    # some pairs with one train and one test item will be discarded
    print("No. training pairs: %i" % Ptrain)
    print("No. test pairs: %i" % Ptest)

    return Ftrain, pair1idxs_tr, pair2idxs_tr, personidxs_tr, prefs_tr, train_points, Ftest, \
           pair1idxs_test, pair2idxs_test, personidxs_test, prefs_test, test_points


def gen_synthetic_personal_prefs(Nfactors, nx, ny, N, Npeople, P, ls, sigma, s, lsy, Npeoplefeatures=4):
    if N > nx * ny:
        N = nx * ny  # can't have more locations than there are grid squares (only using discrete values here)

    # Some random feature values
    xvals = np.random.choice(nx, N, replace=True)[:, np.newaxis]
    yvals = np.random.choice(ny, N, replace=True)[:, np.newaxis]

    # remove repeated coordinates
    for coord in range(N):

        while np.sum((xvals == xvals[coord]) & (yvals == yvals[coord])) > 1:
            xvals[coord] = np.random.choice(nx, 1)
            yvals[coord] = np.random.choice(ny, 1)

    Kt = matern_3_2_from_raw_vals(np.concatenate((xvals.astype(float), yvals.astype(float)), axis=1), ls)
    t = mvn.rvs(cov=Kt/sigma).reshape(nx * ny, 1)

    # Kw = [Kt for _ in range(Nfactors)]
    # Kw = block_diag(*Kw)
    # w = mvn.rvs(cov=Kw/s).reshape(Nfactors, nx * ny).T
    w = np.empty((nx*ny, Nfactors))
    for f in range(Nfactors):
        if np.isscalar(s):
            w[:, f] = mvn.rvs(cov=Kt/s)
        else:
            w[:, f] = mvn.rvs(cov=Kt / s[f])

    # person_features = None
    person_features = np.zeros((Npeople, Npeoplefeatures))
    for i in range(Npeoplefeatures):
        person_features[:, i] = np.random.choice(10, Npeople, replace=True)

    person_features += np.random.rand(Npeople, Npeoplefeatures) * 0.01

    Ky = matern_3_2_from_raw_vals(person_features, lsy)
    # Ky = [Ky for _ in range(Nfactors)]
    # Ky = block_diag(*Ky)
    # y = mvn.rvs(cov=Ky).reshape(Nfactors, Npeople)
    y = np.empty((Nfactors, Npeople))
    for f in range(Nfactors):
       y[f] = mvn.rvs(cov=Ky)

    f_all = w.dot(y) + t

    # divide P between people
    personidxs = np.random.choice(Npeople, P, replace=True)

    # generate pairs indices
    pair1idxs = np.random.choice(N, P, replace=True)
    pair2idxs = np.random.choice(N, P, replace=True)

    # remove indexes of pairs that compare the same data points -- the correct answer is trivial
    while(np.sum(pair1idxs==pair2idxs)):
        matchingidxs = pair1idxs==pair2idxs
        pair2idxs[matchingidxs] = np.random.choice(N, np.sum(matchingidxs), replace=True)

    # generate the discrete labels from the noisy preferences
    g_f = (f_all[pair1idxs, personidxs] - f_all[pair2idxs, personidxs]) / np.sqrt(2)
    phi = norm.cdf(g_f)
    prefs = bernoulli.rvs(phi)

    item_features = np.concatenate((xvals, yvals), axis=1)

    return prefs, item_features, person_features, pair1idxs, pair2idxs, personidxs, f_all, w, t.flatten(), y


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    

    fix_seeds = True
    do_profiling = False
    
    if do_profiling:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(11)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    
    if 'item_features' not in globals():
        #         Npeople = 20
        #         N = 25
        #         P = 100 # pairs per person in test+training set
        #         nx = 5
        #         ny = 5

        Npeople = 8
        N = 16
        P = 5000
        nx = 4
        ny = 4

        Npeoplefeatures = 3
        ls = [10, 5]
        s = 0.1
        sigma = 0.1
        lsy = 2 + np.zeros(Npeoplefeatures)
        Nfactors = 2

        prefs, item_features, person_features, pair1idxs, pair2idxs, personids, latent_f, w, t, y = \
            gen_synthetic_personal_prefs(Nfactors, nx, ny, N, Npeople, P, ls, sigma, s, lsy, Npeoplefeatures)

        # return t as a grid
        t = t.reshape(nx, ny)

        Ptest_percent = 0.2
        Ptest = int(Ptest_percent * pair1idxs.size)
        testpairs = np.random.choice(pair1idxs.shape[0], Ptest, replace=False)
        testidxs = np.zeros(pair1idxs.shape[0], dtype=bool)
        testidxs[testpairs] = True
        trainidxs = np.invert(testidxs)
    
    # if fix_seeds:
    #     np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    # Model initialisation --------------------------------------------------------------------------------------------
    if len(sys.argv) > 1:
        use_svi = sys.argv[1] == 'svi'
    else:
        use_svi = True
    use_t = True
    use_person_features = True
    optimize = False

    ls_initial = np.array(ls)# + np.random.rand(len(ls)) * 10)
    print(("Initial guess of length scale for items: %s, true length scale is %s" % (ls_initial, ls)))
    lsy_initial = np.array(lsy)# + np.random.rand(len(lsy)) * 10)# + 7
    print(("Initial guess of length scale for people: %s, true length scale is %s" % (lsy_initial, lsy)))
    if use_svi:
        model = CollabPrefLearningSVI(2, Npeoplefeatures if use_person_features else 0, ls=ls_initial,
                                      lsy=lsy_initial, use_common_mean_t=use_t,
                                      nfactors=5, ninducing=7, max_update_size=200, delay=25,
                                      shape_s0=1, rate_s0=1, use_lb=True)
    else:
        model = CollabPrefLearningVB(2, Npeoplefeatures if use_person_features else 0, ls=ls_initial, lsy=lsy_initial,
                                     use_common_mean_t=use_t, nfactors=5, use_lb=True)

    if fix_seeds:
        np.random.seed(22)

    model.verbose = True
    model.fit(personids[trainidxs], pair1idxs[trainidxs], pair2idxs[trainidxs], item_features, prefs[trainidxs],
              person_features if use_person_features else None, optimize=optimize)


    print(("Difference between true item length scale and inferred item length scale = %s" % (ls - model.ls)))
    print(("Difference between true person length scale and inferred person length scale = %s" % (lsy - model.lsy)))
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1idxs[testidxs], pair2idxs[testidxs], item_features,
                            person_features if use_person_features else None)
    
    # make the test more difficult: we predict for a person we haven't seen before who has same features as another
    result_new_person = model.predict(
        [np.max(personids) + 1], pair1idxs[testidxs][0:1], pair2idxs[testidxs][0:1],
        item_features,
        np.concatenate((person_features, person_features[personids[0:1], :]), axis=0) if use_person_features
        else None)
    print("Test using new person: %.3f" % result_new_person)
    print("Old prediction: %.3f" % results[0])

    print("Testing prediction of new + old people")
    result_new_old_person = model.predict(
        np.concatenate((personids[testidxs], [np.max(personids) + 1])),
        np.concatenate((pair1idxs[testidxs], pair1idxs[testidxs][0:1])),
        np.concatenate((pair2idxs[testidxs], pair2idxs[testidxs][0:1])),
        item_features,
        np.concatenate((person_features, person_features[personids[0:1], :]), axis=0) if use_person_features
        else None)
    print("Test using new person while predicting old people: %.3f" % result_new_old_person[-1])
    #print("Result is correct = " + str(np.abs(results[0] - result_new_person) < 1e-6)

    if do_profiling:
        pr.disable()
        import datetime
        pr.dump_stats('preference_features_test_svi_%i_%s.profile' % (use_svi, datetime.datetime.now()))
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print((s.getvalue()))

    # Single User model for comparison

    singleusermodel = GPPrefLearning(2, shape_s0=1, rate_s0=1, ls_initial=ls_initial, forgetting_rate=0.7, ninducing=7,
                                     max_update_size=200, delay=25, verbose=True)

    singleusermodel.fit(pair1idxs[trainidxs], pair2idxs[trainidxs], item_features, prefs[trainidxs])

    p_pred_su = singleusermodel.predict(item_features, pair1idxs[testidxs], pair2idxs[testidxs], return_var=False)
    p_pred_round_su = np.round(p_pred_su).astype(int)
    p_pred_su[p_pred_su > (1-1e-6)] = 1 - 1e-6
    p_pred_su[p_pred_su < 1e-6] = 1e-6

    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    from sklearn.metrics import accuracy_score
    
    p_pred = results
    p_pred_round = np.round(results).astype(int)
    p = prefs[testidxs]
       
    print(" --- Preference prediction metrics --- " )
    p_pred[p_pred > (1-1e-6)] = 1 - 1e-6
    p_pred[p_pred < 1e-6] = 1e-6
    print(("Cross entropy error of %.3f" % -np.mean(p * np.log(p_pred.flatten()) + (1-p) * np.log(1 - p_pred.flatten()))))

    from sklearn.metrics import f1_score, roc_auc_score
    print(("F1 score of %.3f" % f1_score(p, p_pred_round)))
    print(('Accuracy: %f' % accuracy_score(p, p_pred_round)))
    print(("ROC of %.3f" % roc_auc_score(p, p_pred)))

    print('Single user model for comparison:')

    print(("Cross entropy error of %.3f" % -np.mean(p * np.log(p_pred_su.flatten()) + (1-p) * np.log(1 - p_pred_su.flatten()))))
    print(("F1 score of %.3f" % f1_score(p, p_pred_round_su)))
    print(('Accuracy: %f' % accuracy_score(p, p_pred_round_su)))
    print(("ROC of %.3f" % roc_auc_score(p, p_pred_su)))

    print(" --- Latent item feature prediction metrics --- " )

    # get the w values that correspond to the coords seen by the model
    widxs = np.ravel_multi_index((
           model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)), dims=(nx, ny))
    w = w[widxs, :]

    # how can we handle the permutations of the features?
    #scipy.factorial(model.Nfactors) / scipy.factorial(model.Nfactors - w.shape[1])
    # remove the features from the model with least variation -- these are the dead features
    wvar = np.var(model.w, axis=0)
    chosen_features = np.argsort(wvar)[-w.shape[1]:]
    w_pred = model.w[:, chosen_features].T.reshape(w.shape[1] * N)

    # w_pred_cov = model.w_cov.reshape(model.Nfactors * N, model.Nfactors, N)
    # w_pred_cov = np.swapaxes(w_pred_cov, 0, 2).reshape(N, model.Nfactors, model.Nfactors, N)
    # w_pred_cov = w_pred_cov[:, chosen_features, :, :][:, :, chosen_features, :]
    # w_pred_cov = w_pred_cov.reshape(N, w.shape[1], w.shape[1] * N)
    # w_pred_cov = np.swapaxes(w_pred_cov, 0, 2).reshape(w.shape[1] * N, w.shape[1] * N)
    #
    # print("w: RMSE of %.3f" % np.sqrt(np.mean((w.T.reshape(N * w.shape[1])-w_pred)**2)))
    # print("w: NLPD error of %.3f" % -mvn.logpdf(w.T.reshape(N * w.shape[1]), mean=w_pred, cov=w_pred_cov))
    #
    # print(" --- Latent person feature prediction metrics --- ")
    #
    # yvar = np.var(model.y, axis=1)
    # chosen_features = np.argsort(yvar)[-y.shape[0]:]
    # y_pred = model.y[chosen_features, :].reshape(y.shape[0] * Npeople)
    #
    # y_pred_cov = model.y_cov.reshape(model.Nfactors * Npeople, model.Nfactors, Npeople)
    # y_pred_cov = np.swapaxes(y_pred_cov, 0, 2).reshape(Npeople, model.Nfactors, model.Nfactors, Npeople)
    # y_pred_cov = y_pred_cov[:, chosen_features, :, :][:, :, chosen_features, :]
    # y_pred_cov = y_pred_cov.reshape(Npeople, w.shape[1], w.shape[1] * Npeople)
    # y_pred_cov = np.swapaxes(y_pred_cov, 0, 2).reshape(w.shape[1] * Npeople, w.shape[1] * Npeople)
    #
    # print("y: RMSE of %.3f" % np.sqrt(np.mean((y.reshape(Npeople * w.shape[1])-y_pred)**2)))
    # print("y: NLPD error of %.3f" % -mvn.logpdf(y.reshape(Npeople * w.shape[1]), mean=y_pred, cov=y_pred_cov))

#     from scipy.stats import kendalltau
#
#     for p in range(Npeople):
#         logging.debug( "Personality features of %i: %s" % (p, str(model.w[p])) )
#         for q in range(Npeople):
#             logging.debug( "Distance between personalities: %f" % np.sqrt(np.sum(model.w[p] - model.w[q])**2)**0.5 )
#             logging.debug( "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0] )

    # visualise the results
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('jet')
    cmap._init()

    # t
    fig = plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.t.flatten()
    scale = np.std(tmap)
    if scale == 0:
        scale = 1
    tmap /= scale
    ax = plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=np.min(tmap), vmax=np.max(tmap), interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: t (item mean)')
    fig.colorbar(ax)

#     plt.figure()
#     tmap = np.zeros((nx, ny))
#     tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(np.diag(model.t_cov))
#     scale = np.std(tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#     plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
#                    vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
#     plt.title('STD at training points: t (item mean)')

    fig = plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = t[model.obs_coords[:, 0].astype(int),
                                                                      model.obs_coords[:, 1].astype(int)].flatten()
    scale = np.std(tmap)
    if scale == 0:
        scale = 1
    tmap /= scale
    ax = plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', vmin=np.min(tmap), vmax=np.max(tmap), interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: t (item mean)')
    fig.colorbar(ax)

    # y
    fig = plt.figure()
    ymap = model.y.T
    scale = np.std(ymap)
    if scale == 0:
        scale = 1.0
    #scale = np.sqrt(model.rate_sy[np.newaxis, :]/model.shape_sy[np.newaxis, :])
    ymap /= scale
    ax = plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]],
               aspect=Nfactors / float(ymap.shape[0]), vmin=np.min(ymap), vmax=np.max(ymap), interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: y (latent features for people)')
    fig.colorbar(ax)

    fig = plt.figure()
    ymap = y.T
    scale = np.std(ymap)
    if scale == 0:
        scale = 1.0
    ymap /= scale
    ax = plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]],
               aspect=Nfactors / float(ymap.shape[0]), vmin=np.min(ymap), vmax=np.max(ymap), interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: y (latent features for people')
    fig.colorbar(ax)

    # w
    scale = np.std(model.w)
    if scale == 0:
        scale = 1.0
    model.w /= scale
    wmin = np.min(model.w)
    wmax = np.max(model.w)

    for f in range(model.Nfactors):
        fig = plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.w[:, f]
        ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=wmin, vmax=wmax, interpolation='none', filterrad=0.01)
        plt.title('predictions at training points: w_%i (latent feature for items)' %f)
        fig.colorbar(ax)

#         fig = plt.figure()
#         wmap = np.zeros((nx, ny))
#         wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(model.w_cov[np.arange(model.N*f, model.N*(f+1)),
#                                                                                    np.arange(model.N*f, model.N*(f+1))])
#         scale = np.std(wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#         wmap /= scale
#         ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]], aspect=None, vmin=-2,
#                    vmax=2, interpolation='none', filterrad=0.01)
#         plt.title('STD at training points: w_%i (latent feature for items)' %f)
#         fig.colorbar(ax)
    scale = np.std(w)
    if scale == 0:
        scale = 1.0
    w /= scale
    wmin = np.min(w)
    wmax = np.max(w)

    for f in range(Nfactors):
        fig = plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = w[:, f]

        ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=wmin, vmax=wmax, interpolation='none', filterrad=0.01)
        plt.title('ground truth at training points: w_%i (latent feature for items' % f)
        fig.colorbar(ax)

    plt.show()