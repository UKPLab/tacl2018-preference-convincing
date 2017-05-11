'''
Test the preference_features module with some simple synthetic data test

Created on 3 Mar 2017

@author: edwin
'''
import logging
import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals, coord_arr_to_1d
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import block_diag
from gp_pref_learning_test import gen_synthetic_prefs
from preference_features import PreferenceComponents

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    

    fix_seeds = True
    do_profiling = False
    
    if do_profiling:
        import cProfile, pstats, StringIO
        pr = cProfile.Profile()
        pr.enable()

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(10)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    
    if 'xvals' not in globals():
        Npeople = 20
        N = 25
        P = 100 # pairs per person in test+training set
        nx = 5
        ny = 5
    # 
    #     Npeople = 200
    #     N = 100
    #     P = 100
    #     nx = 25
    #     ny = 25
    #     
        
        Ptest_percent = 0.2
        pair1idxs = []
        pair2idxs = []
        prefs = []
        personids = []
        xvals = []
        yvals = []
        
        # generate a common prior:
        ls = [10, 5]
        xvals = np.tile(np.arange(nx)[:, np.newaxis], (1, ny)).flatten().astype(float)
        yvals = np.tile(np.arange(ny)[np.newaxis, :], (nx, 1)).flatten().astype(float)
        Kt = matern_3_2_from_raw_vals(np.concatenate((xvals[:, np.newaxis], yvals[:, np.newaxis]), axis=1), ls)
        t = mvn.rvs(cov=Kt).reshape(nx, ny)
        
        Nfactors = 3
        
        Kw = [Kt for _ in range(Nfactors)]
        Kw = block_diag(*Kw)
        w = mvn.rvs(cov=Kw).reshape(Nfactors, nx*ny).T
        
        Npeoplefeatures = 4
        lsy = 2 + np.zeros(Npeoplefeatures)
        #person_features = None
        person_features = np.zeros((Npeoplefeatures, Npeople)) 
        for f in range(Npeoplefeatures):
            person_features[f, :Npeople/2] = -0.2
            person_features[f, Npeople/2:] = 0.2
            person_features[f, :] += np.arange(Npeople)
        
        Ky = matern_3_2_from_raw_vals(person_features.T, lsy)
        Ky = [Ky for _ in range(Nfactors)]
        Ky = block_diag(*Ky)
        y = mvn.rvs(cov=Ky).reshape(Nfactors, Npeople)
        
        xvals = []
        yvals = []
        for p in range(Npeople):
            
            y_p = y[:, p:p+1]
            wy_p = w.dot(y_p).reshape((nx, ny))
            
            f_prior_mean = t + wy_p
            
            _, nx, ny, prefs_p, xvals_p, yvals_p, pair1idxs_p, pair2idxs_p, f, K = gen_synthetic_prefs(f_prior_mean, 
                                                                                    nx, ny, N, P, s=0.0001, ls=ls)
            pair1idxs = np.concatenate((pair1idxs, pair1idxs_p + len(xvals))).astype(int)
            pair2idxs = np.concatenate((pair2idxs, pair2idxs_p + len(yvals))).astype(int)
            prefs = np.concatenate((prefs, prefs_p)).astype(int)
            personids = np.concatenate((personids, np.zeros(len(pair1idxs_p)) + p)).astype(int)
            xvals = np.concatenate((xvals, xvals_p.flatten()))
            yvals = np.concatenate((yvals, yvals_p.flatten()))

        xvals = xvals[:, np.newaxis]
        yvals = yvals[:, np.newaxis]
        item_features = np.concatenate((xvals, yvals), axis=1)
        _, uidxs, inverseidxs = np.unique(coord_arr_to_1d(item_features), return_index=True, return_inverse=True)
        item_features = item_features[uidxs]    
        pair1idxs = inverseidxs[pair1idxs]
        pair2idxs = inverseidxs[pair2idxs]
        Ptest = int(Ptest_percent * pair1idxs.size)
    
        testpairs = np.random.choice(pair1idxs.shape[0], Ptest, replace=False)
        testidxs = np.zeros(pair1idxs.shape[0], dtype=bool)
        testidxs[testpairs] = True
        trainidxs = np.invert(testidxs)
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    # Model initialisation --------------------------------------------------------------------------------------------
    use_svi = True
    ls_initial = np.array(ls) + 5
    print "Initial guess of length scale for items: %s, true length scale is %s" % (ls_initial, ls)
    lsy_initial = np.array(lsy) + 7
    print "Initial guess of length scale for people: %s, true length scale is %s" % (lsy_initial, lsy)
    model = PreferenceComponents(2, Npeoplefeatures, ls=ls_initial, lsy=lsy_initial, nfactors=Nfactors + 5, use_fa=False, 
                                 use_svi=use_svi, delay=1, forgetting_rate=0.9, ninducing=500, max_update_size=5)
    model.verbose = False
    model.min_iter = 1
    model.max_iter = 100
    model.fit(personids[trainidxs], pair1idxs[trainidxs], pair2idxs[trainidxs], item_features, prefs[trainidxs], 
              person_features.T, optimize=True)
#               None, optimize=True)    
    print "Difference between true item length scale and inferred item length scale = %s" % (ls - model.ls)
    print "Difference between true person length scale and inferred person length scale = %s" % (lsy - model.lsy)
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1idxs[testidxs], pair2idxs[testidxs], item_features)
    
    # make the test more difficult: we predict for a person we haven't seen before who has same features as another
    result_new_person = model.predict([np.max(personids) + 1], pair1idxs[testidxs][0:1], pair2idxs[testidxs][0:1], 
                      item_features, np.concatenate((person_features.T, person_features[:, personids[0:1]].T), axis=0))
    print "Test using new person: %.3f" % result_new_person
    print "Old prediction: %.3f" % results[0]
    #print "Result is correct = " + str(np.abs(results[0] - result_new_person) < 1e-6) 
    
    if do_profiling:
        pr.disable()
        import datetime
        pr.dump_stats('preference_features_test_svi_%i_%s.profile' % (use_svi, datetime.datetime.now()))
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
    
    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    from sklearn.metrics import accuracy_score
    
    p_pred = results
    p_pred_round = np.round(results).astype(int)
    p = prefs[testidxs]
       
    print " --- Preference prediction metrics --- " 
    print "Brier score of %.3f" % np.sqrt(np.mean((p-p_pred)**2))
    p_pred[p_pred > (1-1e-6)] = 1 - 1e-6
    p_pred[p_pred < 1e-6] = 1e-6
    print "Cross entropy error of %.3f" % -np.mean(p * np.log(p_pred) + (1-p) * np.log(1 - p_pred))
            
    from sklearn.metrics import f1_score, roc_auc_score
    print "F1 score of %.3f" % f1_score(p, p_pred_round)
    print 'Accuracy: %f' % accuracy_score(p, p_pred_round)
    print "ROC of %.3f" % roc_auc_score(p, p_pred)

    print " --- Latent item feature prediction metrics --- " 
    
    # get the w values that correspond to the coords seen by the model
    widxs = np.ravel_multi_index((
           model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)), dims=(nx, ny))
    w = w[widxs, :]
    
    # how can we handle the permutations of the features?
    #scipy.factorial(model.Nfactors) / scipy.factorial(model.Nfactors - w.shape[1])
    # remove the features from the model with least variation -- these are the dead features
    wvar = np.var(model.w, axis=0)
    chosen_features = np.argsort(wvar)[-w.shape[1]:]
    w_pred = model.w[:, chosen_features]
    w_var = np.diag(model.w_cov)[(np.arange(w.shape[0])[np.newaxis, :]*chosen_features[:, np.newaxis]).flatten()].reshape(
                                                                                         w.shape[1], w.shape[0]).T
    
    print "RMSE of %.3f" % np.sqrt(np.mean((w-w_pred)**2))
    from scipy.stats import norm
    print "NLPD error of %.3f" % -np.mean(norm.logpdf(w, loc=w_pred, scale=np.sqrt(w_var)))
            
    print " --- Latent person feature prediction metrics --- " 

    yvar = np.var(model.y, axis=1)
    chosen_features = np.argsort(yvar)[-y.shape[0]:]
    y_pred = model.y[chosen_features, :].T
    y_var = np.diag(model.y_cov)[(np.arange(y.shape[1])[np.newaxis, :]*chosen_features[:, np.newaxis]).flatten()].reshape(
                                                                                             y.shape[0], y.shape[1]).T

    print "RMSE of %.3f" % np.sqrt(np.mean((y.T-y_pred)**2))
    print "NLPD error of %.3f" % -np.mean(norm.logpdf(y.T, loc=y_pred, scale=np.sqrt(y_var)))   
            
#     from scipy.stats import kendalltau
#      
#     for p in range(Npeople):
#         logging.debug( "Personality features of %i: %s" % (p, str(model.w[p])) )
#         for q in range(Npeople):
#             logging.debug( "Distance between personalities: %f" % np.sqrt(np.sum(model.w[p] - model.w[q])**2)**0.5 )
#             logging.debug( "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0] )
#              
    
    # visualise the results
    import matplotlib.pyplot as plt
                
    cmap = plt.get_cmap('jet')                
    cmap._init()    
    
    # t
    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.t.flatten()
    scale = np.sqrt(model.rate_st/model.shape_st)
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: t (item mean)')

#     plt.figure()
#     tmap = np.zeros((nx, ny))
#     tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(np.diag(model.t_cov))
#     scale = np.std(tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#     plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
#                    vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
#     plt.title('STD at training points: t (item mean)')

    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = t[model.obs_coords[:, 0].astype(int),
                                                                      model.obs_coords[:, 1].astype(int)].flatten()
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: t (item mean)')    
    
    # y
    plt.figure()
    ymap = model.y.T
    scale = np.sqrt(model.rate_sy[np.newaxis, :]/model.shape_sy[np.newaxis, :])
    ymap /= scale
    plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]], 
               aspect=Nfactors / float(ymap.shape[0]), vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: y (latent features for people)')

    plt.figure()
    ymap = y.T
    scale = np.std(ymap)
    ymap /= scale
    plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]], 
               aspect=Nfactors / float(ymap.shape[0]), vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: y (latent features for people')      
       
    # w
    for f in range(model.Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.w[:, f]
        scale = np.sqrt(model.rate_sw[f]/model.shape_sw[f])
        wmap /= scale
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('predictions at training points: w_%i (latent feature for items)' %f)

#         plt.figure()
#         wmap = np.zeros((nx, ny))
#         wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(model.w_cov[np.arange(model.N*f, model.N*(f+1)), 
#                                                                                    np.arange(model.N*f, model.N*(f+1))])        
#         scale = np.std(wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#         wmap /= scale
#         plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]], aspect=None, vmin=-2, 
#                    vmax=2, interpolation='none', filterrad=0.01)
#         plt.title('STD at training points: w_%i (latent feature for items)' %f)

    for f in range(Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = w[:, f]
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('ground truth at training points: w_%i (latent feature for items' % f)  
        
    plt.show()