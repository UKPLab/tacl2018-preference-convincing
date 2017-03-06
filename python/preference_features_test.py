'''
Test the preference_features module with some simple synthetic data test

Created on 3 Mar 2017

@author: edwin
'''
import logging
import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals
from scipy.stats import multivariate_normal as mvn
from gp_pref_learning_test import gen_synthetic_prefs
from preference_features import PreferenceComponents

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    

    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(10)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    Npeople = 200
    Ptest = 20
    pair1idxs = []
    pair2idxs = []
    prefs = []
    personids = []
    xvals = []
    yvals = []
    
    nx = 5
    ny = 5
    
    # generate a common prior:
    ls = [10, 5]
    xvals = np.arange(nx)[:, np.newaxis]
    xvals = np.tile(xvals, (1, ny)).flatten()
    yvals = np.arange(ny)[np.newaxis, :]
    yvals = np.tile(yvals, (nx, 1)).flatten()
    Kt = matern_3_2_from_raw_vals(np.array([xvals, yvals]), ls)
    t = mvn.rvs(cov=Kt).reshape(nx, ny)
    
    Nfactors = 2
    
    Ky = matern_3_2_from_raw_vals(np.arange(Npeople)[np.newaxis, :], [2])
    
    w = np.zeros((nx * ny, Nfactors))
    y = np.zeros((Nfactors, Npeople))
    for f in range(Nfactors):
        w[:, f] = mvn.rvs(cov=Kt).flatten()
        y[f, :] = mvn.rvs(cov=Ky)
    
    for p in range(Npeople):
        
        y_p = y[:, p:p+1]
        wy_p = w.dot(y_p).reshape((nx, ny))
        
        f_prior_mean = t + wy_p
        
        _, nx, ny, prefs_p, xvals_p, yvals_p, pair1idxs_p, pair2idxs_p, f, K = gen_synthetic_prefs(f_prior_mean, nx, ny)
        pair1idxs = np.concatenate((pair1idxs, pair1idxs_p + len(xvals))).astype(int)
        pair2idxs = np.concatenate((pair2idxs, pair2idxs_p + len(yvals))).astype(int)
        prefs = np.concatenate((prefs, prefs_p)).astype(int)
        personids = np.concatenate((personids, np.zeros(len(pair1idxs_p)) + p)).astype(int)
        xvals = np.concatenate((xvals, xvals_p.flatten()))
        yvals = np.concatenate((yvals, yvals_p.flatten()))

    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 

    testpairs = np.random.choice(pair1coords.shape[0], Ptest, replace=False)
    testidxs = np.zeros(pair1coords.shape[0], dtype=bool)
    testidxs[testpairs] = True
    trainidxs = np.invert(testidxs)
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    model = PreferenceComponents([nx,ny], ls=ls, nfactors=Nfactors + 5, use_fa=False, use_svi=True)
    model.verbose = False
    model.fit(personids[trainidxs], pair1coords[trainidxs], pair2coords[trainidxs], prefs[trainidxs])
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1coords[testidxs], pair2coords[testidxs], )
    
    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    from sklearn.metrics import accuracy_score
    
    print 'Accuracy: %f' % accuracy_score(prefs[testidxs], np.round(results))
    
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
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = model.t.flatten()
    scale = np.sqrt(model.rate_st/model.shape_st)
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: t (item mean)')

    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = np.sqrt(np.diag(model.t_cov))
    scale = np.std(tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]])
    plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
    plt.title('STD at training points: t (item mean)')

    plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = t[model.obs_coords[:, 0], model.obs_coords[:, 1]].flatten()
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
    plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]], 
               aspect=Nfactors / float(ymap.shape[0]), vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: y (latent features for people')      
       
    # w
    for f in range(model.Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = model.w[:, f]
        scale = np.sqrt(model.rate_sw[f]/model.shape_sw[f])
        wmap /= scale
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('predictions at training points: w_%i (latent feature for items)' %f)

        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = np.sqrt(model.w_cov[np.arange(model.N*f, model.N*(f+1)), 
                                                                                   np.arange(model.N*f, model.N*(f+1))])        
        scale = np.std(wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]])
        wmap /= scale
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]], aspect=None, vmin=-2, 
                   vmax=2, interpolation='none', filterrad=0.01)
        plt.title('STD at training points: w_%i (latent feature for items)' %f)

    for f in range(Nfactors):
        plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0], model.obs_coords[:, 1]] = w[np.ravel_multi_index((model.obs_coords[:, 0], 
                                                                       model.obs_coords[:, 1]), dims=(nx, ny)), f]
        plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=-2, vmax=2, interpolation='none', filterrad=0.01)
        plt.title('ground truth at training points: w_%i (latent feature for items' % f)  