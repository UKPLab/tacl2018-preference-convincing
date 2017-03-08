'''
Simple synthetic data tests for the GP preference learning module.
Created on 3 Mar 2017

@author: simpson
'''
from scipy.stats import multivariate_normal as mvn, kendalltau, norm, bernoulli
import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals, coord_arr_to_1d
from gp_pref_learning import GPPrefLearning
import logging

def gen_synthetic_prefs(f_prior_mean=None, nx=100, ny=100, N=5, P=100, ls=[10, 10], s=1):
    # f_prior_mean should contain the means for all the grid squares
    # P is number of pairs for training
    # s is inverse precision scale for the latent function.

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
        
    K = matern_3_2_from_raw_vals(np.array([xvals, yvals]), ls)

    # generate the function values for the pairs
    if f_prior_mean is None:
        f = mvn.rvs(cov=K/s) # zero mean        
    else:
        f = mvn.rvs(mean=f_prior_mean[xvals, yvals].flatten(), cov=K/s) # zero mean
    
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
    
    return N, nx, ny, prefs, xvals, yvals, pair1idxs, pair2idxs, f, K

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    
        
    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)
    
    N, nx, ny, prefs, xvals, yvals, pair1idxs, pair2idxs, f, _ = gen_synthetic_prefs()
    pair1coords = np.concatenate((xvals[pair1idxs, :], yvals[pair1idxs, :]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs, :], yvals[pair2idxs, :]), axis=1)   
        
    # separate training and test data
    Ptest = int(len(prefs) * 0.1)
    testpairs = np.random.choice(pair1coords.shape[0], Ptest, replace=False)
    testidxs = np.zeros(pair1coords.shape[0], dtype=bool)
    testidxs[testpairs] = True
    trainidxs = np.invert(testidxs)
    
    xvals_test = np.array([xvals[pair1idxs[testidxs]], xvals[pair2idxs[testidxs]]]).flatten()
    yvals_test = np.array([yvals[pair1idxs[testidxs]], yvals[pair2idxs[testidxs]]]).flatten()
    _, uidxs = np.unique(coord_arr_to_1d(np.concatenate((xvals_test[:, np.newaxis], yvals_test[:, np.newaxis]), axis=1)), 
              return_index=True)
    xvals_test = xvals_test[uidxs][:, np.newaxis]
    yvals_test = yvals_test[uidxs][:, np.newaxis]
    f_test = np.concatenate((f[pair1idxs[testidxs]], f[pair2idxs[testidxs]]))[uidxs]
    
    models = {}
    
    # Create a GPPrefLearning model
    model = GPPrefLearning(2, mu0=0, shape_s0=1, rate_s0=1, ls_initial=[10, 10], use_svi=True, ninducing=100)    
    #model.verbose = True
    model.delay = 1
    
    models['SVI'] = model
    
    # Create a GPPrefLearning model
    model = GPPrefLearning(2, mu0=0, shape_s0=1, rate_s0=1, ls_initial=[10, 10], use_svi=False, ninducing=100)    
    #model.verbose = True
    model.delay = 1    
    
    models['VB'] = model
    
    if fix_seeds:
        np.random.seed() # do this if we want to use a different seed each time to test the variation in results
    
    for modelkey in models:
        model = models[modelkey]
        
        print "--- Running model %s ---" % modelkey
        
        model.fit(pair1coords[trainidxs], pair2coords[trainidxs], prefs[trainidxs], optimize=True)
        print "Final lower bound: %f" % model.lowerbound()
        
        # Predict at the test locations
        fpred, vpred = model.predict_f(np.concatenate((xvals_test, yvals_test), axis=1))
        
        # Compare the observation point values with the ground truth
        obs_coords_1d = coord_arr_to_1d(model.obs_coords)
        test_coords_1d = coord_arr_to_1d(np.concatenate((xvals, yvals), axis=1))
        f_obs = [f[(test_coords_1d==obs_coords_1d[i]).flatten()][0] for i in range(model.obs_coords.shape[0])]
        print "Kendall's tau (observations): %.3f" % kendalltau(f_obs, model.obs_f.flatten())[0]
            
        # Evaluate the accuracy of the predictions
        #print "RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
        #print "NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
        print "Kendall's tau (test): %.3f" % kendalltau(f_test, fpred)[0] 
            
        t = (f[pair1idxs[testidxs]] > f[pair2idxs[testidxs]]).astype(int)
        rho_pred, var_rho_pred = model.predict(pair1coords[testidxs], pair2coords[testidxs])
        rho_pred = rho_pred.flatten()
        t_pred = np.round(rho_pred)
        
        # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
        if fix_seeds:
            np.random.seed(2)    
        
        print "Brier score of %.3f" % np.sqrt(np.mean((t-rho_pred)**2))
        print "Cross entropy error of %.3f" % -np.sum(t * np.log(rho_pred) + (1-t) * np.log(1 - rho_pred))    
        
        from sklearn.metrics import f1_score, roc_auc_score
        print "F1 score of %.3f" % f1_score(t, t_pred)
        print "Accuracy of %.3f" % np.mean(t==t_pred)
        print "ROC of %.3f" % roc_auc_score(t, rho_pred)
        
        # looks like we don't correct the modified order of points in predict() that occurs in the unique locations function 
        