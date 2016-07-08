'''
Created on 10 May 2016

@author: simpson
'''
import numpy as np
from krippendorffalpha import alpha
from preference_features import PreferenceComponents

if __name__ == '__main__':
    
    datadir = './argumentation/outputdata'
    data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',')
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    max_arg_id = np.max(arg_ids)
    
    U = data[:, 1] * max_arg_id + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
    C = data[:, 3]
    L = data[:, 0]
    #print "Krippendorff's alpha for the raw pairwise labels = %.3f" % alpha(U, C, L)
    
    print "Testing Bayesian preference components analysis using real crowdsourced data..."
    personids = data[:, 0]
    upersonids = np.unique(personids)
    Npeople = len(upersonids)
    
    pair1idxs = data[:, 1]
    pair2idxs = data[:, 2]
    prefs = 1.0 - data[:, 3].astype(float) / 2.0 # flip the labels do give the preference for argument 1. Halve the 
    #values so they sit between 0 and 1 inclusive. Labels expressing equal preference will be 0.5.

    #The feature coordinates for the arguments.
    xvals = np.arange(max_arg_id+1)#np.zeros(max_arg_id+1) # ignore the argument features for now, use indices so that
    # the different arguments are not placed at the exact same location -- allows for ordering of the arguments. We 
    # have to combine this with a diagonal covariance function.
    yvals = np.zeros(max_arg_id+1)
    
    nx = 1
    ny = 1
    
    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 
    
    model = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10])
    model.cov_type = 'diagonal'
    model.fit(personids, pair1coords, pair2coords, prefs)
    
    from scipy.stats import kendalltau
    
    for p in range(Npeople):
        print "Personality features of %i: %s" % (p, str(model.x[p]))
        for q in range(Npeople):
            print "Distance between personalities: %f" % np.sqrt(np.sum(model.x[p] - model.x[q])**2)**0.5
            print "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0]