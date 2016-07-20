'''
Created on 10 May 2016

@author: simpson
'''
import logging
logging.basicConfig(level=logging.DEBUG)
    
import numpy as np
#from krippendorffalpha import alpha
from preference_features import PreferenceComponents

if __name__ == '__main__':
    
    # Task A1 ---------------------------------------------------------------------------------------------------------
    datadir = './argumentation/outputdata'
    # load the data with columns: person_ID, arg_1_ID, arg_2_ID, preference_label
    data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',')
    
    Npairs = data.shape[1]
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    max_arg_id = np.max(arg_ids)
    
    #U = data[:, 1] * max_arg_id + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
    #C = data[:, 3]
    #L = data[:, 0]
    #print "Krippendorff's alpha for the raw pairwise labels = %.3f" % alpha(U, C, L)
    
    pair1idxs = data[:, 1]
    pair2idxs = data[:, 2]
    prefs = 1.0 - data[:, 3].astype(float) / 2.0 # flip the labels do give the preference for argument 1. Halve the 
    #values so they sit between 0 and 1 inclusive. Labels expressing equal preference will be 0.5.

    #The feature coordinates for the arguments.
    xvals = np.arange(max_arg_id+1)#np.zeros(max_arg_id+1) # ignore the argument features for now, use indices so that
    # the different arguments are not placed at the exact same location -- allows for ordering of the arguments. We 
    # have to combine this with a diagonal covariance function.
    yvals = np.zeros(max_arg_id+1)
    
    logging.info( "Testing Bayesian preference components analysis using real crowdsourced data...")
    
    nx = 1
    ny = 1
    
    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 

    personids = data[:, 0]
    upersonids = np.unique(personids)
    Npeople = len(upersonids)
    
    # Task A2 ----------------------------------------------------------------------------------------------------------
    
#     from sklearn.cross_validation import KFold
# 
#     kf = KFold(Npairs, 5)
#     
#     for train, test in kf:
#         
#         # Task C1  ----------------------------------------------------------------------------------------------------
#         model = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=True)
#         model.cov_type = 'diagonal'
#         model.fit(personids[train], pair1coords[train], pair2coords[train], prefs[train])
#         model.pickle_me(datadir + '/model_fonly.pkl')
#          
#         model.predict(personids[test], pair1coords[test], pair2coords[test])
        
    # Task A3  ----------------------------------------------------------------------------------------------------
    model = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=True)
    model.cov_type = 'diagonal'
    model.max_iter = 1 # don't run VB till convergence -- gives same results as if running GPs and FA separately
    model.fit(personids, pair1coords, pair2coords, prefs)
    model.pickle_me(datadir + '/model_fonly.pkl')
    no_model_fonly = model # save it for later

# Task B1. VISUALISING THE LATENT PREFERENCE FUNCTIONS ----------------------------------------------------------------

# 1. Put the data into the correct format for visualisation/clustering
fbar = np.zeros(model.t.shape) # posterior means
v = np.zeros(model.t.shape) # posterior variance
for person in model.gppref_models:
    fbar[person, :] = model.f[person][:, 0]
    v[person, :] = model.gppref_models[person].v[:, 0]
fstd = np.sqrt(v)

# 5. Combine all these functions into a mixture distribution to give an overall density for the whole population
from scipy.stats import norm
minf = np.min(fbar - fstd) # min value to plot
maxf = np.max(fbar - fstd) # max value to plot
density_xvals = np.arange(minf, maxf, (maxf-minf) / 100.0 ) # 100 points to plot
density_xvals = np.tile(density_xvals[:, np.newaxis], (1, fbar.shape[1]))

fsum = np.zeros(density_xvals.shape)
for person in range(fbar.shape[0]):
    pidxs = personids == person
    pidxs = np.in1d(xvals, pair1coords[pidxs, 0]) | np.in1d(xvals, pair2coords[pidxs, 0])
    #fsum[:, pidxs] += norm.cdf(density_xvals[:, pidxs], loc=fbar[person:person+1, pidxs], scale=fstd[person:person+1, pidxs])
    fsum[:, pidxs] += norm.pdf(density_xvals[:, pidxs], loc=fbar[person:person+1, pidxs], scale=fstd[person:person+1, pidxs])

#order the points by their midpoints (works for CDF?)
#midpoints = fsum[density_xvals.shape[0]/2, :]
peakidxs = np.argmax(fsum, axis=0)
ordering = np.argsort(peakidxs)
fsum = fsum[:, ordering]

import pickle
with open (datadir + '/fsum.pkl', 'w') as fh:
    pickle.dump(fsum, fh)

#6. 3D plot of the distribution
from matplotlib import pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

idxmatrix = np.arange(fsum.shape[1])
idxmatrix = np.tile(idxmatrix[np.newaxis, :], (density_xvals.shape[0], 1)) # matrix of xvalue indices
ax.plot_surface(density_xvals, idxmatrix, fsum, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# # Plot a bar chart to show each worker's pattern across the whole dataset. This probably kills the computer?
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# personmatrix = np.arange(fbar.shape[0])
# personmatrix = np.tile(personmatrix[:, np.newaxis], (1, fbar.shape[1]))
# 
# ax.bar(personmatrix.flatten(), np.tile(idxmatrix[0:1, :], (personmatrix.shape[0], 1)).flatten(), zs=fbar.flatten(), zdir='y')

#10 and 11: plot the gold standard preference pairs instead.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = model.t.shape[1]
p_hist = np.zeros((3, N**2 ))
for i in range(N):
    print i
    for j in range(N):
        idx = i * N + j
        pairidxs = ((xvals[pair1idxs]==i) & (xvals[pair2idxs]==j)) | ((xvals[pair2idxs]==i) & (xvals[pair1idxs]==j))
        p_hist[0, idx] = np.sum(data[pairidxs, 3] == 0)
        p_hist[1, idx] = np.sum(data[pairidxs, 3] == 1)
        p_hist[2, idx] = np.sum(data[pairidxs, 3] == 2)
        
# sort by mean value
means = np.sum(p_hist * [[-1], [0], [1]], axis=0)
sortbymeans = np.argsort(means)
p_hist = p_hist[:, sortbymeans]
        
pref_xvals = np.tile([[-1], [0], [1]], (1, p_hist.shape[1]))
pair_idxmatrix = np.arange(p_hist.shape[1])
pair_idxmatrix = np.tile(pair_idxmatrix[np.newaxis, :], (pref_xvals.shape[0], 1))
ax.plot_surface(pref_xvals, pair_idxmatrix, p_hist, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)