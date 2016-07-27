'''

For the list of tasks implemented here, see:

https://docs.google.com/spreadsheets/d/15LXSrCcaDURsIYmogt-NEwaakUiITFtiq993TLkflRQ/edit#gid=0

Created on 10 May 2016

@author: simpson
'''
import logging
logging.basicConfig(level=logging.DEBUG)
    
import numpy as np
from krippendorffalpha import alpha
from preference_features import PreferenceComponents
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D    
import pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.mixture import DPGMM
from sklearn.decomposition import FactorAnalysis
from scipy.stats import norm

if __name__ == '__main__':
    
    # Task A1 ---------------------------------------------------------------------------------------------------------
    datadir = './argumentation/outputdata'
    # load the data with columns: person_ID, arg_1_ID, arg_2_ID, preference_label
    data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',')
    
    plotdir = './argumentation/results/'
    
    npairs = data.shape[1]
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    max_arg_id = np.max(arg_ids)
        
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

 
    kf = KFold(npairs, 5)
    
    nfactors_list = [3, 5, 10, 100]
    nmethods = 2 * len(nfactors_list) + 1
    
    results = np.zeros((npairs, nmethods))
    
    k = 0 # count which fold we are in so we can save data     
    for train, test in kf:
        
        m = 0
        
        for nfactors in nfactors_list:
            nflabel = 'nfactors_%i' % nfactors # an extra label to add to plots and filenames            
        
            # Task C1  ----------------------------------------------------------------------------------------------------
            # Hypothesis: allows some personalisation but also sharing data through the means
            model_gpfa = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=True, 
                                              nfactors=nfactors)
            model_gpfa.cov_type = 'diagonal'
            model_gpfa.fit(personids[train], pair1coords[train], pair2coords[train], prefs[train])
            model_gpfa.pickle_me(datadir + '/c1_model_gpfa_%s_%i.pkl' % (nflabel, k))
            
            results_k = model_gpfa.predict(personids[test], pair1coords[test], pair2coords[test])
            results[test, m] = results_k
        
            m += 1 # method index
        
            # Task C3: Baseline with no separate preference functions per user --------------------------------------------
            # Hypothesis: has benefit that there is more data to learn the GP, but no personalisation
            model_base = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=True)
            model_base.cov_type = 'diagonal'
            model_base.fit(np.zeros(len(personids[train])), pair1coords[train], pair2coords[train], prefs) # blank out the user ids
            model_base.pickle_me(datadir + '/c3_model_base_%s_%i.pkl' % (nflabel, k))
    
            results_k = model_base.predict(np.zeros(len(personids[test])), pair1coords[test], pair2coords[test])
            results[test, m] = results_k
            
        # Now run the model but without the FA part; no shared information between people. 
        # Hypothesis: splitting by person results in too little data per person
    
        model_gponly = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], 
                                            verbose=True, nfactors=nfactors)
        model_gponly.cov_type = 'diagonal'
        model_gponly.max_iter = 1 # don't run VB till convergence -- gives same results as if running GPs and FA separately
        model_gponly.fit(personids[train], pair1coords[train], pair2coords[train], prefs[train])
        model_gponly.pickle_me(datadir + '/c1_model_gponly_%s_%i.pkl' % (nflabel, k))
      
        results_k = model_gponly.predict(personids[test], pair1coords[test], pair2coords[test])
        results[test, m] = results_k
        
        m += 1 # method index
        
        m += 1
        k += 1
        
    # Task C2, C4: Compute accuracy metrics -------------------------------------------------------------------------------
    metrics = {}
    metrics['f1'] = np.zeros(nmethods)
    metrics['auc_roc'] = np.zeros(nmethods)
    metrics['log_loss'] = np.zeros(nmethods)
    
    for i in range(nmethods):
        metrics['f1'][i] = f1_score(prefs, np.round(results[i, 0]))
        metrics['auc_roc'][i] = roc_auc_score(prefs, results[i, 0])
        metrics['log_loss'][i] = log_loss(prefs, results[i, 0])
        
    # Task C9/C10: Plotting metrics ---------------------------------------------------------------------------------------
    plt.figure()
    plt.title('F1 Scores with 5-fold Cross Validation')
    ax = plt.bar(metrics['f1'])
    plt.xlabel('Method')
    plt.ylabel('F1 Score')
    ax.set_xticks(np.arange(results.shape[1]))
    ax.set_xticklabels(('PL GPs + FA'))
    
    plt.savefig(plotdir + '/f1scores.eps') 
    
    plt.figure()
    plt.title('AUC of ROC Curve with 5-fold Cross Validation')
    ax = plt.bar(metrics['auc_roc'])
    plt.xlabel('Method')
    plt.ylabel('AUC')
    ax.set_xticks(np.arange(results.shape[1]))
    ax.set_xticklabels(('PL GPs + FA'))
    
    plt.savefig(plotdir + '/auc_roc.eps')
    
    plt.figure()
    plt.title('Cross Entropy Error with 5-fold Cross Validation')
    ax = plt.bar(metrics['log_loss'])
    plt.xlabel('Method')
    plt.ylabel('Cross Entropy')
    ax.set_xticks(np.arange(results.shape[1]))
    ax.set_xticklabels(('PL GPs + FA'))
    
    plt.savefig(plotdir + '/cross_entropy.eps')        

    # Section B. VISUALISING THE LATENT PREFERENCE FUNCTION AND RAW DATA WITHOUT MODELS ---------------------------
    
    nflabel = 'alldata'
    
    # Task A3  ----------------------------------------------------------------------------------------------------
    model_gponly = PreferenceComponents([nx, ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10], verbose=True, 
                                        nfactors=nfactors)
    model_gponly.cov_type = 'diagonal'
    model_gponly.max_iter = 1 # don't run VB till convergence -- gives same results as if running GPs and FA separately
    model_gponly.fit(personids, pair1coords, pair2coords, prefs)
    model_gponly.pickle_me(datadir + '/a3_model_gponly_%s.pkl' % nflabel)    
    
    # Task A1 continued. Put the data into the correct format for visualisation/clustering
    fbar = np.zeros(model_gponly.t.shape) # posterior means
    v = np.zeros(model_gponly.t.shape) # posterior variance
    for person in model_gponly.gppref_models:
        fbar[person, :] = model_gponly.f[person][:, 0]
        v[person, :] = model_gponly.gppref_models[person].v[:, 0]
    fstd = np.sqrt(v)

    # B1. Combine all these functions into a mixture distribution to give an overall density for the whole population
    minf = np.min(fbar - fstd) # min value to plot
    maxf = np.max(fbar - fstd) # max value to plot
    density_xvals = np.arange(minf, maxf, (maxf-minf) / 100.0 ) # 100 points to plot
    density_xvals = np.tile(density_xvals[:, np.newaxis], (1, fbar.shape[1]))
    
    fsum = np.zeros(density_xvals.shape)
    findividual = np.zeros((fsum.shape[0], fsum.shape[1], fbar.shape[0]))
    seenidxs = np.zeros(fbar.shape)
    for person in range(fbar.shape[0]):
        pidxs = personids == person
        pidxs = np.in1d(xvals, pair1coords[pidxs, 0]) | np.in1d(xvals, pair2coords[pidxs, 0])
        #fsum[:, pidxs] += norm.cdf(density_xvals[:, pidxs], loc=fbar[person:person+1, pidxs], scale=fstd[person:person+1, pidxs])
        findividual[:, pidxs, person] = norm.pdf(density_xvals[:, pidxs], loc=fbar[person:person+1, pidxs], scale=fstd[person:person+1, pidxs])
        fsum[:, pidxs] += findividual[:, pidxs, person]
        seenidxs[person, pidxs] = 1
    
    #order the points by their midpoints (works for CDF?)
    #midpoints = fsum[density_xvals.shape[0]/2, :]
    peakidxs = np.argmax(fsum, axis=0)
    ordering = np.argsort(peakidxs)
    fsum = fsum[:, ordering]

    with open (datadir + '/b1_fsum_%s.pkl' % nflabel, 'w') as fh:
        pickle.dump(fsum, fh)
    
    # B2. 3D plot of the distribution

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # B3. Produce/save the plot
    idxmatrix = np.arange(fsum.shape[1])
    idxmatrix = np.tile(idxmatrix[np.newaxis, :], (density_xvals.shape[0], 1)) # matrix of xvalue indices
    ax.plot_surface(density_xvals, idxmatrix, fsum, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.savefig(plotdir + 'b3_fsum_%s.eps' % nflabel)
    
    # B4. Compute variance of the GP means and sort by variance
    fbar_seen = np.empty(fbar.shape) # need to deal properly with the nans
    fbar_seen[:, :] = np.NAN 
    fbar_seen[seenidxs] = fbar[seenidxs]
    fmean_var = np.nanvar(fbar_seen, axis=0) # should exclude the points where people made no classification
    
    peakidxs = np.argmax(fmean_var, axis=0)
    ordering = np.argsort(peakidxs)
    fmean_var = fmean_var[:, ordering]
    
    with open (datadir + '/fmean_var_%s.pkl' % nflabel, 'w') as fh:
        pickle.dump(fmean_var, fh)
    
    # B5. Plot variance in pref function means
    fig = plt.figure()
    plt.plot(np.arange(fmean_var.shape[1]), fmean_var)
    plt.xlabel('Argument Index')
    plt.ylabel('Variance in Latent Pref Function Expected Values')
    plt.title('Variance Expected Latent Preferences Between Different Members of the Crowd')
    plt.savefig(plotdir + 'b5_fsum_%s.eps' % nflabel)
    
    # B6 plot histograms of the gold standard preference pairs. Plots to try:
    # x-axis indexes the argument pairs
    # y-axis indexes the number of observations. 
    # sort by score: number of positive - negative labels
    # Alternatively, summarise this plot somehow?
    fig = plt.figure()
    
    N = model_gponly.t.shape[1]
    p_hist = np.zeros((3, N**2 ))
    p_scat_x = np.zeros((data.shape[0]))
    p_scat_y = np.zeros((data.shape[0]))
    for i in range(N):
        print i
        for j in range(N):
            idx = i * N + j
            pairidxs = ((xvals[pair1idxs]==i) & (xvals[pair2idxs]==j)) | ((xvals[pair2idxs]==i) & (xvals[pair1idxs]==j))
            p_hist[0, idx] = np.sum(data[pairidxs, 3] == 0)
            p_hist[1, idx] = np.sum(data[pairidxs, 3] == 1)
            p_hist[2, idx] = np.sum(data[pairidxs, 3] == 2)
            
            if np.any(pairidxs):
                p_scat_x[pairidxs] = idx
                p_scat_y[pairidxs] = data[pairidxs, 3]
            
    # sort by mean value
    means = np.sum(p_hist * [[-1], [0], [1]], axis=0)
    sortbymeans = np.argsort(means)
    p_hist = p_hist[:, sortbymeans]
            
    # x locations
    x_locs = np.arange(N**2) - 0.5
    
    #plot histogram
    width = 0.3
    plt.bar(x_locs, p_hist[0, :], width, label='1 > 2')
    plt.bar(x_locs + width, p_hist[1, :], width, label='1==2')
    plt.bar(x_locs + 2*width, p_hist[2, :], width, label='1 < 2')
    
    plt.xlabel('Argument Pairs')
    plt.ylabel('Number of labels')
    plt.legend(loc='best')
    plt.title('Histogram of Labels for each Argument')
    
    plt.savefig(plotdir + '/b6_pref_histogram_%s.eps' % nflabel)
    
    #scatter plot
    plt.scatter(p_scat_x, p_scat_y)
    plt.xlabel('Argument Pairs')
    plt.ylabel('Preference Label')
    plt.title('Distribution of Preferences for Arguments')
    
    plt.savefig(plotdir + '/b6_pref_scatter_%s.eps' % nflabel)
    
    # B7 Compute variance in the observed preferences and sort
    mean_p_hist = (-1 * p_hist[0, :] + 1 * p_hist[2, :]) / np.sum(p_hist, axis=0)[np.newaxis, :]
    var_p_hist = (p_hist[0, :] - mean_p_hist)**2 + (p_hist[1, :] - mean_p_hist)**2 + (p_hist[2, :] - mean_p_hist)**2
    var_p_hist /= np.sum(p_hist, axis=0) 
    
    sortbyvar = np.argsort(var_p_hist)
    var_p_hist = var_p_hist[:, sortbyvar]
    
    # B8 Plot Preference pair variance and save
    plt.figure()
    plt.plot(x_locs + 0.5, var_p_hist)
    plt.xlabel('Argument Pairs')
    plt.ylabel('Variance in Pref. Labels')
    plt.title('Variance in Labels Collected for Each Pair')
    
    plt.savefig(plotdir + '/b8_pref_pair_var_%s.eps' % nflabel)
    
    # B9 Plot pref function means as a line graph -- without using a model, this will be very hard to read
    plt.plot(np.arange(N), fbar)
    plt.xlabel('Arguments')
    plt.ylabel('Latent Preference Value')
    plt.title('Expected Latent Preference Functions for Each Person')
    
    plt.savefig(plotdir + '/b9_pref_means_%s.eps' % nflabel)
    
    # Section D: CLUSTER ANALYSIS -------------------------------------------------------------------------------------
    # The data to cluster is stored in fbar.
    
    U = data[:, 1] * max_arg_id + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
    C = data[:, 3]
    L = data[:, 0]
    IAA_all = alpha(U, C, L)    
    
    for nfactors in nfactors_list:
        
        nflabel = 'nfactors_%i' % nfactors # an extra label to add to plots and filenames
    
        # Task D1 -----------------------------------------------------------------------------------------------------
        dpgmm = DPGMM(nfactors)
        dpgmm_labels = dpgmm.fit_predict(fbar)
        dpgmm_proba = dpgmm.predict_proba(fbar)        
    
        # Task D2 -----------------------------------------------------------------------------------------------------
        fa = FactorAnalysis(nfactors)
        fbar_trans = fa.fit_transform(fbar)
        
        # Task D3 -----------------------------------------------------------------------------------------------------
        plt.figure()
        # sum up membership probabilities for DPGMM
        dpgmm_membership = np.sum(dpgmm_proba, axis=0)
        ax = plt.bar(dpgmm_membership)
        ax.set_xticklabels(np.arange(nfactors))
        plt.title('Total membership in Each Cluster')
        plt.xlabel('Cluster Index')
        plt.ylabel('Sum of Membership Probabilities')
        
        plt.savefig(plotdir + '/dpgmm_membership_%s.eps' % nflabel)
        
        # cannot apply this to factor analysis -- can do another plot to show distributions. How fuzzy are memberships?
        plt.figure()
        plt.title('Distribution of People Along Each of %i Factors' % nfactors)
        for k in range(nfactors):
            plt.subplot(nfactors / 4 + 1, 4, k )
            plt.scatter(np.arange(fbar.shape[0]), fbar_trans[:, k])
            plt.xlim(-5 * fbar.shape[0], 6 * fbar.shape[0])
        plt.savefig(plotdir + '/factors_individual_%s.eps' % nflabel)
        
        plt.figure()
        plt.title('Distribution of Cluster Membership Probabilities with %i Clusters' % nfactors)
        for k in range(nfactors):
            plt.subplot(nfactors / 4 + 1, 4, k )
            plt.scatter(np.arange(fbar.shape[0]), dpgmm_proba[:, k])
            plt.xlim(-5 * fbar.shape[0], 6 * fbar.shape[0])
        plt.savefig(plotdir + '/dpgmm_probs_individual_%s.eps' % nflabel)
        
        # Task D4: Plot pairs of components/cluster distributions -- need some way to select pairs if we are going to do this
        plt.figure()
        plt.title('Distribution of People Along Pairs of Factors (%i Factors)' % nfactors)
        for k in range(nfactors):
            for k2 in range(nfactors):
                plt.subplot(nfactors / 4 + 1, 4, k**2 )
                plt.scatter(fbar_trans[:, k], fbar_trans[:, k2])
                plt.xlabel('component %i' % k)
                plt.ylabel('component %i' % k2)
        plt.savefig(plotdir + '/factors_pairs_%s.eps' % nflabel)
        
        plt.figure()
        plt.title('Distribution of People between Pairs of Clusters (%i Clusters)' % nfactors)
        for k in range(nfactors):
            for k2 in range(nfactors):
                plt.subplot(nfactors / 4 + 1, 4, k )
                plt.scatter(dpgmm_proba[:, k], dpgmm_proba[:, k2])
                plt.xlabel('probability of cluster %i' % k)
                plt.ylabel('probability of cluster %i' % k2)                
        plt.savefig(plotdir + '/dpgmm_probs_pairs_%s.eps' % nflabel)
        
        # Task D5, D6: For each cluster, plot f-value means and varaince within the clusters --------------------------
        fbar_archetypes = np.zeros((nfactors, fbar.shape[1]))
            
        fig_dpmeans = plt.figure()
        fig_dpvar = plt.figure()
            
        for k in range(nfactors):
            # use a set of samples from the mixture distribution to approximate the mean.
            weights = dpgmm_proba[:, k][np.newaxis, np.newaxis, :]
            fsum_k = findividual * weights / np.sum(weights)
            fsum_k = fsum_k[:, ordering]
        
            # This is overkill
#             ax = fig.add_subplot(nfactors/4+1, 4,k, projection='3d')    
#             ax.plot_surface(density_xvals, idxmatrix, fsum_k, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, 
#                             antialiased=False)
#             ax.set_xlabel('Arguments')
#             ax.set_ylabel('Latent pref. function value')
#             ax.set_zlabel('Density')

            # Plot the means only
            fmean_k = np.sum(density_xvals * fsum_k / np.sum(fsum_k, axis=0)[np.newaxis, :], axis=0)
            plt.figure(fig_dpmeans)
            plt.plot(fmean_k, label='cluster %s' % k)
            
            # compute the variance
            fvar_k = np.sum((density_xvals - fmean_k[np.newaxis, :])**2 * fsum_k, axis=0) /  np.sum(fsum_k, axis=0)
            plt.figure(fig_dpvar)
            plt.plot(fvar_k, label='cluster %s' % k)    
            
        plt.figure(fig_dpmeans)
        plt.legend(loc='best')
        plt.xlabel('Arguments')
        plt.ylabel('Means of latent preference function')
        plt.title('Latent Function Mean by Cluster')
        plt.savefig(plotdir + 'd5_fmeank_%s.eps' % nflabel)
        
        plt.figure(fig_dpvar)
        plt.legend(loc='best')
        plt.xlabel('Arguments')
        plt.ylabel('Variance of latent preference function')
        plt.title('Latent Function Variance within each Cluster')
        plt.savefig(plotdir + 'd5_fvark_%s.eps' % nflabel)

        # Task D7: For each cluster, plot variance in observed prefs in each cluster ----------------------------------
        for k in range(nfactors):
            # use a set of samples from the mixture distribution to approximate the mean.
            weights = dpgmm_proba[:, k][np.newaxis, np.newaxis, :]
            fsum_k = findividual * weights / np.sum(weights)
            fsum_k = fsum_k[:, ordering]
        
            # Plot the means only
            fmean_k = np.sum(density_xvals * fsum_k / np.sum(fsum_k, axis=0)[np.newaxis, :], axis=0)
            plt.figure(fig_dpmeans)
            plt.plot(fmean_k, label='cluster %s' % k)
            
            # compute the variance
            fvar_k = np.sum((density_xvals - fmean_k[np.newaxis, :])**2 * fsum_k, axis=0) /  np.sum(fsum_k, axis=0)
            plt.figure(fig_dpvar)
            plt.plot(fvar_k, label='cluster %s' % k)    
            
        plt.figure(fig_dpmeans)
        plt.legend(loc='best')
        plt.xlabel('Arguments')
        plt.ylabel('Means of latent preference function')
        plt.title('Latent Function Mean by Cluster')
        plt.savefig(plotdir + 'd5_fmeank_%s.eps' % nflabel)
        
        plt.figure(fig_dpvar)
        plt.legend(loc='best')
        plt.xlabel('Arguments')
        plt.ylabel('Variance of latent preference function')
        plt.title('Latent Function Variance within each Cluster')
        plt.savefig(plotdir + 'd5_fvark_%s.eps' % nflabel)
        

        
        #  This will be a total that incorporates the variance in individual f-values 
        # Task D8: For each cluster, plot variance in predicted prefs in each cluster ---------------------------------
        # Task D9: For each cluster, compute IAA. Plot as bar chart and save ------------------------------------------
        IAA = np.zeros(nfactors)
        labels = ['All People']
        for k in range(nfactors):
            # use a set of samples from the mixture distribution to approximate the mean.
            kpersonidxs = np.argwhere(dpgmm_labels==k)
            kidxs = np.in1d(data[:, 0], kpersonidxs)
            Uk = data[kidxs, 1] * max_arg_id + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
            Ck = data[kidxs, 3] # classifications
            Lk = data[kidxs, 0] # labellers
            
            IAA[k] = alpha(Uk, Ck, Lk)   
            labels.append('%i' % k)
            
        IAA = np.concatenate(([IAA_all], IAA))
            
        plt.figure()
        ax = plt.bar(range(nfactors), IAA)
        plt.xlabel('Cluster Index')
        ax.set_xticklabels(labels)
        plt.ylabel("Krippendorff's Alpha")
        plt.title('Inter-annotator Agreement in each Cluster')
        plt.savefig(plotdir + '/IAA')