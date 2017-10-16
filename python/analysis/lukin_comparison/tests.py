'''
Run tests on the Lukin dataset here. To demonstrate how to use the PreferenceComponents method, 
we do the following simple test:
- load the dataset
- train on the whole dataset
- for a set of test users (for demonstration, we just sample randomly from training set), predict latent features
- for a set of test arguments (for demonstration, we sample pairs randomly from training set), predict latent features
- for a set of test pairs (users + arguments chosen at random from training set), predict the preference rating
- for a set of test triples (users + argument pairs chosen at random from the training set), predict the pairwise label    

TODO: move user analysis stuff to a separate script
TODO: switch to 10-fold cross validation instead of training on all
TODO: Add missing prior belief data
TODO: Check whether the lmh labels are correctly converted to preferences -- in some cases they should be flipped

Created on Sep 25, 2017

@author: simpson
'''

import numpy as np, os, sys
from sklearn.metrics.classification import accuracy_score
import logging
logging.basicConfig(level=logging.DEBUG)

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/lukin_comparison")

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

from preference_features import PreferenceComponentsSVI
from gp_classifier_vb import compute_median_lengthscales

use_entrenched = True

if __name__ == '__main__':
    #load the dataset -----------------------------------------------------------------------------------
    arg_data = np.genfromtxt('./data/lukin/arguments.csv', dtype=float, delimiter=',', skip_header=1)
    arg_ids = arg_data[:, 0].astype(int)
    item_feat = arg_data[:, 1:]
    
    user_data = np.genfromtxt('./data/lukin/users.csv', dtype=float, delimiter=',', skip_header=1)
    user_ids = user_data[:, 0].astype(int)
    person_feat = user_data[:, 1:]
    npersonality_feats = person_feat.shape[1]
    
    pair_data = np.genfromtxt('./data/lukin/ratings.csv', dtype=float, delimiter=',', skip_header=1)
    # should we use the 'entrenched' column as an additional user feature?
    if use_entrenched:
        # double the person feature vector -- each person appears twice, once for entrenched, once for not entrenched
        Npeople = person_feat.shape[0]
        person_feat = np.concatenate((person_feat, person_feat), axis=0)
        entrenched_feat = np.zeros((Npeople*2, 1))
        entrenched_feat[:Npeople] = 1
        person_feat = np.concatenate((person_feat, entrenched_feat), axis=1)
        
        entrenched_labels = pair_data[:, 3].astype(int)
        personIDs_train = np.array([np.argwhere(user_ids==uid)[0][0] + (Npeople*entrenched_labels[i]) for i, uid in 
                                    enumerate(pair_data[:, 0].astype(int))])
    else:
        personIDs_train = np.array([np.argwhere(user_ids==uid)[0][0] for uid in pair_data[:, 0].astype(int)])
    trainids_a1 = np.array([np.argwhere(arg_ids==aid)[0][0] for aid in pair_data[:, 1].astype(int)])
    trainids_a2 = np.array([np.argwhere(arg_ids==aid)[0][0] for aid in pair_data[:, 2].astype(int)])

    prefs_train = pair_data[:, 6] # use the norm labels
    #prefs_train = pair_data[:, 4] # use the lmh labels
    #prefs_train = pair_data[:, 5] # use the lh labels
    
    # Training ---------------------------------------------------------------------------------------------
    #train on the whole dataset
    ndims = item_feat.shape[1]
    ls_initial = compute_median_lengthscales(item_feat)
    person_ls_initial = compute_median_lengthscales(person_feat)
    
    model = PreferenceComponentsSVI(nitem_features=ndims, ls=ls_initial, lsy=person_ls_initial, shape_s0=2, rate_s0=200,
                                    verbose=False, nfactors=20, rate_ls = 1.0 / np.mean(ls_initial), 
                                    uncorrelated_noise=True, use_common_mean_t=False)
    model.max_iter = 200
    model.fit(personIDs_train, trainids_a1, trainids_a2, item_feat, prefs_train, person_feat, optimize=False, 
              nrestarts=1, input_type='zero-centered')
        
    # sanity check: test on the training data
    trpred = model.predict(personIDs_train, trainids_a1, trainids_a2, item_feat, person_feat)
    tracc = accuracy_score(np.round((prefs_train + 1) / 2.0), np.round(trpred))
    print "The model was trained. Testing on the training data gives an accuracy of %.4f" % tracc
        
    # Get some test data ------------------------------------------------------------------------------------
    
    # ***** Section to replace with a matrix of real test data *****
    test_arg_ids = np.arange(100) # PLACEHOLDER -- replace this with the real document IDs loaded from file
    test_item_feat = item_feat[test_arg_ids, :] # PLACEHOLDER -- replace this with the real test document IDs loaded from file
    # ***** End *****
    
    Ntest = len(test_arg_ids) # keep this
    testids = np.arange(Ntest) # keep this -- this is the index into the test_item_feat. In this case, we test all the items
    
    #for a set of test arguments (for demonstration, we sample pairs randomly from training set), predict latent features
    w = model.predict_item_feats(testids, item_feat)
    print w
    
    # Given the argument's latent features, determine the observed features of the most convinced user
    # Chosen method: create a set of simulated users with prototypical personality features
    feat_min = 1
    feat_max = 7
    nfeat_vals = feat_max - feat_min + 1
    feat_range = nfeat_vals**5 # number of different feature combinations, assuming integer values for personality traits
    Npeople = feat_range

    test_people = np.arange(feat_range)
    test_person_feat = np.zeros((feat_range, npersonality_feats))
    f_val = np.zeros(test_person_feat.shape[1]) + feat_min
        
    for p in range(feat_range):
        for f in range(npersonality_feats):
            test_person_feat[p, f] = f_val[f]
            
            if np.mod(p+1, (nfeat_vals ** f) ) == 0:
                f_val[f] = np.mod(f_val[f], nfeat_vals) + 1

    testids = np.tile(testids[:, None], (1, feat_range)).flatten()
    test_people = np.tile(test_people[None, :], (Ntest, 1)).flatten()

    if use_entrenched:
        test_person_feat = np.concatenate((test_person_feat, test_person_feat), axis=0)
        entrenched_feat = np.zeros((Npeople*2, 1))
        entrenched_feat[:Npeople] = 1
        test_person_feat = np.concatenate((test_person_feat, entrenched_feat), axis=1)
        
        test_people = np.concatenate((test_people, test_people + feat_range))
        testids = np.concatenate((testids, testids))
        
        Npeople = Npeople * 2
             
    # predict the ratings from each of the simulated people
    npairs = Npeople * Ntest
    predicted_f = np.zeros(npairs)
    # do it in batches of 500 because there are too many people
    batchsize = 500
    nbatches = int(np.ceil(npairs / float(batchsize)))
    for b in range(nbatches):
        print "Predicting simulated users in batch %i of %i" % (b, nbatches)
        start = batchsize * b
        fin = batchsize * (b + 1)
        if fin > npairs:
            fin = npairs
        predicted_f[start:fin] = model.predict_f(test_people[start:fin], testids[start:fin], 
                                                 test_item_feat, test_person_feat) 
    
    # put into a matrix of row=item, col=person
    predicted_f_mat = np.reshape(predicted_f, (Ntest, Npeople))
    
    # who had the highest preference?
    max_people_idx = np.argmax(predicted_f_mat, axis=1)
        
    # get the features of the max people
    max_people_feats = test_person_feat[max_people_idx, :]
    
    out_data = np.concatenate((test_arg_ids[:, None], max_people_feats), axis=1)
    
    # save to file along with original argument IDs from the test data file
    fmt = '%i, '
    header = 'argID,openness,conscientiousness,extroversion,agreeableness,neuroticism'
    for f in range(npersonality_feats):
        fmt += '%f, '

    if use_entrenched:
        fmt += '%i'
        header += ',entrenched'
        
    if not os.path.isdir('./results/lukin'):
        os.mkdir('./results/lukin')
    np.savetxt('./results/lukin/personalities_for_args.csv', out_data, fmt, ',', header=header)
          