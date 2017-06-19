'''
Incomplete. Should be based on tests.py but separated so we can provide the personalised model experiments separately 
as they will be covered in a different paper.

Created on 19 Jun 2017

@author: simpson
'''
from preference_features import PreferenceComponents

        nfactors = 10

        if 'PersonalisedPrefsBayes' in method:        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, 
                                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba, predicted_f = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                        
        elif 'PersonalisedPrefsUncorrelatedNoise' in method: 
            # Note that this also does not use a common mean to match the Houlsby model.
            # TODO: suspect that with small no. factors, this may be worse, but better with large number in comparison to PersonalisedPrefsBayes with Matern noise GPs.        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                        rate_ls = 1.0 / np.mean(ls_initial_guess), 
                                        use_svi=True, use_fa=False, uncorrelated_noise=True, use_common_mean=False, 
                                        max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                            
        elif 'PersonalisedPrefsFA' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=True, 
                                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                
        elif 'PersonalisedPrefsNoFactors' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, no_factors=True, 
                            max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)
                
        elif 'PersonalisedPrefsNoCommonMean' in method:        
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                        rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, use_common_mean_t=False, 
                        max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat)
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)         
                   
        elif 'IndPrefGP' in method:
            model = PreferenceComponents(nitem_features=ndims, ls=ls_initial_guess, verbose=verbose, nfactors=nfactors, 
                            rate_ls = 1.0 / np.mean(ls_initial_guess), use_svi=True, use_fa=False, no_factors=True, 
                            use_common_mean_t=False, max_update_size=200)
            model.fit(personIDs_train, trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1, 
                      optimize=optimize_hyper, nrestarts=1, input_type='zero-centered')
            proba = model.predict(personIDs_test, testids_a1, testids_a2, items_feat) 
            if folds_regression is not None:
                predicted_f = model.predict_f(personIDs_test, item_idx_ranktest, items_feat)                    
