'''

For the list of tasks implemented here, see:

https://docs.google.com/spreadsheets/d/15LXSrCcaDURsIYmogt-NEwaakUiITFtiq993TLkflRQ/edit#gid=0

Created on 10 May 2016

@author: simpson
'''
import logging
logging.basicConfig(level=logging.DEBUG)
    
import numpy as np
from mpl_toolkits.mplot3d import Axes3D    
from sklearn.cross_validation import KFold

import classification_metrics
from pref_prediction_methods import PredictionTester
from preproc_raw_data import load

if __name__ == '__main__':
    
    datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, personids, \
        npairs, nworkers, ntexts = load()
    
    # Task A2 ---------------------------------------------------------------------------------------------------------
 
    nfactors=5
    
    methods = ['AffProp_Averaging'] # list the names of methods to test here
    nmethods = len(methods) 
    #2 * len(nfactors_list) + 1 # need to increase nmethods if we are trying multiple numbers of factors 
    # -- new implementation will try to optimize the number of factors internally and return only the best results for each method

    # RUN THE VARIOUS PREFERENCE PREDICTION METHODS....
    k = 0 # count which fold we are in so we can save data    

    kf = KFold(npairs, 10)
      
    results = np.zeros((npairs, nmethods)) # the test results only
       
    for train, test in kf:
        logging.info('Running fold %i' % k)
        
        tester = PredictionTester(datadir, k, m, nx, ny, personids, pair1coords, pair2coords, prefs, train, test, results)
        
        for m, method in enumerate(methods):
            if method=='GPFA':
                logging.info('Task C1, GPFA')
             
                # Run the GPFA with this fold
                tester.run_gpfa(nfactors)
            elif method=='CombinedGP':
                # Task C3: Baseline with no separate preference functions per user ----------------------------------------
                logging.info('Task C3, Combined GP')
                 
                tester.run_gp_combined() 
            elif method=='SeparateGP':              
                logging.info('Task C1 part II, Separate GPs (no FA)')
                
                tester.run_gp_separate()
            elif method=='AffProp_Averaging':
                logging.info('Affinity Propagation, then averaging clusters to predict')
                
                tester.run_aff_prop_avg()
        k += 1
          
    metrics = classification_metrics.compute_metrics(nmethods, prefs, results)       
    classification_metrics.plot_metrics(plotdir, metrics, nmethods, methods)
 

             

