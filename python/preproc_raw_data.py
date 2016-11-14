'''
Created on 10 May 2016

@author: simpson
'''
import xmltodict, os
import numpy as np
import logging

def load():
    # Task A1 ---------------------------------------------------------------------------------------------------------
    datadir = './outputdata'
    # load the data with columns: person_ID, arg_1_ID, arg_2_ID, preference_label
    #data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',')
    
    # Random data generation
    N = 1000
    Nitems = 10
    acc = 0.9
#     f = np.array([0, 1, 2]) 
    f = np.random.rand(Nitems) * 10
    data0 = np.random.randint(0, 10, (N,1))
#     data1 = np.zeros((N, 1)).astype(int)#
    data1 = np.random.randint(0, Nitems, (N,1))
#     data1[30:60, :] = 1
#     data1[61:, :] = 2
#     data2 = np.ones((N, 1)).astype(int)
    data2 = np.random.randint(0, Nitems, (N,1))
    correctflag = np.random.rand(N, 1) # use this to introduce noise into the preferences instead of reflecting f precisely
    data3 = 2 * (correctflag < acc) * (f[data1]+0.5 < f[data2]) + 1 * (correctflag < acc) * (np.abs(f[data1] - f[data2]) <=0.5) \
                + (correctflag > acc) * np.random.randint(0, 3, (N, 1))
    logging.debug('Number of neg prefs = %i, no prefs = %i, pos prefs = %i' % (np.sum(data3==0), np.sum(data3==1), np.sum(data3==2)))
    data = np.concatenate((data0, data1, data2, data3), axis=1)
    
    #logging.warning("Subsampling dataset for debugging!!!")
    #data = data[:1000, :]
    
    plotdir = './results/'
    
    npairs = data.shape[0]
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    ntexts = np.max(arg_ids) + 1 # +1 because there can also be an argument with ID 0
    
    pair1idxs = data[:, 1].astype(int)
    pair2idxs = data[:, 2].astype(int)
    prefs = data[:, 3].astype(float) / 2.0 # the labels give the preference for argument 2. Halve the 
    #values so they sit between 0 and 1 inclusive. Labels expressing equal preference will be 0.5.

    #The feature coordinates for the arguments.
    xvals = np.arange(ntexts) # ignore the argument features for now, use indices so that
    # the different arguments are not placed at the exact same location -- allows for ordering of the arguments. We 
    # have to combine this with a diagonal covariance function.
    yvals = np.zeros(ntexts)
    
    logging.info( "Testing Bayesian preference components analysis using real crowdsourced data...")
    
    nx = 1
    ny = 1
    
    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 

    personids = data[:, 0].astype(int)
    upersonids = np.unique(personids)
    nworkers = len(upersonids)
    
    return datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, \
            personids, npairs, nworkers, ntexts
    

def process_list(pairlist):
    
    crowdlabels = []
    
    pairlist = pairlist['annotatedArgumentPair']
    for pair in pairlist:
        for workerlabel in pair['mTurkAssignments']['mTurkAssignment']:
            row = [workerlabel['turkID'], pair['arg1']['id'], pair['arg2']['id'], workerlabel['value']]
            crowdlabels.append(row)
        
    return np.array(crowdlabels)

def translate_to_local(all_labels):
    _, localworkers = np.unique(all_labels[:, 0], return_inverse=True)
    _, localargs = np.unique([all_labels[:, 1], all_labels[:, 2]], return_inverse=True)
    localargs1 = localargs[0:all_labels.shape[0]]
    localargs2 = localargs[all_labels.shape[0]:]
    ulabels = {'a1':0, 'a2':2, 'equal':1}
    locallabels = [ulabels[l] for l in all_labels[:, 3]]
    
    all_labels = np.zeros(all_labels.shape)
    all_labels[:, 0] = localworkers
    all_labels[:, 1] = localargs1
    all_labels[:, 2] = localargs2
    all_labels[:, 3] = locallabels
    return all_labels

if __name__ == '__main__':
    '''
    Read in the original XML file. Extract each element as line in the CSV file with IDs of the text blocks, the user,
    the value they assigned.
    '''
    datadir = '/home/local/UKP/simpson/data/step5-gold-data-all/'
    datafiles = os.listdir(datadir)
    #datafiles = ["testonly.xml"] # try with one file first
    
    outputdir = './argumentation/outputdata'
    
    all_labels = np.empty((0, 4))
    
    for i, f in enumerate(datafiles):
        print "Processing file %i of %i, filename=%s" % (i, len(datafiles), f)
        with open(datadir + f) as ffile:
            doc = xmltodict.parse(ffile.read())
            pairlist = doc['list']
            labels = process_list(pairlist)            
            all_labels = np.concatenate((all_labels, labels), axis=0)
            
    np.savetxt(outputdir + '/all_labels_original.csv', all_labels, fmt='%s, %s, %s, %s', delimiter=',')
    all_labels = translate_to_local(all_labels)
    np.savetxt(outputdir + '/all_labels.csv', all_labels, fmt='%i, %i, %i, %i', delimiter=',')