'''
Created on 10 May 2016

@author: simpson
'''
import xmltodict, os
import numpy as np

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