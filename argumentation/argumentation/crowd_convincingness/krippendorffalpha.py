'''
Created on 10 May 2016

@author: simpson
'''

import numpy as np

def alpha(U, C, L):
    '''
    U - units of analysis, i.e. the data points being labelled
    C - a list of classification labels
    L - a list of labeller IDs
    '''
    M = float(np.unique(L).shape[0])
    N = float(np.unique(U).shape[0])
    
    ulist = np.unique(U)
    
    Dunits = np.zeros(N, dtype=float)
    for i, u in enumerate(U):
        uidxs = U==u
        m_u = float(np.unique(L[uidxs]).shape[0])
        C[uidxs]
        
        for j, l in enumerate(L): 
            Dunits[i] += 1 / (m_u - 1) * np.sum(C[j] - C) 
        
    Dobs = 1 / N * np.sum(Dunits)
    
    alpha = 1 - Dobs / Dexpec