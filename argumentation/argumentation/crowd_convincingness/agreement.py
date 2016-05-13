'''
Created on 10 May 2016

@author: simpson
'''
import numpy as np
from krippendorffalpha import alpha

if __name__ == '__main__':
    
    datadir = './argumentation/outputdata'
    data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',')
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    max_arg_id = np.max(arg_ids)
    
    U = data[:, 1] * max_arg_id + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
    C = data[:, 3]
    L = data[:, 0]
    print alpha(U, C, L)