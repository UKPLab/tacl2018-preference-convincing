'''
Created on 24 Mar 2017

@author: simpson
'''

import os

if __name__ == '__main__':
    dirname = "/home/local/UKP/simpson/data/outputdata/UKPConvArg1-Full-libsvm"
    outputfile = "/home/local/UKP/simpson/git/crowdsourcing_argumentation/data/lingdata/UKPConvArg1-Full-libsvm.txt"
    
    if os.path.isfile(outputfile):
        
        import numpy as np
        
        featurespresent = []
        with open(outputfile, 'r') as ofh:
            lines = ofh.readlines()
            for l, line in enumerate(lines):
                print "%i out of %i" % (l, len(lines))
                for f in line.split('\t')[1:]:
                    feature = f.split(':')[0]
                    if '#' in feature:
                        continue
                    if int(feature) not in featurespresent:
                        featurespresent.append(int(feature))
                    
        print len(featurespresent)
        print np.max(featurespresent)
        
        os.remove(outputfile)
    
    with open(outputfile, 'a') as ofh: 
        for filename in os.listdir(dirname):
            print "writing file %s" % filename
            with open(dirname + "/" + filename) as fh:
                lines = fh.readlines()
            for line in lines:
                ofh.write(line)