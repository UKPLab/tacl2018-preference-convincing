'''
Just use this for debugging with a smaller dataset.
'''

import sys
import os

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")
svm_python_path = '~/libsvm-3.22/python'

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow"))

sys.path.append(os.path.expanduser(svm_python_path))

import tests
    
if __name__ == '__main__':
    tests.dataset_increment = 0     
    tests.datasets = ['UKPConvArgStrict']
    tests.feature_types = ['debug'] # can be 'embeddings' or 'ling' or 'both'
    tests.methods = ['SinglePrefGP_weaksprior'] 
    tests.embeddings_types = ['word_mean']#, 'skipthoughts'] # 'siamese-cbow'] 
                
    default_ls_values = tests.run_test_set(test_on_train=True)