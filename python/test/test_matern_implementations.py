
import numpy as np

# generate some test data -- a large matrix
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

import cProfile

raw = np.random.rand(500, 10000)

K = 1

print('starting...')

pr = cProfile.Profile()
pr.enable()

Ki = np.empty((raw.shape[0] * (raw.shape[0] - 1)) // 2)

for i in range(raw.shape[1]):
    if np.mod(i, 100) == 0:
       print("computing kernel for feature %i" % i)

    xvals = raw[:, i:i + 1]
    xvals = xvals * 3 ** 0.5
    Ki = pdist(xvals, 'euclidean', out=Ki)
    #Ki = np.abs(xvals - xvals.T)

    exp_minusK = np.exp(-Ki)
    Ki = (1.0 + Ki) * exp_minusK

    K *= Ki


def matern1d(xvals):

    xvals = xvals * 3 ** 0.5
    Ki = pdist(xvals, 'euclidean')
    #Ki = np.abs(xvals - xvals.T)

    exp_minusK = np.exp(-Ki)
    Ki = (1.0 + Ki) * exp_minusK
    return Ki

K = squareform(K)

print('done')


pr.disable()

pr.print_stats(sort='calls')

pr.enable()

Kis = Parallel(n_jobs=4, backend='threading')(delayed(matern1d)(raw[:, i:i + 1]) for i in range(raw.shape[1]))
K = np.prod(Kis, axis=0)
K = squareform(K)

print('done')

pr.disable()

pr.print_stats(sort='calls')


dists = pdist((raw), metric='euclidean')
K = dists * np.sqrt(3)
K = (1. + K) * np.exp(-K)

print('done 2')