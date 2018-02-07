'''
TODO: Check that w_cov and y_cov are block-diagonal in terms of each item/person.
TODO: SVI and non-SVI implementations give different answers
TODO: SVI results seem not to converge with small datasets
TODO: non-SVI results seem to have overly small w and y -- perhaps need broader priors (smaller s)
TODO: SVI results seem to vary hugely between runs or on different computers despite same seeds -- need to verify this is true
TODO: Lower bound can go down with SVI -- what about without SVI? Bug in LB or in the SVI algorithm? 
TODO: Lower bound for noise SVI should be computed inside GP code -- is this possible? I think we can just add the missing cov_mu term.

Preference learning model for identifying relevant input features of items and people, plus finding latent 
characteristics of items and people. Can be used to predict preferences or rank items, therefore could be part of
a recommender system. In this case the method uses both collaborative filtering and item-based similarity.

For preference learning, we use a GP as in Chu and Ghahramani. This has been modified to use VB updates to integrate 
into the complete VB framework and allow an SVI adaptation for scalability. 

For finding the latent factors, we modify the model of Archambeau and Bach 2009 to consider correlations between 
items and people using a GP kernel. We assume only one "view" (from their terminology) that best fits the preferences.
Multiple views could be worthwhile if predicting multiple preference functions?
According to A&B 2009, the generative model for probabilistic projection includes several techniques as special cases:
 - diagonal priors on y gives probabilistic factor analysis
 - isotropic priors give probabilistic PCA
 - our model doesn't allow other specific types, but is intended to be used more generally instead
From A&B'09: "PCA suffers from the fact that each principal component is a linear combination of all the original 
variables. It is thus often difficult to interpret the results." A sparse representation would be easier to interpret. 
The aim is to use as few components as are really necessary, and allow priors to determine a trade-off between sparsity
(and interpretability; possibly also avoidance of overfitting/better generalisation) and accuracy of the low-dimensional
representation (avoiding loss). In preference learning, we would like to be able to predict unseen values in a person-item
matrix when we have limited, noisy data. A sparse representation seems applicable as it would avoid overfitting and 
allow personalised noise models to represent complexity so that the shared latent features are more easily interpretable
as common characteristics. 

Our implementation here is similar to the inverse Gamma prior on the weight precision
proposed by A&B'09, but we use a gamma prior that is conjugate to the Gaussian instead. This makes inference simpler
but may have the disadvantage of not enforcing sparseness so strictly -- it is not clear from A&B'09 why they chose
the non-conjugate option. They also use completely independent scale variables for each weight in the weight matrix,
i.e. for each item x factor pair. We have correlations between items with similar features through a kernel function, 
but we also use a common scale parameter for each feature. This induces sparsity over the features, i.e. reduces the
number of features used but means that all items will have an entry for all the important features. It's unclear what
difference this makes -- perhaps features that are weakly supported by small amounts of data for one item will be pushed
to zero by A&B approach, while our approach will allow them to vary more since the feature is important for other items.
The A&B approach may make more sense for representing rare but important features; our approach would not increase their 
scale so that items that do possess the feature will not be able to have a large value and the feature may disappear?
Empirical tests may be needed here. 

The approach is similar to Khan et al. 2014. "Scalable Collaborative Bayesian Preference Learning" but differs in that
we also place priors over the weights and model correlations between different items and between different people.
Our use of priors that encourage sparseness in the features is also different. 

Observed features -- why is it good to use them as inputs to latent features? 
-- we assume some patterns in the observations are common to multiple people, and these manifest as latent features
-- we can use the GP model to map observations to latent features to handle sparsity of data for each item
and person
-- the GP will model dependencies between the input features
An alternative would be a flat model, where the input features for items were added to columns of w, 
and the input features of people created new rows in y. This may make it easier to learn which features are relevant,
but does not help with sparse features because we could not use a GP to smooth and interpolate between items, so 
would need mode observed preference pairs for each item and person to determine their latent feature values.  

For testing effects of no. inducing points, forgetting rate, update size, delay, it would be useful to see accuracy and 
convergence rate.

Created on 2 Jun 2016

@author: simpson
'''

import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import multivariate_normal as mvn, norm, binom
import logging
from gp_classifier_vb import matern_3_2_from_raw_vals, derivfactor_matern_3_2_from_raw_vals, temper_extreme_probs, \
    fractional_convergence
from gp_pref_learning import GPPrefLearning, get_unique_locations, pref_likelihood
from scipy.linalg import block_diag
from scipy.special import gammaln, psi
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern


def expec_output_scale(shape_s0, rate_s0, N, invK, f_mean, m, invK_f_cov=None, f_cov=None):
    # learn the output scale with VB
    shape_s = shape_s0 + 0.5 * N

    if invK_f_cov is None:
        if f_cov is None:
            logging.error('Provide either invK_f_cov or f_cov')
            return
        invK_f_cov = invK.dot(f_cov)

    invK_expecFF = invK_f_cov + invK.dot((f_mean - m).dot(f_mean.T - m.T))
    rate_s = rate_s0 + 0.5 * np.trace(invK_expecFF)

    return shape_s, rate_s


def lnp_output_scale(shape_s0, rate_s0, shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)

    logprob_s = - gammaln(shape_s0) + shape_s0 * np.log(rate_s0) + (shape_s0 - 1) * Elns - rate_s0 * s
    return logprob_s


def lnq_output_scale(shape_s, rate_s, s=None, Elns=None):
    if s is None:
        s = shape_s / rate_s
    if Elns is None:
        Elns = psi(shape_s) - np.log(rate_s)

    lnq_s = - gammaln(shape_s) + shape_s * np.log(rate_s) + (shape_s - 1) * Elns - rate_s * s
    return lnq_s


def update_gaussian(invK, s, Sigma, x):
    cov = np.linalg.inv((invK * s) + Sigma)
    m = cov.dot(x)
    return m, cov


def svi_update_gaussian(invQi_y, mu0_n, mu_u, K_mm, invK_mm, K_nm, Lambda_factor1, K_nn, invQi, prev_invS, prev_invSm,
                        vb_iter, delay, forgetting_rate, N, update_size):
    Lambda_i = Lambda_factor1.dot(invQi).dot(Lambda_factor1.T)

    # calculate the learning rate for SVI
    rho_i = (vb_iter + delay) ** (-forgetting_rate)
    # print "\rho_i = %f " % rho_i

    # weighting. Lambda and 
    w_i = N / float(update_size)

    # S is the variational covariance parameter for the inducing points, u. 
    # Canonical parameter theta_2 = -0.5 * S^-1.
    # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over 
    # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.  
    invS = (1 - rho_i) * prev_invS + rho_i * (w_i * Lambda_i + invK_mm)

    # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y  
    #     invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * invK_mm.dot(K_im.T).dot(invQi).dot(y)
    invSm = (1 - rho_i) * prev_invSm + w_i * rho_i * Lambda_factor1.dot(invQi_y)

    # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
    # L_invS = cholesky(invS.T, lower=True, check_finite=False)
    # B = solve_triangular(L_invS, invK_mm.T, lower=True, check_finite=False)
    # A = solve_triangular(L_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
    # invK_mm_S = A.T
    S = np.linalg.inv(invS)
    invK_mm_S = invK_mm.dot(S)

    # fhat_u = solve_triangular(L_invS, invSm, lower=True, check_finite=False)
    # fhat_u = solve_triangular(L_invS, fhat_u, lower=True, trans=True, check_finite=False, overwrite_b=True)
    fhat_u = S.dot(invSm)
    fhat_u += mu_u

    # TODO: move the K_mm.T.dot(K_nm.T) computation out    
    covpair_uS = K_nm.dot(invK_mm_S)
    fhat = covpair_uS.dot(invSm) + mu0_n
    if K_nn is None:
        C = None
    else:
        covpair = K_nm.dot(invK_mm)
        C = K_nn + (covpair_uS - covpair.dot(K_mm)).dot(covpair.T)
    return fhat, C, invS, invSm, fhat_u, invK_mm_S, S


def expec_pdf_gaussian(K, invK, Elns, N, s, f, mu, f_cov, mu_cov):
    '''
    Expected value of the PDF with respect to the function values with expectation f, the mean values with expectation
    mu, and the inverse covariance scale with expectation s.
    
    Parameters
    ----------
    
    K : covariance matrix (without scaling)
    invK : inverse of the covariance matrix (without scaling)
    Elns : expected log of the covariance inverse scale factor
    N : number of data points
    s : expected covariance inverse scale factor
    f : expected function values
    mu : expected mean values
    f_cov : covariance of the function values
    mu_cov : covariance of the mean parameters; this is needed if the mean is a model parameter inferred using VB
    '''
    _, logdet_K = np.linalg.slogdet(K)
    logdet_Ks = - np.sum(N * Elns) + logdet_K
    invK_expecF = (s * invK).dot(f_cov + (f - mu).dot((f - mu).T) + mu_cov)
    logpf = 0.5 * (- np.log(2 * np.pi) * N - logdet_Ks - np.trace(invK_expecF))

    return logpf


def expec_q_gaussian(f_cov, D):
    _, logdet_C = np.linalg.slogdet(f_cov)
    logqf = 0.5 * (- np.log(2 * np.pi) * D - logdet_C - D)
    return logqf


# noinspection PyAttributeOutsideInit
class PreferenceComponents(object):
    '''
    Model for analysing the latent personality features that affect each person's preferences. Inference using 
    variational Bayes.
    '''

    def __init__(self, nitem_features, nperson_features=0, shape_s0=1, rate_s0=1, shape_ls=1, rate_ls=100, ls=100,
                 shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20,
                 use_common_mean_t=True, kernel_func='matern_3_2'):
        '''
        Constructor
        dims - ranges for each of the observed features of the objects
        mu0 - initial mean for the latent preference function 
        '''
        self.people = None
        self.nitem_features = nitem_features
        self.nperson_features = nperson_features

        shape_s0 = float(shape_s0)
        rate_s0 = float(rate_s0)

        # too much variance leads to extreme values of E[s], which then leads to bad covariance matrices, then crashes
        self.shape_sw0 = shape_s0
        self.rate_sw0 = rate_s0

        self.shape_sy0 = shape_s0
        self.rate_sy0 = rate_s0

        self.shape_st0 = shape_s0
        self.rate_st0 = rate_s0

        # y has different length-scales because it is over user features space
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls

        if ls is not None:
            self.n_wlengthscales = len(
                np.array([ls]).flatten())  # can pass in a single length scale to be used for all dimensions
        else:
            self.n_wlengthscales = self.nitem_features
        self.ls = ls

        self.shape_lsy = shape_lsy
        self.rate_lsy = rate_lsy
        self.lsy = lsy
        if lsy is not None:
            self.n_ylengthscales = len(
                np.array([lsy]).flatten())  # can pass in a single length scale to be used for all dimensions
        else:
            self.n_ylengthscales = self.nperson_features

        self.t_mu0 = 0

        self.conv_threshold = 1e-3
        self.max_iter_G = 10
        self.max_iter = 1000
        self.min_iter = 1
        self.n_converged = 3  # number of iterations while apparently converged (avoids numerical errors)
        self.vb_iter = 0

        self.verbose = verbose

        self.Nfactors = nfactors

        self.use_t = use_common_mean_t

        self._select_covariance_function(kernel_func)

        self.new_obs = True  # flag to indicate that the model has not yet been fitted

    def _select_covariance_function(self, cov_type):
        self.cov_type = cov_type
        if cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2_from_raw_vals
            self.kernel_der = derivfactor_matern_3_2_from_raw_vals
        # the other kernels no longer work because they need to use kernel functions that work with the raw values
        else:
            logging.error('PreferenceComponents: Invalid covariance type %s' % cov_type)

    # FITTING --------------------------------------------------------------------------------------------------------

    def _scaled_Kw(self, y, y_cov, inv_scale):
        Kw = np.zeros((self.N * self.Npeople, self.N * self.Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + f * self.Npeople
            # I think the y_cov term here may cancel out?
            # Makes no sense to have posterior estimates here for y and priors for w? Perhaps we need the thing above, but with priors? Check the maths.
            # Another possibility: we do need separate GP estimates for y and w, but we did it wrong before. The code below has changed the scaling since our last try to reflect this, but we haven't tested with y_gp.

            #yscaling = np.diag(self.y[f, :] ** 2)# + np.diag(self.y_cov[fidxs, :][:, fidxs]))
            yscaling = y[f:f+1, :].T.dot(y[f:f+1, :]) + y_cov[fidxs, :][:, fidxs]

            # yscaling = np.diag(yscaling)
            #
            # blocks = [self.K * yscaling[p] for p in range(self.Npeople)]
            # Kw_f = block_diag(*blocks)
            #
            yscaling = yscaling[None, :, :, None]
            Kw_f = self.K[:, None, None, :] * yscaling
            Kw_f = Kw_f.reshape(self.N, self.Npeople, self.N * self.Npeople)
            Kw_f = np.swapaxes(Kw_f, 0, 2)
            Kw_f = Kw_f.reshape(self.N * self.Npeople, self.N * self.Npeople)

            Kw_f /= inv_scale[f]

            Kw += Kw_f  # Kjoint_f

        Kw += np.eye(Kw.shape[0]) * 1e-6

        return Kw

    def _init_covariance(self):
        self.K = self.kernel_func(self.obs_coords, self.ls) + np.eye(self.N) * 1e-6
        self.invK = np.linalg.inv(self.K)

        blocks = [self.K for _ in range(self.Nfactors)]
        self.Kw = block_diag(*blocks)
        self.invKw = np.linalg.inv(self.Kw)

        blocks = np.tile(self.K[None, :, :], (self.Npeople, 1, 1))
        self.Kt = block_diag(*blocks)

        if self.person_features is None:
            self.Ky = np.diag(np.ones(self.Npeople * self.Nfactors))  # they are all ones
            self.invKy = np.diag(np.ones(self.Npeople * self.Nfactors))  # they are all ones
        else:
            self.lsy = np.zeros(self.nperson_features) + self.lsy
            self.Ky_block = self.kernel_func(self.person_features, self.lsy) + np.eye(self.Npeople) * 1e-6
            self.invKy_block = np.linalg.inv(self.Ky_block)

            blocks = [self.Ky_block for _ in range(self.Nfactors)]
            self.Ky = block_diag(*blocks)
            self.invKy = np.linalg.inv(self.Ky)


    def _init_w(self):
        self.sw_matrix = np.ones(self.Kw.shape) * self.shape_sw0 / self.rate_sw0

        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them
        # i.e. the cluster identifiability problem
        self.w = mvn.rvs(np.zeros(self.N), self.K, self.Nfactors).reshape(self.Nfactors, self.N).T #np.zeros((self.N, self.Nfactors))
        self.w /= (self.shape_sw/self.rate_sw)[None, :]
        self.wy = np.zeros((self.N * self.Npeople, 1))
        self.w_cov = self.Kw / self.sw_matrix

        self.Sigma_w = np.zeros((self.N, self.N, self.Nfactors))

        self.sprior = self.Nfactors * self.shape_sw0 / self.rate_sw0 * self.shape_sy0 / self.rate_sy0
        self.w_gp = GPPrefLearning(self.nitem_features, 0, self.shape_sw0, 1.0/(self.sprior/self.shape_sw0), self.shape_ls, self.rate_ls, self.ls,
                                   fixed_s=True, kernel_func='pre', use_svi=False)
        # delay=self.delay, forgetting_rate=self.forgetting_rate,
        self.w_gp.max_iter_VB = 1
        self.w_gp.min_iter_VB = 1
        self.w_gp.max_iter_G = self.max_iter_G # G needs to converge within each VB iteration otherwise q(w) is very poor and crashes
        self.w_gp.verbose = self.verbose
        self.w_gp.conv_threshold = 1e-5
        self.w_gp.conv_check_freq = 1

        # intialise Q using the prior covariance
        Kw = self._scaled_Kw(np.zeros((self.Nfactors, self.Npeople)), self.Ky / self.shape_sy0 * self.rate_sy0,
                             self.shape_sw / self.rate_sw)

        self.w_gp.set_training_data(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
                                    mu0=np.zeros((self.N*self.Npeople, 1)), K=Kw,
                                    process_obs=self.new_obs, input_type=self.input_type)

    def _init_t(self):
        self.t = np.zeros((self.N, 1))
        self.st = self.shape_st0 / self.rate_st0

        self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0

        self.t_gp = GPPrefLearning(self.nitem_features, 0, 1, 1, self.shape_ls, self.rate_ls, self.ls,
                                   fixed_s=True, kernel_func='pre', use_svi=False)
        # delay=self.delay, forgetting_rate=self.forgetting_rate,
        self.t_gp.max_iter_VB = 1
        self.t_gp.min_iter_VB = 1
        self.t_gp.max_iter_G = self.max_iter_G # G needs to converge within each VB iteration otherwise q(w) is very poor and crashes
        self.t_gp.verbose = self.verbose
        self.t_gp.conv_threshold = 1e-5
        self.t_gp.conv_check_freq = 1

    def _scaled_Ky(self, w, w_cov, inv_scale):
        # Ky uses same layout as Kw
        Ky = np.zeros((self.N * self.Npeople, self.N * self.Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + f * self.N
            #wscaling = np.diag(self.w[:, f]**2)# + np.diag(self.w_cov[fidxs, :][:, fidxs]))#
            wscaling = w[:, f:f+1].dot(w[:, f:f+1].T) + w_cov[fidxs, :][:, fidxs]
            wscaling = wscaling[:, None, None, :]
            Ky_f = self.Ky_block[None, :, :, None] * wscaling

            # wscaling = np.diag(wscaling)

            # blocks = [self.Ky_block * wscaling[n] for n in range(self.N)]
            # Ky_f = block_diag(*blocks)
            # Ky_f = Ky_f.reshape(self.N * self.Npeople, self.N, self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 0, 2)
            # Ky_f = Ky_f.reshape(self.Npeople, self.N, self.N, self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 2, 3)
            # Ky_f = Ky_f.reshape(self.Npeople, self.N, self.N * self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 0, 2)
            # Ky_f = np.swapaxes(Ky_f, 1, 2)
            # Ky_f = Ky_f.reshape(self.Npeople * self.N, self.N * self.Npeople)
            #
            Ky_f = Ky_f.reshape(self.N, self.Npeople, self.N * self.Npeople)
            Ky_f = np.swapaxes(Ky_f, 0, 2)
            Ky_f = Ky_f.reshape(self.N * self.Npeople, self.N * self.Npeople)

            Ky_f /= inv_scale[f]

            Ky += Ky_f  # Kjoint_f
        Ky += np.eye(Ky.shape[0]) * 1e-6

        return Ky

    def _init_y(self):
        self.y = mvn.rvs(np.zeros(self.Npeople), self.Ky_block, self.Nfactors).reshape(self.Nfactors, self.Npeople)
        self.y /= (self.shape_sy/self.rate_sy)[:, None]

        self.sy_matrix = np.ones(self.invKy.shape) * (self.shape_sy0 / float(self.rate_sy0))
        self.y_cov = self.Ky / self.sy_matrix

        self.Sigma_y = np.zeros((self.Npeople, self.Npeople, self.Nfactors))

        self.y_gp = GPPrefLearning(self.nperson_features, 0, self.shape_sy0, 1.0/(self.sprior/self.shape_sy0), self.shape_ls, self.rate_ls, self.ls,
                                   fixed_s=True, kernel_func='pre', use_svi=False)
        # delay=self.delay, forgetting_rate=self.forgetting_rate,
        self.y_gp.max_iter_VB = 1
        self.y_gp.min_iter_VB = 1
        self.y_gp.max_iter_G = self.max_iter_G # G needs to converge within each VB iteration otherwise q(y) is very poor and crashes
        self.y_gp.verbose = self.verbose
        self.y_gp.conv_threshold = 1e-5
        self.y_gp.conv_check_freq = 1

        Ky = self._scaled_Ky(np.zeros((self.N, self.Nfactors)), self.Kw / self.shape_sw0 * self.rate_sw0,
                             self.shape_sy / self.rate_sy)

        self.y_gp.set_training_data(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
                                    mu0=np.zeros((self.N*self.Npeople, 1)), K=Ky,
                                    process_obs=self.new_obs, input_type=self.input_type)

    def _init_params(self):
        if self.Nfactors is None or self.Npeople < self.Nfactors:  # not enough items or people
            self.Nfactors = self.Npeople

        self._init_covariance()

        # put all prefs into a single GP to get a good initial mean estimate t -- this only makes sense if we can also 
        # estimate w y in a sensibel way, e.g. through factor analysis?
        # self.pref_gp[person].fit(items_1_p, items_2_p, prefs_p, mu0_1=mu0_1, mu0_2=mu0_2, process_obs=self.new_obs)

        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0
        self.shape_sy = np.zeros(self.Nfactors) + self.shape_sy0
        self.rate_sy = np.zeros(self.Nfactors) + self.rate_sy0
        self.shape_st = self.shape_st0
        self.rate_st = self.rate_st0

        self.ls = np.zeros(self.nitem_features) + self.ls

        self._init_w()
        self._init_y()
        self._init_t()

    def fit(self, personIDs=None, items_1_coords=None, items_2_coords=None, item_features=None,
            preferences=None, person_features=None, optimize=False, maxfun=20, use_MAP=False, nrestarts=1,
            input_type='binary', use_lb=False):
        """
        Learn the model with data using variational Bayes.

        :param personIDs: a list of the person IDs of the people who expressed their preferences
        :param items_1_coords: if item_features is None, these should be coordinates of the first items in the pairs
        being compared, otherwise these should be indexes into the item_features vector
        :param items_2_coords: if item_features is None, these should be coordinates of the second items in each pair
        being compared, otherwise these should be indexes into the item_features vector
        :param item_features: feature values for the items. Can be None if the items_x_coords provide the feature values as
        coordinates directly.
        :param preferences: the values, 0 or 1 to express that item 1 was preferred to item 2.
        :param person_features:
        :param optimize:
        :param maxfun:
        :param use_MAP:
        :param nrestarts:
        :param input_type:
        :param use_lb:
        :return:
        """
        if optimize:
            return self._optimize(personIDs, items_1_coords, items_2_coords, item_features, preferences,
                                  person_features,
                                  maxfun, use_MAP, nrestarts, input_type)

        if personIDs is not None:  # process new data
            self.new_obs = True  # there are people we haven't seen before
            # deal only with the original IDs to simplify prediction steps and avoid conversions 

            if item_features is None:
                self.obs_coords, pref_v, pref_u = get_unique_locations(items_1_coords, items_2_coords)
            else:
                self.obs_coords = np.array(item_features, copy=False)
                pref_v = np.array(items_1_coords, copy=False)
                pref_u = np.array(items_2_coords, copy=False)
            self.N = self.obs_coords.shape[0]

            upeople = np.unique(personIDs)
            self.personIDs = personIDs
            if person_features is not None:
                self.person_features = np.array(person_features,
                                                copy=False)  # rows per person, columns for feature values
                self.nperson_features = self.person_features.shape[1]
            else:
                self.person_features = None
                self.nperson_features = 0
            if self.person_features is not None:
                self.Npeople = self.person_features.shape[0]
            else:
                self.Npeople = np.max(upeople).astype(int) + 1

            self.preferences = np.array(preferences, copy=False)

            # IDs must be for unique item-user pairs
            self.pref_v = pref_v + (self.N * self.personIDs)
            self.pref_u = pref_u + (self.N * self.personIDs)

            self.tpref_v = pref_v
            self.tpref_u = pref_u

            # the covariance matrices are pre-computed so obs_coords is not needed
            self.dummy_obs_coords = np.empty((self.N * self.Npeople, 1))
            self.tdummy_obs_coords = np.empty((self.N, 1))
        else:
            self.new_obs = False  # do we have new data? If so, reset everything. If not, don't reset the child GPs.

        self.input_type = input_type
        self._init_params()

        # reset the iteration counters
        self.vb_iter = 0
        diff = np.inf
        old_w = np.inf
        old_y = np.inf
        old_lb = -np.inf
        converged_count = 0
        while ((self.vb_iter < self.min_iter) or (converged_count < self.n_converged)) and (
                self.vb_iter < self.max_iter):

            if self.use_t:
                self._expec_t()

            # find the latent components
            self._expec_w()
            self._expec_y()

            self.new_obs = False # observations have now been processed once, only updates are required

            lb = self.lowerbound()

            if not use_lb:
                converged = fractional_convergence(lb, old_lb, self.conv_threshold, True,
                                                   self.vb_iter, self.verbose,
                                                   'Preference Components VB lower bound')
            else:
                converged_w = fractional_convergence(self.w, old_w, self.conv_threshold, False,
                                                     self.vb_iter, self.verbose,
                                                     'Preference Components VB, w ')
                converged_y = fractional_convergence(self.y, old_y, self.conv_threshold,
                                                     self.vb_iter, self.verbose,
                                                     'Preference Components VB, y ')
                converged = converged_w & converged_y

            old_w = self.w
            old_y = self.y
            old_lb = lb

            self.vb_iter += 1

            if converged:
                converged_count += 1
            elif converged_count > 0:  # reset the convergence count as the difference has increased again
                converged_count -= 1

        logging.debug("Preference personality model converged in %i iterations." % self.vb_iter)

    def _expec_t(self):
        Kt = self.Kt / (self.shape_st / self.rate_st)

        mu0 = self.w.dot(self.y).T.reshape(self.N * self.Npeople, 1)
        #mu0 = np.zeros((self.N * self.Npeople, 1))

        self.t_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
                      mu0=mu0, K=Kt,
                      process_obs=self.new_obs, input_type=self.input_type)

        invQ = self.t_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)
        t_prec = np.sum(np.sum(invQ, 2), 1)  # Npeople x Npeople

        z0 = self.t_gp.forward_model(self.t_gp.obs_f) + self.t_gp.G.dot(self.t_gp.mu0 - self.t_gp.obs_f)
        invQ_f = self.t_gp.G.T.dot(np.diag(1.0 / self.t_gp.Q)).dot(self.t_gp.z - z0)  # this subtracts out the t prior
        x = np.sum(invQ_f.reshape(self.Npeople, self.N), 0)
        x = x.reshape(self.N, 1)

        self.t, self.t_cov = update_gaussian(self.invK, self.st, t_prec, x)


        self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.N,
                                                         self.invK, self.t, np.zeros((self.N, 1)),
                                                         f_cov=self.t_cov)

    def _expec_w(self):
        """
        Compute the expectation over the latent features of the items and the latent personality components
        :return:
        """

        Kw = np.zeros((self.N * self.Npeople, self.N * self.Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + f * self.Npeople
            # I think the y_cov term here may cancel out?
            # Makes no sense to have posterior estimates here for y and priors for w? Perhaps we need the thing above, but with priors? Check the maths.
            # Another possibility: we do need separate GP estimates for y and w, but we did it wrong before. The code below has changed the scaling since our last try to reflect this, but we haven't tested with y_gp.

            #yscaling = np.diag(self.y[f, :] ** 2)# + np.diag(self.y_cov[fidxs, :][:, fidxs]))
            yscaling = self.y[f:f+1, :].T.dot(self.y[f:f+1, :]) + self.y_cov[fidxs, :][:, fidxs]

            # yscaling = np.diag(yscaling)
            #
            # blocks = [self.K * yscaling[p] for p in range(self.Npeople)]
            # Kw_f = block_diag(*blocks)
            #
            yscaling = yscaling[None, :, :, None]
            Kw_f = self.K[:, None, None, :] * yscaling
            Kw_f = Kw_f.reshape(self.N, self.Npeople, self.N * self.Npeople)
            Kw_f = np.swapaxes(Kw_f, 0, 2)
            Kw_f = Kw_f.reshape(self.N * self.Npeople, self.N * self.Npeople)

            Kw_f /= (self.shape_sw[f] / self.rate_sw[f])

            Kw += Kw_f  # Kjoint_f

        Kw += np.eye(Kw.shape[0]) * 1e-6

        t = np.tile(self.t, (self.Npeople, 1))

        self.w_gp.s = 1.0
        self.w_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences, mu0=t, K=Kw,
                      process_obs=False, input_type=self.input_type)

        # compute sigma_w
        invQ = self.w_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)

        w_prec = np.zeros((self.N * self.Nfactors, self.N * self.Nfactors))
        for f in range(self.Nfactors):
            for g in range(self.Nfactors):
                #  is to update each factor in turn, which may do a better job of cancelling out unneeded factors.
                yscaling = self.y[f:f+1, :].T.dot(self.y[g:g+1, :]) + self.y_cov[f * self.Npeople + np.arange(self.Npeople), :]\
                                                                  [:, g * self.Npeople + np.arange(self.Npeople)]
                #yscaling = np.diag(self.y[f, :]**2 + np.diag(self.y_cov[f * self.Npeople + np.arange(self.Npeople), :]\
                #                                                  [:, g * self.Npeople + np.arange(self.Npeople)]))

                Sigma_f_g = np.sum(np.sum(yscaling[None, :, :, None] * invQ, 2), 1) # Npeople x Npeople

                fidxs = np.tile(f * self.N + np.arange(self.N)[:, None], (1, self.N))
                gidxs = np.tile(g * self.N + np.arange(self.N)[None, :], (self.N, 1))
                w_prec[fidxs, gidxs] = Sigma_f_g

                if f == g:
                    self.Sigma_w[:, :, f] = Sigma_f_g

        z0 = self.w_gp.forward_model(self.w_gp.obs_f) + self.w_gp.G.dot(self.w_gp.mu0 - self.w_gp.obs_f)
        invQ_f = self.w_gp.G.T.dot(np.diag(1.0 / self.w_gp.Q)).dot(self.w_gp.z - z0)  # this subtracts out the t prior
        x = self.y.dot(invQ_f.reshape(self.Npeople, self.N))
        x = x.reshape(self.N * self.Nfactors, 1)

        # The prediction of w using this method is half the magnitude of obs_f when we have 1 person, 1 factor, y=1
        # The values of the covariance are also half the size.
        # This is correct because the ratio of (y*y + var(y)) / y = 2? The prior covariance for w_gp is twice as large as for the update below,
        # and invQ is similarly divided, mqking w_prec half the size of Q.  Why isn't the resulting self.w 2**2 times smaller
        # than self.w_gp.obs_f? Because of the y*y terms cancelling, that leads us to not needing var(y) in the lines above.
        self.w, self.w_cov = update_gaussian(self.invKw, self.sw_matrix, w_prec, x)
        self.w = np.reshape(self.w, (self.Nfactors, self.N)).T  # w is N x Nfactors

        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + (self.N * f)
            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, self.N,
                                                                   self.invK, self.w[:, f:f + 1], np.zeros((self.N, 1)),
                                                                   f_cov=self.w_cov[fidxs, :][:, fidxs])

            self.sw_matrix[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]

    def _expec_y(self):
        """
        Compute expectation over the personality components using VB

        # Still a problem when expec_w is switched off!
        # When y and w are both fixed, the likelihod still goes down -- why?
        TODO: reinstate SVI

        :return:
        """

        # Ky uses same layout as Kw
        Ky = np.zeros((self.N * self.Npeople, self.N * self.Npeople))

        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + f * self.N
            #wscaling = np.diag(self.w[:, f]**2)# + np.diag(self.w_cov[fidxs, :][:, fidxs]))#
            wscaling = self.w[:, f:f+1].dot(self.w[:, f:f+1].T) + self.w_cov[fidxs, :][:, fidxs]
            wscaling = wscaling[:, None, None, :]
            Ky_f = self.Ky_block[None, :, :, None] * wscaling

            # wscaling = np.diag(wscaling)

            # blocks = [self.Ky_block * wscaling[n] for n in range(self.N)]
            # Ky_f = block_diag(*blocks)
            # Ky_f = Ky_f.reshape(self.N * self.Npeople, self.N, self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 0, 2)
            # Ky_f = Ky_f.reshape(self.Npeople, self.N, self.N, self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 2, 3)
            # Ky_f = Ky_f.reshape(self.Npeople, self.N, self.N * self.Npeople)
            # Ky_f = np.swapaxes(Ky_f, 0, 2)
            # Ky_f = np.swapaxes(Ky_f, 1, 2)
            # Ky_f = Ky_f.reshape(self.Npeople * self.N, self.N * self.Npeople)
            #
            Ky_f = Ky_f.reshape(self.N, self.Npeople, self.N * self.Npeople)
            Ky_f = np.swapaxes(Ky_f, 0, 2)
            Ky_f = Ky_f.reshape(self.N * self.Npeople, self.N * self.Npeople)

            Ky_f /= (self.shape_sy[f] / self.rate_sy[f])

            Ky += Ky_f  # Kjoint_f
        Ky += np.eye(Ky.shape[0]) * 1e-6

        t = np.tile(self.t, (self.Npeople, 1))

        self.y_gp.s = 1.0
        self.y_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences, mu0=t, K=Ky,
                     process_obs=False, input_type=self.input_type)

        # This is to compute q(y)
        invQ = self.y_gp.get_obs_precision()
        invQ = invQ.reshape(self.Npeople * self.N, self.Npeople, self.N)
        invQ = np.swapaxes(invQ, 0, 2).reshape(self.N, self.Npeople, self.Npeople, self.N)

        y_prec = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.Npeople))
        for f in range(self.Nfactors):
            w_cov_f = self.w_cov[f * self.N + np.arange(self.N), :]
            for g in range(self.Nfactors):
                wscaling = w_cov_f[:, g * self.N + np.arange(self.N)] + self.w[:, f:f+1].dot(self.w[:, g:g+1].T)

                Sigma_f_g = np.sum(np.sum(wscaling[:, None, None, :] * invQ, 3), 0) # Npeople x Npeople

                fidxs = np.tile(f * self.Npeople + np.arange(self.Npeople)[:, None], (1, self.Npeople))
                gidxs = np.tile(g * self.Npeople + np.arange(self.Npeople)[None, :], (self.Npeople, 1))

                y_prec[fidxs, gidxs] = Sigma_f_g

                if f == g:
                    self.Sigma_y[:, :, f] = Sigma_f_g

        z0 = self.y_gp.forward_model(self.y_gp.obs_f) + self.y_gp.G.dot(self.y_gp.mu0 - self.y_gp.obs_f)
        invQ_f = self.y_gp.G.T.dot(np.diag(1.0 / self.y_gp.Q)).dot(self.y_gp.z - z0)  # this subtracts out the t prior
        x = self.w.T.dot(invQ_f.reshape(self.Npeople, self.N).T) # here we sum over items
        x = x.reshape(self.Npeople * self.Nfactors, 1)

        self.y, self.y_cov = update_gaussian(self.invKy, self.sy_matrix, y_prec, x)
        self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))  # y is Npeople x Nfactors

        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + (self.Npeople * f)
            self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.Npeople,
                                                                   self.invKy_block, self.y[f:f + 1, :].T,
                                                                   np.zeros((self.Npeople, 1)),
                                                                   f_cov=self.y_cov[fidxs, :][:, fidxs])

            self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]  # sy_rows

    def lowerbound(self):
        rho = self.predict(self.personIDs, self.tpref_v, self.tpref_u, self.obs_coords, self.person_features)
        rho = temper_extreme_probs(rho)
        logrho = np.log(rho)
        lognotrho = np.log(1 - rho)
        # print(logrho[10:20])

        # # Uncomment entire block to use sampling instead of the analytical estimates of rho -- useful to debug predict
        # nsamples = 10000
        #
        # # any way to replace this sampling step? It is very inaccurate.
        #
        # w_samples = mvn.rvs(mean=self.w.T.reshape(self.N * self.Nfactors),
        #                              cov=self.w_cov, size=(nsamples)).T
        #
        # y_samples = mvn.rvs(mean=self.y.reshape(self.Npeople * self.Nfactors),
        #                              cov=self.y_cov, size=(nsamples)).T
        # #
        # # w_samples = np.random.normal(loc=self.w.T.reshape(self.N * self.Nfactors)[:, None],
        # #                              scale=np.sqrt(np.diag(self.w_cov))[:, None], size=(self.N*self.Nfactors, nsamples))
        # #
        # # y_samples = np.random.normal(loc=self.y.reshape(self.Npeople * self.Nfactors)[:, None],
        # #                              scale=np.sqrt(np.diag(self.y_cov))[:, None], size=(self.Npeople*self.Nfactors, nsamples))
        #
        # if self.use_t:
        #     t_samples = np.random.normal(loc=self.t,
        #                              scale=np.sqrt(np.diag(self.t_cov))[:, None],
        #                              size=(self.N, nsamples))
        # else:
        #     t_samples = np.zeros((self.N, nsamples))
        #
        # w_samples = w_samples.reshape(self.Nfactors, self.N, nsamples)
        # y_samples = y_samples.reshape(self.Nfactors, self.Npeople, nsamples)
        #
        # f_samples = [(w_samples[:, :, i].T.dot(y_samples[:, :, i]) + t_samples[:, None, i])
        #                  .T.reshape(self.N * self.Npeople) for i in range(nsamples)]
        # f_samples = np.array(f_samples).T
        #
        # g_f = (f_samples[self.pref_v, :] - f_samples[self.pref_u, :]) / np.sqrt(2)
        # phi = norm.cdf(g_f)  # the probability of the actual observation, which takes g_f as a parameter. In the
        #
        # rho = np.mean(phi, axis=1)[:, np.newaxis]
        #
        # phi = temper_extreme_probs(phi)
        #
        # phi = np.log(phi)
        # notphi = np.log(1 - phi)
        #
        # logrho = np.mean(phi, axis=1)[:, np.newaxis]
        # lognotrho = np.mean(notphi, axis=1)[:, np.newaxis]
        # print(logrho[10:20])

        data_ll = self.w_gp.data_ll(logrho, lognotrho)

        Elnsw = psi(self.shape_sw) - np.log(self.rate_sw)
        Elnsy = psi(self.shape_sy) - np.log(self.rate_sy)
        if self.use_t:
            Elnst = psi(self.t_gp.shape_s) - np.log(self.t_gp.rate_s)
            st = self.t_gp.s
        else:
            Elnst = 0
            st = 1

        sw = self.shape_sw / self.rate_sw
        sy = self.shape_sy / self.rate_sy

        # the parameter N is not multiplied here by Nfactors because it will be multiplied by the s value for each
        # factor and summed inside the function
        logpw = expec_pdf_gaussian(self.Kw, self.invKw, Elnsw, self.N, self.sw_matrix,
                                   self.w.T.reshape(self.N * self.Nfactors, 1), 0, self.w_cov, 0)
        logqw = expec_q_gaussian(self.w_cov, self.N * self.Nfactors)

        if self.use_t:
            logpt = expec_pdf_gaussian(self.K, self.invK, Elnst, self.N, st, self.t, self.t_mu0,
                                       self.t_cov, 0)
            logqt = expec_q_gaussian(self.t_cov, self.N)
        else:
            logpt = 0
            logqt = 0

        logpy = expec_pdf_gaussian(self.Ky, self.invKy, Elnsy, self.Npeople, self.sy_matrix,
                                   self.y.reshape(self.Npeople * self.Nfactors, 1), 0, self.y_cov, 0)
        logqy = expec_q_gaussian(self.y_cov, self.Npeople * self.Nfactors)

        logps_y = 0
        logqs_y = 0
        logps_w = 0
        logqs_w = 0
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f],
                                        Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])

            logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f], sy[f],
                                        Elnsy[f])
            logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f], sy[f], Elnsy[f])

        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst)
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)

        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy + logps_y - logqs_y
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = data_ll + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw / self.rate_sw))
            logging.debug('s_y=%s' % (self.shape_sy / self.rate_sy))
            logging.debug('s_t=%.2f' % (self.shape_st / self.rate_st))

        logging.debug('likelihood=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (data_ll, w_terms, y_terms, t_terms))

        if self.verbose:
            logging.debug("Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb))

        logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
        logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
        logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))

        return lb

    # PREDICTION --------------------------------------------------------------------------------------------------

    def predict(self, personids, items_1_coords, items_2_coords, item_features=None, person_features=None):

        if item_features is None:
            coords_1 = items_1_coords
            coords_2 = items_2_coords
        else:
            coords_1 = item_features[items_1_coords]
            coords_2 = item_features[items_2_coords]

        if person_features is None and self.person_features is not None:
            logging.debug('No person features provided -- assuming same people as during training')
            person_features = self.person_features

        coords, item_0_idxs, item_1_idxs = get_unique_locations(coords_1, coords_2)
        personids = np.array(personids)

        Npeople = person_features.shape[0]
        N = coords.shape[0]

        # set personids and itemids to None to compute all pairs
        mu, f_cov = self.predict_f(coords, person_features, return_cov=True)
        mu = mu.T.reshape(N * Npeople, 1)

        pref_v = item_0_idxs + (N * personids)
        pref_u = item_1_idxs + (N * personids)

        predicted_prefs = pref_likelihood(mu, f_cov[pref_v, pref_v] + f_cov[pref_u, pref_u] - f_cov[pref_v, pref_u]
                                          - f_cov[pref_u, pref_v], subset_idxs=[], v=pref_v, u=pref_u)

        return predicted_prefs

    def predict_f_item_person(self, itemids, personids, item_features, person_features=None, return_cov=False):
        if return_cov:
            predicted_f, cov_f = self.predict_f(item_features, person_features, return_cov)
            predicted_f = predicted_f[itemids, personids]
            idxs = np.ravel_multi_index((itemids, personids), (predicted_f.shape[0], predicted_f.shape[1]))
            cov_f = cov_f[idxs, :][:, idxs]
            return predicted_f, cov_f
        else:
            predicted_f = self.predict_f(item_features, person_features, return_cov)
            predicted_f = predicted_f[itemids, personids]
            return predicted_f

    def predict_f(self, item_features, person_features=None, return_cov=False):
        y, cov_y = self._predict_y(person_features)

        t, w, cov_t, cov_w = self._predict_w_t(item_features)

        N = item_features.shape[0]
        Npeople = y.shape[1]

        predicted_f = (w.dot(y) + t)

        if not return_cov:
            return predicted_f

        cov_f = np.tile(cov_t, (Npeople, Npeople))

        # covariance of a product of two independent gaussians (product-Gaussian distribution)
        for f in range(self.Nfactors):
            cov_f += (cov_y[f][:, None, :, None] * cov_w[f][None, :, None, :]).reshape(N * Npeople, N * Npeople)

            yscaling = y[f:f + 1, :].T.dot(y[f:f + 1, :])
            cov_wf = cov_w[f][None, :, None, :] * yscaling[:, None, :, None]
            cov_wf = cov_wf.reshape(N * Npeople, Npeople * N)

            wscaling = (w[:, f:f + 1].dot(w[:, f:f + 1].T))
            cov_yf = cov_y[f][:, None, :, None,] * wscaling[None, :, None, :]
            cov_yf = cov_yf.reshape(N * Npeople, Npeople * N)

            cov_f += cov_wf + cov_yf

        return predicted_f, cov_f

    def _predict_w_t(self, coords_1):
        # kernel between pidxs and t
        K = self.kernel_func(coords_1, self.ls, self.obs_coords)
        K_starstar = self.kernel_func(coords_1, self.ls, coords_1)
        covpair = K.dot(self.invK)
        N = coords_1.shape[0]

        if self.use_t:
            # use kernel to compute t
            invKt = self.invK.dot(self.t)
            t_out = K.dot(invKt)

            cov_t = K_starstar + covpair.dot(self.t_cov + self.K * self.t_gp.s).dot(covpair.T)
        else:
            t_out = np.zeros((N, 1))

            cov_t = np.zeros((N, N))

        w_out = K.dot(self.invK.dot(self.w))  # N x Nfactors

        cov_w = np.zeros((self.Nfactors, N, N))
        for f in range(self.Nfactors):
            fidxs = np.arange(self.N) + self.N * f
            cov_w[f] = K_starstar  * self.rate_sw[f] / self.shape_sw[f] + \
               covpair.dot(self.w_cov[fidxs, :][:, fidxs] - self.K * self.rate_sw[f] / self.shape_sw[f]).dot(covpair.T)

        return t_out, w_out, cov_t, cov_w

    def _predict_y(self, person_features):
        if person_features is None:  # predict for the input people
            y_out = self.y
            cov_y = np.array([np.diag(self.y_cov[f])[:, None] for f in range(self.Nfactors)])
            return y_out, cov_y

        Ky = self.kernel_func(person_features, self.lsy, self.person_features)
        Ky_starstar = self.kernel_func(person_features, self.lsy, person_features)
        covpair = Ky.dot(self.invKy_block)
        Npeople = person_features.shape[0]

        y_out = Ky.dot(self.invKy_block.dot(self.y.T)).T  # Nfactors x N

        cov_y = np.zeros((self.Nfactors, Npeople, Npeople))
        for f in range(self.Nfactors):
            fidxs = np.arange(self.Npeople) + self.Npeople * f
            cov_y[f] = Ky_starstar * self.rate_sy[f] / self.shape_sy[f] + covpair.dot(self.y_cov[fidxs, :][:, fidxs]
                                                    - self.Ky_block * self.rate_sy[f] / self.shape_sy[f]).dot(covpair.T)

        return y_out, cov_y

    # OPTIMIZATION ------------------------------------------------------------------------------------------------

    def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None,
                  maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):

        max_iter = self.max_iter
        self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features,
                 input_type=input_type)
        self.max_iter = max_iter

        min_nlml = np.inf
        best_opt_hyperparams = None
        best_iter = -1

        logging.debug("Optimising item length-scale for all dimensions")

        nfits = 0  # number of calls to fit function

        # optimise each length-scale sequentially in turn
        final_r = 0
        for r in range(nrestarts):
            # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML 
            # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg, 
            # or trust-ncg, which may be faster still?
            if person_features is None:
                initialguess = np.log(self.ls)
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, use_MAP,),
                               jac=self.nml_jacobian, method='L-BFGS-B',
                               options={'maxiter': maxfun, 'gtol': 0.1 / self.nitem_features})
            else:
                initialguess = np.append(np.log(self.ls), np.log(self.lsy))
                logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
                logging.debug("Initial person length-scale guess in restart %i: %s" % (r, self.lsy))
                res = minimize(self.neg_marginal_likelihood, initialguess, args=('both', -1, use_MAP,),
                               jac=self.nml_jacobian, method='L-BFGS-B',
                               options={'maxiter': maxfun, 'gtol': 0.1 / self.nitem_features})

            opt_hyperparams = res['x']
            nlml = res['fun']
            nfits += res['nfev']

            if nlml < min_nlml:
                min_nlml = nlml
                best_opt_hyperparams = opt_hyperparams
                best_iter = r

            # choose a new lengthscale for the initial guess of the next attempt
            if r < nrestarts - 1:
                self.ls = gamma.rvs(self.shape_ls, scale=1.0 / self.rate_ls, size=len(self.ls))
                if person_features is not None:
                    self.lsy = gamma.rvs(self.shape_lsy, scale=1.0 / self.rate_lsy, size=len(self.lsy))

            final_r = r

        if best_iter < final_r:
            # need to go back to the best result
            if person_features is None:  # don't do this if further optimisation required anyway
                self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, use_MAP=False)

        logging.debug(
            "Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
        if self.person_features is not None:
            logging.debug(
                "Chosen person length-scale %s, used %i evals of NLML over %i restarts" % (self.lsy, nfits, nrestarts))

        logging.debug("Optimal hyper-parameters: item = %s, person = %s" % (self.ls, self.lsy))
        return self.ls, self.lsy, -min_nlml  # return the log marginal likelihood

    def neg_marginal_likelihood(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf

        # set the correct hyperparameters according to lstype
        if lstype == 'item':
            if dimension == -1 or self.n_wlengthscales == 1:
                self.ls[:] = np.exp(hyperparams)
            else:
                self.ls[dimension] = np.exp(hyperparams)

        elif lstype == 'person':
            if dimension == -1 or self.n_ylengthscales == 1:
                self.lsy[:] = np.exp(hyperparams)
            else:
                self.lsy[dimension] = np.exp(hyperparams)

        elif lstype == 'fa':
            new_Nfactors = int(np.round(np.exp(hyperparams)))

        elif lstype == 'both' and dimension <= 0:  # can be zero if single length scales or -1 to do all
            # person and item
            self.ls[:] = np.exp(hyperparams[:self.nitem_features])
            self.lsy[:] = np.exp(hyperparams[self.nitem_features:])

        else:
            logging.error("Invalid length-scale type for optimization.")

        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf

        # make sure we start again -- fit should set the value of parameters back to the initial guess
        if lstype != 'fa' or new_Nfactors != self.Nfactors:  # don't rerun if the number of factors is same.
            self.fit()
        marginal_log_likelihood = self.lowerbound()
        if use_MAP:
            log_model_prior = self.ln_modelprior()
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood

        if lstype == 'person':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.lsy))
            else:
                logging.debug(
                    "LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.lsy[dimension]))
        elif lstype == 'item':
            if dimension == -1:
                logging.debug("LML: %f, %s length-scales = %s" % (lml, lstype, self.ls))
            else:
                logging.debug(
                    "LML: %f, %s length-scale for dim %i = %.3f" % (lml, lstype, dimension, self.ls[dimension]))
        elif lstype == 'both':
            logging.debug("LML: %f, item length-scales = %s, person length-scales = %s" % (lml, self.ls, self.lsy))
        return -lml

    def _gradient_dim(self, lstype, d, dimension):
        der_logpw_logqw = 0
        der_logpy_logqy = 0
        der_logpt_logqt = 0
        der_logpf_logqf = 0

        # compute the gradient. This should follow the MAP estimate from chu and ghahramani. 
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate 
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):

            dKdls = self.K * self.kernel_der(self.obs_coords, self.ls, dimension)
            invK_w = self.invK.dot(self.w)  # N x Nfactors

            for f in range(self.Nfactors):
                fidxs = np.arange(self.N) + (self.N * f)

                swf = self.shape_sw[f] / self.rate_sw[f]
                invKs_Cf = swf * self.invK.dot(self.w_cov[fidxs, :][:, fidxs])
                invK_wf = invK_w[:, f:f + 1]

                Sigma = self.Sigma_w[:, :, f]

                der_logpw_logqw += 0.5 * (invK_wf.T.dot(dKdls).dot(invK_wf) * swf -
                                          np.trace(invKs_Cf.dot(Sigma).dot(dKdls / swf)))

            if self.use_t:
                invKs_t = self.invK.dot(self.t) * self.t_gp.s
                invKs_C = self.t_gp.s * self.invK.dot(self.t_cov)

                der_logpt_logqt = 0.5 * (invKs_t.T.dot(dKdls).dot(invKs_t) -
                                         np.trace(invKs_C.dot(self.t_gp.get_obs_precision()).dot(dKdls / self.t_gp.s)))

        elif (lstype == 'person' or (
                lstype == 'both' and d >= self.nitem_features)) and self.person_features is not None:
            dKdls = self.Ky * self.kernel_der(self.person_features, self.lsy, dimension)
            invK_y = self.invKy.dot(self.y.T)  # Npeople x Nfactors

            for f in range(self.Nfactors):
                fidxs = np.arange(self.Npeople) + (self.Npeople * f)

                syf = self.shape_sy[f] / self.rate_sy[f]
                invKs_Cf = syf * self.invKy.dot(self.y_cov[fidxs, :][:, fidxs])
                invK_yf = invK_y[:, f:f + 1]

                Sigma = self.Sigma_y[:, :, f]

                der_logpy_logqy += 0.5 * (invK_yf.T.dot(dKdls).dot(invK_yf) * syf -
                                          np.trace((invKs_Cf * Sigma).dot(dKdls / syf)))

        return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf

    def nml_jacobian(self, hyperparams, lstype, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf

        needs_fitting = self.new_obs

        if lstype == 'item':
            if dimension == -1 or self.n_wlengthscales == 1:
                if np.any(np.abs(self.ls - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.ls[:] = np.exp(hyperparams)
                dimensions = np.arange(len(self.ls))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.ls[dimension] = np.exp(hyperparams)
                dimensions = [dimension]
        elif lstype == 'person':
            if dimension == -1 or self.n_ylengthscales == 1:
                if np.any(np.abs(self.lsy - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.lsy[:] = np.exp(hyperparams)
                dimensions = np.arange(len(self.lsy))
            else:
                if np.any(np.abs(self.ls[dimension] - np.exp(hyperparams)) > 1e-4):
                    needs_fitting = True
                    self.lsy[dimension] = np.exp(hyperparams)
                dimensions = [dimension]
        elif lstype == 'both' and dimension <= 0:

            hyperparams_w = hyperparams[:self.nitem_features]
            hyperparams_y = hyperparams[self.nitem_features:]

            if np.any(np.abs(self.ls - np.exp(hyperparams_w)) > 1e-4):
                needs_fitting = True
                self.ls[:] = np.exp(hyperparams_w)

            if np.any(np.abs(self.lsy - np.exp(hyperparams_y)) > 1e-4):
                needs_fitting = True
                self.lsy[:] = np.exp(hyperparams_y)

            dimensions = np.append(np.arange(len(self.ls)), np.arange(len(self.lsy)))
        else:
            logging.error("Invalid optimization setup.")

        if np.any(np.isinf(self.ls)):
            return np.inf
        if np.any(np.isinf(self.lsy)):
            return np.inf

        # make sure we start again -- fit should set the value of parameters back to the initial guess
        if needs_fitting:
            self.fit()

        # num_jobs = multiprocessing.cpu_count()
        # mll_jac = Parallel(n_jobs=num_jobs)(delayed(self._gradient_dim)(lstype, d, dim)
        #                                      for d, dim in enumerate(dimensions))
        # mll_jac = np.array(mll_jac, order='F')
        mll_jac = np.zeros(len(dimensions), dtype=float)
        for d, dim in enumerate(dimensions):
            mll_jac[d] = self._gradient_dim(lstype, d, dim)

        if len(mll_jac) == 1:  # don't need an array if we only compute for one dimension
            mll_jac = mll_jac[0]
        elif (lstype == 'item' and self.n_wlengthscales == 1) or (lstype == 'person' and self.n_ylengthscales == 1):
            mll_jac = np.sum(mll_jac)
        elif lstype == 'both':
            if self.n_wlengthscales == 1:
                mll_jac[:self.nitem_features] = np.sum(mll_jac[:self.nitem_features])
            if self.n_ylengthscales == 1:
                mll_jac[self.nitem_features:] = np.sum(mll_jac[self.nitem_features:])

        if use_MAP:  # gradient of the log prior
            log_model_prior_grad = self.ln_modelprior_grad()
            lml_jac = mll_jac + log_model_prior_grad
        else:
            lml_jac = mll_jac
        logging.debug("Jacobian of LML: %s" % lml_jac)
        if self.verbose:
            logging.debug("...with item length-scales = %s, person length-scales = %s" % (self.ls, self.lsy))
        return -lml_jac  # negative because the objective function is also negated

    def ln_modelprior(self):
        # Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls * np.log(self.rate_ls) \
                 + (self.shape_ls - 1) * np.log(self.ls) - self.ls * self.rate_ls

        lnp_gpy = - gammaln(self.shape_lsy) + self.shape_lsy * np.log(self.rate_lsy) \
                  + (self.shape_lsy - 1) * np.log(self.lsy) - self.lsy * self.rate_lsy

        return np.sum(lnp_gp) + np.sum(lnp_gpy)

        # UTILITY FUNCTIONS ----------------------------------------------------------------------------------------------

    def pickle_me(self, filename):
        import pickle
        from copy import deepcopy
        with open(filename, 'w') as fh:
            m2 = deepcopy(self)
            for p in m2.pref_gp:
                m2.pref_gp[p].kernel_func = None  # have to do this to be able to pickle
            pickle.dump(m2, fh)

        #
# class PreferenceNoComponentFactors(PreferenceComponents):
#     '''
#     Class for preference learning with multiple users, where each user has a GP. No sharing of information between users
#     and no latent components.
#     '''
#     def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1,
#                  shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20,
#                  use_common_mean_t=True, kernel_func='matern_3_2',
#                  max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
#
#
#         PreferenceComponents.__init__(self, nitem_features, nperson_features, mu0, shape_s0, rate_s0,
#                  shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors,
#                  use_common_mean_t, kernel_func,
#                  max_update_size, ninducing, forgetting_rate, delay)
#         self.use_t = False
#
#     def _init_w(self):
#         self.w = np.zeros((self.N, self.Nfactors))
#
#     def _init_y(self):
#         self.y = np.zeros((self.Nfactors, self.Npeople))
#
#     def _predict_w_t(self, coords_1):
#         # kernel between pidxs and t
#         w1 = np.zeros((coords_1.shape[0], self.Nfactors))
#         t1 = np.zeros((coords_1.shape[0], 1))
#
#         return t1, w1
#
#     def _predict_y(self, _, Npeople):
#         return super(PreferenceNoComponentFactors, self)._predict_y(None, Npeople)
#
#     def _gradient_dim(self, lstype, d, dimension):
#         der_logpf_logqf = 0
#
#         if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
#             for p in self.pref_gp:
#                 der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
#
#         return der_logpf_logqf
#
#     def _expec_f_p(self, p, mu0_output):
#         f, _ = self.pref_gp[p].predict_f(
#                 out_idxs=self.coordidxs[p] if self.vb_iter==0 else None,
#                 out_feats=self.obs_coords if self.vb_iter==0 else None,
#                 mu0_output=mu0_output,
#                 reuse_output_kernel=True)
#         self.f[self.coordidxs[p], p] = f.flatten()
#
#     def _expec_w(self):
#         return
#
#     def lowerbound(self):
#         f_terms = 0
#
#         for p in self.pref_gp:
#             f_terms += self.pref_gp[p].lowerbound()
#             if self.verbose:
#                 logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
#
#         lb = f_terms
#
#         if self.verbose:
#             logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
#
#         return lb
#
# class PreferenceComponentsFA(PreferenceComponents):
#     # Factor Analysis
#     def _init_w(self):
#         self.fa = FactorAnalysis(n_components=self.Nfactors)
#         self.w = np.zeros((self.N, self.Nfactors))
#
#     def _init_y(self):
#         self.y = np.ones((self.Nfactors, self.Npeople))
#
#     def _optimize(self, personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features=None,
#                  maxfun=20, use_MAP=False, nrestarts=1, input_type='binary'):
#
#         max_iter = self.max_iter
#         self.fit(personIDs, items_1_coords, items_2_coords, item_features, preferences, person_features, input_type=input_type)
#         self.max_iter = max_iter
#
#         min_nlml = np.inf
#         best_opt_hyperparams = None
#         best_iter = -1
#
#         logging.debug("Optimising item length-scale for all dimensions")
#
#         nfits = 0 # number of calls to fit function
#
#         # optimise each length-scale sequentially in turn
#         for r in range(nrestarts):
#             # try to do it using the conjugate gradient method instead. Requires Jacobian (gradient) of LML
#             # approximation. If we also have Hessian or Hessian x arbitrary vector p, we can use Newton-CG, dogleg,
#             # or trust-ncg, which may be faster still?
#             initialguess = np.log(self.ls)
#             logging.debug("Initial item length-scale guess in restart %i: %s" % (r, self.ls))
#             res = minimize(self.neg_marginal_likelihood, initialguess, args=('item', -1, use_MAP,),
#                    jac=self.nml_jacobian, method='L-BFGS-B', options={'maxiter':maxfun, 'gtol': 0.1 / self.nitem_features})
#
#             opt_hyperparams = res['x']
#             nlml = res['fun']
#             nfits += res['nfev']
#
#             if nlml < min_nlml:
#                 min_nlml = nlml
#                 best_opt_hyperparams = opt_hyperparams
#                 best_iter = r
#
#             # choose a new lengthscale for the initial guess of the next attempt
#             if r < nrestarts - 1:
#                 self.ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls, size=len(self.ls))
#
#         if best_iter < r:
#             # need to go back to the best result
#             self.neg_marginal_likelihood(best_opt_hyperparams, 'item', -1, use_MAP=False)
#
#         logging.debug("Chosen item length-scale %s, used %i evals of NLML over %i restarts" % (self.ls, nfits, nrestarts))
#
#         initialguess = np.log(self.Nfactors)
#         res = minimize(self.neg_marginal_likelihood, initialguess, args=('fa', -1, use_MAP,),
#                method='Nelder-Mead', options={'maxfev':maxfun, 'xatol':np.mean(self.ls) * 1e100, 'return_all':True})
#         min_nlml = res['fun']
#         logging.debug("Optimal number of factors = %s, with initialguess=%i and %i function evals" % (self.Nfactors,
#                                                                        int(np.exp(initialguess)), res['nfev']))
#
#         logging.debug("Optimal hyper-parameters: item = %s" % (self.ls))
#         return self.ls, self.lsy, -min_nlml
#
#     def _gradient_dim(self, lstype, d, dimension):
#         der_logpw_logqw = 0
#         der_logpy_logqy = 0
#         der_logpt_logqt = 0
#         der_logpf_logqf = 0
#
#         # compute the gradient. This should follow the MAP estimate from chu and ghahramani.
#         # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate
#         # gradient when converged due to the coordinate ascent method.
#         if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
#             for p in self.pref_gp:
#                 der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
#
#         elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):
#             return 0
#
#         return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf
#
#     def _predict_w_t(self, coords_1):
#         if self.cov_type == 'matern_3_2':
#             kernel = Matern(self.ls)
#         else:
#             logging.error('Kernel not implemented for FA')
#             return 0
#
#         w1 = np.zeros((coords_1.shape[0], self.Nfactors))
#         for f in range(self.Nfactors):
#             w_gp = GPR(kernel=kernel, optimizer=None)
#             w_gp.fit(self.obs_coords, self.w[:, f])
#             w1[:, f] = w_gp.predict(coords_1, return_std=False)
#
#         t_gp = GPR(kernel, optimizer=None)
#         t_gp.fit(self.obs_coords, self.t)
#         t1 = t_gp.predict(coords_1, return_std=False)
#
#         return t1, w1, np.zeros((self.N, self.N)), np.zeros((self.N*self.Nfactors, self.N*self.Nfactors))
#
#     def _predict_y(self, person_features, Npeople):
#
#         y1 = np.zeros((self.Nfactors, Npeople))
#         if person_features is None:
#             return y1
#
#         if self.cov_type == 'matern_3_2':
#             kernel = Matern(self.ls)
#         else:
#             logging.error('Kernel not implemented for FA')
#             return 0
#
#         for f in range(self.Nfactors):
#             y_gp = GPR(kernel=kernel, optimizer=None)
#             y_gp.fit(self.person_features, self.y[f, :])
#             y1[f, :] = y_gp.predict(person_features, return_std=False)
#
#         return y1, np.zeros((self.Npeople, self.Npeople))
#
#     def _expec_w(self):
#         '''
#         Compute the expectation over the latent features of the items and the latent personality components
#         '''
#         self.y = self.fa.fit_transform(self.f.T).T
#         self.w = self.fa.components_.T
#         self.wy = self.w.dot(self.y)
#         return
#
#     def _expec_t(self):
#         self.t = self.fa.mean_[:, np.newaxis]
#         return
#
#     def lowerbound(self):
#         f_terms = 0
#
#         for p in self.pref_gp:
#             f_terms += self.pref_gp[p].lowerbound()
#             if self.verbose:
#                 logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
#
#         lb = np.sum(self.fa.score_samples(self.f.T)) + f_terms
#         if self.verbose:
#             logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
#         return lb
#
# class PreferenceComponentsSVI(PreferenceComponents):
#
#     def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1,
#                  shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20,
#                  use_common_mean_t=True, kernel_func='matern_3_2',
#                  max_update_size=10000, ninducing=500, forgetting_rate=0.9, delay=1.0):
#
#         self.use_svi_people = False # this gets switched on later if we have features and correlations between people
#         self.t_mu0_u = 0
#
#         super(PreferenceComponentsSVI, self).__init__(nitem_features, nperson_features, mu0, shape_s0, rate_s0,
#                  shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors, use_common_mean_t,
#                  kernel_func, max_update_size, ninducing, forgetting_rate, delay)
#
#     def _init_w(self):
#         self.w = np.zeros((self.N, self.Nfactors))
#
#     def _init_y(self):
#         if not self.use_svi_people:
#             return super(PreferenceComponentsSVI, self)._init_y()
#
#         #self.y = np.mod(np.arange(self.Npeople), self.Nfactors).astype(float) + 1
#         #self.y /= np.max(self.y)
#         self.y = self.Ky_nm_block.dot(self.invKy_mm_block).dot(self.y_u.T).T
#
#         if not self.use_svi_people:
#             self.y_cov = np.diag(np.ones(self.Npeople*self.Nfactors)) # use ones to avoid divide by zero
#         else:
#             self.y_cov = np.array(self.rate_sy / self.shape_sy).flatten()
#             self.Sigma_y = np.zeros((self.Npeople, self.Npeople, self.Nfactors))
#
#     def _init_t(self):
#         self.t_mu0 = np.zeros((self.N, 1)) + self.t_mu0
#         self.t = np.copy(self.t_mu0)
#         self.t_cov = self.K / (self.shape_st / self.rate_st)#np.diag(np.ones(self.N))
#         self.Sigma_t = np.zeros((self.N, self.N))
#
#     def _init_params(self):
#         if self.person_features is not None:
#             self.use_svi_people = True
#         super(PreferenceComponentsSVI, self)._init_params()
#         self._choose_inducing_points()
#
#     # merge this with below... should not be possible to use SVI with the child GPs without SVI for the factors.
#     # make sure correct setting is used for the child GPs in their constructor...
#     def _choose_inducing_points(self):
#         # choose a set of inducing points -- for testing we can set these to the same as the observation points.
#         nobs = self.obs_coords.shape[0]
#
#         self.update_size = self.max_update_size # number of observed points in each stochastic update
#         if self.update_size > nobs:
#             self.update_size = nobs
#
#         if self.ninducing > nobs:
#             self.ninducing = nobs
#             self.inducing_coords = self.obs_coords
#         else:
#             init_size = 300
#             if self.ninducing < init_size:
#                 init_size = self.ninducing
#             kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing, random_state=345)
#             kmeans.fit(self.obs_coords)
#
#             self.inducing_coords = kmeans.cluster_centers_
#
#         self.w_invSm = np.zeros((self.ninducing * self.Nfactors, 1), dtype=float)# theta_1
#         self.w_invS = np.zeros((self.ninducing * self.Nfactors, self.ninducing * self.Nfactors), dtype=float) # theta_2
#
#         self.t_invSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1
#         self.t_invS = np.diag(np.ones(self.ninducing, dtype=float)) # theta_2
#
#         self.t_mu0_u = np.zeros((self.ninducing, 1)) + self.t_mu0_u
#
#         self.K_mm = self.kernel_func(self.inducing_coords, self.ls) + 1e-6 * np.eye(self.ninducing) # jitter
#         self.invK_mm = np.linalg.inv(self.K_mm)
#         self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords)
#
#         self.Lambda_factor_t = np.diag(np.ones(self.K_mm.shape[0])) # self.invK_mm.dot(self.K_mm.T)
#
#         blocks = [self.K_mm for _ in range(self.Nfactors)]
#         self.Kw_mm = block_diag(*blocks)
#         blocks = [self.invK_mm for _ in range(self.Nfactors)]
#         self.invKw_mm = block_diag(*blocks)
#         blocks = [self.K_nm for _ in range(self.Nfactors)]
#         self.Kw_nm = block_diag(*blocks)
#
#         self.w_u = np.zeros((self.ninducing, self.Nfactors))
#         #self.w_u = mvn.rvs(np.zeros(self.Nfactors * self.ninducing), cov=self.Kw_mm).reshape((self.Nfactors, self.ninducing)).T
#         #self.w_u *= (self.shape_sw/self.rate_sw)[np.newaxis, :]
#         #self.w_u = 2 * (np.random.rand(self.ninducing, self.Nfactors) - 0.5) * self.rate_sw / self.shape_sw #np.zeros((self.ninducing, self.Nfactors))
#         self.t_u = np.zeros((self.ninducing, 1))
#         self.f_u = np.zeros((self.ninducing, self.Npeople))
#
#         # For y
#         if self.person_features is None:
#             self.use_svi_people = False
#             return
#
#         self.y_update_size = self.max_update_size # number of inducing points in each stochastic update
#         if self.y_update_size > self.Npeople:
#             self.y_update_size = self.Npeople
#
#         self.y_ninducing = self.ninducing
#         if self.y_ninducing > self.people.shape[0]:
#             self.y_ninducing = self.people.shape[0]
#
#         init_size = 300
#         if self.y_ninducing > init_size:
#             init_size = self.y_ninducing
#         kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.y_ninducing)
#         kmeans.fit(self.person_features)
#
#         self.y_inducing_coords = kmeans.cluster_centers_
#
#         self.y_invSm = np.zeros((self.y_ninducing * self.Nfactors, 1), dtype=float)# theta_1
#         self.y_invS = np.diag(np.ones(self.y_ninducing * self.Nfactors, dtype=float)) # theta_2
#
#         self.Ky_mm_block = self.kernel_func(self.y_inducing_coords, self.lsy)
#         self.Ky_mm_block += 1e-6 * np.eye(len(self.Ky_mm_block)) # jitter
#         blocks = [self.Ky_mm_block for _ in range(self.Nfactors)]
#         self.Ky_mm = block_diag(*blocks)
#
#         self.invKy_mm_block = np.linalg.inv(self.Ky_mm_block)
#         blocks = [self.invKy_mm_block for _ in range(self.Nfactors)]
#         self.invKy_mm = block_diag(*blocks)
#
#         self.Ky_nm_block = self.kernel_func(self.person_features, self.lsy, self.y_inducing_coords)
#         blocks = [self.Ky_nm_block for _ in range(self.Nfactors)]
#         self.Ky_nm = block_diag(*blocks)
#
#         self.Lambda_factor_y = self.invKy_mm.dot(self.Ky_nm.T)
#
#         self.y_u = mvn.rvs(np.zeros(self.y_ninducing), self.Ky_mm_block, self.Nfactors)
#         self.y_u /= (self.shape_sy / self.rate_sy)[:, None]
#
#         #np.random.rand(self.Nfactors, self.y_ninducing) - 0.5 #np.zeros((self.Nfactors, self.y_ninducing))
#
#     def _gradient_dim(self, lstype, d, dimension):
#         der_logpw_logqw = 0
#         der_logpy_logqy = 0
#         der_logpt_logqt = 0
#         der_logpf_logqf = 0
#
#         # compute the gradient. This should follow the MAP estimate from chu and ghahramani.
#         # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate
#         # gradient when converged due to the coordinate ascent method.
#         if lstype == 'item' or (lstype == 'both' and d < self.nitem_features):
#             dKdls = self.K_mm * self.kernel_der(self.inducing_coords, self.ls, dimension)
#             # try to make the s scale cancel as much as possible
#             invK_w = self.invK_mm.dot(self.w_u)
#             invKs_C = self.invKws_mm_S
#             N = self.ninducing
#
#             self._compute_sigma_w()
#
#             for f in range(self.Nfactors):
#                 fidxs = np.arange(N) + (N * f)
#                 invK_wf = invK_w[:, f]
#                 invKs_C_f = invKs_C[fidxs, :][:, fidxs]
#                 sw = self.shape_sw[f] / self.rate_sw[f]
#                 Sigma_w_f = self.Sigma_w[:, :, f]
#                 der_logpw_logqw += 0.5 * (invK_wf.T.dot(dKdls).dot(invK_wf) * sw -
#                                     np.trace(invKs_C_f.dot(Sigma_w_f).dot(dKdls / sw)))
#
#             if self.use_t:
#                 invKs_t = self.inv_Kts_mm.dot(self.t_u)
#                 invKs_C = self.invKts_mm_S
#
#                 self._compute_sigma_t()
#
#                 der_logpt_logqt = 0.5 * (invKs_t.T.dot(dKdls).dot(invKs_t) -
#                             np.trace(invKs_C.dot(self.Sigma_t).dot(dKdls / self.shape_st * self.rate_st)))
#
#             for p in self.pref_gp:
#                 der_logpf_logqf += self.pref_gp[p].lowerbound_gradient(dimension)
#
#         elif lstype == 'person' or (lstype == 'both' and d >= self.nitem_features):
#             if self.person_features is None:
#                 pass
#             elif not self.use_svi_people:
#                 dKdls, invK_y, invKs_C, N = self._gradient_dim_people_terms(dimension)
#             else:
#                 dKdls = self.Ky_mm_block * self.kernel_der(self.y_inducing_coords, self.lsy, dimension)
#                 invK_y = self.invKy_mm_block.dot(self.y_u.T)
#                 invKs_C = self.invKys_mm_S
#                 N = self.y_ninducing
#
#                 self._compute_sigma_y()
#
#             for f in range(self.Nfactors):
#                 fidxs = np.arange(N) + (N * f)
#                 invK_yf = invK_y[:, f]
#                 invKs_C_f = invKs_C[fidxs, :][:, fidxs]
#                 sy = self.shape_sy[f] / self.rate_sy[f]
#                 Sigma_y_f = self.Sigma_y[:, :, f]
#                 der_logpy_logqy += 0.5 * (invK_yf.T.dot(dKdls).dot(invK_yf) * sy -
#                                     np.trace(invKs_C_f.dot(Sigma_y_f).dot(dKdls / sy)))
#
#         return der_logpw_logqw + der_logpy_logqy + der_logpt_logqt + der_logpf_logqf
#
#     def predict_item_feats(self, items_coords, item_features=None):
#         if item_features is None:
#             coords = items_coords
#         else:
#             coords = item_features[items_coords]
#
#         K1 = self.kernel_func(coords, self.ls, self.inducing_coords)
#         w1 = K1.dot(self.invK_mm).dot(self.w_u)
#
#         return w1
#
#     def _predict_w_t(self, coords_1):
#
#         # kernel between pidxs and t
#         K1 = self.kernel_func(coords_1, self.ls, self.inducing_coords)
#
#         # use kernel to compute t.
#         t1 = K1.dot(self.invK_mm).dot(self.t_u)
#
#         # kernel between pidxs and w -- use kernel to compute w. Don't need Kw_mm block-diagonal matrix
#         w1 = K1.dot(self.invK_mm).dot(self.w_u)
#
#         K11 = self.kernel_func(coords_1, self.ls, coords_1)
#
#         covpair = K1.dot(self.invK_mm)
#         covpair_uS = K1.dot(self.invKts_mm_S)
#         cov_t = K11 + (covpair_uS - covpair.dot(self.Kts_mm)).dot(covpair.T)
#
#         blocks = [K11 for _ in range(self.Nfactors)]
#         K11 = block_diag(*blocks)
#
#         blocks = [K1 for _ in range(self.Nfactors)]
#         K1 = block_diag(*blocks)
#
#         covpair = K1.dot(self.invKw_mm)
#         covpair_uS = K1.dot(self.invKws_mm_S)
#         cov_w = K11 + (covpair_uS - covpair.dot(self.Kws_mm)).dot(covpair.T)
#
#         return t1, w1, cov_t, cov_w
#
#     def _predict_y(self, person_features, Npeople):
#         if self.use_svi_people and person_features is not None:
#             Ky = self.kernel_func(person_features, self.lsy, self.y_inducing_coords)
#             # use kernel to compute y
#             invKy_train = self.invKy_mm_block
#             y_train = self.y_u.reshape(self.Nfactors, self.y_ninducing).T
#
#             # use kernel to compute y
#             y = Ky.dot(invKy_train).dot(y_train)
#
#             covpair = Ky.dot(self.invKy_mm)
#             covpair_uS = Ky.dot(self.invKys_mm_S)
#             cov = self.kernel_func(person_features, self.lsy, person_features) + (covpair_uS - covpair.dot(self.Kys_mm)).dot(covpair.T)
#
#             return y.T, cov
#         else:
#             return super(PreferenceComponentsSVI, self)._predict_y(person_features, Npeople)
#
#     def _expec_f_p(self, p, mu0_output):
#         f, _ = self.pref_gp[p].predict_f(
#                 out_idxs=self.coordidxs[p] if self.vb_iter==0 else None,
#                 out_feats=self.obs_coords if self.vb_iter==0 else None,
#                 mu0_output=mu0_output, reuse_output_kernel=True)
#         self.f[self.coordidxs[p], p] = f.flatten()
#         self.f_u[:, p:p+1] = self.w_u.dot(self.y[:, p:p+1]) + self.t_u
#
#     def _update_wy_plus_t(self, p):
#         super(PreferenceComponentsSVI, self)._update_wy_plus_t(p)
#
#         for f in range(self.Nfactors):
#             fidxs = np.arange(self.N) + (f * self.N)
#             pidx = f * self.Npeople + p
#             self.pref_gp[p].cov_mu0_mm += self.wS[fidxs, :][:, fidxs] * self.y_cov[pidx, pidx] + \
#                             self.wS[fidxs, :][:, fidxs] * self.y[f, p:p+1]**2 + \
#                             self.w_u[:, f]**2 * self.y_cov[pidx, pidx]
#         if self.use_t:
#             self.pref_gp[p].cov_mu0_mm += self.tS
#
#     def _expec_f(self):
#         self._update_sample()
#         super(PreferenceComponentsSVI, self)._expec_f()
#
#     def _compute_sigma_w(self):
#         self.Sigma_w = np.zeros((self.ninducing, self.ninducing, self.Nfactors))
#         psample = np.arange(self.ninducing)
#
#         for p in self.pref_gp:
#             prec_p = self.invK_mm * self.pref_gp[p].s
#             y_p = self.y[:, p:p+1]
#
#             if self.use_svi_people:
#                 yidxs = p + self.y_ninducing * np.arange(self.Nfactors)
#             else:
#                 yidxs = p + self.Npeople * np.arange(self.Nfactors)
#
#             _, Sigma_w_p = self._compute_sigma_w_p(self.ninducing, y_p, yidxs, prec_p, psample)
#             self.Sigma_w += Sigma_w_p
#
#     def _expec_w(self):
#         '''
#         Compute the expectation over the latent features of the items and the latent personality components
#         '''
#         # Put a GP prior on w with covariance K/gamma and mean 0
#         N = self.update_size
#         Nobs_counter = 0
#         Nobs_counter_i = 0
#
#         x = np.zeros((N, self.Nfactors))
#         Sigma = np.zeros((N * self.Nfactors, N * self.Nfactors))
#
#         for p in self.pref_gp:
#             if self.use_svi_people and p not in self.pdata_idx_i:
#                 continue
#
#             pidxs = self.coordidxs[p]
#             y_p = self.y[:, p:p+1]
#             Nobs_counter += len(pidxs)
#             psample = np.in1d(self.data_idx_i, pidxs)
#             pidxs = self.data_idx_i[psample]
#             if not len(pidxs):
#                 continue
#             Nobs_counter_i += len(pidxs)
#
#             prec_p = self.pref_gp[p].get_obs_precision()
#             invQ_f = prec_p.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :])
#             x_p = y_p.T * invQ_f
#             x[psample, :] += x_p
#
#             if self.use_svi_people:
#                 yidxs = p + self.y_ninducing * np.arange(self.Nfactors)
#             else:
#                 yidxs = p + self.Npeople * np.arange(self.Nfactors)
#
#             Sigma_p, _ = self._compute_sigma_w_p(N, y_p, yidxs, prec_p, psample)
#             Sigma += Sigma_p
#
#         x = x.T.flatten()[:, np.newaxis]
#
#         widxs_i = (np.tile(self.data_idx_i[:,None], (1,self.Nfactors)) + np.arange(self.Nfactors)[None,:]).flatten()
#         Lambda_factor_w_i = self.invKw_mm.dot(self.Kw_nm[widxs_i, :].T)
#
#         self.w, _, self.w_invS, self.w_invSm, self.w_u, self.invKws_mm_S, self.wS = svi_update_gaussian(x, 0, 0,
#             self.Kws_mm, self.inv_Kws_mm, self.Kws_nm, Lambda_factor_w_i, None, Sigma, self.w_invS,
#             self.w_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)
#
#         self.w = np.reshape(self.w, (self.Nfactors, self.N)).T # w is N x Nfactors
#         self.w_u = np.reshape(self.w_u, (self.Nfactors, self.ninducing)).T # w is N x Nfactors
#
#         for f in range(self.Nfactors):
#             fidxs = np.arange(self.ninducing) + (self.ninducing * f)
#             self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0,
#                     self.ninducing, self.invK_mm, self.w_u[:, f:f+1], np.zeros((self.ninducing, 1)),
#                     self.invKws_mm_S[fidxs, :][:, fidxs] / self.shape_sw[f] * self.rate_sw[f])
#             fidxs = np.arange(self.N) + (self.N * f)
#
#         self._expec_y()
#         self.wy = self.w.dot(self.y)
#
#     def _compute_sigma_y(self):
#         self.Sigma_y = np.zeros((self.y_ninducing, self.y_ninducing, self.Nfactors))
#         pidxs = np.arange(self.ninducing)
#         for p in self.pref_gp:
#             prec_f = self.invK_mm * self.pref_gp[p].s
#             self.Sigma_y[np.arange(self.y_ninducing), np.arange(self.y_ninducing), :] += self._compute_sigma_y_p(
#                     self.ninducing, self.w_u, self.Kws_mm.dot(self.invKws_mm_S), prec_f, pidxs)[
#                         np.arange(self.Nfactors), np.arange(self.Nfactors)]
#
#     def _expec_y(self):
#         '''
#         Compute expectation over the personality components using VB
#         '''
#         Npeople = self.Npeople
#         Sigma = np.zeros((self.Nfactors * Npeople, self.Nfactors * Npeople))
#         x = np.zeros((Npeople, self.Nfactors))
#
#         Nobs_counter = 0
#         Nobs_counter_i = 0
#
#         pidx = 0
#
#         w_cov = self.Kws_mm.dot(self.invKws_mm_S)
#         w = self.w_u
#         N = self.ninducing
#
#         for p in self.pref_gp:
#             pidxs = self.coordidxs[p]
#             Nobs_counter += len(pidxs)
#             if self.use_svi_people and p not in self.pdata_idx_i:
#                 continue
#             Nobs_counter_i += len(pidxs)
#             pidxs = np.arange(N)
#
#             prec_f = self.pref_gp[p].get_obs_precision()
#             invQ_f = prec_f.dot(self.f[pidxs, p:p+1] - self.t[pidxs, :])
#
#             Sigma_p = self._compute_sigma_y_p(N, w, w_cov, prec_f, pidxs)
#             sigmaidxs = np.arange(self.Nfactors) * Npeople + pidx
#             Sigmarows = np.zeros((self.Nfactors, Sigma.shape[1]))
#             Sigmarows[:, sigmaidxs] = Sigma_p
#             Sigma[sigmaidxs, :] += Sigmarows
#
#             x[pidx, :] = w.T.dot(invQ_f).T
#             pidx += 1
#
#         x = x.T.flatten()[:, np.newaxis]
#
#         if not self.use_svi_people:
#             self.Sigma_y = Sigma
#
#             # y_cov is same format as K and Sigma with rows corresponding to (f*Npeople) + p where f is factor index from 0
#             # and p is person index
#             self.y_cov = np.linalg.inv(self.invKy * self.sy_matrix + Sigma)
#             self.y = self.y_cov.dot(x)
#
#             # y is Nfactors x Npeople
#             self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
#
#             for f in range(self.Nfactors):
#                 fidxs = np.arange(self.Npeople) + (self.Npeople * f)
#                 self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0,
#                                                     self.Npeople, self.invKy, self.y[f:f+1, :].T,
#                                                     np.zeros((self.Npeople, 1)), f_cov=self.y_cov[fidxs, :][:, fidxs])
#
#                 self.sy_matrix[fidxs, :] = self.shape_sy[f] / self.rate_sy[f] # sy_rows
#         else: # SVI implementation
#             self.y, _, self.y_invS, self.y_invSm, self.y_u, self.invKys_mm_S, self.yS = svi_update_gaussian(x, 0, 0,
#                 self.Kys_mm, self.inv_Kys_mm, self.Kys_nm, self.Lambda_factor_y, None, Sigma, self.y_invS,
#                 self.y_invSm, self.vb_iter, self.delay, self.forgetting_rate, Nobs_counter, Nobs_counter_i)
#
#             # y is Nfactors x Npeople
#             self.y = np.reshape(self.y, (self.Nfactors, self.Npeople))
#             self.y_u = np.reshape(self.y_u, (self.Nfactors, self.y_ninducing))
#             self.y_cov = self.Ky_nm.dot(self.Kys_mm.dot(self.invKys_mm_S)).dot(self.Ky_nm.T)
#
#             for f in range(self.Nfactors):
#                 fidxs = np.arange(self.y_ninducing) + (self.y_ninducing * f)
#                 self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0,
#                     self.y_ninducing, self.invKy_mm_block, self.y_u[f:f+1, :].T, np.zeros((self.y_ninducing, 1)),
#                     self.invKys_mm_S[fidxs, :][:, fidxs] / self.shape_sy[f] * self.rate_sy[f])
#                 fidxs = np.arange(self.Npeople) + (self.Npeople * f)
#
#     def _compute_sigma_t(self):
#         self.Sigma_t = np.zeros((self.ninducing, self.ninducing))
#
#         for p in self.pref_gp:
#             prec_f = self.invK_mm * self.pref_gp[p].s
#             self.Sigma_t += prec_f
#
#     def _expec_t(self):
#         if not self.use_t:
#             return
#
#         N = self.update_size
#         Nobs_counter = 0
#         Nobs_counter_i = 0
#
#         Sigma = np.zeros((N, N))
#         x = np.zeros((N, 1))
#
#         #size_added = 0
#         for p in self.pref_gp:
#             pidxs = self.coordidxs[p]
#
#             Nobs_counter += len(pidxs)
#             psample = np.in1d(self.data_idx_i, pidxs)
#             pidxs = self.data_idx_i[psample]
#             if not len(pidxs):
#                 continue
#             Nobs_counter_i += len(pidxs)
#
#             prec_f = self.pref_gp[p].s
#             invQ_f = (self.f[pidxs, p:p+1] - self.t[pidxs, :]) * prec_f
#             x[psample, :] += invQ_f
#
#             sigmarows = np.zeros((np.sum(psample), N))
#             sigmarows[:, psample] = prec_f
#             Sigma[psample, :] += sigmarows
#
#         Lambda_factor_t = self.invK_mm.dot(self.K_nm[self.data_idx_i, :].T)
#
#         self.t, _, self.t_invS, self.t_invSm, self.t_u, self.invKts_mm_S, self.tS = svi_update_gaussian(x,
#             self.t_mu0, self.t_mu0_u, self.Kts_mm, self.inv_Kts_mm, self.Kts_nm, Lambda_factor_t,
#             None, Sigma, self.t_invS, self.t_invSm, self.vb_iter, self.delay,
#             self.forgetting_rate, Nobs_counter, Nobs_counter_i)
#
#         self.t_cov_u = self.Kts_mm.dot(self.invKts_mm_S)
#
#         self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, self.ninducing,
#             self.invK_mm, self.t_u, np.zeros((self.ninducing, 1)), self.invKts_mm_S / self.shape_st * self.rate_st)
#
#     def _update_sample(self):
#         self._update_sample_idxs()
#
#         sw_mm = np.zeros((self.Nfactors * self.ninducing, self.Nfactors * self.ninducing), dtype=float)
#         sw_nm = np.zeros((self.Nfactors * self.N, self.Nfactors * self.ninducing), dtype=float)
#         for f in range(self.Nfactors):
#             fidxs = np.arange(self.ninducing) + (self.ninducing * f)
#             sw_mm[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]
#             fidxs = np.arange(self.N) + (self.N * f)
#             sw_nm[fidxs, :] = self.shape_sw[f] / self.rate_sw[f]
#
#         st = self.shape_st / self.rate_st
#
#         self.Kws_mm = self.Kw_mm / sw_mm
#         self.inv_Kws_mm  = self.invKw_mm * sw_mm
#         self.Kws_nm = self.Kw_nm  / sw_nm
#
#         self.Kts_mm = self.K_mm / st
#         self.inv_Kts_mm  = self.invK_mm * st
#         self.Kts_nm = self.K_nm / st
#
#         if self.use_svi_people:
#             sy_mm = np.zeros((self.Nfactors * self.y_ninducing, self.Nfactors * self.y_ninducing), dtype=float)
#             sy_nm = np.zeros((self.Nfactors * self.Npeople, self.Nfactors * self.y_ninducing), dtype=float)
#             for f in range(self.Nfactors):
#                 fidxs = np.arange(self.y_ninducing) + (self.y_ninducing * f)
#                 sy_mm[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]
#                 fidxs = np.arange(self.Npeople) + (self.Npeople * f)
#                 sy_nm[fidxs, :] = self.shape_sy[f] / self.rate_sy[f]
#
#             self.Kys_mm = self.Ky_mm / sy_mm
#             self.inv_Kys_mm  = self.invKy_mm * sy_mm
#             self.Kys_nm = self.Ky_nm / sy_nm
#
#     def _update_sample_idxs(self):
#         self.data_idx_i = np.sort(np.random.choice(self.N, self.update_size, replace=False))
#
#         if self.use_svi_people:
#             self.pdata_idx_i = np.sort(np.random.choice(self.Npeople, self.y_update_size, replace=False))
#
#     def lowerbound(self):
#         f_terms = 0
#         y_terms = 0
#
#         for p in self.pref_gp:
#             f_terms += self.pref_gp[p].lowerbound()
#             if self.verbose:
#                 logging.debug('s_f^%i=%.2f' % (p, self.pref_gp[p].s))
#
#         logpw = mvn.logpdf(self.w_u.T.flatten(), cov=self.Kws_mm) # this line is slow
#         logqw = mvn.logpdf(self.w_u.T.flatten(), mean=self.w_u.T.flatten(), cov=self.Kws_mm.dot(self.invKws_mm_S),
#                            allow_singular=True) # this line is slow
#
#         if self.use_t:
#             logpt = mvn.logpdf(self.t_u.flatten(), cov=self.Kts_mm)
#             logqt = mvn.logpdf(self.t_u.flatten(), mean=self.t_u.flatten(), cov=self.Kts_mm.dot(self.invKts_mm_S))
#         else:
#             logpt = 0
#             logqt = 0
#
#         if self.use_svi_people:
#             logpy = mvn.logpdf(self.y_u.flatten(), cov=self.Kys_mm)
#             logqy = mvn.logpdf(self.y_u.flatten(), mean=self.y_u.flatten(), cov=self.Kys_mm.dot(self.invKys_mm_S),
#                                allow_singular=True)
#         else:
#             if self.person_features is not None:
#                 logpy = mvn.logpdf(self.y.flatten(), cov=self.Ky / self.sy_matrix)
#             else:
#                 logpy = 0
#                 for f in range(self.Nfactors):
#                     logpy += np.sum(norm.logpdf(self.y[f, :], scale=np.sqrt(self.rate_sy[f] / self.shape_sy[f])))
#             logqy = mvn.logpdf(self.y.flatten(), mean=self.y.flatten(), cov=self.y_cov)
#
#         logps_y = 0
#         logqs_y = 0
#         logps_w = 0
#         logqs_w = 0
#         for f in range(self.Nfactors):
#             logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f])
#             logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f])
#
#             logps_y += lnp_output_scale(self.shape_sy0, self.rate_sy0, self.shape_sy[f], self.rate_sy[f])
#             logqs_y += lnq_output_scale(self.shape_sy[f], self.rate_sy[f])
#
#         logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st)
#         logqs_t = lnq_output_scale(self.shape_st, self.rate_st)
#
#         w_terms = logpw - logqw + logps_w - logqs_w
#         y_terms += logpy - logqy + logps_y - logqs_y
#         t_terms = logpt - logqt + logps_t - logqs_t
#
#         lb = f_terms + t_terms + w_terms + y_terms
#
#         if self.verbose:
#             logging.debug('s_w=%s' % (self.shape_sw/self.rate_sw))
#             logging.debug('s_y=%s' % (self.shape_sy/self.rate_sy))
#             logging.debug('s_t=%.2f' % (self.shape_st/self.rate_st))
#             logging.debug('fterms=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (f_terms, w_terms, y_terms, t_terms))
#
#         if self.verbose:
#             logging.debug( "Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb) )
#
#         if self.verbose:
#             logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
#             logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
#             logging.debug("y: %.2f, %.2f" % (np.min(self.y), np.max(self.y)))
#
#         return lb
#
