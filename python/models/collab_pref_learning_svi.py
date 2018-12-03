"""
Scalable implementation of collaborative Gaussian process preference learning using stochastic variational inference.
Scales to large sets of observations (preference pairs) and numbers of items and users.

The results look different to the non-SVI version. There is a difference in how G is computed inside expec_t --
in the non-SVI version, it is computed separately for each observation location, and the obs_f estimates used to compute
it do not have any shared component across people because the value of t is computed by aggregating across people
outside the child GP. With the SVI implementation, the aggregation is done at each step by the inducing points, so that
inside the iterations of t_gp, there is a common t value when computing obs_f for all people. I think both are valid
approximations considering they are using slightly different values of obs_f to compute the updates. Differences may
accumulate from small differences in the approximations.

"""
import datetime

import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
import logging
from gp_pref_learning import GPPrefLearning, pref_likelihood
from scipy.linalg import block_diag
from scipy.special import psi, binom
from sklearn.cluster import MiniBatchKMeans

from collab_pref_learning_vb import CollabPrefLearningVB, expec_output_scale, expec_pdf_gaussian, expec_q_gaussian, \
    temper_extreme_probs, lnp_output_scale, lnq_output_scale


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

def inducing_to_observation_moments(Ks_mm, invK_mm, K_nm, fhat_mm, mu0, S=None, K_nn=None, full_cov=True):
    covpair = K_nm.dot(invK_mm)
    fhat = covpair.dot(fhat_mm) + mu0

    if S is None:
        C = None
    elif full_cov:
        covpairS = covpair.dot(S)  # C_nm

        if K_nn is None:
            C = None
        else:
            C = K_nn + (covpairS - covpair.dot(Ks_mm)).dot(covpair.T)

    else:
        C = K_nn + np.sum(covpair.dot(S - Ks_mm) * covpair, axis=1)

    return fhat, C

class CollabPrefLearningSVI(CollabPrefLearningVB):

    def __init__(self, nitem_features, nperson_features=0, mu0=0, shape_s0=1, rate_s0=1,
                 shape_ls=1, rate_ls=100, ls=100, shape_lsy=1, rate_lsy=100, lsy=100, verbose=False, nfactors=20,
                 use_common_mean_t=True, kernel_func='matern_3_2',
                 max_update_size=500, ninducing=500, forgetting_rate=0.9, delay=1.0, use_lb=True):

        self.max_update_size = max_update_size
        self.ninducing_preset = ninducing
        self.forgetting_rate = forgetting_rate
        self.delay = delay

        self.conv_threshold_G = 1e-5

        self.t_mu0 = mu0

        super(CollabPrefLearningSVI, self).__init__(nitem_features, nperson_features, shape_s0, rate_s0,
                                                    shape_ls, rate_ls, ls, shape_lsy, rate_lsy, lsy, verbose, nfactors, use_common_mean_t,
                                                    kernel_func, use_lb=use_lb)

        if use_lb:
            self.n_converged = 10 # due to stochastic updates, take more iterations before assuming convergence

    def _init_covariance(self):
        self.shape_sw = np.zeros(self.Nfactors) + self.shape_sw0
        self.rate_sw = np.zeros(self.Nfactors) + self.rate_sw0

    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        self.update_size = self.max_update_size # number of observed points in each stochastic update
        if self.update_size > self.nobs:
            self.update_size = self.nobs

        # Inducing points for items -----------------------------------------------------------

        self.ninducing = self.ninducing_preset

        if self.ninducing >= self.obs_coords.shape[0]:
            self.ninducing = self.obs_coords.shape[0]
            self.inducing_coords = self.obs_coords
        else:
            init_size = 300
            if init_size < self.ninducing:
                init_size = self.ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing)
            kmeans.fit(self.obs_coords)

            self.inducing_coords = kmeans.cluster_centers_

        # Kernel over items (used to construct priors over w and t)
        if self.verbose:
            logging.debug('Initialising K_mm')
        self.K_mm = self.kernel_func(self.inducing_coords, self.ls) + \
                    (1e-4 if self.cov_type=='diagonal' else 1e-6) * np.eye(self.ninducing) # jitter
        self.invK_mm = np.linalg.inv(self.K_mm)
        if self.verbose:
            logging.debug('Initialising K_nm')
        self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords)

        # Related to w, the item components ------------------------------------------------------------
        # posterior expected values
        # self.w_u = mvn.rvs(np.zeros(self.ninducing), self.K_mm, self.Nfactors).reshape(self.Nfactors, self.ninducing)
        # self.w_u /= (self.shape_sw / self.rate_sw)[:, None]
        # self.w_u = self.w_u.T
        self.w_u = np.zeros((self.ninducing, self.Nfactors))
        # self.w_u[np.arange(self.ninducing), np.arange(self.ninducing)] = 1.0

        # moments of distributions over inducing points for convenience
        # posterior covariance
        self.wS = np.array([self.invK_mm * self.shape_sw0 / self.rate_sw0 for _ in range(self.Nfactors)])
        self.winvS = np.array([self.invK_mm * self.shape_sw0 / self.rate_sw0 for _ in range(self.Nfactors)])
        self.winvSm = np.zeros((self.ninducing, self.Nfactors))

        # Inducing points for people -------------------------------------------------------------------
        if self.person_features is None:
            self.y_ninducing = self.Npeople

            # Prior covariance of y
            self.Ky_mm_block = np.ones(self.y_ninducing) * (1+1e-4 if self.cov_type=='diagonal' else 1+1e-6)  # jitter
            self.invKy_mm_block = self.Ky_mm_block
            self.Ky_nm_block = np.diag(self.Ky_mm_block)

            # posterior covariance
            self.yS = np.array([self.Ky_mm_block for _ in range(self.Nfactors)])
            self.yinvS = np.array([self.invKy_mm_block for _ in range(self.Nfactors)])
            self.yinvSm = np.zeros((self.y_ninducing, self.Nfactors))

            if self.y_ninducing <= self.Nfactors:
                # give each person a factor of their own, with a little random noise so that identical users will
                # eventually get clustered into the same set of factors.
                self.y_u = np.zeros((self.Nfactors, self.y_ninducing))
                self.y_u[:self.y_ninducing, :] = np.eye(self.y_ninducing)
                self.y_u += np.random.rand(*self.y_u.shape) * 1e-6
            else:
                # positive values
                self.y_u = norm.rvs(0, 1, (self.Nfactors, self.y_ninducing))**2

        else:
            self.y_ninducing = self.ninducing_preset

            if self.y_ninducing >= self.Npeople:
                self.y_ninducing = self.Npeople
                self.y_inducing_coords = self.person_features
            else:
                init_size = 300
                if self.y_ninducing > init_size:
                    init_size = self.y_ninducing
                kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.y_ninducing)
                kmeans.fit(self.person_features)

                self.y_inducing_coords = kmeans.cluster_centers_

            # Kernel over people used to construct prior covariance for y
            if self.verbose:
                logging.debug('Initialising Ky_mm')
            self.Ky_mm_block = self.y_kernel_func(self.y_inducing_coords, self.lsy)
            self.Ky_mm_block += (1e-4 if self.cov_type=='diagonal' else 1e-6) * np.eye(self.y_ninducing) # jitter

            # Prior covariance of y
            self.invKy_mm_block = np.linalg.inv(self.Ky_mm_block)

            if self.verbose:
                logging.debug('Initialising Ky_nm')
            self.Ky_nm_block = self.y_kernel_func(self.person_features, self.lsy, self.y_inducing_coords)

            # posterior covariance
            self.yS = np.array([self.Ky_mm_block for _ in range(self.Nfactors)])
            self.yinvS = np.array([self.invKy_mm_block for _ in range(self.Nfactors)])
            self.yinvSm = np.zeros((self.y_ninducing, self.Nfactors))

            # posterior means
            if self.y_ninducing <= self.Nfactors:
                # give each person a factor of their own, with a little random noise so that identical users will
                # eventually get clustered into the same set of factors.
                self.y_u = np.zeros((self.Nfactors, self.y_ninducing))
                self.y_u[:self.y_ninducing, :] = np.eye(self.y_ninducing)
                self.y_u += np.random.rand(*self.y_u.shape) * 1e-6
            else:
                self.y_u = mvn.rvs(np.zeros(self.y_ninducing), self.Ky_mm_block, self.Nfactors) ** 2

        if self.Nfactors == 1:
            self.y_u = self.y_u[None, :]

        # Related to t, the item means -----------------------------------------------------------------
        self.t_u = np.zeros((self.ninducing, 1))  # posterior means
        self.tS = None

        if self.use_t:
            self.tinvSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1/posterior covariance dot means
            self.tinvS = self.invK_mm * self.shape_st0 / self.rate_st0 # theta_2/posterior covariance


    def _post_sample(self, K_nm, invK_mm, w_u, wS, t_u, tS,
                     Ky_nm, invKy_mm, y_u, y_var, v, u, expectedlog=False):

        # sample the inducing points because we don't have full covariance matrix. In this case, f_cov should be Ks_nm
        nsamples = 500

        if wS.ndim == 3:
            w_samples = np.array([mvn.rvs(mean=w_u[:, f], cov=wS[f], size=(nsamples))
                              for f in range(self.Nfactors)])
        else:
            w_samples = np.array([mvn.rvs(mean=w_u[:, f], cov=wS, size=(nsamples))
                                  for f in range(self.Nfactors)])

        if self.use_t:
            if np.isscalar(t_u):
                t_u = np.zeros(tS.shape[0]) + t_u
            else:
                t_u = t_u.flatten()

            t_samples = mvn.rvs(mean=t_u, cov=tS, size=(nsamples))

        N = y_u.shape[1]
        if np.isscalar(y_var):
            y_var = np.zeros((self.Nfactors * N)) + y_var
        else:
            y_var = y_var.flatten()

        y_samples = np.random.normal(loc=y_u.flatten()[:, None], scale=np.sqrt(y_var)[:, None],
                                     size=(N * self.Nfactors, nsamples)).reshape(self.Nfactors, N, nsamples)

        # w_samples: F x nsamples x N
        # t_samples: nsamples x N
        # y_samples: F x Npeople x nsamples

        if K_nm is not None:
            covpair_w = K_nm.dot(invK_mm)
            w_samples = np.array([covpair_w.dot(w_samples[f].T).T for f in range(self.Nfactors)])  # assume zero mean
            if self.use_t:
                t_samples = K_nm.dot(invK_mm).dot(t_samples.T).T  # assume zero mean

            if self.person_features is not None:
                covpair_y = Ky_nm.dot(invKy_mm)
                y_samples = np.array([covpair_y.dot(y_samples[f]) for f in range(self.Nfactors)])  # assume zero mean

        if self.use_t:
            f_samples = np.array([w_samples[:, s, :].T.dot(y_samples[:, :, s]).T + t_samples[s][None, :]for s in range(nsamples)])
        else:
            f_samples = np.array([w_samples[:, s, :].T.dot(y_samples[:, :, s]).T for s in range(nsamples)])

        f_samples = f_samples.reshape(nsamples, self.N * self.Npeople).T

        phi = pref_likelihood(f_samples, v=v, u=u)
        phi = temper_extreme_probs(phi)
        notphi = 1 - phi

        if expectedlog:
            phi = np.log(phi)
            notphi = np.log(notphi)

        m_post = np.mean(phi, axis=1)[:, np.newaxis]
        not_m_post = np.mean(notphi, axis=1)[:, np.newaxis]
        v_post = np.var(phi, axis=1)[:, np.newaxis]
        v_post = temper_extreme_probs(v_post, zero_only=True)
        # fix extreme values to sensible values. Don't think this is needed and can lower variance?
        v_post[m_post * (1 - not_m_post) <= 1e-7] = 1e-8

        return m_post, not_m_post, v_post

    def _estimate_obs_noise(self):

        # to make a and b smaller and put more weight onto the observations, increase v_prior by increasing rate_s0/shape_s0
        m_prior, not_m_prior, v_prior = self._post_sample(self.K_nm, self.invK_mm,
                np.zeros((self.ninducing, self.Nfactors)), self.K_mm * self.rate_sw0 / self.shape_sw0,
                self.t_mu0, self.K_mm * self.rate_st0 / self.shape_st0,
                self.Ky_nm_block, self.invKy_mm_block, np.zeros((self.Nfactors, self.y_ninducing)), 1,
                self.pref_v, self.pref_u)

        # find the beta parameters
        a_plus_b = 1.0 / (v_prior / (m_prior*not_m_prior)) - 1
        a = (a_plus_b * m_prior)
        b = (a_plus_b * not_m_prior)

        nu0 = np.array([b, a])
        # Noise in observations
        nu0_total = np.sum(nu0, axis=0)
        obs_mean = (self.z + nu0[1]) / (1 + nu0_total)
        var_obs_mean = obs_mean * (1 - obs_mean) / (1 + nu0_total + 1)  # uncertainty in obs_mean
        Q = (obs_mean * (1 - obs_mean) + var_obs_mean)
        Q = Q.flatten()
        return Q

    def _init_w(self):
        # initialise the factors randomly -- otherwise they can get stuck because there is nothing to differentiate them
        # i.e. the cluster identifiability problem
        # self.w = np.zeros((self.N, self.Nfactors))
        self.w = self.K_nm.dot(self.invK_mm).dot(self.w_u)
        # save for later
        batchsize = 500
        nbatches = int(np.ceil(self.N / float(batchsize) ))

        self.Kw_file_tag = ''#datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if self.N > 500:
            self.Kw = np.memmap('./Kw_%s.tmp' % self.Kw_file_tag, dtype=float, mode='w+', shape=(self.N, self.N))
        else:
            self.Kw = np.zeros((self.N, self.N))

        for b in range(nbatches):

            end1 = (b+1)*batchsize
            if end1 > self.N:
                end1 = self.N

            for b2 in range(nbatches):

                end2 = (b2+1)*batchsize
                if end2 > self.N:
                    end2 = self.N

                self.Kw[b*batchsize:(b+1)*batchsize, :][:, b2*batchsize:(b2+1)*batchsize] = self.kernel_func(
                    self.obs_coords[b*batchsize:end1, :], self.ls, self.obs_coords[b2*batchsize:end2, :])

        if self.N > 500:
            self.Kw.flush()

        self.Sigma_w = np.zeros((self.Nfactors, self.ninducing, self.ninducing))

        if not self.new_obs:
            return

        self.Q = self._estimate_obs_noise()

    def _init_t(self):
        self.t = np.zeros((self.N, 1))
        self.st = self.shape_st0 / self.rate_st0

        if not self.use_t:
            return

    def _init_y(self):
        if self.person_features is None:
            self.y = self.y_u
        else:
            self.y = self.Ky_nm_block.dot(self.invKy_mm_block).dot(self.y_u.T).T
        self.y_var = np.ones((self.Nfactors, self.Npeople))

        # save for later
        batchsize = 500
        nbatches = int(np.ceil(self.Npeople / float(batchsize) ))

        self.Ky_file_tag = ''#datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if self.Npeople > 500:
            self.Ky = np.memmap('./Kw_%s.tmp' % self.Ky_file_tag, dtype=float, mode='w+', shape=(self.Npeople, self.Npeople))
        else:
            self.Ky = np.zeros((self.Npeople, self.Npeople))

        for b in range(nbatches):

            end1 = (b+1)*batchsize
            if end1 > self.Npeople:
                end1 = self.Npeople

            for b2 in range(nbatches):

                end2 = (b2+1)*batchsize
                if end2 > self.Npeople:
                    end2 = self.Npeople

                if self.person_features is not None:
                    self.Ky[b*batchsize:(b+1)*batchsize, :][:, b2*batchsize:(b2+1)*batchsize] = self.y_kernel_func(
                        self.person_features[b*batchsize:end1, :], self.lsy, self.person_features[b2*batchsize:end2, :])
                elif b == b2:
                    self.Ky[b*batchsize:(b+1)*batchsize, :][:, b2*batchsize:(b2+1)*batchsize] = np.eye(end1 - b*batchsize, dtype=float)
                else:
                    self.Ky[b * batchsize:(b + 1) * batchsize, :][:, b2 * batchsize:(b2 + 1) * batchsize] = np.zeros(
                        (end1 - b * batchsize, end2 - b2 * batchsize), dtype=float)

        if self.Npeople > 500:
            self.Ky.flush()

        if self.person_features is None:
            self.Sigma_y = np.zeros((self.Nfactors, self.y_ninducing))
        else:
            self.Sigma_y = np.zeros((self.Nfactors, self.y_ninducing, self.y_ninducing))

    def _init_params(self):
        if self.Nfactors is None or self.Npeople < self.Nfactors:  # not enough items or people
            self.Nfactors = self.Npeople

        self._init_covariance()

        # initialise the inducing points first
        self._choose_inducing_points()

        self.ls = np.zeros(self.nitem_features) + self.ls

        self.G = -np.inf

        self._init_w()
        self._init_y()
        self._init_t()


    def _compute_jacobian(self):
        self.obs_f = (self.w.dot(self.y) + self.t).T.reshape(self.N * self.Npeople, 1)

        phi, g_mean_f = pref_likelihood(self.obs_f, v=self.pref_v, u=self.pref_u, return_g_f=True)  # first order Taylor series approximation
        J = 1 / (2 * np.pi) ** 0.5 * np.exp(-g_mean_f ** 2 / 2.0) * np.sqrt(0.5)
        J = J[self.data_obs_idx_i, :]

        s = (self.pref_v[self.data_obs_idx_i, None] == self.joint_idx_i[None, :]).astype(int) - \
            (self.pref_u[self.data_obs_idx_i, None] == self.joint_idx_i[None, :]).astype(int)

        J_df = J * s

        return J_df


    def _expec_t(self):

        self._update_sample()

        if not self.use_t:
            return

        N = self.ninducing

        # self.t_gp.s = self.st
        # self.t_gp.fit(self.pref_v, self.pref_u, self.dummy_obs_coords, self.preferences,
        #               mu0=mu0, K=None, process_obs=self.new_obs, input_type=self.input_type)

        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = self.nobs / float(self.update_size)

        self.prev_tinvS = self.tinvS
        self.prev_tinvSm = self.tinvSm

        covpair = self.invK_mm.dot(self.K_nm[self.w_idx_i].T)

        for G_iter in range(self.max_iter_G):

            oldG = self.G
            self.G = self._compute_jacobian() # P_i x (N_i*Npeople_i)

            # we need to map from the real Npeople points to the inducing points.
            GinvQG = (self.G.T/self.Q[None, self.data_obs_idx_i]).dot(self.G)

            self.Sigma_t = covpair.dot(GinvQG).dot(covpair.T)

            # need to get invS for current iteration and merge using SVI weighted sum
            self.tinvS = (1-rho_i) * self.prev_tinvS + rho_i * (self.invK_mm*self.shape_st/self.rate_st + w_i * self.Sigma_t)

            z0 = pref_likelihood(self.obs_f, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                 - self.G.dot(self.t[self.w_idx_i, :]) # P x NU_i

            invQ_f = (self.G.T / self.Q[None, self.data_obs_idx_i]).dot(self.z[self.data_obs_idx_i] - z0)
            x = covpair.dot(invQ_f)

            # need to get x for current iteration and merge using SVI weighted sum
            self.tinvSm = (1-rho_i) * self.prev_tinvSm + rho_i * w_i * x

            self.tS = np.linalg.inv(self.tinvS)
            self.t_u = self.tS.dot(self.tinvSm)

            diff = np.max(np.abs(oldG - self.G))

            if self.verbose:
                logging.debug("expec_t: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

        self.t, _ = inducing_to_observation_moments(self.Kts_mm, self.invK_mm, self.K_nm, self.t_u, self.t_mu0)

        self.shape_st, self.rate_st = expec_output_scale(self.shape_st0, self.rate_st0, N,
                                                         self.invK_mm, self.t_u, np.zeros((N, 1)),
                                                         f_cov=self.tS)
        self.st = self.shape_st / self.rate_st


    def _expec_w(self):
        """
        Compute the expectation over the latent features of the items and the latent personality components
        """
        # Put a GP prior on w with covariance K/gamma and mean 0
        N = self.ninducing

        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = self.nobs / float(self.update_size)

        self.prev_winvS = self.winvS
        self.prev_winvSm = self.winvSm

        covpair = self.invK_mm.dot(self.K_nm[self.w_idx_i].T)

        for G_iter in range(self.max_iter_G):

            oldG = self.G
            if G_iter > 0:
                self.G = self.G * 0.1  + self._compute_jacobian() * 0.9
            else:
                self.G = self._compute_jacobian()

            # we need to map from the real Npeople points to the inducing points.
            invQGT = self.G.T/self.Q[None, self.data_obs_idx_i]

            for f in range(self.Nfactors):
                # scale the precision by y
                ycov_idxs = [np.argwhere(self.p_idx_i == idx).flatten()[0] for idx in self.y_idx_i]
                scaling_f = self.y[f:f+1, self.y_idx_i].T.dot(self.y[f:f+1, self.y_idx_i]) + \
                            self.y_cov_i[f][ycov_idxs, :][:, ycov_idxs]

                Sigma_w_f = covpair.dot(invQGT.dot(self.G) * scaling_f).dot(covpair.T)

                # need to get invS for current iteration and merge using SVI weighted sum
                self.winvS[f] = (1-rho_i) * self.prev_winvS[f] + rho_i * (self.invK_mm*self.shape_sw[f]
                            / self.rate_sw[f] + w_i * Sigma_w_f)

                z0 = pref_likelihood(self.obs_f, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                     - self.G.dot(self.w[self.w_idx_i, f:f+1] * self.y[f:f+1, self.y_idx_i].T) # P x NU_i

                invQ_f = (self.y[f:f+1, self.y_idx_i].T * self.G.T / self.Q[None, self.data_obs_idx_i]).dot(
                    self.z[self.data_obs_idx_i] - z0)

                x = covpair.dot(invQ_f)

                # need to get x for current iteration and merge using SVI weighted sum
                self.winvSm[:, f] = (1-rho_i) * self.prev_winvSm[:, f] + rho_i * w_i * x.flatten()

                self.wS[f] = np.linalg.inv(self.winvS[f])
                self.w_u[:, f] = self.wS[f].dot(self.winvSm[:, f])

                self.w[:, f:f+1], _ = inducing_to_observation_moments(self.K_mm / self.shape_sw[f] * self.rate_sw[f],
                                    self.invK_mm, self.K_nm, self.w_u[:, f:f+1], 0)

                self.obs_f = (self.w.dot(self.y) + self.t).T.reshape(self.N * self.Npeople, 1)

            diff = np.max(np.abs(oldG - self.G))

            if self.verbose:
                logging.debug("expec_w: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

        if self.verbose:
            logging.debug('Computing w_cov_i')
        Kw_i = self.Kw[self.n_idx_i, :][:, self.n_idx_i]
        K_nm_i = self.K_nm[self.n_idx_i]

        self.w_cov_i = np.zeros((self.Nfactors, self.n_idx_i.shape[0], self.n_idx_i.shape[0]))

        covpair = K_nm_i.dot(self.invK_mm)
        sw = self.shape_sw / self.rate_sw

        for f in range(self.Nfactors):
            print('w_cov_i: wS: %f' % np.min(np.diag(self.wS[f])))

            self.w_cov_i[f] = Kw_i / sw[f] + covpair.dot(self.wS[f] - self.K_mm/sw[f]).dot(covpair.T)

            self.shape_sw[f], self.rate_sw[f] = expec_output_scale(self.shape_sw0, self.rate_sw0, N,
                                                       self.invK_mm, self.w_u[:, f:f + 1], np.zeros((N, 1)),
                                                       f_cov=self.wS[f])


    def _expec_y(self):
        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        w_i = np.sum(self.nobs) / float(self.update_size)

        self.prev_yinvSm = self.yinvSm
        self.prev_yinvS = self.yinvS

        if self.person_features is not None:
            covpair = self.invKy_mm_block.dot(self.Ky_nm_block[self.y_idx_i].T)
        else:
            covpair = np.eye(self.Npeople)[self.y_idx_i, :].T

        if self.verbose:
            logging.debug('_expec_y: starting update.')

        for G_iter in range(self.max_iter_G):
            oldG = self.G

            self.G = self.G * 0.1 + self._compute_jacobian() * 0.9

            invQG =  self.G.T / self.Q[None, self.data_obs_idx_i]

            for f in range(self.Nfactors):

                wcov_idxs = [np.argwhere(self.n_idx_i == idx).flatten()[0] for idx in self.w_idx_i]
                scaling_2 = self.w[self.w_idx_i, f:f+1].dot(self.w[self.w_idx_i, f:f+1].T) + self.w_cov_i[f][wcov_idxs, :][:, wcov_idxs]
                Sigma_y_f =  covpair.dot(scaling_2 * invQG.dot(self.G)).dot(covpair.T)

                # need to get invS for current iteration and merge using SVI weighted sum
                if self.person_features is not None:
                    self.yinvS[f] = (1-rho_i) * self.prev_yinvS[f] + rho_i * (
                        self.invKy_mm_block + w_i * Sigma_y_f)
                else:
                    Sigma_y_f = np.diag(Sigma_y_f)
                    print(np.min(np.diag(scaling_2)))
                    self.yinvS[f] = (1 - rho_i) * self.prev_yinvS[f] + rho_i * (1 + w_i * Sigma_y_f)

                z0 = pref_likelihood(self.obs_f, v=self.pref_v[self.data_obs_idx_i], u=self.pref_u[self.data_obs_idx_i]) \
                     - self.G.dot(self.w[self.w_idx_i, f:f+1] * self.y[f:f+1, self.y_idx_i].T)

                invQ_f = (self.w[self.w_idx_i, f:f+1] * self.G.T / self.Q[None, self.data_obs_idx_i]).dot(
                    self.z[self.data_obs_idx_i] - z0)

                x = covpair.dot(invQ_f)

                # need to get x for current iteration and merge using SVI weighted sum
                self.yinvSm[:, f] = (1-rho_i) * self.prev_yinvSm[:, f] + rho_i * w_i * x.flatten()

                if self.person_features is None:
                    self.yS[f] = 1.0 / self.yinvS[f]
                    self.y_u[f] = (self.yS[f].T * self.yinvSm[:, f]).T
                    self.y[f] = self.y_u[f]
                else:
                    self.yS[f] = np.linalg.inv(self.yinvS[f])
                    self.y_u[f] = self.yS[f].dot(self.yinvSm[:, f])

                    yf, _ = inducing_to_observation_moments(self.Ky_mm_block,
                            self.invKy_mm_block, self.Ky_nm_block, self.y_u[f:f+1, :].T, 0)
                    self.y[f:f + 1] = yf.T

                self.obs_f = (self.w.dot(self.y) + self.t).T.reshape(self.N * self.Npeople, 1)

            diff = np.max(np.abs(oldG - self.G))

            if self.verbose:
                logging.debug("expec_y: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break


    def _update_sample_idxs(self):

        self.data_obs_idx_i = np.sort(np.random.choice(self.nobs, self.update_size, replace=False))

        data_idx_i = np.zeros((self.N, self.Npeople), dtype=bool)
        data_idx_i[self.tpref_v[self.data_obs_idx_i], self.personIDs[self.data_obs_idx_i]] = True
        data_idx_i[self.tpref_u[self.data_obs_idx_i], self.personIDs[self.data_obs_idx_i]] = True

        separate_idx_i = np.argwhere(data_idx_i.T)
        self.y_idx_i = separate_idx_i[:, 0]
        self.w_idx_i = separate_idx_i[:, 1]
        self.joint_idx_i = self.w_idx_i + (self.N * self.y_idx_i)

        self.n_idx_i, pref_idxs = np.unique([self.tpref_v[self.data_obs_idx_i], self.tpref_u[self.data_obs_idx_i]],
                                            return_inverse=True)
        self.p_idx_i, _ = np.unique(self.personIDs[self.data_obs_idx_i], return_inverse=True)
        pref_idxs = pref_idxs.reshape(2, self.update_size)

        # the index into n_idx_i for each of the selected prefs
        self.pref_v_w_idx = pref_idxs[0]#np.array([np.argwhere(self.n_idx_i == n).flatten() for n in self.tpref_v[self.data_obs_idx_i]])
        self.pref_u_w_idx = pref_idxs[1]#np.array([np.argwhere(self.n_idx_i == n).flatten() for n in self.tpref_u[self.data_obs_idx_i]])

        if self.verbose:
            logging.debug('Computing y_cov_i')

        self.y_cov_i = np.zeros((self.Nfactors, self.p_idx_i.shape[0], self.p_idx_i.shape[0]))

        if self.person_features is not None:
            Ky_i = self.Ky[self.p_idx_i, :][:, self.p_idx_i]
            K_nm_i = self.Ky_nm_block[self.p_idx_i]
            covpair = K_nm_i.dot(self.invKy_mm_block)

            for f in range(self.Nfactors):
                self.y_cov_i[f] = Ky_i + covpair.dot(self.yS[f] - self.Ky_mm_block).dot(covpair.T)
        else:
            for f in range(self.Nfactors):
                self.y_cov_i[f] = np.diag(self.yS[f][self.p_idx_i])
                print('y_cov_i: %f' % np.min(self.y_cov_i[f]))

    def _update_sample(self):

        self._update_sample_idxs()

        if self.use_t:
            self.Kts_mm = self.K_mm / self.st

        self.G = -np.inf # need to reset G because we have a new sample to compute it for

    def data_ll(self, logrho, lognotrho):
        bc = binom(np.ones(self.z.shape), self.z)
        logbc = np.log(bc)
        lpobs = np.sum(self.z * logrho + (1 - self.z) * lognotrho)
        lpobs += np.sum(logbc)

        data_ll = lpobs
        return data_ll

    def _logpD(self):
        # K_star, um_minus_mu0, uS, invK_mm, v, u
        if self.person_features is None:
            y_var = self.yS
        else:
            y_var = np.array([np.diag(self.yS[f]) for f in range(self.Nfactors)])

        print('yvar: ')
        print(np.min(y_var))

        logrho, lognotrho, _ = self._post_sample(self.K_nm, self.invK_mm, self.w_u, self.wS, self.t_u, self.tS,
                                     self.Ky_nm_block, self.invKy_mm_block, self.y_u,
                                     y_var,
                                     self.pref_v, self.pref_u, expectedlog=True)


        data_ll = self.data_ll(logrho, lognotrho)

        return data_ll

    def lowerbound(self):

        data_ll = self._logpD()

        Elnsw = psi(self.shape_sw) - np.log(self.rate_sw)
        if self.use_t:
            Elnst = psi(self.shape_st) - np.log(self.rate_st)
            st = self.st
        else:
            Elnst = 0
            st = 1

        sw = self.shape_sw / self.rate_sw

        # the parameter N is not multiplied here by Nfactors because it will be multiplied by the s value for each
        # factor and summed inside the function
        logpw = np.sum([expec_pdf_gaussian(self.K_mm, self.invK_mm, Elnsw[f], self.ninducing,
                    self.shape_sw[f] / self.rate_sw[f], self.w_u[:, f:f+1], 0, self.wS[f], 0)
                        for f in range(self.Nfactors)])

        logqw = np.sum([expec_q_gaussian(self.wS[f], self.ninducing * self.Nfactors) for f in range(self.Nfactors)])

        if self.use_t:
            logpt = expec_pdf_gaussian(self.K_mm, self.invK_mm, Elnst, self.ninducing, st, self.t_u, self.t_mu0,
                                       0, 0) - 0.5 * self.ninducing
            logqt = expec_q_gaussian(self.tS, self.ninducing)
        else:
            logpt = 0
            logqt = 0

        logpy = np.sum([expec_pdf_gaussian(self.Ky_mm_block, self.invKy_mm_block, 0, self.y_ninducing, 1,
                                   self.y_u[f:f+1, :].T, 0, self.yS[f], 0) for f in range(self.Nfactors)])
        logqy = np.sum([expec_q_gaussian(self.yS[f], self.y_ninducing * self.Nfactors) for f in range(self.Nfactors)])

        logps_w = 0
        logqs_w = 0
        for f in range(self.Nfactors):
            logps_w += lnp_output_scale(self.shape_sw0, self.rate_sw0, self.shape_sw[f], self.rate_sw[f], sw[f],
                                        Elnsw[f])
            logqs_w += lnq_output_scale(self.shape_sw[f], self.rate_sw[f], sw[f], Elnsw[f])

        logps_t = lnp_output_scale(self.shape_st0, self.rate_st0, self.shape_st, self.rate_st, st, Elnst)
        logqs_t = lnq_output_scale(self.shape_st, self.rate_st, st, Elnst)

        w_terms = logpw - logqw + logps_w - logqs_w
        y_terms = logpy - logqy
        t_terms = logpt - logqt + logps_t - logqs_t

        lb = data_ll + t_terms + w_terms + y_terms

        if self.verbose:
            logging.debug('s_w=%s' % (self.shape_sw / self.rate_sw))
            logging.debug('s_t=%.2f' % (self.shape_st / self.rate_st))

        if self.verbose:
            logging.debug('likelihood=%.3f, wterms=%.3f, yterms=%.3f, tterms=%.3f' % (data_ll, w_terms, y_terms, t_terms))

        logging.debug("Iteration %i: Lower bound = %.3f, " % (self.vb_iter, lb))

        if self.verbose:
            logging.debug("t: %.2f, %.2f" % (np.min(self.t), np.max(self.t)))
            logging.debug("w: %.2f, %.2f" % (np.min(self.w), np.max(self.w)))
            logging.debug("y: %f, %f" % (np.min(self.y), np.max(self.y)))

        return lb

    def _predict_w_t(self, coords_1, return_cov=True):

        # kernel between pidxs and t
        if self.verbose:
            logging.debug('Computing K_nm in predict_w_t')
        K = self.kernel_func(coords_1, self.ls, self.inducing_coords)
        if self.verbose:
            logging.debug('Computing K_nn in predict_w_t')
        K_starstar = self.kernel_func(coords_1, self.ls, coords_1)
        covpair = K.dot(self.invK_mm)
        N = coords_1.shape[0]

        # use kernel to compute t.
        if self.use_t:
            t_out = K.dot(self.invK_mm).dot(self.t_u)

            covpair_uS = covpair.dot(self.tS)
            if return_cov:
                cov_t = K_starstar * self.rate_st / self.shape_st + (covpair_uS -
                                                                     covpair.dot(self.Kts_mm)).dot(covpair.T)
            else:
                cov_t = None
        else:
            t_out = np.zeros((N, 1))
            if return_cov:
                cov_t = np.zeros((N, N))
            else:
                cov_t = None

        # kernel between pidxs and w -- use kernel to compute w. Don't need Kw_mm block-diagonal matrix
        w_out = K.dot(self.invK_mm).dot(self.w_u)

        if return_cov:
            cov_w = np.zeros((self.Nfactors, N, N))
            for f in range(self.Nfactors):
                cov_w[f] = K_starstar  * self.rate_sw[f] / self.shape_sw[f] + \
                                covpair.dot(self.wS[f] - self.K_mm * self.rate_sw[f] / self.shape_sw[f]).dot(covpair.T)
        else:
            cov_w = None

        return t_out, w_out, cov_t, cov_w

    def predict_t(self, item_features):
        '''
        Predict the common consensus function values using t
        '''
        if item_features is None:
            # reuse the training points
            t = self.t
        else:
            # use kernel to compute t.
            if self.use_t:
                # kernel between pidxs and t
                if self.verbose:
                    logging.debug('Computing K_nm in predict_t')
                K = self.kernel_func(item_features, self.ls, self.inducing_coords)
                t = K.dot(self.invK_mm).dot(self.t_u)
            else:
                N = item_features.shape[0]
                t = np.zeros((N, 1))

        return t

    def predict_common(self, item_features, item_0_idxs, item_1_idxs):
        '''
        Predict the common consensus pairwise labels using t.
        '''
        if not self.use_t:
            return np.zeros(len(item_0_idxs))

        if self.verbose:
            logging.debug('Computing K_nm in predict_common')

        if item_features is None:
            item_features = self.obs_coords

        K = self.kernel_func(item_features, self.ls, self.inducing_coords)
        if self.verbose:
            logging.debug('Computing K_nn in predict_common')
        K_starstar = self.kernel_func(item_features, self.ls, item_features)
        covpair = K.dot(self.invK_mm)
        covpair_uS = covpair.dot(self.tS)

        t_out = K.dot(self.invK_mm).dot(self.t_u)
        cov_t = K_starstar * self.rate_st / self.shape_st + (covpair_uS - covpair.dot(self.Kts_mm)).dot(covpair.T)

        predicted_prefs = pref_likelihood(t_out, cov_t[item_0_idxs, item_1_idxs]
                                          + cov_t[item_0_idxs, item_1_idxs]
                                          - cov_t[item_0_idxs, item_1_idxs]
                                          - cov_t[item_0_idxs, item_1_idxs],
                                          subset_idxs=[], v=item_0_idxs, u=item_1_idxs)

        return predicted_prefs

    def _y_var(self):
        if self.person_features is None:
            return self.yS

        v = np.array([inducing_to_observation_moments(self.Ky_mm_block, self.invKy_mm_block, self.Ky_nm_block,
                                       self.y_u[f:f+1, :].T, 0, S=self.yS[f], K_nn=1.0, full_cov=False)[1]
                      for f in range(self.Nfactors)])
        return v

    def _predict_y(self, person_features, return_cov=True):

        if person_features is None and self.person_features is None:

            if return_cov:
                cov_y = np.zeros((self.Nfactors, self.y_ninducing, self.y_ninducing))
                for f in range(self.Nfactors):
                    cov_y[f] = self.yS[f]
            else:
                cov_y = None

            return self.y_u, cov_y

        elif person_features is None:
            person_features = self.person_features
            Ky = self.Ky_nm_block
            Ky_starstar = self.Ky_nm_block.dot(self.invKy_mm_block).dot(self.Ky_nm_block.T)

        else:
            if self.verbose:
                logging.debug('Computing Ky_nm in predict_y')
            Ky = self.y_kernel_func(person_features, self.lsy, self.y_inducing_coords)
            if self.verbose:
                logging.debug('Computing Ky_nn in predict_y')
            Ky_starstar = self.y_kernel_func(person_features, self.lsy, person_features)

        covpair = Ky.dot(self.invKy_mm_block)
        Npeople = person_features.shape[0]

        y_out = Ky.dot(self.invKy_mm_block).dot(self.y_u.T).T

        if return_cov:
            cov_y = np.zeros((self.Nfactors, Npeople, Npeople))
            for f in range(self.Nfactors):
                cov_y[f] = Ky_starstar + covpair.dot(self.yS[f] - self.Ky_mm_block).dot(covpair.T)
        else:
            cov_y = None

        return y_out, cov_y

    def _compute_gradients_all_dims(self, lstype, dimensions):
        mll_jac = np.zeros(len(dimensions), dtype=float)

        if lstype == 'item' or (lstype == 'both'):
            common_term = np.sum(np.array([(self.w_u[:, f:f+1].dot(self.w_u[:, f:f+1].T) + self.wS[f]).dot(
                self.shape_sw[f] / self.rate_sw[f] * self.invK_mm) - np.eye(self.ninducing)
                for f in range(self.Nfactors)]), axis=0)
            if self.use_t:
                common_term += (self.t_u.dot(self.t_u.T) + self.tS).dot(self.shape_st / self.rate_st * self.invK_mm) \
                               - np.eye(self.ninducing)

            for dim in dimensions[:self.nitem_features]:
                if self.verbose and np.mod(dim, 1000)==0:
                    logging.debug('Computing gradient for %s dimension %i' % (lstype, dim))
                mll_jac[dim] = self._gradient_dim(self.invK_mm, common_term, 'item', dim)

        if (lstype == 'person' or (lstype == 'both')) and self.person_features is not None:
            common_term = np.sum(np.array([(self.y_u[f:f+1].T.dot(self.y_u[f:f+1,:]) + self.yS[f]).dot(self.invKy_mm_block)
                                           - np.eye(self.y_ninducing) for f in range(self.Nfactors)]), axis=0)

            for dim in dimensions[self.nitem_features:]:
                if self.verbose and np.mod(dim, 1000)==0:
                    logging.debug('Computing gradient for %s dimension %i' % (lstype, dim))
                mll_jac[dim + self.nitem_features] = self._gradient_dim(self.invKy_mm_block, common_term, 'person', dim)

        return mll_jac

    def _gradient_dim(self, invK_mm, common_term, lstype, dimension):
        # compute the gradient. This should follow the MAP estimate from chu and ghahramani.
        # Terms that don't involve the hyperparameter are zero; implicit dependencies drop out if we only calculate
        # gradient when converged due to the coordinate ascent method.
        if lstype == 'item':
            dKdls = self.K_mm * self.kernel_der(self.inducing_coords, self.ls, dimension)
        elif lstype == 'person' and self.person_features is not None:
            dKdls = self.Ky_mm_block * self.kernel_der(self.y_inducing_coords, self.lsy, dimension)

        return 0.5 * np.trace(common_term.dot(dKdls).dot(invK_mm))