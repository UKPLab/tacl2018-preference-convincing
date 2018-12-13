
import numpy as np
import logging

from collab_pref_learning_svi import CollabPrefLearningSVI, inducing_to_observation_moments
from gp_pref_learning import pref_likelihood

from collab_pref_learning_vb import expec_output_scale

class CollabPrefLearningFITC(CollabPrefLearningSVI):

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
            Qfitc = np.diag(self.Q[self.data_obs_idx_i]) + (self.G*(
                np.ones(self.w_idx_i.shape[0]) -
                np.diag(self.K_nm[self.w_idx_i].dot(self.invK_mm).dot(self.K_nm[self.w_idx_i].T))
                )[None, :] * self.rate_st/self.shape_st).dot(self.G.T)
            GinvQG = self.G.T.dot(np.linalg.inv(Qfitc)).dot(self.G)

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

            for f in range(self.Nfactors):
                # scale the precision by y
                scaling_f = self.y[f:f+1, self.y_idx_i].T.dot(self.y[f:f+1, self.y_idx_i]) + \
                            self.y_cov_i[f][self.uy_idx_i, :][:, self.uy_idx_i]

                Qfitc = np.diag(self.Q[self.data_obs_idx_i]) + (self.G*(
                    np.ones(self.w_idx_i.shape[0]) -
                    np.diag(self.K_nm[self.w_idx_i].dot(self.invK_mm).dot(self.K_nm[self.w_idx_i].T))
                )[None, :] * self.rate_sw[f] / self.shape_sw[f]).dot(self.G.T)

                Sigma_w_f = covpair.dot(self.G.T.dot(np.linalg.inv(Qfitc)).dot(self.G) * scaling_f).dot(covpair.T)

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
        Kw_i = self.Kw[self.uw_i, :][:, self.uw_i]
        K_nm_i = self.K_nm[self.uw_i]

        self.w_cov_i = np.zeros((self.Nfactors, self.uw_i.shape[0], self.uw_i.shape[0]))

        covpair = K_nm_i.dot(self.invK_mm)
        sw = self.shape_sw / self.rate_sw

        for f in range(self.Nfactors):
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
            covpair = self.invKy_mm.dot(self.Ky_nm[self.y_idx_i].T)
        else:
            covpair = np.eye(self.Npeople)[self.y_idx_i, :].T

        if self.verbose:
            logging.debug('_expec_y: starting update.')

        for G_iter in range(self.max_iter_G):
            oldG = self.G

            self.G = self.G * 0.1 + self._compute_jacobian() * 0.9

            for f in range(self.Nfactors):

                scaling_f = self.w[self.w_idx_i, f:f+1].dot(self.w[self.w_idx_i, f:f+1].T) \
                            + self.w_cov_i[f][self.uw_idx_i, :][:, self.uw_idx_i]

                Qfitc = np.diag(self.Q[self.data_obs_idx_i]) + (self.G*(
                    np.ones(self.y_idx_i.shape[0]) -
                    np.diag(self.Ky_nm[self.y_idx_i].dot(self.invKy_mm).dot(self.Ky_nm[self.y_idx_i].T))
                )[None, :] * self.rate_sy[f] / self.shape_sy[f]).dot(self.G.T)

                Sigma_y_f =  covpair.dot(scaling_f * self.G.T.dot(np.linalg.inv(Qfitc)).dot(self.G)).dot(covpair.T)

                # need to get invS for current iteration and merge using SVI weighted sum
                if self.person_features is not None:
                    self.yinvS[f] = (1-rho_i) * self.prev_yinvS[f] + rho_i * (self.shape_sy[f] / self.rate_sy[f] *
                                                                              self.invKy_mm + w_i * Sigma_y_f)
                else:
                    Sigma_y_f = np.diag(Sigma_y_f)
                    self.yinvS[f] = (1 - rho_i) * self.prev_yinvS[f] + rho_i * (self.shape_sy[f] / self.rate_sy[f]
                                                                                + w_i * Sigma_y_f)

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

                    yf, _ = inducing_to_observation_moments(self.Ky_mm_block / self.shape_sy[f] * self.rate_sy[f],
                                                            self.invKy_mm, self.Ky_nm, self.y_u[f:f + 1, :].T, 0)

                    self.y[f:f + 1] = yf.T

                self.obs_f = (self.w.dot(self.y) + self.t).T.reshape(self.N * self.Npeople, 1)

            diff = np.max(np.abs(oldG - self.G))

            if self.verbose:
                logging.debug("expec_y: iter %i, G-diff=%f" % (G_iter, diff))

            if diff < self.conv_threshold_G:
                break

        for f in range(self.Nfactors):
            self.shape_sy[f], self.rate_sy[f] = expec_output_scale(self.shape_sy0, self.rate_sy0, self.y_ninducing,
                                                                   self.invKy_mm, self.y_u[f:f + 1, :], np.zeros((self.y_ninducing, 1)),
                                                                   f_cov=self.yS[f])
