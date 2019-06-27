import numpy as np
from scipy.stats import norm

label = 1

# z has a normal distribution with
mu = 0
sigmasq = 1

# we want to compute the expected log likelihood:
def loglikelihood(z):
    return np.log(norm.cdf(z))

def likelihood(z):
    return norm.cdf(z)

# to take an expectation, we sample z:
samplez = norm.rvs(mu, sigmasq**0.5, 2000)

ell = np.mean(loglikelihood(samplez))

print(ell)

# if we compute the log likelihood of the expectation:
lle = loglikelihood(mu / np.sqrt(1 + sigmasq))
print(lle)

z_i = mu+norm.pdf(mu)/(1-likelihood(mu))
print(z_i)

lle = -0.5 * (mu**2 + sigmasq + np.log(2*np.pi) + z_i**2) + mu*z_i
print(lle)