import sys
# If you want to re-run, just modify this to where you put the NS_NFMC fork of the PyMC3 repo.
sys.path.insert(1, '/home/richard/pymc3_dev/')
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import pandas as pd

print(f"Running on PyMC3 v{pm.__version__}")

az.style.use("arviz-darkgrid")


rg_model = pm.Model()
x = np.arange(10)
y = 2 * x + 1 + np.random.normal(scale=0.1)

with rg_model:

    a = pm.Normal('a', mu=2, sigma=1)
    b = pm.Normal('b', mu=1, sigma=1)

    like = pm.Normal('like', mu=a*x+b, sigma=0.1, observed=y)

    start = {'a': 2.0, 'b': 1.0}
    rg_trace = pm.sample_nfmc(1000, init_method='adam', start=start, local_thresh=3, local_step_size=0.5, 
                              local_grad=True, init_local=True, nf_local_iter=20, nf_iter=20, chains=1,
                              frac_validate=0.2, alpha=(0,0), parallel=False, bw_factor=2.0,
                              k_trunc=0.25, pareto=True, iteration=5, norm_tol=0.05)
    rg_az_trace = az.from_pymc3(rg_trace)

'''
data = pd.read_csv(pm.get_data("radon.csv"))
data["log_radon"] = data["log_radon"].astype(theano.config.floatX)
county_names = data.county.unique()
county_idx = data.county_code.values

n_counties = len(data.county.unique())

with pm.Model() as hierarchical_model:
    # Hyperpriors for group nodes
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=100)
    sigma_a = pm.HalfNormal("sigma_a", 5.0)
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=100)
    sigma_b = pm.HalfNormal("sigma_b", 5.0)

    # Intercept for each county, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are
    # vectors of length n_counties).
    a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_counties)
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_counties)

    # Model error
    eps = pm.HalfCauchy("eps", 5.0)

    radon_est = a[county_idx] + b[county_idx] * data.floor.values

    # Data likelihood
    radon_like = pm.Normal("radon_like", mu=radon_est, sigma=eps, observed=data.log_radon)

with hierarchical_model:
    start = {'mu_a': 1.5, 'mu_b': -0.7, 'sigma_a': 0.3, 'sigma_b': 0.3, 'eps': 0.7,
             'a': np.random.normal(loc=1.5, scale=0.3, size=n_counties),
             'b': np.random.normal(loc=-0.7, scale=0.3, size=n_counties)}
    hierarchical_nf_trace = pm.sample_nfmc(500, init_method='adam', start=start, local_thresh=4, 
                                           local_step_size=0.5, local_grad=True, init_local=True, 
                                           full_local=False, nf_local_iter=10, nf_iter=40, chains=1,  
                                           frac_validate=0.2, alpha=(0,0), parallel=False,
                                           NBfirstlayer=True, bw_factor=2.0, k_trunc=0.5, pareto=True, 
                                           iteration=5, norm_tol=0.05)
    hierarchical_az_nf_trace = az.from_pymc3(hierarchical_nf_trace)
'''
