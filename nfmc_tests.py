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
import scipy

print(f"Running on PyMC3 v{pm.__version__}")

az.style.use("arviz-darkgrid")

'''
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

n = 250
np.random.seed(1111)
# Draw precision matrix from the Wishart distribution, with n degrees of freedom and identity scale.
wish = scipy.stats.wishart(df=n, scale=np.eye(n))
A = wish.rvs()
detA = np.linalg.det(A)
print(np.linalg.inv(A))

def gaussian(x):
    log_like = (
        -0.5 * n * tt.log(2 * np.pi)
        - 0.5 * (x).T.dot(A).dot(x)
    )
    
    return log_like

with pm.Model() as model:
    
    #X = pm.Flat('X', shape=n, testval=0)
    X = pm.Uniform('X', lower=-5, upper=5, shape=n, testval=0)
    llk = pm.Potential("llk", gaussian(X))
    #X = pm.MvNormal('X', mu=np.zeros(n), tau=A, shape=n)
    
with model:
    
    g_start = {'X': np.zeros(n)}
    g_trace = pm.sample_nfmc(draws=500, init_draws=5000, resampling_draws=1000, 
                             init_method='advi', start=g_start, local_thresh=1, 
                             local_step_size=0.5, init_EL2O='scipy', mean_field_EL2O=True, 
                             use_hess_EL2O=False, EL2O_optim_method='adam', EL2O_draws=250,
                             local_grad=True, init_local=True, full_local=False, nf_local_iter=20,
                             nf_iter=40, chains=1,  frac_validate=0.2, alpha=(0,0), parallel=False,
                             NBfirstlayer=True, k_trunc=0.5, iteration=10, norm_tol=0.03,
                             bw_factor_min=0.5, bw_factor_max=5.0, bw_factor_num=21, 
                             max_line_search=2, adam_steps=2)
    g_az_trace = az.from_pymc3(g_trace)
