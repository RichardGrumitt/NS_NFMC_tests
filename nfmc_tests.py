import sys
# If you want to re-run, just modify this to where you put the NS_NFMC fork of the PyMC3 repo.
sys.path.insert(1, '/home/richard/pymc3_dev/')
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

print(f"Running on PyMC3 v{pm.__version__}")

az.style.use("arviz-darkgrid")

rg_model = pm.Model()
x = np.arange(10)
y = 2 * x + 1 + np.random.normal(scale=0.1)

with rg_model:

    a = pm.Normal('a', mu=2, sigma=1)
    b = pm.Normal('b', mu=1, sigma=1)

    like = pm.Normal('like', mu=a*x+b, sigma=0.1, observed=y)
    
    rg_trace = pm.sample_nfmc(1000, init_method='full_rank', local_thresh=3, local_step_size=0.1, 
                              local_grad=True, init_local=True, nf_local_iter=0, nf_iter=20, chains=1,
                              frac_validate=0.2, alpha=(0,0), parallel=False, bw_factor=2.0,
                              k_trunc=0.25, pareto=True, iteration=5)
    rg_az_trace = az.from_pymc3(rg_trace)
