#!/usr/bin/env python
# coding: utf-8

# # Bayesian additive regression tree
# 
# <a href="http://www-stat.wharton.upenn.edu/~edgeorge/Research_papers/BART%20June%2008.pdf">paper</a>
# 
# a sum-of-trees model for approximating an unknown function
# 
# To avoid overfitting, BART uses a regularization prior that forces each tree to be able to explain only a limited subset of the relationships between the covariates and the predictor variable.
# 
# ### BART:a sum-of-trees model + a regularization prior on the parameters of that model
# 
# sum-of-trees model
# 
# $y=f(x)+ \epsilon$..$f(x)$->$E(y|x)=h(x)=\sum_{j}^{m}g_{j}(x)$, $g(x)$ us a regression tree
# for a regression tree, the value $\mu_{j}$ of the terminal mode is assigned to a sample with $X={x_{1},x_{2},....x_{p}}$, so we can have $g(x;T,M)$ with T is the nodes and decision rules, M is the $\mu$, and 
# $y=\sum_{j}^{m}g(x;T_{j},M_{j})+\epsilon$, different trees have different $T_{j},M_{j}$
# 
# When every terminal node assignment depends on just a single component of  x, the sum-of-trees model reduces to a simple additive function, a sum of step functions of the individual components of x. m is the number of parameters, with how many trees, So how to set the m
# 
# #### the choice of m
# a fast and robust option is to choose m=200,then maybe check if a couple of other values makes any difference, to look if the performance improve with the increased m
# 
# A regularization prior
# Without such a regularizing influence, large tree components would overwhelm the rich structure,thereby limiting the advantages of the additive representation both in terms of function approximation and computation.
# <img src="image/bart_prior.png" width="800" height="400">
# 
# #### prior for T
# 1.the depth of the tree
# 2.distribution splitting variables-- uniform
# 3.the distribution on the splitting rule assignment in each interior node
# 
# ####  prior for $\mu|T$
# using a Gaussian distribution N(0,$\sigma$),$\sigma=\frac{0.5}{k\sqrt(m)}$, with k and m control the individual tree effects, which should not be too large
# 
# #### prior for $\sigma$
# the inverse chi-square distribution,For automatic use, Chipman et al. (2010) recommend the default setting 3,0.9 with $\nu,\lambda$
# 

# ### Using backfitting MCMC algorithm to sampling
# 
# chanllenge is to sample from the posterior $(T_{j},M_{j}|T_{(j)},M_{(j)},\sigma)$ in Gibbs sampler
# 
# what is the relationship between $T_{j},M_{j}$ and $T_{(j)},M_{(j)}$,  $R_{j}=y-\sum_{k<>j}g(x;T_{k},M_{k})$
# 
# for $T_{j},M_{j}|R_{j},\sigma$, it can be sampled $M_{j}|T_{j},R_{j},\sigma, T_{j}|R_{j},\sigma$
# 
# in $T_{j}|R_{j},\sigma$ is without $M_{j}$, which can be integrated, as $R_{j}=g(x;T_{j},M_{j})$ based on $T_{j},M_{j}$
# 
# #### How to draw $T_{j}$, also a MH algorithm with always accepting
# 
# propose a new tree based on the current tree, like add a terminal node and change decision rules of the nodes
# 
# so, first,have a proposal $T_{j}^{*}$, and we have $M_{j}^{*} and have R_{j}^{*}, which is used to the next draw of $T_{j}$
# each tree has its modification and end the gibbs,
# 

# the post samples can give approximation to the y and can be used to test partial dependance
# 
# and the frequency of component x to be used as the splitting rules can help do the variable selection

# #### BART+ probit =>classification
# 
# $P(Y=1|x)=\Phi(G(x))$... without $\sigma, G(x)=\sum g(x;M,T)$
# 
# with $M^{*}and T^{*}$, we can get G(x) then use $Z_{i}$ n samples to get appromation to P(Y=1|x), so Z is N(G(x),1)

# ### BART code
# use pymc3
# use BayesTree R library

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import arviz as az
from scipy.special import expit


# In[4]:


az.style.use('arviz-grayscale')
plt.rcParams["figure.dpi"] = 300
np.random.seed(5453)
viridish = [(0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 1.0),
            (0.1843137254901961, 0.4196078431372549, 0.5568627450980392, 1.0),
            (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 1.0),
            (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 1.0)]


# In[14]:


data = pd.read_csv("bikes_hour.csv")
data = data[::50]
data.sort_values(by='hr', inplace=True)
data.hr.values.astype(float)

X = np.atleast_2d(data["hr"]).T
Y = data["cnt"]

with pm.Model() as bart_g:
    σ= pm.HalfNormal('σ', Y.std())
    μ = pm.BART('μ', X, Y, m=50)
    y = pm.Normal('y', μ, σ, observed=Y)
    idata_bart_g = pm.sample(2000, chains=1, return_inferencedata=True)


# In[18]:


space_in = pd.read_csv("space_influenza.csv")
X = np.atleast_2d(space_in["age"]).T
Y = space_in["sick"]


Y_jittered = np.random.normal(Y, 0.02)
plt.plot(X[:,0], Y_jittered, ".", alpha=0.5)






