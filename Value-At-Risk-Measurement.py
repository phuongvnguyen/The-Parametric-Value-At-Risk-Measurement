#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{The Value-At-Risk Measurements }}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $$\text{1. Issue}$$
# 
# One of the most frequently used aspects of the volatility models is to measure the Value-At-Risk (VaR). This project attempts to use the GARCH model to measure the VaR.
# 
# $$\text{2. Methodology}$$
# 
# The GARCH model specification
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%.
# 
# 
# $$\text{3. Dataset}$$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/The-Value-At-Risk-Forecasting/blob/master/mydata.xlsx
# 
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[1]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[4]:


data = pd.read_excel("mydata.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[5]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[6]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[7]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[8]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[9]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[10]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[11]:


daily_return.index


# ### Plotting returns

# In[17]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(daily_return.Return['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns', fontsize=15,fontweight='bold'
             ,color='b')
plt.title('20/09/2007- 29/12/2017',fontsize=13,fontweight='bold',
          color='b')
plt.ylabel('Return (%)',fontsize=10)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=10,fontweight='normal',color='k')


# # Modelling GARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 

# In[ ]:


for row in daily_return.index: 
    print(row)


# In[51]:


#garch = arch_model(daily_return,mean='AR',lags=5,
 #                  vol='GARCH',dist='studentst',
  #              p=1, o=0, q=1)
garch = arch_model(daily_return,vol='Garch', p=1, o=0, q=1, dist='skewt')
results_garch = garch.fit(last_obs='2016-12-30', update_freq=1,disp='on')
print(results_garch.summary())


# # Estimating the VaR
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%.
# 
# ## Computing the quantiles
# 
# The quantiles, $q_{\alpha}$, can be computed using the ppf method of the distribution attached to the model. The quantiles, $q_{\alpha}$, are given below.

# In[60]:


quantiles_VaRgarch = garch.distribution.ppf([0.01, 0.05], results_garch.params[-2:])
print(Bold+'The quantiles at 1% and 5% are given as follows'+End)
print(quantiles_VaRgarch)


# ## Computing the conditional mean and volatilitie

# In[94]:


forecasts_VaRgarch = results_garch.forecast(start='2017-01-03')
cond_mean_VaRgarch = forecasts_VaRgarch.mean['2017':]
cond_var_VaRgarch = forecasts_VaRgarch.variance['2017':]


# ## Computing the Value-At-Risk (VaR)

# In[104]:


value_at_risk = -cond_mean_VaRgarch.values - np.sqrt(cond_var_VaRgarch).values * quantiles_VaRgarch[None, :]

value_at_risk = pd.DataFrame(
    value_at_risk, columns=['1%', '5%'], index=cond_var_VaRgarch.index)

value_at_risk.head(5)


# # Visualizing the VaR vs actual values
# ## Picking actual data

# In[102]:


rets_2017= daily_return['2017':].copy()
rets_2017.name = 'Return'
rets_2017.head(5)


# ## Plotting

# In[110]:


fig=plt.figure(figsize=(12,5))
plt.plot(value_at_risk['1%'] ,LineWidth=2,
         linestyle='--',label='VaR returns at 1%')
plt.plot(value_at_risk['5%'] ,LineWidth=2,
         linestyle=':',label='VaR returns at 5%')
plt.plot(rets_2017['Return'] ,LineWidth=2,
         linestyle='-',label='Actual return')
plt.suptitle('The Daily GARCH-based Value-At-Risk (VaR) Measurements of the Vingroup stock', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.autoscale(enable=True,axis='both',tight=True)
plt.legend()


# In[ ]:




