'''
functions for model
'''

import os
from os import walk
import pickle
import numpy as np
import pandas as pd
import random
import math
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import fsolve
from scipy.stats import norm
from tqdm import tqdm
import warnings
# alternative to optimisation by curvefit
from scipy.optimize import differential_evolution
from scipy.signal import butter, lfilter
from scipy import stats
from ipywidgets import interact, FloatSlider

import matplotlib.pyplot as plt

# for plotly graph display offline inside the notebook itself.
import plotly.offline as pyo
from plotly.offline import init_notebook_mode
import os


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(y, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    yfilter = lfilter(b, a, y)
    return yfilter

def lowpass(y, cutoff, fs, order=6):
    # example
    # order = 6
    # fs = 1000/dt       # sample rate, Hz
    # cutoff = 1000      # desired cutoff frequency of the filter, Hz
    # y = lowpass(y, cutoff=cutoff, fs=fs, order=6)
    b, a = butter_lowpass(cutoff, fs, order)
    yfilter = butter_lowpass_filter(y, cutoff, fs, order)    
    return yfilter

def mid_value(arr):
    middle_index = len(arr) // 2
    closest_value = arr[middle_index]
    return closest_value

# estimates rise and decays for initial fits
def tau_estimators(x, y, percent=90):
    # Find the maximum value of the signal
    ymax = np.max(y)

    # Find the index of the maximum value
    ind = np.argmax(y)

    # Find the threshold values for the rise and decay phases
    threshold_lower = ymax * (1 - percent/100)
    threshold_upper = ymax * percent/100

    # Find the indices where the signal crosses the thresholds for rise
    ind1 = mid_value(np.where(y[:ind] >= threshold_lower)[0])
    if any(value >= threshold_upper for value in y[:ind]):
        ind2 = mid_value(np.where(y[:ind] >= threshold_upper)[0])
    else: 
        ind2 = ind

    # Find the indices where the signal crosses the thresholds for decay
    ind3 = mid_value(np.where(y[ind:] <= threshold_upper)[0] + ind)
    ind4 = mid_value(np.where(y[ind:] <= threshold_lower)[0] + ind)

    trise = x[ind2] - x[ind1]
    T2 = x[ind4] - x[ind3] # tdecay
    if trise <= 0: # fixes very rare error when ind2 < ind1
        trise = 0.2 * T2
    # trise = T1*T2/(T1+T2)
    T1 = trise * T2 / (T2 - trise) # good estimate for T1
    return round(T1,3), round(T2,3)


def params_order5(X):
    if X[3] < X[1]:
        X[0:2], X[2:4], X[4]  = X[2:4].copy(), X[0:2].copy(), X[4]  
    return X

def params_order6(X):
    if X[4] < X[2]:
        X[0], X[3] = X[3], X[0]
        X[2], X[4] = X[4], X[2]
    return X

def params_order7(X):
    if X[5] < X[2]:
        X[0:3], X[3:6], X[6] = X[3:6].copy(), X[0:3].copy(), X[6]
    return X

def params_order9(X):
    if X[6] < X[2]:
        X[0:4], X[4:8], X[8] = X[4:8].copy(), X[0:4].copy(), X[8]
    return X

def find_absolute_max(y):
    max_value = np.max(np.abs(y))
    return max_value * np.sign(y[np.abs(y).argmax()])

def alpha_cf(x, a1, b1):
    return a1 * x * np.exp(-b1 * x)

def alpha(params, t):
    # Extract the parameters
    a1, tau1 = params
    # Compute alpha function
    y = a1 * t * np.exp(-t / tau1) 
    return y

# _alt versions take a1_max not a1 ie a1_max etc is the actual maximum of the alpha synapse
def alpha_alt(params, t):
    # Extract the parameters
    a1_max, tau1 = params
    a1 = a1_max/tau1 * np.exp(1)
    # Compute the sum of two alpha functions
    y = a1 * t * np.exp(-t / tau1) 
    return y

def alpha2_cf(x, a1, b1, a2, b2):
    return a1 * x * np.exp(-b1 * x) + a2 * x * np.exp(-b2 * x)

def alpha2(params, t):
    # Extract the parameters
    a1, tau1, a2, tau2 = params
    # Compute the sum of two alpha functions
    y = a1 * t * np.exp(-t / tau1) + a2 * t * np.exp(-t / tau2)
    return y

# _alt versions take a1_max not a1 ie a1_max etc is the actual maximum of the alpha synapse
def alpha2_alt(params, t):
    # Extract the parameters
    a1_max, tau1, a2_max, tau2 = params
    a1 = a1_max/tau1 * np.exp(1)
    a2 = a2_max/tau2 * np.exp(1)
    # Compute the sum of two alpha functions
    y = a1 * t * np.exp(-t / tau1) + a2 * t * np.exp(-t / tau2)
    return y

def product_cf(x, a1, b1, b2):
    return a1 * (1 - np.exp(-b1*x)) * np.exp(-b2*x)

def product(params, t):
    # Extract the parameters
    a1, tau1, tau2 = params
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2)
    return y 

def product_alt(params, t):
    # Extract the parameters
    a1_max, tau1, tau2 = params
    f = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2) 
    a1 = a1_max / f
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2)
    return y

def product2_cf(x, a1, b1, b2, a2, c1, c2):
    return a1 * (1 - np.exp(-b1*x)) * np.exp(-b2*x) + a2 * (1 - np.exp(-c1*x)) * np.exp(-c2*x)

def product2(params, t):
    # Extract the parameters
    a1, tau1, tau2, a2, tau3, tau4 = params
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) + a2 * (1 - np.exp(-t/tau3)) * np.exp(-t/tau4)
    return y

def product2_alt(params, t):
    # Extract the parameters
    a1_max, tau1, tau2, a2_max, tau3, tau4 = params
    f1 =  ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2) 
    a1 = a1_max / f1
    f2 = ((tau3/(tau3+tau4)) ** (tau3/tau4)) * tau4/(tau3+tau4)
    a2 = a2_max / f2
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) + a2 * (1 - np.exp(-t/tau3)) * np.exp(-t/tau4)
    return y

def product2_1r_cf(x, a1, b1, b2, a2, c2):
    return a1 * (1 - np.exp(-b1*x)) * np.exp(-b2*x) + a2 * (1 - np.exp(-b1*x)) * np.exp(-c2*x)

def product2_1r(params, t):
    # Extract the parameters
    a1, tau1, tau2, a2, tau4 = params
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) + a2 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau4)
    return y

def product2_1r_alt(params, t):
    # Extract the parameters
    a1_max, tau1, tau2, a2_max, tau4 = params
    f1 = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    a1 = a1_max / f1
    f2 = ((tau1/(tau1+tau4)) ** (tau1/tau4)) * tau4/(tau1+tau4)
    a2 = a2_max / f2
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) + a2 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau4)
    return y

def nll_alpha(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = alpha(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def nll_alpha2(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = alpha2(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def nll_product(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = product(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def nll_product2(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = product2(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def nll_product2_1r(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = product2_1r(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def fit_alpha(x, y, p0, bounds = [(0, None), (0, None), (0, None)]):
    # Perform MLE to fit talpha function
    result = minimize(nll_alpha, p0, args=(x, y), bounds=bounds)  # method='L-BFGS-B'
    return result.x

def fit_alpha2(x, y, p0, bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]):
    # Perform MLE to fit the sum of two alpha functions
    # Add bound for nonnegative sd
    result = minimize(nll_alpha2, p0, args=(x, y), bounds=bounds)  # method='L-BFGS-B'
    return result.x

def fit_product(x, y, p0, bounds = [(0, None), (0, None), (0, None), (0, None)] ):
    # Add bound for nonnegative sd
    result = minimize(nll_product, p0, args=(x, y), bounds=bounds)  # method='L-BFGS-B'
    return result.x

# def fit_product2(x, y, p0, bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)] ):
#     result = minimize(nll_product2, p0, args=(x, y), bounds=bounds)  # method='L-BFGS-B'
#     return result.x

def fit_product2(x, y, p0, bounds = None, constraints=None ):
    result = minimize(nll_product2, p0, args=(x, y), bounds=bounds, constraints=constraints)  # method='L-BFGS-B'
    return result.x

# Define inequality constraint: tpeak1 < tpeak2 
def tpeak_fun(tau1, tau2):
    return tau1 * np.log((tau1 + tau2) / tau1)

# # Define inequality constraint: rise tau1 < tau3 
# def constraint1(params):
#     return params[4] - params[1]

# Define inequality constraint: rise tau1_rise < tau2_rise
# if f > 1 then second rise time is at least f times larger than first rise time
def constraint1(p0, factor=1):
    tau1_rise = p0[1] * p0[2] / (p0[1] + p0[2])
    tau2_rise = p0[4] * p0[5] / (p0[4] + p0[5])
    return tau2_rise - factor*tau1_rise 

def tpeak_fun(tau1, tau2):
    return tau1 * np.log((tau1+tau2)/tau1)

def tau_rise_fun(tau1, tau2):
    return tau1 * tau2 / (tau1 + tau2)

# # Define inequality constraint: rise tpeak1 < tpeak2
# def constraint1(p0):
#     tpeak1 = tpeak_fun(p0[1],p0[2])
#     tpeak2 = tpeak_fun(p0[4],p0[5])
#     return tpeak2 - tpeak1
# Define inequality constraint: rise tau2 < tau4 
def constraint2(p0, factor=1):
    return p0[5] - factor*p0[2]

def fit_product2_1r(x, y, p0, bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)] ):
    result = minimize(nll_product2_1r, p0, args=(x, y), bounds=bounds)  # method='L-BFGS-B'
    return result.x

# converts parameter output to 'max' format where amplitude values are the response maximum
def alpha_conversion(pars):
    pars_out = pars.copy()
    a1 = pars[0]
    tau1 = pars[1]
    a1_max = a1 * tau1 / np.exp(1)
    pars_out[0] = a1_max
    return pars_out

# converts parameter output to 'absolute' from 'max' 
def alpha_conversion_reverse(pars):
    pars_out = pars.copy()
    a1_max = pars[0]
    tau1 = pars[1]
    a1 = a1_max *  np.exp(1) / tau1
    pars_out[0] = a1
    return pars_out

# alpha_kinetics calculates rise or decay
# if p[1] > p[0] the rise
# if p[1] < p[0] the decay
def alpha_kinetics(t, tau, initial_guess, p):
    
    # Defining the function for 100*p % rise
    def func_p(t, tau, p):
        return t * np.exp(-t/tau) - p * tau * np.exp(-1)

    p1 = p[0]
    p2 = p[1]

    # Solving the equations numerically
    t1 = fsolve(func_p, initial_guess, args=(tau, p1))
    t2 = fsolve(func_p, initial_guess, args=(tau, p2))

    # Calculating the 20-80% rise time
    return t2 - t1


def FITalpha(x, y, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=10000, p0_algorithm='trf', cv=0.4, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False):

    if initial_guess is not None:
        # convert to standard A1, tau1 form for fits
        initial_guess = alpha_conversion_reverse(initial_guess)
        N = 1 # provides solution for these starting values

    # Perform the fitting
    # Adjusted initial parameter guess
    # supress any errors associated with overflow encountered in exp
    # if max is true then function will return the maximum ie peak value of the response
    
    # if y contains NAN thenfind index of first and remove all points after (this situation can occur if imported
    # spreadsheet has different lengths of responses
    x,y = preprocess(x,y)
    
    # Perform the fitting        
    if initial_guess is None:
        if start_method == 'DE':
            params_start = DE_p0(x, y, method='alpha', percent=percent, cv=cv, N=N)
        elif start_method == 'LS':
    #         params_start = CF_p0(x, y, method='alpha', initial_guess=initial_guess, maxfev=maxfev, percent=percent)
            params_start = LS_p0(x, y, method='alpha', initial_guess=initial_guess, maxfev=maxfev, percent=percent, algorithm=p0_algorithm, cv=cv, N=N)
        elif start_method is None: 
            params_start = random_initial_guess(x=x, y=y, method='alpha', percent=percent, cv=cv, N=N)
    else:
        params_start = [initial_guess] 
    
    N = len(params_start)
    
    # find fits by MLE and isolate best fit with minimum ss
    dt = x[1] - x[0]
    sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    fits = []
    ss = []
    # switching off warnings to avoid runtime warning: divide by zero encountered in divide
    # only seems to occur when initial_guess=None; when initial_guess is None start values are not filtered through 'LS' or 'DE" methods
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            params = params_start[ii]
            p0 = [params[0], params[1], sd]  # 6 params plus term for sd
            bounds = [(0, None), (0, None), (0, None)]
            params_fit = fit_alpha(x, y, p0, bounds)
            fits.append(params_fit)
            y_fit = alpha(params_fit[0:2], x)
            ss.append(np.sum((y - y_fit)**2))
    idx = ss.index(min(ss))  
    params_fit = fits[idx]
    p0 = params_start[idx]

    # Generate the fitted curve
    y_fit = alpha(params_fit[:-1], x)
    # consider making a more accurate y_fit with more accurate x
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    # outputs are in raw form. If max is TRUE will convert a1 to a1_max, the max value of the response ie the peak
    params_out = alpha_conversion(params_fit).tolist()
    
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area = params_out[0] * params_out[1] * np.exp(1)
        rd = [alpha_kinetics(x, tau=params_out[1], initial_guess=params_out[1]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out[1], initial_guess=params_out[1]*2, p=decay)[0]
             ]
    elif kinetics == 'fit':
        # add area under curve
        area = np.trapz(y_fit, x)
        # percent rise and decay times 
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = alpha(params_fit[:-1], x1)
        rd = rise_and_decay_times(x=x1, y=y1, rise=rise, decay=decay)
    
    params_out.insert(2, rd[0])
    params_out.insert(3, rd[1])
    # area
    params_out.insert(4, area)
    start_out = alpha_conversion(p0)
    if initial_guess is None and start_method is not None:
        start_out = start_out.tolist() 

    # Column names with Unicode symbols
    columns=['peak', '\u03C4', f'rise{int(100*rise[0])}-{int(100*rise[1])}', f'decay{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    if criterion=='BIC':
        bic = IC(params_fit, x, y, criterion=criterion, func=nll_alpha)
        if plot:
            fig.show()
            out = params_out[0:len(columns)]
            out.append(bic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'BIC:'])
        if show_results:
            return params_out, start_out, bic

    elif criterion=='AIC':
        aic = IC(params_fit, x, y, criterion=criterion, func=nll_alpha)
        if plot:
            fig.show()
            out = params_out[0:len(columns)]
            out.append(aic)
            df = output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'AIC:'])
            df
        if show_results:
            return params_out, start_out, aic
    else:
        if plot:
            fig.show()
            out = params_out[0:len(columns)]
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:'])
        if show_results:
            return params_out, start_out

# converts parameter output to 'max' format where amplitude values are the response maximum
def alpha2_conversion(pars):
    pars_out = pars.copy()
    a1 = pars[0]
    tau1 = pars[1]
    a1_max = a1 * tau1 / np.exp(1)
    pars_out[0] = a1_max
    a2 = pars[2]
    tau2 = pars[3]
    a2_max = a2 * tau2 / np.exp(1)
    pars_out[2] = a2_max
    return pars_out

# converts parameter output to 'absolute' from 'max' 
def alpha2_conversion_reverse(pars):
    pars_out = pars.copy()
    a1_max = pars[0]
    tau1 = pars[1]
    a1 = a1_max *  np.exp(1) / tau1
    pars_out[0] = a1
    a2_max = pars[2]
    tau2 = pars[3]
    a2 = a2_max *  np.exp(1) / tau2
    pars_out[2] = a2
    return pars_out


def FITalpha2(x, y, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=10000, p0_algorithm='trf', cv=0.4, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False):
    # Perform the fitting
    # Adjusted initial parameter guess
    # supress any errors associated with overflow encountered in exp
    
    if initial_guess is not None:
        # convert to standard A1, tau1, A2, tau2 ...form
        initial_guess = alpha2_conversion_reverse(initial_guess)
        N = 1 # provides solution for these starting values

    # if y contains NAN then find index of first and remove all points after (this situation can occur if imported
    # spreadsheet has different lengths of responses
    x,y = preprocess(x,y)
    
    # Perform the fitting        
    if initial_guess is None:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if start_method == 'DE':
                params_start = DE_p0(x, y, method='alpha2', percent=percent, cv=cv, N=N)
            elif start_method == 'LS':
        #         params_start = CF_p0(x, y, method='alpha2', initial_guess=initial_guess, maxfev=maxfev, percent=percent)
                params_start = LS_p0(x, y, method='alpha2', initial_guess=initial_guess, maxfev=maxfev, percent=percent, algorithm=p0_algorithm, cv=cv, N=N)
            elif start_method is None: 
                params_start = random_initial_guess(x=x, y=y, method='product2', percent=percent, cv=cv, N=N)
            # Keep only those arrays that do not contain any NaN values
            params_start = [arr for arr in params_start if not np.isnan(arr).any()]
    else:
        params_start = [initial_guess]
    
    N = len(params_start)    

    # find fits by MLE and isolate best fit with minimum ss
    dt = x[1] - x[0]
    sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    fits = []
    ss = []
    # switching off warnings to avoid runtime warning: divide by zero encountered in divide
    # only seems to occur when initial_guess=None; when initial_guess is None start values are not filtered through 'LS' or 'DE" methods
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            params = params_start[ii]
            p0 = [params[0], params[1], params[2], params[3], sd]  # 4 params plus term for sd
            p0 = params_order5(p0)
            bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
            params_fit = fit_alpha2(x, y, p0, bounds)
            fits.append(params_fit)
            y_fit = alpha2(params_fit[0:4], x)
            ss.append(np.sum((y - y_fit)**2))
    idx = ss.index(min(ss))  
    params_fit = fits[idx]
    p0 = params_start[idx]
    
    # Generate the fitted curve
    y_fit = alpha2(params_fit[:-1], x)

    y_fit1 = alpha(params_fit[0:2], x)
    y_fit2 = alpha(params_fit[2:4], x)
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit (sum)', line=dict(color='indianred', dash='dot')))

    fig.add_trace(go.Scatter(x=x, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    # outputs are in raw form so convert a1 to a1_max, the max value of the response ie the peak
    params_out = alpha2_conversion(params_fit).tolist()
    start_out = alpha2_conversion(p0)
    
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area1 = params_out[0] * params_out[1] * np.exp(1)
        area2 = params_out[2] * params_out[3] * np.exp(1)
        rd1 = [alpha_kinetics(x, tau=params_out[1], initial_guess=params_out[1]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out[1], initial_guess=params_out[1]*2, p=decay)[0]
             ]
        rd2 = [alpha_kinetics(x, tau=params_out[3], initial_guess=params_out[3]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out[3], initial_guess=params_out[3]*2, p=decay)[0]
             ]
    # get rise and decay + area from fits curves
    elif kinetics == 'fit':
        # add area under curve
        area1 = np.trapz(y_fit1, x)
        area2 = np.trapz(y_fit2, x)
        # percent rise and decay times 
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = alpha(params_fit[0:2], x1)
        y2 = alpha(params_fit[2:4], x1)
        rd1 = rise_and_decay_times(x=x1, y=y1, rise=rise, decay=decay)
        rd2 = rise_and_decay_times(x=x1, y=y2, rise=rise, decay=decay)
            
    params_out.insert(2, rd1[0])
    params_out.insert(3, rd1[1])
    params_out.insert(6, rd2[0])
    params_out.insert(7, rd2[1])
    # area

    params_out.insert(4, area1)
    params_out.insert(9, area2)
        
    columns=['peak', '\u03C4', f'rise{int(100*rise[0])}-{int(100*rise[1])}', f'decay{int(100*decay[0])}-{int(100*decay[1])}', 'area']
        
    if criterion=='BIC':
        bic = IC(params_fit, x, y, criterion=criterion, func=nll_alpha2)
        if plot:
            fig.show()
            out = params_out[0:10]
            out.append(bic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'BIC:'])
        if show_results:
            return params_out, start_out, bic
 
    elif criterion=='AIC':
        aic = IC(params_fit, x, y, criterion=criterion, func=nll_alpha2)
        if plot:
            fig.show()
            out = params_out[0:10]
            out.append(aic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'AIC:'])
        if show_results:
            return params_out, start_out, aic
    else:
        if plot:
            fig.show()
            out = params_out[0:10]
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:'])
        if show_results:
            return params_out, start_out

def product_conversion(pars):
    a1 = pars[0]
    tau1 = pars[1]
    tau2 = pars[2]
    f = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    tau_rise = tau1 * tau2 / (tau1 + tau2)
    tau_decay = tau2
    t_peak = tau1 * np.log((tau1+tau2)/tau1)
    a1_max = a1 * f
    area = a1 * ( tau2**2 / (tau1 + tau2) )
    pars_out = [a1_max, tau_rise, tau_decay, t_peak, area]
    return pars_out

# for inputing an initial guess of form [A1_peak, tau1_rise, tau1_decay, A2_peak, tau2_rise, tau2_decay]
def product_conversion_reverse(pars):
    a1_max = pars[0]
    tau1_rise = pars[1]
    tau1_decay = pars[2]
    f1 = (tau1_decay/tau1_rise) ** (-tau1_rise/(tau1_decay-tau1_rise))  * (1 - tau1_rise/tau1_decay)

    tau1 = tau1_rise*tau1_decay / (tau1_decay - tau1_rise)
    tau2 = tau1_decay

#     t1_peak = tau1_decay * tau1_rise / (tau1_decay - tau1_rise)  * np.log(tau1_decay/tau1_rise)
    a1 = a1_max / f1
    
    pars_out = [a1, tau1, tau2]
    
    return pars_out

def output_results(fits, col_names, dp=3):
    # Rounding values
    fits = np.round(fits, dp)
    # Printing the names alongside the rounded values
    for name, value in zip(col_names, fits):
        print(f"{name}: {value}")

# product_kinetics calculates rise or decay
# if p[1] > p[0] the rise
# if p[1] < p[0] the decay
# inital guess is [t1, t2] where t2 > t1
def product_kinetics(t, tau_rise, tau_decay, initial_guess, p):
    f = (tau_decay/tau_rise) ** (-tau_rise/(tau_decay-tau_rise))  * (1 - tau_rise/tau_decay)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        def func_p(t, tau_rise, tau_decay, p, f):
            return np.exp(-t/tau_decay) - np.exp(-t/tau_rise) - f * p
        # Defining the function for 100*p % rise

        p1 = p[0]
        p2 = p[1]

        # Solving the equations numerically
        t1 = fsolve(func_p, initial_guess[0], args=(tau_rise, tau_decay, p1, f))
        t2 = fsolve(func_p, initial_guess[1], args=(tau_rise, tau_decay, p2, f))

        # Calculating the 20-80% rise time/80-20% decay time

        return t2 - t1

# estimates time points for rise and decays for initial fits
def rise_decay_estimators(x, y, rise=[0.2, 0.8], decay=[0.8, 0.2]):
    # Find the maximum value of the signal
    ymax = np.max(y)

    # Find the index of the maximum value
    ind = np.argmax(y)

    # Find the threshold values for the rise
    y1 = ymax * rise[0]
    y2 = ymax * rise[1]

    # Find the indices where the signal crosses the thresholds for rise
    # Check if the slice y[:ind] is empty
    if len(y[:ind]) == 0:
        # Assign a default value to ind1 if y[:ind] is empty
        ind1 = 0  # Or any other value you'd like to assign in this case
        ind2 = ind
    else:
        # If y[:ind] is not empty, proceed as normal
        ind1 = (np.abs(y[:ind] - y1)).argmin()
        ind2 = (np.abs(y[:ind] - y2)).argmin()
    
#     trise = x[ind2] - x[ind1]
    
    # Find the threshold values for the decay
    y3 = ymax * decay[0]
    y4 = ymax * decay[1]
    
    # Find the indices where the signal crosses the thresholds for decay
    # must correct for very slow decays
    if min(y[ind:]) > y4: # extrapolate?
        ind3 = ind
        ind4 = len(y[ind:]) - 1
    else:
        ind3 = (np.abs(y[ind:] - y3)).argmin() + ind
        ind4 = (np.abs(y[ind:] - y4)).argmin() + ind
#     tdecay = x[ind4] - x[ind3]    
    
    return [x[ind1], x[ind2], x[ind3], x[ind4]]

def rise_and_decay_times(x, y, tpeak=None, rise=[0.2, 0.8], decay=[0.8, 0.2], dp=3):
    
    if tpeak is None:
        ymax = max(y)
        ind = (np.abs(y - ymax)).argmin()
        tpeak = x[ind]
    else:
        ind = (np.abs(x - tpeak)).argmin()
        ymax = y[ind]
    
    y1 = rise[0] * ymax
    y2 = rise[1] * ymax

    # Get the indices where signal is closest to 20% and 80% of peak value before tpeak
    rise_ind1 = np.argmin(np.abs(np.array(y[:ind]) - y1))
    rise_ind2 = np.argmin(np.abs(np.array(y[:ind]) - y2))

    # Get the indices where signal is closest to 80% and 20% of peak value after tpeak
    y3 = decay[0] * ymax
    y4 = decay[1] * ymax
    decay_ind1 = np.argmin(np.abs(np.array(y[ind:]) - y3)) + ind  # fix here
    decay_ind2 = np.argmin(np.abs(np.array(y[ind:]) - y4)) + ind  # and here

    rise_time = x[rise_ind2] - x[rise_ind1]
    decay_time = x[decay_ind2] - x[decay_ind1]

    return np.round(rise_time, dp), np.round(decay_time, dp)

def extend_product(params, x, p=0.2, increment=100):
    # Calculate ymax
    y = product(params, x)
    ymax = np.max(y)
    ind = (np.abs(y - ymax)).argmin()
    dx = x[1] - x[0]
    while True:
        # Calculate y_fit1 with the current x1
        y = product(params, x)
        # Check the condition
        if p * ymax < np.min(y[ind:]):
            # If the condition is not met, extend x1 and repeat
            x = np.arange(0, x[-1]+increment, dx)  # Extend x1 by the defined increment
        else:
            # If the condition is met, break the loop
            break
    return x, y


def FITproduct(x, y, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=10000, p0_algorithm='trf', cv=0.4, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False):
        
    if initial_guess is not None:
        # convert to standard A1, tau1, tau2 ...form
        initial_guess = product_conversion_reverse(initial_guess)
        N = 1 # provides solution for these starting values

    # if y contains NAN then find index of first and remove all points after (this situation can occur if imported
    # spreadsheet has different lengths of responses
    x,y = preprocess(x,y)
    
    # Perform the fitting        
    if initial_guess is None:
        if start_method == 'DE':
            params_start = DE_p0(x, y, method='product', percent=percent, cv=cv, N=N)
        elif start_method == 'LS':
            params_start = LS_p0(x, y, method='product', initial_guess=initial_guess, maxfev=maxfev, percent=percent, algorithm=p0_algorithm, cv=cv, N=N)
        elif start_method is None: 
            params_start = random_initial_guess(x=x, y=y, method='product', percent=percent, cv=cv, N=N)
    
        # Keep only those arrays that do not contain any NaN values
        params_start = [arr for arr in params_start if not np.isnan(arr).any()]
    else:
        params_start = [initial_guess]
    
    N = len(params_start)    

    # find fits by MLE and isolate best fit with minimum ss
    dt = x[1] - x[0]
    sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    fits = []
    ss = []
    # switching off warnings to avoid runtime warning: divide by zero encountered in divide
    # only seems to occur when initial_guess=None; when initial_guess is None start values are not filtered through 'LS' or 'DE" methods
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            params = params_start[ii]
            p0 = [params[0], params[1], params[2], sd]  # 3 params plus term for sd
            bounds = [(0, None), (0, None), (0, None), (0, None)]
            params_fit = fit_product(x, y, p0, bounds)
            fits.append(params_fit)
            y_fit = product(params_fit[0:3], x)
            ss.append(np.sum((y - y_fit)**2))
    idx = ss.index(min(ss))  
    params_fit = fits[idx]
    p0 = params_start[idx]
    p0 = np.append(p0, sd)
    
    # Generate the fitted curve
    y_fit = product(params_fit[:-1], x)

    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    params_out = product_conversion(params_fit)
    start_out = p0
    start_out = product_conversion(start_out[0:3])

    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        # area returned by product_conversion
        ests = rise_decay_estimators(x=x, y=y_fit, rise=rise, decay=decay)
        rd = [product_kinetics(x, tau_rise=params_out[1], tau_decay=params_out[2], initial_guess=ests[0:2], p=rise)[0], 
              product_kinetics(x, tau_rise=params_out[1], tau_decay=params_out[2], initial_guess=ests[2:4], p=decay)[0]
             ]
    elif kinetics == 'fit':
        # add area under curve
        area = np.trapz(y_fit, x)
        # percent rise and decay times 
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = product(params_fit[:-1], x1)
        rd = rise_and_decay_times(x=x1, y=y1, rise=rise, decay=decay)
         # area
        params_out[4] = area

    params_out.insert(4, rd[0])
    params_out.insert(5, rd[1])

    # Column names with Unicode symbols
    columns = ['peak', 'τrise', 'τdecay', 'tpeak', f'rise{int(100*rise[0])}-{int(100*rise[1])}', f'decay{int(100*decay[0])}-{int(100*decay[1])}', 'area']    
    if criterion=='BIC':
        bic = IC(params_fit, x, y, criterion=criterion, func=nll_product)
        if plot:
            fig.show()
            out = params_out[0:7]
            out.append(bic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'BIC:'])
        if show_results:
            return np.round(params_out,dp), np.round(start_out,dp), np.round(params_fit,dp), np.round(bic,dp)
    elif criterion=='AIC':
        aic = IC(params_fit, x, y, criterion=criterion, func=nll_product)
        if plot:
            fig.show()
            out = params_out[0:7]
            out.append(aic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'AIC:'])
        if show_results:
            return np.round(params_out,dp), np.round(start_out,dp), np.round(params_fit,dp), np.round(aic,dp)
    else:
        if plot:
            fig.show()
            out = params_out[0:7]
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:'])
        if show_results:
            return np.round(params_out,dp), np.round(start_out,dp), np.round(params_fit,dp)

def product2_conversion(pars):
    a1 = pars[0]
    tau1 = pars[1]
    tau2 = pars[2]
    f1 = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    tau1_rise = tau1 * tau2 / (tau1 + tau2)
    tau1_decay = tau2
    t1_peak = tau1 * np.log((tau1+tau2)/tau1)
    a1_max = a1 * f1
    area1 = a1 * ( tau2**2 / (tau1 + tau2) )
        
    a2 = pars[3]
    tau3 = pars[4]
    tau4 = pars[5]
    f2 = ((tau3/(tau3+tau4)) ** (tau3/tau4)) * tau4/(tau3+tau4)
    tau2_rise = tau3 * tau4 / (tau3 + tau4)
    tau2_decay = tau4
    t2_peak = tau3 * np.log((tau3+tau4)/tau3)
    a2_max = a2 * f2
    area2 = a2 * ( tau4**2 / (tau3 + tau4) )

    pars_out = [a1_max, tau1_rise, tau1_decay, t1_peak, area1, a2_max, tau2_rise, tau2_decay, t2_peak, area2]
    
    return pars_out

# for inputing an initial guess of form [A1_peak, tau1_rise, tau1_decay, A2_peak, tau2_rise, tau2_decay]
def product2_conversion_reverse(pars):
    a1_max = pars[0]
    tau1_rise = pars[1]
    tau1_decay = pars[2]
    f1 = (tau1_decay/tau1_rise) ** (-tau1_rise/(tau1_decay-tau1_rise))  * (1 - tau1_rise/tau1_decay)

    tau1 = tau1_rise*tau1_decay / (tau1_decay - tau1_rise)
    tau2 = tau1_decay

#     t1_peak = tau1_decay * tau1_rise / (tau1_decay - tau1_rise)  * np.log(tau1_decay/tau1_rise)
    a1 = a1_max / f1
    
    a2_max = pars[3]
    tau2_rise = pars[4]
    tau2_decay = pars[5]
    f2 = (tau2_decay/tau2_rise) ** (-tau2_rise/(tau2_decay-tau2_rise))  * (1 - tau2_rise/tau2_decay)

    tau3 = tau2_rise*tau2_decay / (tau2_decay - tau2_rise)
    tau4 = tau2_decay

#     t2_peak = tau2_decay * tau2_rise / (tau2_decay - tau2_rise)  * np.log(tau2_decay/tau2_rise)
    a2 = a2_max / f2

    pars_out = [a1, tau1, tau2, a2, tau3, tau4]
    
    return pars_out


def FITproduct2(x, y, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', cv=0.4, N=30, slow_rise_constraint=True, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False):
    
    if initial_guess is not None:
        # convert to standard A1, tau1, tau2 ...form
        initial_guess = product2_conversion_reverse(initial_guess)
        N = 1 # provides solution for these starting values
        
    # if y contains NAN then find index of first and remove all points after (this situation can occur if imported
    # spreadsheet has different lengths of responses
    x,y = preprocess(x,y)
    
    # Perform the fitting        
    if initial_guess is None:
        if start_method == 'DE':
            params_start = DE_p0(x, y, method='product2', percent=percent, cv=cv, N=N)
        elif start_method == 'LS':
    #         params_start = CF_p0(x, y, method='product2', initial_guess=initial_guess, maxfev=maxfev, percent=percent)
            params_start = LS_p0(x, y, method='product2', initial_guess=initial_guess, maxfev=maxfev, percent=percent, algorithm=p0_algorithm,  cv=cv, N=N)
        elif start_method is None: 
            params_start = random_initial_guess(x=x, y=y, method='product2', percent=percent, cv=cv, N=N)

        # Keep only those arrays that do not contain any NaN values
        params_start = [arr for arr in params_start if not np.isnan(arr).any()]
    else:
        params_start = [initial_guess]
    
    N = len(params_start)    
    
    # find fits by MLE and isolate best fit with minimum ss
    dt = x[1] - x[0]
    sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    fits = []
    ss = []
    # switching off warnings to avoid runtime warning: divide by zero encountered in divide
    # only seems to occur when initial_guess=None; when initial_guess is None start values are not filtered through 'LS' or 'DE" methods
    
    # if slow_rise_constraint=True no solutions are possible where a response with the slowest decay has a faster rise
    # ie one rewsponse will be 'fast', the other 'slow'
    
    factor_value = 1  # determines how much larger second trise or tdecay is relative to the first trise or tdecay
    # currently not an option; value set to one

    if slow_rise_constraint:
        constraints = (
            {'type': 'ineq', 'fun': lambda p0: constraint1(p0, factor=factor_value)},
            {'type': 'ineq', 'fun': lambda p0: constraint2(p0, factor=factor_value)}  
        )
    else:
        constraints=None
    
#     if slow_rise_constraint:
        
#         constraints = (
#             {'type': 'ineq', 'fun': constraint1},
#             {'type': 'ineq', 'fun': constraint2}
#         )
        
#     else:
#         constraints=None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            params = params_start[ii]
            p0 = [params[0], params[1], params[2], params[3], params[4], params[5], sd]  # 6 params plus term for sd
            p0 = params_order7(p0)
            bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
            params_fit = fit_product2(x, y, p0, bounds, constraints=constraints)
            fits.append(params_fit)
            y_fit = product2(params_fit[0:6], x)
            ss.append(np.sum((y - y_fit)**2))
    
    idx = ss.index(min(ss))  
    params_fit = fits[idx]
    p0 = params_start[idx]
    p0 = np.append(p0, sd)

    # Generate the fitted curve
    y_fit = product2(params_fit[:-1], x)
    y_fit1 = product(params_fit[0:3], x)
    y_fit2 = product(params_fit[3:6], x)
    
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit (sum)', line=dict(color='indianred', dash='dot')))

    fig.add_trace(go.Scatter(x=x, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    params_out = product2_conversion(params_fit)
    start_out = params_order7(p0)
    start_out = product2_conversion(start_out[0:6])
    
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        # area returned by product2_conversion
        x1, y1 = extend_product(params_fit[0:3], x, p=decay[1], increment=100)
        ests1 = rise_decay_estimators(x=x1, y=y1, rise=rise, decay=decay)
        rd1 = [product_kinetics(x, tau_rise=params_out[1], tau_decay=params_out[2], initial_guess=ests1[0:2], p=rise)[0], 
              product_kinetics(x, tau_rise=params_out[1], tau_decay=params_out[2], initial_guess=ests1[2:4], p=decay)[0]
             ]
        x2, y2 = extend_product(params_fit[3:6], x, p=decay[1], increment=100)
        ests2 = rise_decay_estimators(x=x2, y=y2, rise=rise, decay=decay)
        rd2 = [product_kinetics(x, tau_rise=params_out[6], tau_decay=params_out[7], initial_guess=ests2[0:2], p=rise)[0], 
               product_kinetics(x, tau_rise=params_out[6], tau_decay=params_out[7], initial_guess=ests2[2:4], p=decay)[0]
             ]    
        
    elif kinetics == 'fit':
        # add area under curve
        area1 = np.trapz(y_fit1, x)
        area2 = np.trapz(y_fit2, x)
        # percent rise and decay times 
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        
        y1 = product(params_fit[0:3], x1)
        y2 = product(params_fit[3:6], x1)
        
        rd1 = rise_and_decay_times(x=x1, y=y1, rise=rise, decay=decay)
        rd2 = rise_and_decay_times(x=x1, y=y2, rise=rise, decay=decay)
        # area
        params_out[4] = area1
        params_out[9] = area2

    params_out.insert(4, rd1[0])
    params_out.insert(5, rd1[1])

    params_out.insert(11, rd2[0])
    params_out.insert(12, rd2[1])
    
    # Column names with Unicode symbols
    columns = ['peak', 'τrise', 'τdecay', 'tpeak', f'rise{int(100*rise[0])}-{int(100*rise[1])}', f'decay{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    if criterion=='BIC':
        bic = IC(params_fit, x, y, criterion=criterion, func=nll_product2)
        if plot:
            fig.show()
            out = params_out[0:14]
            out.append(bic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'BIC:'])
        if show_results:
            return np.round(params_out, dp), np.round(start_out,dp), np.round(params_fit,dp), np.round(bic,dp)
    elif criterion=='AIC':
        aic = IC(params_fit, x, y, criterion=criterion, func=nll_product2)
        if plot:
            fig.show()
            out = params_out[0:14]
            out.append(aic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'AIC:'])
        if show_results:
            return np.round(params_out, dp), np.round(start_out,dp), np.round(params_fit,dp), np.round(aic,dp)
    else:
        if plot:
            fig.show()
            out = params_out[0:14]
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:'])       
        if show_results:
            return np.round(params_out, dp), np.round(start_out,dp), np.round(params_fit,dp)
        
def product2_1r_conversion(pars):
    a1 = pars[0]
    tau1 = pars[1]
    tau2 = pars[2]
    f1 = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    tau1_rise = tau1 * tau2 / (tau1 + tau2)
    tau1_decay = tau2
    t1_peak = tau1 * np.log((tau1+tau2)/tau1)
    a1_max = a1 * f1

    a2 = pars[3]
    tau4 = pars[4]
    f2 = ((tau1/(tau1+tau4)) ** (tau1/tau4)) * tau4/(tau1+tau4)
    tau2_rise = tau1 * tau4 / (tau1 + tau4)
    tau2_decay = tau4
    t2_peak = tau1 * np.log((tau1+tau4)/tau1)
    a2_max = a2 * f2

    pars_out = [a1_max, tau1_rise, tau1_decay, t1_peak, a2_max, tau2_rise, tau2_decay, t2_peak]
    
    return pars_out


def FITproduct2_1r(x, y, criterion=None, plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', cv=0.4, N=30):
    # if y contains NAN thenfind index of first and remove all points after (this situation can occur if imported
    # spreadsheet has different lengths of responses
    x,y = preprocess(x,y)
    
    # Perform the fitting        
    if start_method == 'DE':
        params_start = DE_p0(x, y, method='product2_1r', percent=percent)
    elif start_method == 'LS':
#         params_start = CF_p0(x, y, method='product2', initial_guess=initial_guess, maxfev=maxfev, percent=percent)
        params_start = LS_p0(x, y, method='product2_1r', initial_guess=initial_guess, maxfev=maxfev, percent=percent, algorithm=p0_algorithm)
    elif start_method is None: 
        params_start = random_initial_guess(x=x, y=y, method='product2_1r', percent=percent, cv=cv, N=N)
    
    # Keep only those arrays that do not contain any NaN values
    params_start = [arr for arr in params_start if not np.isnan(arr).any()]
    N = len(params_start)    

    # find fits by MLE and isolate best fit with minimum ss
    dt = x[1] - x[0]
    sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    fits = []
    ss = []
    # switching off warnings to avoid runtime warning: divide by zero encountered in divide
    # only seems to occur when initial_guess=None; when initial_guess is None start values are not filtered through 'LS' or 'DE" methods
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            params = params_start[ii]
            p0 = [params[0], params[1], params[2], params[3], params[4], sd]  # 6 params plus term for sd
            p0 = params_order6(p0)
            bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
            params_fit = fit_product2_1r(x, y, p0, bounds)
            fits.append(params_fit)
            y_fit = product2_1r(params_fit[0:5], x)
            ss.append(np.sum((y - y_fit)**2))
    idx = ss.index(min(ss))  
    params_fit = fits[idx]
    p0 = params_start[idx]
    p0 = np.append(p0, sd)
    
    # Generate the fitted curve
    y_fit = product2_1r(params_fit[:-1], x)

    y_fit1 = product(params_fit[0:3], x)
    y_fit2 = product([params_fit[3], params_fit[1], params_fit[4]], x)
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit (sum)', line=dict(color='indianred', dash='dot')))

    fig.add_trace(go.Scatter(x=x, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    params_out = product2_1r_conversion(params_fit)
    start_out = params_order6(p0)
    params_fit = params_fit.tolist() 
    col_names = ['peak', 'τrise', 'τdecay', 'tpeak']

    if criterion=='BIC':
        bic = IC(params_fit, x, y, criterion=criterion, func=nll_product2_1r)
        if plot:
            fig.show()
            # Printing the formatted output strings for fast and slow components
            col_header, rows_fast = output_results(params_out[0:4], col_names, dp=dp)
            _, rows_slow = output_results(params_out[4:8], col_names, dp=dp)
            print(f"       {col_header}")
            print(f"fast:  {rows_fast}")
            print(f"slow:  {rows_slow}")
            print("BIC:", np.round(bic, dp))
        return params_out, start_out, params_fit, bic
    elif criterion=='AIC':
        aic = IC(params_fit, x, y, criterion=criterion, func=nll_product2_1r)
        if plot:
            fig.show()
            # Printing the formatted output strings for fast and slow components
            col_header, rows_fast = output_results(params_out[0:4], col_names, dp=dp)
            _, rows_slow = output_results(params_out[4:8], col_names, dp=dp)
            print(f"       {col_header}")
            print(f"fast:  {rows_fast}")
            print(f"slow:  {rows_slow}")
            print("AIC:", np.round(aic,dp))
        return params_out, start_out, params_fit, aic
    else:
        if plot:
            fig.show()
            fig.show()
            # Printing the formatted output strings for fast and slow components
            col_header, rows_fast = output_results(params_out[0:4], col_names, dp=dp)
            _, rows_slow = output_results(params_out[4:8], col_names, dp=dp)
            print(f"       {col_header}")
            print(f"fast:  {rows_fast}")
            print(f"slow:  {rows_slow}")
        return params_out, start_out, params_fit, 
    
def IC(params_fit, x, y, criterion='BIC', func=nll_alpha):
    # Given a set of candidate models for the data, the preferred model is the one with the minimum AIC/BIC value.
    if criterion=='BIC':
        # Calculate Bayesian information criterion or BIC
        n = len(y)
        k = len(params_fit) - 1  # Subtract 1 to exclude the standard deviation parameter
        nll = func(params_fit, x, y)
        ic = 2 * nll + k * np.log(n) # k*log(n) - 2*LL
    elif criterion=='AIC':
        # Calculate Akaike information criterion or AIC
        k = len(params_fit) - 1  # Subtract 1 to exclude the standard deviation parameter
        nll = func(params_fit, x, y)
        ic = 2 * k + 2 * nll # 2*k - 2*LL
    return ic


# SSE_ and p0_ functions to calculate starting values 
def SSE_alpha(pars, x, y):
    yfit = alpha(pars, x)
    return np.sum((y - yfit) ** 2.0)

def SSE_alpha2(pars, x, y):
    yfit = alpha2(pars, x)
    return np.sum((y - yfit) ** 2.0)

def SSE_product(pars, x, y):
    yfit = product(pars, x)
    return np.sum((y - yfit) ** 2.0)

def SSE_product2(pars, x, y):
    yfit = product2(pars, x)
    return np.sum((y - yfit) ** 2.0)

def SSE_product2_1r(pars, x, y):
    yfit = product2_1r(pars, x)
    return np.sum((y - yfit) ** 2.0)

def p0_start(bounds, x, y, func):
    result = differential_evolution(func, bounds=bounds, args=(x, y), tol=0.01)
    return result.x

# setting bounds help to reduce the parameter space
def random_initial_guess(x, y, method='alpha', percent=80, cv=0.4, N=10):
    ymax = find_absolute_max(y)
    Tnet, T2 = tau_estimators(x, y, percent=percent)
    if method == 'alpha':
        a1 = ymax*np.exp(1)/T2 # this is the function a1 calculated from the response max ie a1_max or ymax
        initial = [a1, T2] 
        initial = start_log_normal(input=initial, cv=cv, N=N)

    elif method == 'alpha2':
        a1 = ymax*np.exp(1)/T2 
        initial = [a1, T2, a1, T2] 
        initial = start_log_normal(input=initial, cv=cv, N=N)

    elif method == 'product':
        T1 = Tnet*T2/(T2-Tnet)
        f1 = ((T1/(T1+T2)) ** (T1/T2)) * T2/(T1+T2) 
        a1 = ymax / f1
        initial = [a1, T1, T2]
        initial = start_log_normal(input=initial, cv=cv, N=N)
        
    elif method == 'product2':
        T1 = Tnet*T2/(T2-Tnet)
        f1 = ((T1/(T1+T2)) ** (T1/T2)) * T2/(T1+T2)
        a1 = ymax / f1
        initial = [a1, T1, T2, a1, T1, T2] 
        initial = start_log_normal(input=initial, cv=cv, N=N)

    elif method == 'product2_1r':
        T1 = Tnet*T2/(T2-Tnet)
        f1 = ((T1/(T1+T2)) ** (T1/T2)) * T2/(T1+T2)
        a1 = ymax / f1
        initial = [a1, T1, T2, a1, T2]
        initial = start_log_normal(input=initial, cv=cv, N=N)

    return initial

# setting bounds help to reduce the parameter space
def DE_p0(x, y, method='alpha', percent=80, cv=0.4, N=10):
    initial = random_initial_guess(x=x, y=y, method=method, percent=percent, cv=cv, N=N)
    if method == 'alpha':
        SSE = SSE_alpha
        bounds=[]
        for ii in range(N):
            bounds.append([(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1])])

    elif method == 'alpha2':
        SSE = SSE_alpha2
        bounds=[]
        for ii in range(N):
            bounds.append([(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1]),(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1])])

    elif method == 'product':
        SSE = SSE_product
        bounds=[]
        for ii in range(N):
            bounds.append([(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1]), (0.01, 10*initial[ii][2])])

    elif method == 'product2':
        SSE = SSE_product2
        bounds=[]
        for ii in range(N):
            bounds.append([(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1]), (0.01, 10*initial[ii][2]), (0, 10*initial[ii][3]), (0.01, 10*initial[ii][4]), (0.01, 10*initial[ii][5])])

    elif method == 'product2_1r':
        SSE = SSE_product2_1r
        bounds=[]
        for ii in range(N):
            bounds.append([(0, 10*initial[ii][0]), (0.01, 10*initial[ii][1]), (0.01, 10*initial[ii][2]), (0, 10*initial[ii][3]), (0.01, 10*initial[ii][4])])
    output = []
    for ii in range(N):
        p0 = p0_start(bounds[ii], x, y, SSE)
        output.append(p0)
    return output

# fund optimised starting values by least square curve fitting
def CF_p0(x, y, method='alpha', initial_guess=None, maxfev=10000, percent=80):
    ymax = find_absolute_max(y)
    Tnet, T2 = tau_estimators(x, y, percent=percent)
    if method == 'alpha':
        if initial_guess is None:
            a1 = ymax*np.exp(1)/T2
            initial_guess = [a1, 1/T2] 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0, p0_covariance = curve_fit(alpha_cf, x, y, p0=initial_guess, maxfev=maxfev)
        except RuntimeError:
            print("Optimal starting values not found within the maximum number of function evaluations")
            p0 = np.full(len(initial_guess), math.nan)
        out = (p0[0], 1/p0[1])
        return np.array(out)
    
    elif method == 'alpha2':
        if initial_guess is None:
            a1 = ymax*np.exp(1)/T2
            initial_guess = [a1, 1/T2, a1, 1/T2] 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0, p0_covariance = curve_fit(alpha2_cf, x, y, p0=initial_guess, maxfev=maxfev)
        except RuntimeError:
            print("Optimal starting values not found within the maximum number of function evaluations")
            p0 = np.full(len(initial_guess), math.nan)
        out = (p0[0], 1/p0[1], p0[2], 1/p0[3])
        return np.array(out)
    
    elif method == 'product':
        if initial_guess is None:
            T1 = Tnet*T2/(T2-Tnet)
            f1 = ((T1/(T1+T2)) ** (T1/T2)) * T2/(T1+T2)
            a1 = ymax / f1
            initial_guess = [a1, 1/T1, 1/T2] 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0, p0_covariance = curve_fit(product_cf, x, y, p0=initial_guess, maxfev=maxfev)
        except RuntimeError:
            print("Optimal starting values not found within the maximum number of function evaluations")
            p0 = np.full(len(initial_guess), math.nan)
        out = (p0[0], 1/p0[1], 1/p0[2])
        return np.array(out)
    
    elif method == 'product2':
        if initial_guess is None:
            T1 = Tnet*T2/(T2-Tnet)
            f1 = ((T1/(T1+T2)) ** (T1/T2)) * T2/(T1+T2)
            a1 = ymax / f1
            initial_guess = [a1, 1/T1, 1/T2, a1, 1/T1, 1/T2] 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0, p0_covariance = curve_fit(product2_cf, x, y, p0=initial_guess, maxfev=maxfev)
        except RuntimeError:
            print("Optimal starting values not found within the maximum number of function evaluations")
            p0 = np.full(len(initial_guess), math.nan)
        out = (p0[0], 1/p0[1], 1/p0[2], p0[3], 1/p0[4], 1/p0[5])
        return np.array(out)

     

# calculate residuals    
# func = alpha, alpha2, product, product2
def residuals(pars, x, y, func):
    yfit = func(pars, x)
    return y - yfit


# get N randomised starting values assuming they are drawn from a lognormal distribution
# with mean values given by input and coffecient of variation by cv

def start_log_normal(input, cv=0.4, N=10):
    # Initialize the output list
    output = []

    # Calculate the mean and standard deviation of the logarithm of the distribution
    sigma = np.sqrt(np.log(1 + (cv**2)))
    mu = np.log(input) - 0.5 * (sigma**2)

    # Generate values
    for ii in range(N):
        values = [np.random.lognormal(mean=m, sigma=sigma) for m in mu]
        output.append(values)
    
    return output


# find optimised starting values by least square curve fitting
# setting bounds help to reduce the parameter space
# uses least_squares to get starting values
# By default, least_squares uses the Trust Region Reflective algorithm, which is a 
# derivative-free algorithm designed for solving nonlinear least squares problems. 
# This algorithm combines the trust region method with a reflective strategy to handle 
# both bound and unbound constraints on the variables
# for least_squares bounds are arranged as [(lower), (upper)]
def LS_p0(x, y, method='alpha', initial_guess=None, maxfev=10000, percent=80, algorithm='trf', cv=0.4, N=10, weighted=False):
    if initial_guess is None:
        initial = random_initial_guess(x=x, y=y, method=method, percent=percent, cv=cv, N=N)
    
        if method == 'alpha':
            func = alpha
            bounds=[]
            for ii in range(N):
                bounds.append([(0, 0.01), tuple([x * 10 for x in initial[ii]])])

        elif method == 'alpha2':
            func = alpha2
            bounds=[]
            for ii in range(N):
                bounds.append([(0, 0.01, 0, 0.01), tuple([x * 10 for x in initial[ii]])])

        elif method == 'product':
            func = product
            bounds=[]
            for ii in range(N):
                bounds.append([(0, 0.01, 0.01), tuple([x * 10 for x in initial[ii]])])
            
        elif method == 'product2':
            func = product2
            bounds=[]
            for ii in range(N):
                bounds.append([(0, 0.01, 0.01, 0, 0.01, 0.01), tuple([x * 10 for x in initial[ii]])])

        elif method == 'product2_1r':
            func = product2_1r
            bounds=[]
            for ii in range(N):
                bounds.append([(0, 0.01, 0.01, 0, 0.01), tuple([x * 10 for x in initial[ii]])])

    output = []
    if algorithm == 'trf':
        for ii in range(N):
                res = least_squares(residuals, x0=initial[ii], loss="linear", args=(x, y, func), bounds=bounds[ii], gtol=1e-8, method='trf', max_nfev=maxfev)
                output.append(res["x"])
    elif algorithm == 'lm':
        for ii in range(N):
            res = least_squares(residuals, x0=initial[ii], loss="linear", args=(x, y, func), gtol=1e-8, method='lm', max_nfev=maxfev)
            output.append(res["x"])
    elif algorithm == 'dogbox':
        for ii in range(N):
            res = least_squares(residuals, x0=initial[ii], loss="linear", args=(x, y, func), gtol=1e-8, method='dogbox')
            output.append(res["x"])
    return output


def min_index(x):
    xmin = min(x)
    ind = x.index(xmin)
    return ind

def extract_summary_stats(data_dict, key, confidence_interval=0.95, dp=3, col_names=None):
    """
    Extracts summary statistics for a specific key from a dictionary of data.

    Parameters:
        data_dict (dict): Dictionary containing the data.
        key (str): Key for which to extract the summary statistics.
        confidence_interval (float): Confidence interval (default: 0.95).
        dp (int): Number of decimal places for rounding (default: 5).

    Returns:
        dict: Summary statistics for the specified key.
    """
    data = data_dict[key]
    mean_val = np.nanmean(data, axis=0)
    median_val = np.nanmedian(data, axis=0)
    std_val = np.nanstd(data, axis=0)

    upper_ci = np.percentile(data, [(1 + confidence_interval) / 2 * 100], axis=0)
    lower_ci = np.percentile(data, [(1 - confidence_interval) / 2 * 100], axis=0)

    summary_stats = {
        'upper ci': np.round(upper_ci[0], dp),
        'mean': np.round(mean_val, dp),        
        'lower ci': np.round(lower_ci[0], dp),
        'median': np.round(median_val, dp),
        'sd': np.round(std_val, dp)
    }
    df = pd.DataFrame.from_dict(summary_stats, orient='index')
    if col_names is not None:
        df.columns = col_names
    return df.round(decimals=dp)

def IC_model(data_dict, model, criterion='BIC'):
    """
    Calculates the proportion selecting a specific model based on BIC or AIC outcomes.
    
    Parameters:
        data_dict (dict): Dictionary containing the data.
        model (int): Model for which to calculate the proportion.
        criterion (str): Criterion to use for model selection ('BIC' or 'AIC').
    
    Returns:
        float: Proportion of the specified model based on BIC or AIC outcomes.
    """
    # Validate criterion
    if criterion not in ['BIC', 'AIC']:
        raise ValueError("Invalid entry: criterion must be either 'BIC' or 'AIC'.")
    
    # Dynamically create the criterion_list based on the keys in data_dict
    criterion_list = [values for key, values in data_dict.items() if key.startswith(criterion)]
    
    # Check if criterion_list is empty, and if so, raise an error
    if not criterion_list:
        raise ValueError(f"No keys in data_dict match the specified criterion: '{criterion}'")
    
    criterion_outcome = []
    for values in zip(*criterion_list):
        criterion_outcome.append(min_index(list(values)))
    
    p = criterion_outcome.count(model) / len(criterion_outcome)
    
    return p

# Helper function to find the index of the minimum value in a list
def min_index(values_list):
    return values_list.index(min(values_list))


# function to preprocess data
# removes baseline, makes y relative to baseline and rescales x for fitting
def data_preprocessor(file_path, stim_time, neg=True):
    out = []
    # Read the CSV file into a NumPy array
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    # Extract the first column
    X = data[:, 0]
    Y = data[:, 1]
    # specify if negative going make the signal positive
    if neg:
        Y = -Y
    # Find the index of the element with the smallest difference
    idx = np.argmin(np.abs(X - stim_time))
    
    x = X[idx:len(X)]
    y = Y[idx:len(Y)] - np.mean(Y[0:idx])

    # rescale x
    dx = x[1] - x[0]
    x = np.arange(0, len(x)*dx, dx)     
    
    return x,y

def output_results(params, col_names, dp=3):
    values = [f"{p:.{dp}f}" for p in params]
    widths = [max(len(col), len(val)) + 2 for col, val in zip(col_names, values)]
    row_format = "  ".join([f"{{:<{width}}}" for width in widths])
    col_header = row_format.format(*col_names)
    rows = row_format.format(*values)
    return col_header, rows


def idx_finder(x, x_target):
    """
    Find the index of the value in the numpy array that's closest to the target value.

    Parameters:
    - arr array of numbers.
    - target (float/int): Target value.

    Returns:
    - int: Index of the nearest value.
    """
    x = np.array(x)
    differences = np.abs(x - x_target)
    return np.argmin(differences)

# calculates single response when params is [peak, trise, tdecay] 
def product_alt2(params, t):
    # Extract the parameters
    a1_max, tnet1, tau2 = params
    tau1 = tnet1*tau2/(tau2-tnet1)
    f1 = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    a1 = a1_max / f1
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) 
    return y

# calculates repsonse for 2 products  when params is [peak1, trise1, tdecay1, peak2, trise2, tdecay2] 
def product2_alt2(params, t):
    # Extract the parameters
    a1_max, tnet1, tau2, a2_max, tnet3, tau4 = params
    tau1 = tnet1*tau2/(tau2-tnet1)
    tau3 = tnet3*tau4/(tau4-tnet3)   
    
    f1 = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2)
    a1 = a1_max / f1
    f2 = ((tau3/(tau3+tau4)) ** (tau3/tau4)) * tau4/(tau3+tau4)
    a2 = a2_max / f2
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2) + a2 * (1 - np.exp(-t/tau3)) * np.exp(-t/tau4)
    return y

# calculates nll when params is [peak1, trise1, tdecay1] 
def nll_product_alt2(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = product_alt2(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

# calculates nll when params is [peak1, trise1, tdecay1, peak2, trise2, tdecay2] 
def nll_product2_alt2(params, x, y):
    # Compute the negative log-likelihood
    sd = params[-1]
    y_pred = product2_alt2(params[:-1], x)
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    # Calculate the negative log-likelihood
    nLL = -LL
    return nLL

def round_up(x, np=10):
    return np*round(x/np)
 
def FITproduct_batch(x, df, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', cv=0.4, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve'):
    fits_list = []
    start_list = []
    fit_list = []
    if criterion == 'BIC':
        BIC_list = []
    elif criterion == 'AIC':
        AIC_list = []
    
    for column in df.columns:
        if criterion in ('BIC', 'AIC'):
            fits, start, fit, crit = FITproduct(x, df[column].values, criterion=criterion , plot=plot, 
            initial_guess=initial_guess, start_method=start_method, percent=percent, dp=dp, 
            maxfev=maxfev, p0_algorithm=p0_algorithm, cv=cv, N=N, rise=rise, decay=decay, kinetics=kinetics, 
            show_results=True)
        else:
            fits, start, fit = FITproduct(x, df[column].values, criterion=criterion , plot=plot, 
            initial_guess=initial_guess, start_method=start_method, percent=percent, dp=dp, 
            maxfev=maxfev, p0_algorithm=p0_algorithm, cv=cv, N=N, rise=rise, decay=decay, kinetics=kinetics,
            show_results=True)
        
        # Storing the results in lists
        fits_list.append(fits)
        start_list.append(start)
        fit_list.append(fit)
        if criterion == 'BIC':
            BIC_list.append(crit)
        elif criterion == 'AIC':
            AIC_list.append(crit)
              
    output1 = pd.DataFrame(fits_list)
    column_names1 = ['peak', '\u03C4rise', '\u03C4decay', 'tpeak', f'rise{int(100*rise[0])}-{int(100*rise[1])}', f'decay{int(100*decay[0])}-{int(100*decay[1])}', 'area']  
    output1.columns = column_names1

    column_names2 = ['peak1', '\u03C4rise', '\u03C4decay', 'tpeak', 'area'] # customize the names based on your data
    output2 = pd.DataFrame(start_list)
    output2.columns = column_names2
    
    output3 = pd.DataFrame(fit_list)
    # remove last column (this is the fitted standard deviation of the noise that arises by maximum likelihood estimation)
    column_names3 = ['A1', '\u03C4\u2081', '\u03C4\u2082', '\u03C3']  # customize names based on data
    output3.columns = column_names3
  
    if criterion == 'BIC':
        output4 = pd.DataFrame(BIC_list)
        output4.columns = ['BIC']
    elif criterion == 'AIC':
        output4 = pd.DataFrame(AIC_list)
        output4.columns = ['AIC']
    
    if criterion in ('BIC', 'AIC'):
        return output1, output2, output3, output4
    else:
        return output1, output2, output3
    
# Assuming FITproduct2 is already defined and available for use
def FITproduct2_batch(x, df, criterion='AIC', plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', cv=0.4, N=30, slow_rise_constraint=True, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve'):
    fits_list = []
    start_list = []
    fit_list = []
    if criterion == 'BIC':
        BIC_list = []
    elif criterion == 'AIC':
        AIC_list = []
    
    for column in df.columns:
        if criterion in ('BIC', 'AIC'):
            fits, start, fit, crit = FITproduct2(x, df[column].values, criterion=criterion , plot=plot, 
            initial_guess=initial_guess, start_method=start_method, percent=percent, dp=dp, 
            maxfev=maxfev, p0_algorithm=p0_algorithm, cv=cv, N=N, slow_rise_constraint=slow_rise_constraint,
            rise=rise, decay=decay, kinetics=kinetics, show_results=True)
        else:
            fits, start, fit = FITproduct2(x, df[column].values, criterion=criterion , plot=plot, 
            initial_guess=initial_guess, start_method=start_method, percent=percent, dp=dp, 
            maxfev=maxfev, p0_algorithm=p0_algorithm, cv=cv, N=N, slow_rise_constraint=slow_rise_constraint,
            rise=rise, decay=decay, kinetics=kinetics, show_results=True)
        
        # Storing the results in lists
        fits_list.append(fits)
        start_list.append(start)
        fit_list.append(fit)
        if criterion == 'BIC':
            BIC_list.append(crit)
        elif criterion == 'AIC':
            AIC_list.append(crit)
              
    output1 = pd.DataFrame(fits_list)
    column_names1 = ['peak1', '\u03C4rise1', '\u03C4decay1', 'tpeak1', 
                      f'rise1{int(100*rise[0])}-{int(100*rise[1])}', 
                      f'decay1{int(100*decay[0])}-{int(100*decay[1])}', 
                      'area1',
                      'peak2', '\u03C4rise2', '\u03C4decay2', 'tpeak2', 
                      f'rise2{int(100*rise[0])}-{int(100*rise[1])}', f'decay2{int(100*decay[0])}-{int(100*decay[1])}', 
                      'area2'] 
    output1.columns = column_names1

    column_names2 = ['peak1', '\u03C4rise1', '\u03C4decay1', 'tpeak1', 'area1', 'peak2', '\u03C4rise2', '\u03C4decay2', 'tpeak2', 'area2'] # customize the names based on your data
    output2 = pd.DataFrame(start_list)
    output2.columns = column_names2
    
    output3 = pd.DataFrame(fit_list)
    # remove last column (this is the fitted standard deviation of the noise that arises by maximum likelihood estimation)
    column_names3 = ['A1', '\u03C4\u2081', '\u03C4\u2082', 'A2', '\u03C4\u2083', '\u03C4\u2084', '\u03C3']  # customize names based on data
    output3.columns = column_names3
  
    if criterion == 'BIC':
        output4 = pd.DataFrame(BIC_list)
        output4.columns = ['BIC']
    elif criterion == 'AIC':
        output4 = pd.DataFrame(AIC_list)
        output4.columns = ['AIC']
    
    if criterion in ('BIC', 'AIC'):
        return output1, output2, output3, output4
    else:
        return output1, output2, output3



def FITproduct_widget(x, y, criterion='SS', initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', slider_factor=3, cv=0.4, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve'):

    x,y = preprocess(x,y)
    
    fits, start, fit = FITproduct(x=x, y=y, criterion=None, plot=False, initial_guess=initial_guess, start_method=start_method, 
        percent=percent, dp=dp, maxfev=maxfev, p0_algorithm=p0_algorithm, cv=cv, N=N, rise=rise, decay=decay, 
        kinetics=kinetics, show_results=True)
    
    # Define a function to execute the fitting and plot
    def interactive_fit(peak1, trise1, tdecay1, criterion='blank'):
        params = [peak1, trise1, tdecay1]
        y_fit = product_alt2(params, x)
        
        params.append(fit[3])
        if criterion=='BIC':
            bic = np.round(IC(params, x, y, criterion=criterion, func=nll_product_alt2), 2)
        elif criterion=='AIC':
            aic = np.round(IC(params, x, y, criterion=criterion, func=nll_product_alt2), 2)
        elif criterion=='SS':
            ss = np.round(np.sum((y - y_fit)**2), 2)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, color='lightgray', linewidth=0.5)
        plt.plot(x, y_fit, color='indianred', linewidth=1.5, linestyle=':')
        plt.xlabel('time (ms)')
        plt.ylabel('response')
        plt.title('interactive curve fitting')
        plt.grid(False)

        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 20))

#         plt.xlim(left=0)
#         plt.ylim(bottom=0)

        # Displaying the BIC value inside the plot
        if criterion=='BIC':
            plt.text(0.80, 0.95, f'BIC: {bic}', transform=plt.gca().transAxes, verticalalignment='top')
        elif criterion=='AIC':
            plt.text(0.80, 0.95, f'AIC: {aic}', transform=plt.gca().transAxes, verticalalignment='top')
        elif criterion=='SS':
            plt.text(0.80, 0.95, f'SS: {ss}', transform=plt.gca().transAxes, verticalalignment='top')             

        plt.show()
        
        
    peak1_slider = FloatSlider(value=fits[0], min=0, max=round_up(slider_factor*fits[0], 10), step=0.01, description='peak1')
    trise1_slider = FloatSlider(value=fits[1], min=0, max=round_up(slider_factor*fits[1], 1), step=0.01, description='trise1')
    tdecay1_slider = FloatSlider(value=fits[2], min=0, max=round_up(slider_factor*fits[2], 1), step=0.01, description='tdecay1')

    if criterion is None:
        _ = interact(interactive_fit, peak1=peak1_slider, trise1=trise1_slider, tdecay1=tdecay1_slider)
    else:
        _ = interact(interactive_fit, peak1=peak1_slider, trise1=trise1_slider, tdecay1=tdecay1_slider, criterion=criterion)

        
def FITproduct2_widget(x, y, criterion='SS', initial_guess=None, start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', slider_factor=3, cv=0.4, N=30, slow_rise_constraint=True, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve'):

    x,y = preprocess(x,y)
    
    fits, start, fit = FITproduct2(x=x, y=y, criterion=None, plot=False, initial_guess=initial_guess, 
        start_method=start_method, percent=percent, dp=dp, maxfev=maxfev, p0_algorithm=p0_algorithm, 
        cv=cv, N=N, slow_rise_constraint=slow_rise_constraint, rise=rise, decay=decay, 
        kinetics=kinetics, show_results=True)
    
    # Define a function to execute the fitting and plot
    def interactive_fit(peak1, trise1, tdecay1, peak2, trise2, tdecay2, criterion='blank'):
        params = [peak1, trise1, tdecay1, peak2, trise2, tdecay2]
        y_fit = product2_alt2(params, x)
        y_fit1 = product_alt2(params[0:3], x)
        y_fit2 = product_alt2(params[3:6], x)

        params.append(fit[6])
        if criterion=='BIC':
            bic = np.round(IC(params, x, y, criterion=criterion, func=nll_product2_alt2), 2)
        elif criterion=='AIC':
            aic = np.round(IC(params, x, y, criterion=criterion, func=nll_product2_alt2), 2)
        elif criterion=='SS':
            ss = np.round(np.sum((y - y_fit)**2), 2)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, color='lightgray', linewidth=0.5)
        plt.plot(x, y_fit, color='indianred', linewidth=1.5, linestyle=':')
        plt.plot(x, y_fit1, color='slateblue', linewidth=1.5, linestyle=':')
        plt.plot(x, y_fit2, color='slateblue', linewidth=1.5, linestyle=':')
        plt.xlabel('time (ms)')
        plt.ylabel('response')
        plt.title('interactive curve fitting')
        plt.grid(False)

        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 20))

#         plt.xlim(left=0)
#         plt.ylim(bottom=0)

        # Displaying the BIC value inside the plot
        if criterion=='BIC':
            plt.text(0.80, 0.95, f'BIC: {bic}', transform=plt.gca().transAxes, verticalalignment='top')
        elif criterion=='AIC':
            plt.text(0.80, 0.95, f'AIC: {aic}', transform=plt.gca().transAxes, verticalalignment='top')
        elif criterion=='SS':
            plt.text(0.80, 0.95, f'SS: {ss}', transform=plt.gca().transAxes, verticalalignment='top')             

        plt.show()
        
        
    peak1_slider = FloatSlider(value=fits[0], min=0, max=round_up(slider_factor*fits[0], 10), step=0.01, description='peak1')
    trise1_slider = FloatSlider(value=fits[1], min=0, max=round_up(slider_factor*fits[1], 1), step=0.01, description='trise1')
    tdecay1_slider = FloatSlider(value=fits[2], min=0, max=round_up(slider_factor*fits[2], 1), step=0.01, description='tdecay1')

    peak2_slider = FloatSlider(value=fits[7], min=0, max=round_up(slider_factor*fits[7], 10), step=0.01, description='peak2')
    trise2_slider = FloatSlider(value=fits[8], min=0, max=round_up(slider_factor*fits[8], 1), step=0.01, description='trise2')
    tdecay2_slider = FloatSlider(value=fits[9], min=0, max=round_up(slider_factor*fits[9], 1), step=0.01, description='tdecay2')

    if criterion is None:
        _ = interact(interactive_fit, peak1=peak1_slider, trise1=trise1_slider, tdecay1=tdecay1_slider, peak2=peak2_slider, trise2=trise2_slider, tdecay2=tdecay2_slider)
    else:
        _ = interact(interactive_fit, peak1=peak1_slider, trise1=trise1_slider, tdecay1=tdecay1_slider, peak2=peak2_slider, trise2=trise2_slider, tdecay2=tdecay2_slider, criterion=criterion)
        
def idx_finder(arr, target):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - target))


def load_data(wd=None, folder='data', filename='your_file.xlsx', dt=None, stim_time=100, baseline=True, time=True):
    
    # Get the working directory
    if wd is None:
        wd = os.getcwd()
 
    # Construct the file path
    file_path = os.path.join(wd, folder, filename)
    
    _, extension = os.path.splitext(filename)
    
    # Load the DataFrame
    if extension.lower() == '.xlsx':
        df = pd.read_excel(file_path)
    elif extension.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    if time:
        dt = df.iloc[:, 0].values[1] - df.iloc[:, 0].values[0]
        stim_time = stim_time - df.iloc[:, 0].values[0]
    N = df.shape[0]
    t = np.arange(0, N*dt, dt)
    
    # Find the index corresponding to the stimulus time
    idx1 = idx_finder(t, stim_time)
    
    # Slice the DataFrame based on the found index
    df2 = df.iloc[idx1:]
    if baseline:
        df3 = df.iloc[:idx1]
        bl = df3.mean()
        df2 = df2 - bl
    
    if time:
        t = df2.iloc[:, 0].values
        t = t - t[0]
        df2 = df2.drop(df2.columns[0], axis=1)
    else:
        N = df2.shape[0]
        t = np.arange(0, N*dt, dt)
    
    # rename cols from zero
    df2.columns = range(len(df2.columns))
    # Resetting the index of the DataFrame
    df2.reset_index(drop=True, inplace=True)
    
    # determine if positive or negative going
    sign_abs = df2.apply(lambda col: (1 if col[col.abs().idxmax()] > 0 else -1), axis=0)
    
    # Check if all signs are the same
    all_same_sign = len(sign_abs.unique()) == 1
    
    if all_same_sign:
        df2 = df2 * sign_abs
    else:
        raise ValueError("Error: The maximum absolute values of the responses have different signs")
    return t, df2

# Finding the index of the first NaN in y
def preprocess(x,y):
    first_nan_index = None
    for index, yi in enumerate(y):
        if yi != yi:  # This condition will be True for NaN values, since NaN != NaN
            first_nan_index = index
            break

    # If a NaN value is found, create new arrays that stop at the first NaN
    if first_nan_index is not None:
        new_x = x[:first_nan_index]
        new_y = y[:first_nan_index]
    else:
        new_x = x
        new_y = y
    
    return new_x, new_y

def output_fun(fit, columns=None, row_labels=None):
    """
    Create and print a DataFrame with specified values, columns, and row labels.
    
    Parameters:
    - fit: A list of numerical values to be added to the DataFrame.
    - columns: A list of column names.
    - row_labels: A list of labels for the rows. Default is None.
    
    Returns:
    - DataFrame: The created DataFrame.
    """
    
    # Reshape the fit list to have multiple rows if necessary
    fit_reshaped = [fit[i:i+len(columns)] for i in range(0, len(fit), len(columns))]
    
    # Creating the DataFrame
    df = pd.DataFrame(fit_reshaped, columns=columns)
    
    # Assigning row labels if provided
    if row_labels:
        df.index = row_labels
    
    df = df.replace(np.nan, '')
    # Displaying the DataFrame
    print(df)

def create_column_slices(ncols, slice_size=3):
    columns_slices_to_process = []
    
    # Creating slices of columns
    for i in range(0, ncols, slice_size):
        # Ensuring the last slice captures the remaining columns if they are less than slice_size
        end_index = min(i + slice_size, ncols)
        columns_slices_to_process.append((i, end_index))
        
    return columns_slices_to_process


def product_conversion_df(df, dp=3):
    nrows, ncols = df.shape
    rows = list(range(nrows))
    columns_slices = create_column_slices(ncols-1)
    
    out = []  # To store the final output
    
    for row in rows:
        out1 = []  # To store the output of each row before flattening
        
        for col_slice in columns_slices:
            # Unpack the column slice tuple into start and end
            start_col, end_col = col_slice
            values = [df.iloc[row, col] for col in range(start_col, end_col)]
            # Apply the product_conversion and round the results
            result = product_conversion(values)
            result = [round(val, dp) for val in result]  # Round each value in the result
            out1.append(result)
        
        # Flatten the list of results and append it to the final output
        out2 = [item for sublist in out1 for item in sublist]
        out.append(out2)
    
    # Convert the final output to a DataFrame
    df_new = pd.DataFrame(out)
    
    # Add the last column of df to df_new
    df_new[df_new.shape[1]] = df.iloc[:, -1]  # Using the number of columns in df_new as the new column name
    nrows, ncols = df_new.shape
    if ncols == 6:
        df_new.columns = ['peak', '\u03C4rise', '\u03C4decay', 'tpeak', 'area', '\u03C3'] 
 
    if ncols == 11:
        df_new.columns = ['peak1', '\u03C4rise1', '\u03C4decay1', 'tpeak1', 'area1',
                      'peak2', '\u03C4rise2', '\u03C4decay2', 'tpeak2', 'area2', '\u03C3'] 
   
    return df_new


def plot1(df1, method=0, model='product2', lwd=0.8, xrange=(0, 300), yrange=(0, 1000), 
        width=600,height=600, pairing = True):
    #     if method = 0 then plots tau else plots 80-20 decay
    # Data for the scatter plots
    if model == 'product2':
        if method == 0:
            x1 = df1.iloc[:, 2].values
            x2 = df1.iloc[:, 9].values
        else: 
            x1 = df1.iloc[:, 5].values
            x2 = df1.iloc[:, 12].values
        y1 = df1.iloc[:, 0].values
        y2 = df1.iloc[:, 7].values
    else:
        method=1
        x1 = df1.iloc[:, 5].values
        x2 = df1.iloc[:, 12].values
        y1 = df1.iloc[:, 0].values
        y2 = df1.iloc[:, 7].values


    # Creating the figure
    fig = go.Figure()

    # Adding the scatter plots to the figure
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(color='slateblue', size=lwd*8, opacity=0.6), name='fast'))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', marker=dict(color='indianred', size=lwd*8, opacity=0.6), name='slow'))

    # Add dotted lines if pairing is enabled
    if pairing:
        for i in range(len(x1)):
            fig.add_trace(go.Scatter(x=[x1[i], x2[i]], y=[y1[i], y2[i]],
                                     mode='lines', line=dict(color='darkgray', width=lwd, dash='dot'),
                                     showlegend=False))

    if method == 0:
        title = 'peak amplitude (pA) vs τdecay (ms)'
        x_title = 'τdecay (ms)'
    else:
        title = 'peak amplitude (pA) vs decay time'
        x_title = '80-20 decay (ms)'
    
    # Updating the layout
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_title,
        yaxis_title='peak amplitude (pA)',
        xaxis=dict(
            range=xrange,
            showline=True,
            showgrid=False,
            linewidth=lwd, 
            showticklabels=True,
            linecolor='black',
            ticks='outside',
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
        ),
        yaxis=dict(
            range=yrange,
            showline=True,
            showgrid=False,
            linewidth=lwd, 
            showticklabels=True,
            linecolor='black',
            ticks='outside',
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
        ),
        plot_bgcolor='white',
        # Set the size of the figure
        width=width,  
        height=width 
    )

    return fig

# function to merge these results together

def save_results(df, wd=None, folder='example data', filename='your_filename', out_name='summary', ext='xlsx'):
    """
    Save results in a DataFrame to a CSV or Excel file.
    
    :param df: DataFrame to save.
    :param wd: Working directory, defaults to the current working directory.
    :param folder: Directory where the file will be saved, defaults to 'example data'.
    :param filename: Name of the file to save, without extension.
    :param ext: Extension of the file, 'csv' for CSV file and 'xlsx' for Excel file. Defaults to 'xlsx'.
    """
    wd = wd or os.getcwd()
    file_path = os.path.join(wd, folder, f"{filename}_{out_name}.{ext}")

    # Ensure the folder exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check the file extension and save accordingly
    if ext == 'csv':
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif ext == 'xlsx':
        df.to_excel(file_path, index=False)
    else:
        raise ValueError("Unsupported file extension. Please use 'csv' or 'xlsx'.")

def save_fig(fig, wd=None, folder='example data', filename='your_filename', out_name='fig', ext='svg'):
    """
    Save a plotly figure to a file.

    Parameters:
    fig (plotly.graph_objs.Figure): The plotly figure to save.
    wd (str, optional): The working directory where the file will be saved. 
                        Defaults to the current working directory.
    folder (str, optional): The name of the folder within the working directory to save the file. 
                            Defaults to 'example data'.
    filename (str, optional): The base name of the file (without extension). 
                              Defaults to 'your_filename'.
    ext (str, optional): The file extension (type of file to save as). 
                         Defaults to 'svg' for SVG file format.
    """
    # If wd is None, use the current working directory
    wd = wd or os.getcwd()

    # Add the extension to the filename
    # This constructs the full filename including the chosen extension
    filename_with_ext = f"{filename}_{out_name}.{ext}"

    # Construct the full file path by combining the working directory, folder, and filename
    file_path = os.path.join(wd, folder, filename_with_ext)

    # Ensure the target folder exists, create it if it does not
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the Plotly figure to the specified file path
    fig.write_image(file_path)
    
def WBplot(data, wid=0.2, cap=0.05, xlab='', ylab='amplitude (pA)', xrange=(0.5, 2.5), yrange=(0, 300), lwd=0.8, 
           amount=0.01, plot_width=400, plot_height=600):
    
    unique_x = data['x'].unique()
    fig = go.Figure()

    for i in unique_x:
        d = data[data['x'] == i]['y']
        q1 = np.percentile(d, 25, method='linear')
        q3 = np.percentile(d, 75, method='linear')
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        d_filtered = d[(d >= lower_bound) & (d <= upper_bound)]
        median_val = np.median(d_filtered)
        min_val = np.min(d_filtered)
        max_val = np.max(d_filtered)

        # Box
        fig.add_shape(type="rect", x0=i - wid, y0=q1, x1=i + wid, y1=q3, line=dict(color="black", width=lwd))

        # Median line
        fig.add_trace(go.Scatter(x=[i - wid*1.1, i + wid*1.1], y=[median_val, median_val], mode='lines', line=dict(color='black', width=3*lwd)))

        # Whiskers and caps
        fig.add_trace(go.Scatter(x=[i, i], y=[q1, min_val], mode='lines', line=dict(color='black', width=lwd)))
        fig.add_trace(go.Scatter(x=[i, i], y=[q3, max_val], mode='lines', line=dict(color='black', width=lwd)))
        fig.add_trace(go.Scatter(x=[i - cap, i + cap], y=[min_val, min_val], mode='lines', line=dict(color='black', width=lwd)))
        fig.add_trace(go.Scatter(x=[i - cap, i + cap], y=[max_val, max_val], mode='lines', line=dict(color='black', width=lwd)))

    # Add jitter to x values
    np.random.seed(42)
    data['x_jitter'] = data['x'] + np.random.uniform(-amount, amount, size=data.shape[0])

    # Colors for different x values
    colors = {1: 'slateblue', 2: 'indianred'}

    # Add individual data points with jitter
    for i in unique_x:
        subset_data = data[data['x'] == i]
        fig.add_trace(go.Scatter(
            x=subset_data['x_jitter'],
            y=subset_data['y'],
            mode='markers',
            marker=dict(color=colors[i], size=lwd*8, opacity=0.6),  # Adjusted size and color
            showlegend=False
        ))

    # Connect data points within subjects
    subjects = data['s'].unique()
    for subj in subjects:
        subset_data = data[data['s'] == subj]
        fig.add_trace(go.Scatter(
            x=subset_data['x_jitter'],
            y=subset_data['y'],
            mode='lines',
            line=dict(color='darkgray', width=lwd, dash='dot'),
            showlegend=False
        ))

    # Determine the tick values within the xrange
    tickvals = [x for x in range(int(np.ceil(xrange[0])), int(np.floor(xrange[1])) + 1)]

    # Set layout, remove legend, set plot size, and customize axes
    fig.update_layout(
        xaxis=dict(
            title=xlab,
            range=xrange,
            tickvals=tickvals,
            showline=True,  
            linewidth=lwd,  
            linecolor='black',  
            mirror=False,  
            ticks='outside',  
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
            showgrid=False  
        ),
        yaxis=dict(
            title=ylab,
            range=yrange,
            showline=True,  
            linewidth=lwd,  
            linecolor='black',  
            mirror=False,  
            ticks='outside',
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
            showgrid=False  
        ),
        showlegend=False,
        width=plot_width,
        height=plot_height,
        plot_bgcolor='white' 
    )

    return fig

def df_(df, name='peak'):
    # Initialize lists for x, y, and subjects
    x = []
    y = []
    subjects = []

    # Initialize a counter for name='peak' columns
    col_counter = 1

    # Iterate over columns that start with name='peak'
    for col_name in df.columns:
        if col_name.lower().startswith(name):
            for row_num, value in enumerate(df[col_name]):
                x.append(col_counter)  # Use the counter for 'peak' columns
                y.append(value)
                subjects.append(row_num + 1)  # Assuming each row is a different subject
            col_counter += 1  # Increment the counter for each 'peak' column

    # Create the new DataFrame
    data = pd.DataFrame({'s': subjects, 'x': x, 'y': y})
    return data

def example_plot(x, y, params_fit):

    # Generate the fitted curves
    y_fit = product2(params_fit[:-1], x)
    y_fit1 = product(params_fit[0:3], x)
    y_fit2 = product(params_fit[3:6], x)

    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x, y=y, name='response', line=dict(color='lightgray')))

    # Add the fitted curves
    fig.add_trace(go.Scatter(x=x, y=y_fit, name='fit (sum)', line=dict(color='indianred', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(
        xaxis_title='ms',
        yaxis_title='response',
    )

    return fig
