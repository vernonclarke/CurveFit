'''
functions for model
'''

# run this to import all necessary functions
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

# for plotly graph display offline inside the notebook itself.
import plotly.offline as pyo
from plotly.offline import init_notebook_mode


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

# _alt versions take a1_max not a1 ie a1_max etc is the actual maximum of the alpha synapse
def alpha_alt(params, t):
    # Extract the parameters
    a1_max, tau1 = params
    a1 = a1_max/tau1 * np.exp(1)
    # Compute the sum of two alpha functions
    y = a1 * t * np.exp(-t / tau1) 
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

def product_alt(params, t):
    # Extract the parameters
    a1_max, tau1, tau2 = params
    f = ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2) 
    a1 = a1_max / f
    y = a1 * (1 - np.exp(-t/tau1)) * np.exp(-t/tau2)
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

def model_n(params, ISI, n, x, model='alpha'):
    """
    generate n responses with a specified delay between them and return both
    individual responses and their combined sum

    parameters:
    params (list): list of a values followed by a single b value for each alpha response
    dt (float): time step for the x values
    ISI (float): stimulus interval in ms
    n (int): number of responses.
    x (np.array): array of x values
    model (str): type of model ('alpha', 'alpha2', 'product', 'product2')

    returns:
    tuple (list of np.arrays, np.array): 
        - a list of individual responses 
        - combined response
    """
    individual_responses = []
    
    # Combined response initialized to zero
    combined_response = np.zeros_like(x)

    if model == 'alpha':
        func = alpha_alt
        # Extract b from the end of params and create [a, b] pairs
        b = params[-1]
        pars_list = [[a, b] for a in params[:-1]]
    elif model == 'alpha2':
        func = alpha2_alt
        # Extract b from the end of params and create [a, b] pairs
        b = params[n]
        d = params[-1]
        pars_list = [[a, b, c, d] for a,c in zip(params[:n], params[n+1:-1])]
    elif model == 'product':
        func = product_alt
        # Extract b from the end of params and create [a, b] pairs
        b = params[-2]
        c = params[-1]
        pars_list = [[a, b, c] for a in params[:-2]]    
    elif model == 'product2':
        func = product2_alt
        # Extract b from the end of params and create [a, b] pairs
        b = params[n]
        c = params[n+1]
        e = params[-2]
        f = params[-1]
        pars_list = [[a, b, c, d, e, f] for a,d in zip(params[:n], params[n+2:-2])]
    else:
        raise ValueError(f"unknown model type: {model_type}")

    # index for delay calculation
    dt = x[1] - x[0]
    idx = int(ISI / dt)

    for i, pars in enumerate(pars_list):
        # Calculate delay for the current response
        delay = idx * i

        # Generate response with delay
        x_delayed = np.arange(0, (max(x) - ISI * i + dt), dt)
        y_delayed = func(pars, x_delayed)  

        # Pad with zeros to align with the original x array
        y_delayed_padded = np.concatenate([np.zeros(delay), y_delayed])

        # Add to the list of individual responses
        individual_responses.append(y_delayed_padded[:len(x)])  # Truncate to match the length of x

        # Add the delayed response to the total combined response
        combined_response += y_delayed_padded[:len(combined_response)]  # Truncate to match the length of y

    return individual_responses, combined_response


def find_absolute_max_ISI(x, y, n, ISI):
    """
    Find peaks for each alpha response based on the combined response, 
    number of responses, and frequency.

    Parameters:
    x (np.array): x values.
    y (np.array): response.
    n (int): Number of responses.
    frequency (float): Frequency in Hz.
    dt (float): Time step for the x values.

    Returns:
    List: Peaks for each alpha response.
    """
    peaks = []
    dt = x[1] - x[0]
    idx = int(ISI / dt)  # Index increment per ISI

    for i in range(n):
        # Calculate start and end indices for each segment
        start_idx = i * idx
        end_idx = min(start_idx + idx, len(y))

        # Segment the combined response
        segment = y[start_idx:end_idx]

        # Find the peak within this segment
        if len(segment) > 0:
            max_value = np.max(np.abs(segment))
            peak = max_value * np.sign(segment[np.abs(segment).argmax()])
            peaks.append(peak)
        else:
            peaks.append(0)

    return peaks


def tau_estimators_ISI(x, y, n, ISI, percent=90):
    """
    Estimate rise (T1) for the first response and decay (T2) based on the last response in the train.

    Parameters:
    x (np.array): Array of x values.
    y (np.array): Combined alpha response.
    n (int): Number of responses.
    ISI (float): ISI of the responses.
    percent (float): Percentage threshold for estimating T1 and T2.

    Returns:
    Tuple: (T1, T2) estimates for the first and last response in the train.
    """
    
    # Number of indices per response
    dt = x[1] - x[0]
    idx = int(ISI / dt)  

    # Isolate the first and last response based on n and ISI
    first_response = y[:idx]
    last_response = y[(n-1)*idx:]

    # Estimate T1 using the first response
    ymax_first = np.max(first_response)
    ind_first = np.argmax(first_response)    
    # Find the threshold values for the rise and decay phases
    threshold_first_lower = ymax_first * (1 - percent/100)
    threshold_first_upper = ymax_first * percent/100
    
    ind1_first = mid_value(np.where(y[:ind_first] >= threshold_first_lower)[0])
    if any(value >= threshold_first_upper for value in y[:ind_first]):
        ind2_first = mid_value(np.where(y[:ind_first] >= threshold_first_upper)[0])
    else: 
        ind2 = ind_first
    
    trise = x[ind2_first] - x[ind1_first]

    # Estimate T2 using the last response
    ymax_last = np.max(last_response)
    ind_last = np.argmax(last_response) + (n-1)*idx
    
    threshold_last_lower = ymax_last * (1 - percent/100)
    threshold_last_upper = ymax_last * percent/100

    ind3_last = mid_value(np.where(y[ind_last:] <= threshold_last_upper)[0] + ind_last)
    ind4_last = mid_value(np.where(y[ind_last:] <= threshold_last_lower)[0] + ind_last)
    
    T2 = x[ind4_last] - x[ind3_last]

    if trise <= 0:  # Fix for the rare case where T1 is non-positive
        trise = 0.2 * T2

    T1 = trise * T2 / (T2 - trise) # good estimate for T1
    
    return round(T1,3), round(T2,3)


def rise_and_decay_times_ISI(x, y, n, ISI, rise=[0.2, 0.8], decay=[0.8, 0.2], dp=3):

    # Number of indices per response
    dt = x[1] - x[0]
    idx = int(ISI / dt)  

    # Isolate the first and last response based on n and ISI
    y1st = y[:idx]
    ynth = y[(n-1)*idx:]

    # Estimate T1 using the first response
    ymax1 = np.max(y1st)
    ind1 = (np.abs(y1st - ymax1)).argmin()
#     tpeak1 = x[ind1]
        
    y1 = rise[0] * ymax1
    y2 = rise[1] * ymax1

    # Get the indices where signal is closest to 20% and 80% of peak value before tpeak
    rise_ind1 = np.argmin(np.abs(np.array(y[:ind1]) - y1))
    rise_ind2 = np.argmin(np.abs(np.array(y[:ind1]) - y2))
    
    # Estimate T2 using the nth response
    ymax_nth = np.max(ynth)
    indn = (np.abs(ynth - ymax_nth)).argmin() + (n-1)*idx
#     tpeak_nth = x[indn]
    
    # Get the indices where signal is closest to 80% and 20% of peak value after tpeak
    y3 = decay[0] * ymax_nth
    y4 = decay[1] * ymax_nth
    decay_ind1 = np.argmin(np.abs(np.array(y[indn:]) - y3)) + indn  # fix here
    decay_ind2 = np.argmin(np.abs(np.array(y[indn:]) - y4)) + indn  # and here

    rise_time = x[rise_ind2] - x[rise_ind1]
    decay_time = x[decay_ind2] - x[decay_ind1]

    return np.round(rise_time, dp), np.round(decay_time, dp)


# estimates time points for rise and decays for initial fits
def rise_decay_estimators_ISI(x, y, n, ISI, bl=0, rise=[0.2, 0.8], decay=[0.8, 0.2]):

    # Number of indices per response
    dt = x[1] - x[0]
    idx = int(ISI / dt)  

    # Isolate the first and last response based on n and ISI
    y1st = y[:idx]
    ynth = y[(n-1)*idx:]

    # Estimate T1 using the first response
    ymax1 = np.max(y1st)
    ind1 = (np.abs(y1st - ymax1)).argmin()
#     tpeak1 = x[ind1]
        
    y1 = rise[0] * ymax1
    y2 = rise[1] * ymax1

    # Get the indices where signal is closest to 20% and 80% of peak value before tpeak
    rise_ind1 = np.argmin(np.abs(np.array(y[:ind1]) - y1))
    rise_ind2 = np.argmin(np.abs(np.array(y[:ind1]) - y2))
    
    # Estimate T2 using the nth response
    ymax_nth = np.max(ynth)
    indn = (np.abs(ynth - ymax_nth)).argmin() + (n-1)*idx
#     tpeak_nth = x[indn]
    
    # Get the indices where signal is closest to 80% and 20% of peak value after tpeak
    y3 = decay[0] * ymax_nth
    y4 = decay[1] * ymax_nth
    decay_ind1 = np.argmin(np.abs(np.array(y[indn:]) - y3)) + indn  # fix here
    decay_ind2 = np.argmin(np.abs(np.array(y[indn:]) - y4)) + indn  # and here

    return [x[rise_ind1]-bl, x[rise_ind2]-bl, x[decay_ind1]-bl, x[decay_ind2]-bl]

def random_initial_guess_ISI(x, y, n, ISI, model='alpha', percent=80, cv=0.4, N=10):
    
    ymax = find_absolute_max_ISI(x=x, y=y, n=n, ISI=ISI)
    T1, T2 = tau_estimators_ISI(x=x, y=y, n=n, ISI=ISI, percent=percent)
    
    if model == 'alpha':
        initial = ymax
        initial.append(T2) 
        initial = start_log_normal(input=initial, cv=cv, N=N)

    elif model == 'alpha2':
        initial = [y / 2 for y in ymax]
        initial.append(T2) 
        initial = initial + initial
        initial = start_log_normal(input=initial, cv=cv, N=N)

    elif model == 'product':
#         T1 = Tnet*T2/(T2-Tnet)
        initial = ymax + [T1, T2]
        initial = start_log_normal(input=initial, cv=cv, N=N)
        
    elif model == 'product2':
        initial = [y / 2 for y in ymax]
        initial = initial + [T1, T2]
        initial = initial + initial
        initial = start_log_normal(input=initial, cv=cv, N=N)
    
    return initial

# SSE_ and p0_ functions to calculate starting values 
def SSE_n(pars, ISI, n, x, y, model='alpha'):
    _, yfit = model_n(params=pars, ISI=ISI, n=n, x=x, model=model)
    return np.sum((y - yfit) ** 2.0)


# setting bounds help to reduce the parameter space
def nDE_ISI(x, y, n, ISI, model='alpha', initial_guess=None, percent=80, cv=0.4, N=10):
    
    if initial_guess is None:
        initial = random_initial_guess_ISI(x=x, y=y, n=n, ISI=ISI, model=model, percent=percent, cv=cv, N=N) 
#         initial = random_initial_guess(x=x, y=y, model=model, percent=percent, cv=cv, N=N) 
    else:
        initial = [initial_guess]
        N = 1

    if model == 'alpha':
        bounds=[]
        for ii in range(N):
            b = [(0, 10 * val) for val in initial[ii]]
            b[-1] = (0.01, 10 * initial[ii][-1]) 
            bounds.append(b)
    
    elif model == 'alpha2':
        bounds=[]
        for ii in range(N):
            b = [(0, 10 * val) for val in initial[ii]]
            b[n] = (0.01, 10 * initial[ii][-n])  
            b[-1] = (0.01, 10 * initial[ii][-1])  
            bounds.append(b)
            
    elif model == 'product':
        bounds=[]
        for ii in range(N):
            b = [(0, 10 * val) for val in initial[ii]]
            b[-2:] = [(0.01, 10 * initial[ii][-2]), (0.01, 10 * initial[ii][-1])] 
            bounds.append(b)
            
    elif model == 'product2':
        bounds=[]
        for ii in range(N):
            b = [(0, 10 * val) for val in initial[ii]]
            b[n:n+2] = [(0.01, 10 * initial[ii][-2]), (0.01, 10 * initial[ii][-1])] 
            b[-2:] = [(0.01, 10 * initial[ii][-2]), (0.01, 10 * initial[ii][-1])] 
            bounds.append(b)

    output1 = []
    output2 = []
    # minimized SSE
    SSE = lambda pars, x, y: SSE_n(pars, ISI, n, x, y, model=model)
    for ii in range(N):
        res = differential_evolution(func=SSE, bounds=bounds[ii], args=(x, y), tol=0.01)
        output1.append(res.x)
        output2.append(res.fun)
    return output1, output2

def LS_constr1(p0, n, factor=1):
    tau1_rise = p0[n] * p0[n+1] / (p0[n] + p0[n+1])
    tau2_rise = p0[-2] * p0[-1] / (p0[-2] + p0[-1])
    return tau2_rise - factor*tau1_rise 

def LS_constr2(p0, n, factor=1):
    return p0[-1] - factor*p0[n]
        
def nLS_ISI(x, y, n, ISI, model='alpha', initial_guess=None, percent=80, cv=0.2, N=10, slow_rise_constraint=True):
    
    if initial_guess is None:
        initial = random_initial_guess_ISI(x=x, y=y, n=n, ISI=ISI, model=model, percent=percent, cv=cv, N=N) 
    else:
        initial = [initial_guess]
        N = 1       

    if slow_rise_constraint:
        factor_value = 1
        # define constraints
        constraints = (
            {'type': 'ineq', 'fun': lambda p0: LS_constr1(p0=p0, n=n, factor=factor_value)},
            {'type': 'ineq', 'fun': lambda p0: LS_constr2(p0=p0, n=n, factor=factor_value)}  
            )
        method='SLSQP'
    else:
        constraints=None
        method='L-BFGS-B'
            
    output1 = []
    output2 = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            p0 = initial[ii]
            bounds = [(1e-03, None) for _ in p0]
            res = fit_LS(x=x, y=y, n=n, ISI=ISI, p0=p0, bounds=bounds, model=model, method=method, constraints=constraints)                
            output1.append(res.x)
            output2.append(res.fun)

    return output1, output2


def fit_LS(x, y, n, ISI, p0, bounds, model='alpha', method='L-BFGS-B', constraints=None):
    """
    if constraints = 'None' then method=L-BFGS-B'
    if using constraints then only the optimisation method 'SLSQP' supports bounds and constraints
    """
    # minimise least squares   
    result = minimize(resids, p0, args=(x, y, ISI, n, model), method=method, bounds=bounds, constraints=constraints)
    return result

def resids(params, x, y, ISI, n, model='alpha'):
    """
    compute the sum of the squares of the residuals for models

    Parameters:
    params (list): list of parameters for the model
    x (np.array): array of x values
    y (np.array): array of y values
    ISI (float): inter-Stimulus Interval in ms
    n (int): number of responses
    model (str): type of model ('alpha', 'alpha2', 'product', 'product2').

    Returns:
    float: ss
    """
    # Compute negative log-likelihood
    y_pred = model_n(params=params, ISI=ISI, n=n, x=x, model=model)[1]
    ss = np.sum((y - y_pred)**2)
    return ss

# fits, ss = nLS_ISI(x=x, y=y, n=n, ISI=ISI, model='product2', slow_rise_constraint=True, cv=0.2, N=30)

def MLE_constr1(p0, n, factor=1):
    tau1_rise = p0[n] * p0[n+1] / (p0[n] + p0[n+1])
    tau2_rise = p0[-3] * p0[-2] / (p0[-3] + p0[-2])
    return tau2_rise - factor*tau1_rise 

def MLE_constr2(p0, n, factor=1):
    return p0[-2] - factor*p0[n]

        
def nMLE_ISI(params_start, x, y, n, ISI, sd, model='alpha', slow_rise_constraint=False):
    N = len(params_start)
    # find fits by MLE and isolate best fit with minimum neg loglik
    fits = []
    nLL = []
    factor_value = 1  # determines how much larger second trise or tdecay is relative to the first trise or tdecay
    # currently not an option; value set to one

    # define constraints
    if slow_rise_constraint:
        constraints = (
            {'type': 'ineq', 'fun': lambda p0: MLE_constr1(p0=p0, n=n, factor=factor_value)},
            {'type': 'ineq', 'fun': lambda p0: MLE_constr2(p0=p0, n=n, factor=factor_value)}  
            )
        method='SLSQP'
    else:
        constraints=None
        method='L-BFGS-B'

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for ii in range(N):
            pars = params_start[ii]
            # pars = product2_params(pars, n)
            p0 = np.insert(pars, len(pars), sd) # params plus term for sd
            bounds = [(0, None) for _ in p0]
            result = fit_n(x=x, y=y, n=n, ISI=ISI, p0=p0, bounds=bounds, model=model, method=method, constraints=constraints)
            params_fit = result.x
            fits.append(params_fit)
            y_fit = model_n(params_fit[:-1], ISI, n, x, model=model)[1]            
            nLL.append(result.fun)
            
    return fits, nLL

def nFIT(x, y, n=1, ISI=50, bl=0, model='product2', criterion='AIC', plot=True, initial_guess=None, fit_method='LS', percent=80, dp=3, cv=0.2, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False, slow_rise_constraint=True, nll_calculation='ols'):
    """
    perform batch fitting of a specified model to multiple data columns

    parameters:
    x (array-like): x values for the fitting process in ms
    y (array-like): y values for the data to be fitted
    n (int): number of stimulations in a train
    ISI (float): inter-stimulus interval of responses in a train
    bl (float): baseline period in ms (time to first stimulus)
    model (str): type of model to fit ('alpha', 'alpha2', 'product', 'product2')
    criterion (str): criterion for model evaluation ('AIC' or 'BIC')
    plot (bool): whether to plot the fitted results
    initial_guess (array-like): initial guess for fitting parameters
    fit_method (str): method used for fitting ('LS' for least squares, etc)
    percent (float): percentage for initial parameter estimation (rise and decay)
    dp (int): decimal places for rounding
    cv (float): coefficient of variation for initial guesses
    N (int): number of iterations for fitting
    rise (list): relative start and end of peak for rise time calculation
    decay (list): relative start and end from peak for decay time calculation
    kinetics (str): method for calculating kinetics ('solve' or 'fit')
    show_results (bool): whether to show the fitting results
    slow_rise_constraint (bool): whether to apply a constraint for slow rise (product2 LS or MLE)
    nll_calculation (str): method for calculating negative log-likelihood ('ols' or 'normal') for model evaluation

    returns:
    output1 (DataFrame): dataframe containing the fitting results for each column of df
    output2 (DataFrame): dataframe containing the AIC or BIC values for each column of df, if applicable
    """
    
    if model != 'product2':
        slow_rise_constraint=False
    
    x_orig = x; y_orig = y
    dt = x[1] - x[0]
    ids = int(bl/dt) - 1

    if bl == 0:
        sd = np.std(y[int(len(y) - 5 / dt):int(len(y))])
    else:
        sd = np.std(y[0:ids])
        x = x[ids:]; y = y[ids:]

    if initial_guess is not None:
        N = 1 
        
    params_fit = nFITs(x=x, y=y, n=n, ISI=ISI, sd=sd, model=model, initial_guess=initial_guess, fit_method=fit_method, percent=percent, cv=cv, N=N, slow_rise_constraint=slow_rise_constraint)
    
    if model == 'alpha':
        out = alpha_output(params_fit=params_fit, x=x, y=y, x_orig=x_orig, y_orig=y_orig, ids=ids, n=n, ISI=ISI, bl=bl, criterion=criterion, plot=plot, dp=dp, rise=rise, decay=decay, kinetics=kinetics, show_results=show_results, nll_calculation=nll_calculation)
    elif model == 'alpha2':
        out = alpha2_output(params_fit=params_fit, x=x, y=y, x_orig=x_orig, y_orig=y_orig, ids=ids, n=n, ISI=ISI, bl=bl, criterion=criterion, plot=plot, dp=dp, rise=rise, decay=decay, kinetics=kinetics, show_results=show_results, nll_calculation=nll_calculation)
    elif model == 'product':
        out = product_output(params_fit=params_fit, x=x, y=y, x_orig=x_orig, y_orig=y_orig, ids=ids, n=n, ISI=ISI, bl=bl, criterion=criterion, plot=plot, dp=dp, rise=rise, decay=decay, kinetics=kinetics, show_results=show_results, nll_calculation=nll_calculation)
    elif model == 'product2':
        out = product2_output(params_fit=params_fit, x=x, y=y, x_orig=x_orig, y_orig=y_orig, ids=ids, n=n, ISI=ISI, bl=bl, criterion=criterion, plot=plot, dp=dp, rise=rise, decay=decay, kinetics=kinetics, show_results=show_results, nll_calculation=nll_calculation)

    return out
    
def nFITs(x, y, n=5, ISI=50, sd=5, model='product2', initial_guess=None, fit_method='LS', percent=80, cv=0.2, N=30, slow_rise_constraint=True):
    
    x,y = preprocess(x,y)

    # Perform fitting by either DE or LS
    if fit_method == 'DE':
        fits, sse = nDE_ISI(x=x, y=y, n=n, ISI=ISI, model=model, initial_guess=initial_guess, percent=percent, cv=cv, N=N)
        idx = sse.index(min(sse))
    else:
        fits, ss = nLS_ISI(x=x, y=y, n=n, ISI=ISI, model=model, initial_guess=initial_guess, percent=percent, cv=cv, N=N, slow_rise_constraint=slow_rise_constraint)
        idx = ss.index(min(ss))
    
    N = len(fits) 
    if fit_method != 'MLE':
        params_fit = fits[idx]
        params_fit = np.insert(params_fit, len(params_fit), sd) # params plus term for sd
        params_out = params_fit.tolist()    

    else:
        # find fits by MLE and isolate best fit with minimum neg loglik
        
        fits, nLL = nMLE_ISI(params_start=fits, x=x, y=y, n=n, ISI=ISI, sd=sd, model=model, slow_rise_constraint=slow_rise_constraint) 
        
        idx = nLL.index(min(nLL))
        params_fit = fits[idx]
    
    return params_fit

def alpha_output(params_fit, x, y, x_orig, y_orig, ids, n=5, ISI=50, bl=0, criterion='AIC', plot=True, dp=3, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False, nll_calculation='ols'):    
    
    # Generate the fitted curve
    y_fit = model_n(params_fit[:-1], ISI, n, x, model='alpha')[1]
    params_out = params_fit.tolist()
    
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area = 0
        for ii in range(len(params_out)-2):
            area = area + params_out[ii] * params_out[-2] * np.exp(1)    
        rd = [alpha_kinetics(x, tau=params_out[-2], initial_guess=params_out[-2]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out[-2], initial_guess=params_out[-2]*2, p=decay)[0]
             ]
    elif kinetics == 'fit':
        # add area under curve
        area = np.trapz(y_fit, x)
        # percent rise and decay times 
        dt = x[1] - x[0]
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = model_n(params_fit[:-1], ISI, n, x1, model='alpha')[1]
        rd = rise_and_decay_times_ISI(x=x1, y=y1, n=n, ISI=ISI, rise=rise, decay=decay)
    
    params_out.insert(len(params_out)-1, rd[0])
    params_out.insert(len(params_out)-1, rd[1])
    params_out.insert(len(params_out)-1, area)
   
    # plots
    if bl != 0:
        y_fit = np.concatenate([np.zeros(ids), y_fit])

    # consider making a more accurate y_fit with more accurate x
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    # Set the axis labels
    fig.update_layout(width=1000, height=500, xaxis_title='ms', yaxis_title='response')
    
    # Column names with Unicode symbols
    columns1 = [f'A{i}' for i in range(1, n + 1)]
    columns2 = ['\u03C4', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    columns = columns1 + columns2
    out = params_out[0:len(columns)]
    
    if criterion=='BIC':
        bic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='alpha', nll_calculation=nll_calculation)
        out.append(bic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'BIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(bic,dp)
    elif criterion=='AIC':
        aic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='alpha', nll_calculation=nll_calculation)
        out.append(aic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'AIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(aic,dp)
    else:
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:'])
        if show_results:
            out.append(params_fit[-1])
            return np.round(np.array(out),dp)

def alpha2_output(params_fit, x, y, x_orig, y_orig, ids, n=5, ISI=50, bl=0, criterion='AIC', plot=True, dp=3, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False, nll_calculation='ols'):
    
    # Generate the fitted curve
    y_fit = model_n(params_fit[:-1], ISI, n, x, model='alpha2')[1]
    params_out = params_fit.tolist()

    out1 = params_out[:n+1]
    out2 = params_out[n+1:-1]
    
    if out1[-1] < out2[-1]:
        params_out1 = out1; params_out2 = out2
    else:
        params_out1 = out2; params_out2 = out1

    y_fit1 = model_n(params_out1, ISI, n, x, model='alpha')[1]
    y_fit2 = model_n(params_out2, ISI, n, x, model='alpha')[1]

    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area1 = 0
        for ii in range(len(params_out1)-1):
            area1 = area1 + params_out1[ii] * params_out1[-1] * np.exp(1)   

        area2 = 0
        for ii in range(len(params_out2)-1):
            area2 = area2 + params_out2[ii] * params_out2[-1] * np.exp(1) 

        rd1 = [alpha_kinetics(x, tau=params_out1[-1], initial_guess=params_out1[-1]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out1[-1], initial_guess=params_out1[-1]*2, p=decay)[0]
             ]

        rd2 = [alpha_kinetics(x, tau=params_out2[-1], initial_guess=params_out2[-1]/2, p=rise)[0], 
              alpha_kinetics(x, tau=params_out2[-1], initial_guess=params_out2[-1]*2, p=decay)[0]
             ]

    elif kinetics == 'fit':
        # add area under curve
        area1 = np.trapz(y_fit1, x)
        area2 = np.trapz(y_fit2, x)
        # percent rise and decay times 
        dt = x[1] - x[0]
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = model_n(params_out1, ISI, n, x1, model='alpha')[1]
        y2 = model_n(params_out2, ISI, n, x1, model='alpha')[1]
        rd1 = rise_and_decay_times_ISI(x=x1, y=y1, n=n, ISI=ISI, rise=rise, decay=decay)
        rd2 = rise_and_decay_times_ISI(x=x1, y=y2, n=n, ISI=ISI, rise=rise, decay=decay)

    params_out1.insert(len(params_out1), rd1[0])
    params_out1.insert(len(params_out1), rd1[1])
    params_out1.insert(len(params_out1), area1)    
    params_out2.insert(len(params_out2), rd2[0])
    params_out2.insert(len(params_out2), rd2[1])
    params_out2.insert(len(params_out2), area2)       

    # plots
    if bl != 0:
        y_fit = np.concatenate([np.zeros(ids), y_fit])
        y_fit1 = np.concatenate([np.zeros(ids), y_fit1])
        y_fit2 = np.concatenate([np.zeros(ids), y_fit2])

    # consider making a more accurate y_fit with more accurate x
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    fig.add_trace(go.Scatter(x=x_orig, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(width=1000, height=500, xaxis_title='ms', yaxis_title='response')

    # Column names with Unicode symbols
    columns1 = [f'A{i}' for i in range(1, n + 1)]
    columns2 = ['\u03C4', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    columns = columns1 + columns2
    
    out1 = params_out1[0:len(columns)]
    out2 = params_out2[0:len(columns)]
    out = out1 + out2
    
    if criterion=='BIC':
        bic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='alpha2', nll_calculation=nll_calculation)
        out.append(bic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'BIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(bic,dp)
    elif criterion=='AIC':
        aic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='alpha2', nll_calculation=nll_calculation)
        out.append(aic)
        if plot:
            fig.show()
            out.append(aic)
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'AIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(aic,dp)            
    else:
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:'])
        if show_results:
            out.append(params_fit[-1])
            return np.round(np.array(out),dp)
        
def product_output(params_fit, x, y, x_orig, y_orig, ids, n=5, ISI=50, bl=0, criterion='AIC', plot=True, dp=3, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False, nll_calculation='ols'):

    # Generate the fitted curve
    y_fit = model_n(params_fit[:-1], ISI, n, x, model='product')[1]
    params_out = params_fit.tolist()
    
    tau_rd = product_tau(tau1=params_out[-3], tau2=params_out[-2])
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area = 0
        f = product_f(tau1=params_out[-3], tau2=params_out[-2])
        for ii in range(len(params_out)-3):
            area = area + product_area(A=params_out[ii], tau1=params_out[-3], tau2=params_out[-2]) 
        ests = rise_decay_estimators_ISI(x=x, y=y_fit, n=n, ISI=ISI, bl=bl, rise=rise, decay=decay)
        rd = [product_kinetics(x, tau_rise=tau_rd[0], tau_decay=tau_rd[1], initial_guess=ests[0:2], p=rise)[0], 
              product_kinetics(x, tau_rise=tau_rd[0], tau_decay=tau_rd[1], initial_guess=ests[2:4], p=decay)[0]
              ]

    elif kinetics == 'fit':
        # add area under curve
        area = np.trapz(y_fit, x)
        # percent rise and decay times 
        dt = x[1] - x[0]
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = model_n(params_fit[:-1], ISI, n, x1, model='product')[1]
        rd = rise_and_decay_times_ISI(x=x1, y=y1, n=n, ISI=ISI, rise=rise, decay=decay)


    # calculate tpeak
    tpeak = product_tpeak(tau1=params_out[-3], tau2=params_out[-2])
      
    # replace T1 with Trise
    params_out[-3] = tau_rd[0]
    # insert tpeak
    params_out.insert(len(params_out)-1, tpeak)
    # insert rise and decay
    params_out.insert(len(params_out)-1, rd[0])
    params_out.insert(len(params_out)-1, rd[1])
    # insert area
    params_out.insert(len(params_out)-1, area)

    # plots
    if bl != 0:
        y_fit = np.concatenate([np.zeros(ids), y_fit])

    # consider making a more accurate y_fit with more accurate x
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    # Set the axis labels
    fig.update_layout(width=1000, height=500, xaxis_title='ms', yaxis_title='response')

    # Column names with Unicode symbols
    columns1 = [f'A{i}' for i in range(1, n + 1)]
    columns2 = ['τrise', 'τdecay', 'tpeak', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    columns = columns1 + columns2    
    out = params_out[0:len(columns)]

    if criterion=='BIC':
        bic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='product', nll_calculation=nll_calculation)
        out.append(bic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'BIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(bic,dp)

    elif criterion=='AIC':
        aic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='product', nll_calculation=nll_calculation)
        out.append(aic)
        if plot:
            fig.show()
            df = output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:', 'AIC:'])
            df
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(aic,dp)

    else:
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fit:'])
        if show_results:
            out.append(params_fit[-1])
            return np.round(np.array(out),dp) 
        
def product2_output(params_fit, x, y, x_orig, y_orig, ids, n=5, ISI=50, bl=0, criterion='AIC', plot=True, dp=3, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', show_results=False, nll_calculation='ols'):
        
    # Generate the fitted curve
    y_fit = model_n(params_fit[:-1], ISI, n, x, model='product2')[1]
    params_out = params_fit.tolist()    

    out1 = params_out[:n+2]
    out2 = params_out[n+2:-1]
    
    # sort by rises
    if product_tau(tau1=out1[-2], tau2=out1[-1]) < product_tau(tau1=out2[-2], tau2=out2[-1]):
        params_out1 = out1; params_out2 = out2
    else:
        params_out1 = out2; params_out2 = out1

    y_fit1 = model_n(params_out1, ISI, n, x, model='product')[1]
    y_fit2 = model_n(params_out2, ISI, n, x, model='product')[1]

    tau_rd1 = product_tau(tau1=params_out1[-2], tau2=params_out1[-1])
    tau_rd2 = product_tau(tau1=params_out2[-2], tau2=params_out2[-1])
    
    # get rise and decay + area by solving / equation
    if kinetics == 'solve':
        area1 = 0
        f1 = product_f(tau1=params_out1[-2], tau2=params_out1[-1])
        for ii in range(len(params_out1)-2):
            area1 = area1 + product_area(A=params_out1[ii], tau1=params_out1[-2], tau2=params_out1[-1]) 
        ests1 = rise_decay_estimators_ISI(x=x, y=y_fit1, n=n, ISI=ISI, bl=bl, rise=rise, decay=decay)
        rd1 = [product_kinetics(x, tau_rise=tau_rd1[0], tau_decay=tau_rd1[1], initial_guess=ests1[0:2], p=rise)[0], 
              product_kinetics(x, tau_rise=tau_rd1[0], tau_decay=tau_rd1[1], initial_guess=ests1[2:4], p=decay)[0]
              ]
        
        area2 = 0
        f2 = product_f(tau1=params_out2[-2], tau2=params_out2[-1])
        for ii in range(len(params_out2)-2):
            area2 = area2 + product_area(A=params_out2[ii], tau1=params_out2[-2], tau2=params_out2[-1]) 
        ests2 = rise_decay_estimators_ISI(x=x, y=y_fit2, n=n, ISI=ISI, bl=bl, rise=rise, decay=decay)
        rd2 = [product_kinetics(x, tau_rise=tau_rd2[0], tau_decay=tau_rd2[1], initial_guess=ests2[0:2], p=rise)[0], 
              product_kinetics(x, tau_rise=tau_rd2[0], tau_decay=tau_rd2[1], initial_guess=ests2[2:4], p=decay)[0]
              ]

    elif kinetics == 'fit':
        # add area under curve
        area1 = np.trapz(y_fit1, x)
        area2 = np.trapz(y_fit2, x)
        # percent rise and decay times 
        dt = x[1] - x[0]
        if dp > 4:
            x1 = np.arange(x[0], x[-1], dt/10**4) 
        else:
            x1 = np.arange(x[0], x[-1], dt/10**dp) 
        y1 = model_n(params_out1, ISI, n, x1, model='product')[1]
        y2 = model_n(params_out2, ISI, n, x1, model='product')[1]
        rd1 = rise_and_decay_times_ISI(x=x1, y=y1, n=n, ISI=ISI, rise=rise, decay=decay)
        rd2 = rise_and_decay_times_ISI(x=x1, y=y2, n=n, ISI=ISI, rise=rise, decay=decay)

    # calculate tpeaks
    tpeak1 = product_tpeak(tau1=params_out1[-2], tau2=params_out1[-1])
    tpeak2 = product_tpeak(tau1=params_out2[-2], tau2=params_out2[-1])
    
    # replace T1 with Trise
    params_out1[-2] = tau_rd1[0]
    params_out2[-2] = tau_rd2[0]
    
    # insert tpeak
    params_out1.insert(len(params_out1), tpeak1)
    params_out2.insert(len(params_out2), tpeak2)
    
    # insert rise and decay
    params_out1.insert(len(params_out1), rd1[0])
    params_out1.insert(len(params_out1), rd1[1])
    
    params_out2.insert(len(params_out2), rd2[0])
    params_out2.insert(len(params_out2), rd2[1])
    
    # insert area
    params_out1.insert(len(params_out1), area1)
    params_out2.insert(len(params_out2), area2)    

    # plots
    if bl != 0:
        y_fit = np.concatenate([np.zeros(ids), y_fit])
        y_fit1 = np.concatenate([np.zeros(ids), y_fit1])
        y_fit2 = np.concatenate([np.zeros(ids), y_fit2])

    # consider making a more accurate y_fit with more accurate x
    # Create the figure
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig, name='response', line=dict(color='lightgray')))

    # Add the fitted curve
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit, name='fit', line=dict(color='indianred', dash='dot')))

    fig.add_trace(go.Scatter(x=x_orig, y=y_fit1, name='fit1', line=dict(color='slateblue', dash='dot')))
    fig.add_trace(go.Scatter(x=x_orig, y=y_fit2, name='fit2', line=dict(color='slateblue', dash='dot')))

    # Set the axis labels
    fig.update_layout(width=1000, height=500, xaxis_title='ms', yaxis_title='response')
    
    # Column names with Unicode symbols
    columns1 = [f'A{i}' for i in range(1, n + 1)]
    columns2 = ['τrise', 'τdecay', 'tpeak', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    columns = columns1 + columns2
    
    out1 = params_out1[0:len(columns)]
    out2 = params_out2[0:len(columns)]
    out = out1 + out2

    if criterion=='BIC':
        bic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='product2', nll_calculation=nll_calculation)
        out.append(bic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'BIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(bic,dp)

    elif criterion=='AIC':
        aic = IC_n(params_fit, x, y, ISI, n, criterion=criterion, model='product2', nll_calculation=nll_calculation)
        out.append(aic)
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:', 'AIC:'])
        if show_results:
            out[-1] = params_fit[-1]
            return np.round(np.array(out),dp), np.round(aic,dp)
            
    else:
        if plot:
            fig.show()
            output_fun(np.round(out,dp), columns=columns, row_labels = ['fast:', 'slow:'])
        if show_results:
            out.append(params_fit[-1])
            return np.round(np.array(out),dp)
        
        
def mid_value(arr):
    middle_index = len(arr) // 2
    closest_value = arr[middle_index]
    return closest_value

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

# nll_calculation can be 'ols' or 'normal'
def IC_n(params_fit, x, y,  ISI, n, criterion='BIC', model='alpha', nll_calculation='ols'):
    if nll_calculation=='ols':
        func = nll_n2
    elif nll_calculation=='normal':
        func = nll_n
    
    # Given a set of candidate models for the data, the preferred model is the one with the minimum AIC/BIC value.
    if criterion=='BIC':
        # Calculate Bayesian information criterion or BIC
        n = len(y)
        k = len(params_fit) - 1  # Subtract 1 to exclude the standard deviation parameter
        nll = func(params_fit, x, y, ISI, n, model=model)
        ic = 2 * nll + k * np.log(n) # k*log(n) - 2*LL
    elif criterion=='AIC':
        # Calculate Akaike information criterion or AIC
        k = len(params_fit) - 1  # Subtract 1 to exclude the standard deviation parameter
        nll = func(params_fit, x, y, ISI, n, model=model)
        ic = 2 * k + 2 * nll # 2*k - 2*LL
    return ic

def nll_n(params, x, y, ISI, n, model='alpha'):
    """
    compute the negative log-likelihood for models

    Parameters:
    params (list): list of parameters for the model
    x (np.array): array of x values
    y (np.array): array of y values
    ISI (float): inter-stimulus interval in ms
    n (int): number of responses
    model (str): type of model ('alpha', 'alpha2', 'product', 'product2').

    Returns:
    float: negative log-likelihood value.
    """
    # Extract standard deviation and model function based on model_type
    sd = params[-1]

    # Compute negative log-likelihood
    y_pred = model_n(params=params[:-1], ISI=ISI, n=n, x=x, model=model)[1]
    LL = np.sum(norm.logpdf(y, y_pred, sd))
    nLL = -LL
    return nLL

def nll_n2(params, x, y, ISI, n, model='alpha'):
    """
    Compute the negative log-likelihood for models using the OLS method.

    Parameters:
    params (list): List of parameters for the model.
    x (np.array): Array of x values.
    y (np.array): Array of y values.
    ISI (float): Inter-Stimulus Interval in ms.
    n (int): Number of responses.
    model (str): Type of model ('alpha', 'alpha2', 'product', 'product2').

    Returns:
    float: Negative log-likelihood value.
    """
    # Calculate the model predictions
    y_pred = model_n(params=params[:-1], ISI=ISI, n=n, x=x, model=model)[1]

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y - y_pred) ** 2)

    # Number of data points
    num_points = len(x)

    # Number of parameters (excluding the last one, which is assumed to be sd)
    num_params = len(params) - 1

    # Estimate of the error variance (sigma squared)
    sigma_squared = rss / (num_points - num_params)

    # Calculate the negative log-likelihood using the RSS
    nLL = num_points * np.log(2 * np.pi * sigma_squared) / 2 + rss / (2 * sigma_squared)

    return nLL
    
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
    
def fit_n(x, y, n, ISI, p0, bounds, model='alpha', method='L-BFGS-B', constraints=None):
    """
    if constraints = 'None' then method=L-BFGS-B'
    if using constraints then only the optimisation method 'SLSQP' supports bounds and constraints
    """
    # Perform MLE to fit alpha function    
    result = minimize(nll_n, p0, args=(x, y, ISI, n, model), method=method, bounds=bounds, constraints=constraints)
    return result

def product_f(tau1, tau2):
    return ((tau1/(tau1+tau2)) ** (tau1/tau2)) * tau2/(tau1+tau2) 

def product_area(A, tau1, tau2):
    f = product_f(tau1=tau1, tau2=tau2)
    return A / f * tau2**2 / (tau1 + tau2)

# gives tau_rise and decay
def product_tau(tau1, tau2):
    return [tau1*tau2/(tau1+tau2), tau2] 
    
# gives tpeak
def product_tpeak(tau1, tau2):
    return tau1 * np.log((tau1 + tau2)/tau1)

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

def nFITbatch(x, df, n=1, ISI=50, bl=0, model='product2', criterion='AIC', plot=True, initial_guess=None, fit_method='LS', percent=80, dp=3, cv=0.2, N=30, rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve', slow_rise_constraint=True, nll_calculation='ols'):    
    """
    perform batch fitting of a specified model to multiple data columns

    parameters:
    x (array-like): x values for the fitting process in ms
    df (DataFrame): dataframe where each column represents a different dataset to be fitted
    n (int): number of stimulations in a train
    ISI (float): inter-stimulus interval of responses in a train
    bl (float): baseline period in ms (time to first stimulus)
    model (str): type of model to fit ('alpha', 'alpha2', 'product', 'product2')
    criterion (str): criterion for model evaluation ('AIC' or 'BIC')
    plot (bool): whether to plot the fitted results
    initial_guess (array-like): initial guess for fitting parameters
    fit_method (str): method used for fitting ('LS' for least squares, etc)
    percent (float): percentage for initial parameter estimation (rise and decay)
    dp (int): decimal places for rounding
    cv (float): coefficient of variation for initial guesses
    N (int): number of iterations for fitting
    rise (list): relative start and end of peak for rise time calculation
    decay (list): relative start and end from peak for decay time calculation
    kinetics (str): method for calculating kinetics ('solve' or 'fit')
    show_results (bool): whether to show the fitting results
    slow_rise_constraint (bool): whether to apply a constraint for slow rise (product2 LS or MLE)
    nll_calculation (str): method for calculating negative log-likelihood ('ols' or 'normal') for model evaluation

    returns:
    output1 (DataFrame): dataframe containing the fitting results for each column of df
    output2 (DataFrame): dataframe containing the AIC or BIC values for each column of df, if applicable
    """
    
    fits_list = []
    criterion_list = []
    
    for column in df.columns:
        try:
            if criterion in ('BIC', 'AIC'):
                fits, crit_value = nFIT(x=x, y=df[column].values, n=n, ISI=ISI, bl=bl, model=model, criterion=criterion, plot=plot, initial_guess=initial_guess, fit_method=fit_method, percent=percent, dp=dp, cv=cv, N=N, rise=rise, decay=decay, kinetics=kinetics, show_results=True, slow_rise_constraint=slow_rise_constraint, nll_calculation=nll_calculation)
                criterion_list.append(crit_value)
            else:
                fits = nFIT(x=x, y=df[column].values, n=n, ISI=ISI, bl=bl, model=model, criterion=criterion, plot=plot, initial_guess=initial_guess, fit_method=fit_method, percent=percent, dp=dp, cv=cv, N=N, rise=rise, decay=decay, kinetics=kinetics, show_results=True, slow_rise_constraint=slow_rise_constraint, nll_calculation=nll_calculation)

            fits_list.append(fits)
        except Exception as e:
            print(f"An error occurred with column {column}: {e}")
            fits_list.append(None)
            criterion_list.append(None)

    output1 = pd.DataFrame(fits_list, columns=get_column_names(model=model, n=n, rise=rise, decay=decay))
  
    if criterion in ('BIC', 'AIC'):
        output2 = pd.DataFrame(criterion_list, columns=[criterion])
        return output1, output2
    else:
        return output1

def get_column_names(model, n, rise=[0.2, 0.8], decay=[0.8, 0.2]):
    """
    generate column names for the output df based on the model type

    parameters:
    model (str): Type of model ('alpha', 'alpha2', 'product', 'product2')
    n (int): number of stimulations in a train
    rise (list): relative start and end of peak for rise time calculation
    decay (list): relative start and end from peak for decay time calculation

    returns:
    list: List of column names.
    """
    # Base columns for amplitude parameters
    amplitude_columns = [f'A{i}' for i in range(1, n + 1)]

    # Additional columns based on model
    if model == 'alpha' or model == 'alpha2':
        extra_columns = ['τ', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']
    elif model == 'product' or model == 'product2':
        extra_columns = ['τrise', 'τdecay', 'tpeak', f'r{int(100*rise[0])}-{int(100*rise[1])}', f'd{int(100*decay[0])}-{int(100*decay[1])}', 'area']

    # Concatenate and repeat columns if necessary
    if model == 'alpha2' or model == 'product2':
        columns = amplitude_columns + extra_columns + amplitude_columns + extra_columns + ['\u03C3']
    else:
        columns = amplitude_columns + extra_columns + ['\u03C3']

    return columns

def extract_summary_stats(data_dict, key, confidence_interval=0.95, dp=3, col_names=None):
    """
    extracts summary statistics for a specific key from a dictionary of data

    Parameters:
        data_dict (dict): dictionary containing the data
        key (str): key for which to extract the summary statistics
        confidence_interval (float): confidence interval (default: 0.95)
        dp (int): number of decimal places for rounding (default: 3)

    Returns:
        dict: summary statistics for the specified key
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
    calculates the proportion selecting a specific model based on BIC or AIC outcomes
    
    Parameters:
        data_dict (dict): dictionary containing the data
        model (int): model for which to calculate the proportion
        criterion (str): criterion to use for model selection ('BIC' or 'AIC')
    
    Returns:
        float: proportion of the specified model based on BIC or AIC outcomes
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
