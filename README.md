-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
# <center>CurveFit</center>
#### This is a work in progress

vernon.clarke@northwestern.edu

## Instructions for setting up and running Jupyter Notebook

## A. Check if Python is Installed

To determine if Python is installed on your system and to check its version, follow these steps:

1. **Open your Terminal**:
   - On Windows: Search for command prompt or PowerShell in the start menu.
   - On MacOS: Press `cmd + cpace` to open spotlight search and type 'terminal'.
   - On Linux: Search for terminal in your applications menu or press `ctrl + alt + T`.

2. **Check Python version**:
   ```bash
   python --version
   ```

If Python is installed, the version number will be displayed

## Install Python (if not already installed)

Follow these instructions based on your operating system:

### On Windows

- **Download Python**: Navigate to the [official Python website](https://python.org) and download the latest version for Windows.
- **Install Python**: Open the downloaded installer. Ensure you select the "Add Python to PATH" option during installation for easier command-line access.

### On MacOS

- **Download Python**: Visit the [official Python website](https://python.org) to download Python for MacOS.
- Alternatively, use **Homebrew** to install Python by opening Terminal and running:
    ```bash
    brew install python3
    ```

### On Linux

Python is usually pre-installed on Linux. If you need to install or update it, use the package manager for your distribution:

- **On Ubuntu** (and Debian-based systems):
    ```bash
    sudo apt update
    sudo apt install python3
    ```
- **On Fedora** (and RHEL-based systems):
    ```bash
    sudo dnf install python3
    ```
- **On Arch Linux**:
    ```bash
    sudo pacman -S python3
    ```

## B1. Setting up Jupyter Notebook without Anaconda/Miniconda

#### In this example a directory called Jupyter (this can be named whatever you like) has been created within the documents folder

### Create 'fitting' environment <span style="font-size: 80%;">(do once to set up)</span>:

1. **Open Terminal**

2. **Navigate to the Jupyter Directory**
    ```bash
    cd documents
    cd Jupyter
    ```

3. **Create a new virtual environment named 'fitting'**
    ```bash
    python3 -m venv fitting
    ```

4. **Activate the 'fitting' environment**
   - On MacOS and Linux:
     ```bash
     source fitting/bin/activate
     ```
   - On Windows:
     ```bash
     .\fitting\Scripts\activate
     ```

5. **Install the required packages**
    ```bash
    pip install jupyter numpy pandas matplotlib openpyxl plotly scipy tqdm ipywidgets numba
    ```

6. **Quit terminal**
    ```bash
    deactivate
    exit
    ```
    
7. **Close the terminal window/tab**

### To Run  <span style="font-size: 80%;">(do this every time you want to run the code)</span>:

1. **Open terminal**

2. **Navigate back to the Jupyter directory and activate the 'fitting' environment**
    ```bash
    cd documents
    cd Jupyter
    source fitting/bin/activate  # On Windows use: .\fitting\Scripts\activate
    ```

3. **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

4. **Exit Jupyter Notebook when finished**
    ```bash
    ctrl+C
    ```

5. **Deactivate the virtual environment**
    ```bash
    deactivate
    ```

6. **Close the terminal window/tab**
   

## B2. Setting up Jupyter Notebook with Anaconda/Miniconda

**An alternative to the above is to use Anaconda or Miniconda**

Anaconda and Miniconda are Python distributions that include the Python interpreter along with a suite of other tools and libraries. These can be particularly useful for scientific computing, data analysis, and machine learning projects.

Even if Python is already installed, you may prefer to use Anaconda/Miniconda.

- **Anaconda**: Download the installer from the [Anaconda website](https://www.anaconda.com/products/individual).
- **Miniconda**: Download the installer from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

Run the downloaded installer and follow the instructions to set up your Python environment.


### create 'fitting' environment <span style="font-size: 80%;">(do once to set up)</span>:

1. **Open Terminal**

2. **Navigate to Jupyter directory**
    ```bash
    cd documents
    cd Jupyter
    ```

3. **Create a new conda environment named 'fitting'**
    ```bash
    conda create -n fitting
    ```

4. **Activate the 'fitting' environment**
    ```bash
    conda activate fitting
    ```

5. **Install the required packages**
    ```bash
    conda install -n fitting sqlite jupyter numpy pandas matplotlib openpyxl plotly scipy tqdm ipywidgets numba
    ```

6. **Quit Terminal**
    ```bash
    exit
    ```

7. **Close the terminal window/tab**

### To run:

1. **Open Terminal**

2. **Navigate back to the Jupyter directory and activate the 'fitting' environment**
    ```bash
    cd documents
    cd Jupyter
    conda activate fitting
    ```

3. **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

4. **Exit Jupyter Notebook when finished**
    ```bash
    ctrl+C
    ```

5. **Close the terminal window/tab**

-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
## <center>These routines are designed to provide fits for product and alpha functions</center>


### product function:

$$
\boldsymbol{ y = A (1 - e^{-t/\tau_1}) e^{-t/\tau_2} }
$$

This equation can be written in the alternative form:

$$
\boldsymbol{ y = A  (e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}}) }
$$


where 

$$
\boldsymbol{ \tau_{rise} = \tau_1 \tau_2 / (\tau_1 + \tau_2) }
$$

$$
\boldsymbol{ \tau_{decay} = \tau_2 }
$$


### sum of two product functions:


$$
y = A_1 (1 - e^{-t/\tau_1}) e^{-t/\tau_2} + A_2 (1 - e^{-t/\tau_3}) e^{-t/\tau_4} 
$$

This equation can be written in the alternative form:

$$
y = A_1  (e^{-t/\tau_{decay1}} - e^{-t/\tau_{rise1}}) + A_2  (e^{-t/\tau_{decay2}} - e^{-t/\tau_{rise2}}) 
$$

where     

$$
\tau_{rise1} = \frac{\tau_1 \tau_2}{\tau_1 + \tau_2} 
$$

$$
\tau_{rise2} = \frac{\tau_3 \tau_4}{\tau_3 + \tau_4}
$$

$$
\tau_{decay1} = \tau_2 
$$

$$
\tau_{decay2} = \tau_4
$$


### alpha function:

$$
y = A t e^{-t/\tau} 
$$

### sum of two alpha functions:

$$
y = A_1 t e^{-t/\tau_1} + A_2 t e^{-t/\tau_2} 
$$

## <center>Solutions for the product function</center>

### Product function takes the form:

$$
y = A (1 - e^{-t/\tau_1}) e^{-t/\tau_2} 
$$

This equation can be written:

$$
y = A  (e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}}) 
$$

where    
 
$$
\tau_{rise} = \tau_1 \tau_2 / (\tau_1 + \tau_2)
$$

$$
\tau_{decay} = \tau_2
$$

### Time to peak of response for product function:

In order to calculate $t_{peak}$, differentiate y with respect to t:

$$
\frac{dy}{dt} = A \left( \frac{e^{-t/\tau_{rise}}}{\tau_{rise}} - \frac{e^{-t/\tau_{decay}}}{\tau_{decay}} \right) 
$$

The time of the peak of the response $t = t_{peak}$ can be found by solving $\frac{dy}{dt} = 0$:

$$
0 = A \left( \frac{e^{-t_{peak}/\tau_{rise}}}{\tau_{rise}} - \frac{e^{-t_{peak}/\tau_{decay}}}{\tau_{decay}} \right)
$$

simplifying:

$$
\frac{e^{-t_{peak}/\tau_{decay}}}{\tau_{decay}} = \frac{e^{-t_{peak}/\tau_{rise}}}{\tau_{rise}}
$$

cross-multiplying:

$$
\frac{\tau_{decay}}{\tau_{rise}} = \frac{e^{-t_{peak}/\tau_{decay}}}{e^{-t_{peak}/\tau_{rise}}}
$$

simplifying:

$$
\frac{\tau_{decay}}{\tau_{rise}} = {e^{t_{peak} \left(\frac{\tau_{decay} - \tau_{rise}}{\tau_{decay} \cdot \tau_{rise}}\right)}}
$$

taking the natural logarithm of both sides and rearranging gives an expression for the time to peak $t_{peak}$:

$$
t_{peak} = \frac{\tau_{decay} \tau_{rise}}{\tau_{decay} - \tau_{rise}} ln\left(\frac{\tau_{decay}}{\tau_{rise}}\right)
$$

substituting for $\tau_{rise}$ and $\tau_{decay}$ gives an equivalent form in terms of $\tau_1$ and $\tau_2$:

$$
t_{peak} = \tau_1 ln\left(\frac{\tau_1 + \tau_2}{\tau_1}\right)
$$

To find the peak of the response $A_{peak}$, find solution where $t = t_{peak}$

$$
\boldsymbol{ A_{peak} = Af }
$$

where the fraction f is given by 

$$
f = {e^{-t_{peak}/\tau_{decay}} - e^{-t_{peak}/\tau_{rise}}}
$$

f in terms of $\tau_{decay}$ and $\tau_{rise}$: 

$$
f = {e^{-\frac{\tau_{rise}}{\tau_{decay} - \tau_{rise}} ln\left(\frac{\tau_{decay}}{\tau_{rise}}\right)} - e^{-\frac{\tau_{decay} }{\tau_{decay} - \tau_{rise}} ln\left(\frac{\tau_{decay}}{\tau_{rise}}\right)}}
$$

since $e^{-xlog(y)} = e^{log(y^{-x})} = y^{-x}$

$$
\boldsymbol{ f = {\left( \left(\frac{\tau_{decay}}{\tau_{rise}}\right)^{-\frac{\tau_{rise}}{\tau_{decay}-\tau_{rise}}} \right) \left( 1 - \frac{\tau_{rise}}{\tau_{decay}}\right) } }
$$

similarly in terms of $\tau_1$ and $\tau_2$: 

$$
\boldsymbol{ f = {\left( \left( \frac{\tau_1}{\tau_1+\tau_2} \right)^{\frac{\tau_1}{\tau_2}} \right) \frac{\tau_2}{\tau_1+\tau_2}} }
$$

### Area under the curve for the product function:

To find the area under the curve for the equation:

$$ 
y = A(e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}}) 
$$

Integrate the function with respect to t then calculate the integral of y from 0 to $\infty$ (i.e. the area under the curve):

$$ 
\text{Area} = \int_{0}^{\infty} A(e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}})dt 
$$

Solve this integral:

$$
\text{Area} = \int_{0}^{\infty} A(e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}})dt = A \left[ -\tau_{decay} e^{-t/\tau_{decay}} + \tau_{rise} e^{-t/\tau_{rise}} \right]_0^{\infty} 
$$   

$$
= A (\tau_{decay} - \tau_{rise}) 
$$    

Area under the curve is given by

$$ 
\boldsymbol{ \text{Area} = A (\tau_{decay} - \tau_{rise}) = \frac{A_{peak}}{f} (\tau_{decay} - \tau_{rise}) }
$$

Similarly, in terms of $\tau_1$ and $\tau_2$:

$$ 
\boldsymbol{ \text{Area} = A \left(\frac{\tau_2^2}{\tau_1 + \tau_2}\right) = \frac{A_{peak}}{f} \left(\frac{\tau_2^2}{\tau_1 + \tau_2}\right) }
$$

where $A_{peak}$ is the peak amplitude and f is as previously defined (see above)

### Rise and decay kinetics:
Let p be the relative amplitudes of the response at some time t such that $p = y / A_{peak}$
rearranging the equation:

This equation can be written:

$$
y = A_{peak}f(e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}}) 
$$

gives:

$$
e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}} -fp = 0
$$

This equation can be solved for t using a numerical root-finding algorithm (e.g. using an iterative method with an initialised variables as the starting value for the iteration):

Solving for $p_1$ and $p_2$ gives $t_1$ and $t_2$

if $p_2 > p_1$ then rise time is given by:

$$
rise_{p_1 - p_2} =  t_2 - t_1
$$

if $p_1 > p_2$ then decay time is given by:

$$
decay_{p_1 - p_2} =  t_2 - t_1
$$

for instance if $p_1 = 0.2$ and $p_2 = 0.8$ then $t_2$ - $t_1$ gives the 20 - 80 % rise time

likewise if $p_1 = 0.9$ and $p_2 = 0.1$ then $t_2$ - $t_1$ gives the 90 - 10 % decay time

## <center>Solutions for the alpha function</center>

### alpha function takes the form:

$$
y = A t e^{-t/\tau} 
$$

The solutions for the peak response ($A_{peak}$), time to peak ($t_{peak}$) and area are easier to calculate:

### Time to peak of response for alpha function:

In order to calculate $t_{peak}$, differentiate y with respect to t:

$$
\frac{dy}{dt} = A \left(1 - \frac{t}{\tau} \right) e^{-t/\tau} 
$$

The time of the peak of the response $t = t_{peak}$ can be found by solving $ \frac{dy}{dt} = 0 $:

$$
0 = A \left(1 - \frac{t}{\tau} \right) e^{-t/\tau} 
$$

simplifying:

$$
\boldsymbol{ t_{peak} = \tau }
$$

To find the peak of the response $A_{peak}$, find solution where $t = t_{peak}$

$$
\boldsymbol{ A_{peak} = A \tau e^{-1} }
$$

Substituting for $A = \frac{A_{peak}} {\tau} e$ in original equation gives an often used form of the alpha function 

$$
\boldsymbol{ y = A_{peak} \frac{t}{\tau} e^{1-t/\tau} }
$$

### Area under the curve for the alpha function:

To find the area under the curve for the equation:

$$ 
y = A t e^{-t/\tau} 
$$

Integrate the function with respect to t then calculate the integral of y from 0 to $\infty$ (i.e. the area under the curve):

$$ 
\text{Area} = \int_{0}^{\infty} A t e^{-t/\tau} dt 
$$

Solve this integral:

$$
\lim_{T \to \infty} \int_{0}^{T} Ate^{-t/\tau} dt = \lim_{T \to \infty} \left[ -A\tau e^{-t/\tau}(t+\tau) \right]_0^T 
$$ 

$$ 
= \lim_{T \to \infty} \left[ -A\tau e^{-T/\tau}(T+\tau) + A\tau^2e^{0} \right] 
$$

$$
\boldsymbol{ \text{Area} = A\tau^2 = A_{peak} \tau e^1 }
$$
    
### Rise and decay kinetics:
Let p be the relative amplitudes of the response at some time t such that $p = y / A_{peak}$
rearranging the equation:

$$
\boldsymbol{ y = A_{peak} \frac{t}{\tau} e^{1-t/\tau} }
$$

gives:

$$
 te^{-t/\tau} - p\tau e^{-1} = 0
$$

This equation can be solved for t using a numerical root-finding algorithm (e.g. using an iterative method with an initialised variables as the starting value for the iteration):

Solving for $p_1$ and $p_2$ gives $t_1$ and $t_2$

$$
rise_{p_1 - p_2} =  t_2 - t_1
$$

if $p_1 > p_2$ then decay time is given by:

$$
decay_{p_1 - p_2} =  t_2 - t_1
$$

for instance if $p_1 = 0.2$ and $p_2 = 0.8$ then $t_2$ - $t_1$ gives the 20 - 80 % rise time

likewise if $p_1 = 0.9$ and $p_2 = 0.1$ then $t_2$ - $t_1$ gives the 90 - 10 % decay time

## <center>Strategy<center>

1. **find approximate starting values:**
    - Using `tau_estimators` (20-80% rise and 80-20% decay)

2. **get optimised starting parameter values:**
    - Using `curvefit` least squares method (default) or `differential_evolution` 
   
   **advantages of `least squares` / `differential_evolution`:**   
   
   - **Differential Evolution:**
     - *Global Optimization:* Explores the entire solution space, suitable for multiple minima or maxima.
     - *Non-Differentiable Objective Functions:* Handles non-differentiable or discontinuous functions.
     - *Reduced Sensitivity to Initial Conditions:* Less sensitive to initial guesses in complex problems.
     - *No Derivative Information Required:* Useful when derivatives are hard to obtain or functions are noisy.
     
   - **Least Squares:**
     - *Suitable for Linear Regression:* Has a closed-form solution for linear relationships.
     - *Interpretability:* Provides interpretable coefficients in regression.
     - *Efficiency for Well-Behaved Problems:* Efficient where the solution space is smooth and well-defined.
     - *Mathematical Foundation:* Widely used due to its strong mathematical basis.

3. **get final fits by Maximum Likelihood Estimation (MLE) using optimised starting parameters:**
    - From step 2 as starting values
    
4. **output:**
    - Initial approximate starting values from step 2
    - Fit in form [ $A_1$, $\tau_1$, $\tau_2$, $A_2$, $\tau_3$, $\tau_4$, $\sigma$ ]; act as starting values for MLE fit; accurate as 'least squares' or 'differential evolution' curve fits
    - Fits [ $A_{peak_1}$, $\tau_{rise_1}$, $\tau_{decay_1}$, $A_{peak_2}$, $\tau_{rise_2}$, $\tau_{decay_2}$ ]
    - Model information criterion if chosen

## <center>Maximum Likelihood Estimation (MLE)<center>

Maximum Likelihood Estimation (MLE) is a statistical method used for estimating the parameters of a model. Given a statistical model with some unknown parameters, MLE aims to find the parameter values that maximize the likelihood function. The likelihood function measures how well the model explains the observed data. Stages involved in MLE:

#### 1. **Defining the Model:**
   - Statistical model represented by a probability distribution that is defined by some parameters. This could be a normal distribution, Poisson distribution, etc.

#### 2. **Likelihood Function:**
   - The likelihood function is defined as the probability of observing the given data as a function of the parameters of the model. 
   - Mathematically, it can be written as $L(\theta | X)$, where $\theta$ represents the parameters, and $X$ represents the data.

#### 3. **Maximization:**
   - Find the values of the parameters that maximize this likelihood function. In other words, find the values of $\theta$ that make the observed data most probable.

#### 4. **Log-Likelihood:**
   - Often, it is mathematically easier to work with the natural logarithm of the likelihood function, known as the log-likelihood. 
   - Taking the logarithm simplifies the mathematics (turns products into sums) and doesn’t change the location of the maximum.

#### 5. **Optimization:**
   - Optimization techniques, such as gradient descent, are used to find the values of the parameters that maximize the log-likelihood.
   - The first derivative (slope) of the log-likelihood is set to zero, and solving this equation gives the maximum likelihood estimates.

### Example:

Consider a sample of data $X = {x_1, x_2, ..., x_n}$ from a normal distribution with unknown mean $\mu$ and known variance $\sigma^2$. The likelihood function is given by:

$$
L(\mu | X) = \prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

Taking the natural logarithm (log-likelihood) gives:

$$
\log(L(\mu | X)) = -\frac{n}{2} \log(2 \pi \sigma^2) - \sum_{i=1}^{n} \frac{(x_i - \mu)^2}{2\sigma^2}
$$

### Key Points:
- MLE finds the parameter values that make the observed data most probable under the assumed model.
- MLE estimates are found by maximizing the likelihood or log-likelihood function.
- MLE is widely used in various statistical models and machine learning algorithms for parameter estimation.

### Model selection using Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC)

### 1. **Penalization of Model Complexity:**
   - **BIC and AIC:** Both criteria penalize model complexity. They help in avoiding overfitting by penalizing models with more parameters.
   - **Sum of Squares:** SS does not penalize model complexity. A model with more parameters might fit the data better by capturing the noise as a pattern, leading to overfitting.

### 2. **Model Comparison:**
   - **BIC and AIC:** They allow for the comparison of non-nested models, providing a means to select the best model among a set of candidates.
   - **Sum of Squares:** It’s primarily a goodness-of-fit measure and does not facilitate model comparison directly.

### 3. **Information Loss:**
   - **BIC and AIC:** They estimate the loss of information when a particular model is used to represent the process generating the data. Lower values indicate less information loss.
   - **Sum of Squares:** SS is a measure of the discrepancy between the observed and predicted values but does not account for information loss.

### 4. **Asymptotic Consistency:**
   - **BIC:** BIC is consistent, meaning it will select the true model as the sample size approaches infinity (if the true model is among the candidates).
   - **AIC:** AIC is not always consistent, but it tends to be more efficient with smaller sample sizes.
   - **Sum of Squares:** SS does not have this property.

### 5. **Applicability:**
   - **BIC and AIC:** They are applicable in broader contexts and are used for model selection in various statistical and machine learning models.
   - **Sum of Squares:** Primarily used in the context of regression and analysis of variance.

### **Summary:**
BIC and AIC have advantages in model comparison, penalization of complexity, and estimation of information loss, making them more suitable for model selection than using the 'sum of squares'

### Using the code:
#### step by step guide

1. **Load data using function `load_data`**

    ```python
    x, df = load_data(wd=None, folder='data', filename='your_file.xlsx', dt=0.1, 
        stim_time=100, baseline=True, time=True)
    ```

   **Inputs:**
    - `wd`: working directory (if ignored then function assumes data is located within the current working directory)
    - `folder`: name of folder within main working directory that contains filename
    - `filename`: with extension (can be *.xlsx or *.csv file)
    - `dt`: If time=True then dt is ignored. If time=False, then uses dt to create time vector 
    - `stim_time`: must be relative to the start time if time column is not provided. Returned output removes all baselines
    - `baseline`: if True then uses all values are zeroed to the baseline (average of all given values before stim_time)
    - `time`: if True then takes first column as time. Stim_time must be the time in this column that stimulation occurs. If this column is present and starts at t = 200 ms and stim occurs 100 ms after this then stim_time = 300. If time is False then program will create a time vector based on number of samples and dt and will assume time starts at t = 0. In this example stim_time is relative to t[0] i.e. stim_time = 100 ms
   
   **Outputs:**
    - `x`: i.e. time
    - `df`: i.e. a dataframe that contains the processed traces: baseline is removed and traces are relative to baseline if baseline = True. If input file is negative going, output is positive (to simplify fitting routines). If the absolute maximum values of different traces have different signs then function returns an error (i.e. the assumption is all responses are either negative or positive going)


2. **Analyse one trace using function `FITproduct2`**

   ```python
   x, df = load_data(wd=None, folder='data', filename='your_file.xlsx', dt=0.1, 
       stim_time=100, baseline=True, time=True)
   
   fits, start, fit = FITproduct2(x=x, y=df.iloc[:, 0].values, criterion=None, 
       plot=True, initial_guess=None, start_method='LS', percent=80, dp=3,
       maxfev=100000, p0_algorithm='trf', cv=0.4, N=30, slow_rise_constraint=True, 
       rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve')
    
   fits, start, fit, BIC = FITproduct2(x, y, criterion='BIC', 
       plot=True, initial_guess=None, start_method='LS', percent=80, dp=3, 
       maxfev=100000, p0_algorithm='trf', cv=0.4, N=30, slow_rise_constraint=True, 
       rise=[0.2, 0.8], decay=[0.8, 0.2], kinetics='solve')
   ```   
   **Inputs:**
    - `x`: array represents time
    - `y`: array represents response obtained from df above; for nth column in df y = df.iloc[:, n-1].values
    - `criterion`: defaults to None, If criterion = 'BIC' or 'AIC' then returns Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC), respectively. In a similar manner to the sum of the squares of the residuals ('sum of squares'), lower values indicate a better fit. Has the advantage over 'sum of squares' that it can be used to compare fits that arise from models with different numbers of fitted parameters (e.g. one or sum of two product fits)
    - `plot`: if True (default) then returns a plot showing original response in grey with fits superimposed 
    - `initial_guess`: defaults to None. If None then function should calculate appropriate starting values; if provided, only useful if start_method='LS'. Use in form initial_guess=[70, 3, 7, 70, 6, 100] where values are estimates for [ $A_1$, $\tau_1$, $\tau_2$, $A_2$, $\tau_3$, $\tau_4$ ]. In general the solution is usually better if set to None.
    - `start_method`: can be set to 'LS' for least squares. This option can be further altered by changing p0_algorithm. Other options are 'DE' for differential evolution and None. If None then takes initial random starting values and does 'MLE' without the intermediate step to refine the starting values. The latter can be considered a pure MLE method whereas setting start_method to either 'LS' or 'DE' is a hybrid. Obviously, if do not want MLE, this function also provides the (start) values obtained for this fitting provided start_method is either 'LS' or 'DE'. 
    - `percent`: default set to 80%. If start_method='DE' or 'LS' + initial_guess = None then if percent = 80, function calculates 'rough' initial starting values based on the amplitude and the 20-80% rise and 80-20% decay times. These are then fed into a separate function to calculate 'good' starting values using 'LS' or "DE' curve fitting.
    - `dp`: number of decimal results are returned to; default is 3
    - `maxfev`: default is 100000; maximum number of iterations that the algorithm should run to try and find the parameter values that minimize the difference between the predicted and actual values (i.e., that best fits the data)
    - `p0_algorithm`: by default set to 'trf'. When using least_squares to get starting values (i.e. start_method='LS'), least_squares will use the Trust Region Reflective algorithm, which is a derivative-free algorithm designed for solving nonlinear least squares problems. This algorithm combines the trust region method with a reflective strategy to handle both bound and unbound constraints on the variables. Other options include 'lm' and 'dogbox'
    - `cv`: the cv or coefficient of variation is defined as the ratio of the standard deviation ($\sigma$) to the mean ($\mu$). The initial estimate for the starting values is based on response amplitude, 20-80% rise and 80-20% decay times. These values are used to create a lognormal distribution of starting values $\sigma = \sqrt{ \log(1 + (\text{cv}^2)) }$ and $\mu = \log(x) - 0.5\sigma^2$ where x is the parameter being randomized. A lognormal distribution was chosen because it ensures starting values cannot be negative. 
    - `N`: by default set to 30. N is the number of randomized starting values that are used to initialize the fits. if N > 1 then the function will compare solutions to determine the fit with minimised sum of squares. The greater N, then the higher the probability that the presented solution represents the best solution and not just a local minimum. With randomized starting values, the optimization algorithm may find different local minima in different runs. Setting N in the range 10 to 30 represents a good comprise between finding the best fit and function efficiency.
    - `slow_rise_contraint`:  default is True. If True then MLE rejects any solutions with underlying fits can comprise the fastest rise combined with the slowest decay and vice versa. If True then guarantees the 'fast' fittted response will have the faster rise AND decay and the slow component the slower rise AND decay
    - `rise`: The function is designed to return time constants related to exponential rise and decay (either as ${\tau}rise$ and ${\tau}decay$ or ${\tau}1$ and ${\tau}2$; nb ${\tau}decay = {\tau}2)$. Neither ${\tau}rise $ or ${\tau}1$ truly define rise time and both require careful interpretation. As an alternative, the function will also return the % rise and decay times. Default setting is $\text{rise} = [0.2, 0.8]$ which will determine the 20 - 80% rise time.
    - `decay`: Similarly to rise above, the default for this input is $\text{default} = [0.8, 0.2]$ which returns the 80 - 20 % decay time
    - `kinetics`: Default for kinetics is 'solve'. If using 'solve', the function solves an equation iteratively to determine rise, decay time constants;  area under the fit is calculated using the exact form of the equation given above. In practice, calculating rise and decay uses 'fsolve' from 'scipy.optimize' a numerical root-finding algorithm with estimated starting values. If set to 'fit', the function determines rise, decay and area from the underlying fits. If one fit is especially slow, 'solve' method will be more accurate. The 'fit' method is limited by the time axis (i.e. if response hasn't decayed sufficiently to 20% of peak then 80-20% decay cannot be determined and, instead' the time from 80% peak to the end is returned. 

   **Algorithms in `least_squares` method:** (i.e. start_method='LS')
    - **Trust Region Reflective ('trf')**
       - **Applicability:** Suitable for both bound and unbound problems. Especially powerful when the variables have constraints.
       - **Method:** Uses a trust-region algorithm. It can handle large-scale problems efficiently.
       - **Robustness:** It is quite robust and capable of handling sparse Jacobians, making it suitable for problems with a large number of variables.
    - **Levenberg-Marquardt ('lm')**
       - **Applicability:** Best used when there are no bounds on the variables. It’s more traditional and well-established.
       - **Method:** A combination of the gradient descent and Gauss-Newton methods. It doesn’t handle bound constraints inherently.
       - **Robustness:** It’s less robust compared to 'trf' when dealing with constraints but is powerful for smooth, unconstrained problems.
    - **Dogleg ('dogbox')**
       - **Applicability:** Suitable for smaller-scale problems with bound constraints.
       - **Method:** A trust-region algorithm that works efficiently when the number of observations is much greater than the number of variables.
       - **Robustness:** It’s versatile and can be more efficient than 'trf' for some specific small-scale problems.
       

3. **Examine fits for one trace using function `FITproduct2_widget`**
   ```python
   x, df = load_data(wd=None, folder='data', filename='your_file.xlsx', dt=0.1,
       stim_time=100, baseline=True, time=True)
   ``` 
   ```python   
   FITproduct2_widget(x=x, y=df.iloc[:, 0].values, criterion='SS', initial_guess=None,
       start_method='LS', percent=80, dp=3, maxfev=100000, p0_algorithm='trf', slow_rise_constraint=True)
   ```   
   **Inputs:**
    - function inputs are the same as function FITproduct2; only difference is criterion can be set to 'SS" which returns the sum of the squares (of the residuals), in addition to AIC or BIC


4. **Batch analysis using function `FITproduct2_batch`**
   ```python
   x, df = load_data(wd=None, folder='data', filename='your_file.xlsx', dt=0.1, 
       stim_time=100, baseline=True, time=True)
   ``` 
   ```python
   fits, start, fits2  =  FITproduct2_batch(x=x, df=df, criterion=None, 
       plot=True, initial_guess=None, start_method='LS', percent=80, dp=5,
       maxfev=100000, p0_algorithm='trf', slow_rise_constraint=True)
   ```  
   ```python
   fits, start, fits2, BIC =  FITproduct2_batch(x=x, df=df, criterion='BIC', 
       plot=True, initial_guess=None, start_method='LS', percent=80, dp=5,
       maxfev=100000, p0_algorithm='trf', slow_rise_constraint=True)        
   ``` 
   **Inputs:**
    - function inputs are the same as function FITproduct2 with one exception:
    - `df`: dataframe obtained from function load_data; replaces 'y' as input in FITproduct2

### `curve_fit` function in `scipy.optimize`

The `curve_fit` function located within `scipy.optimize` is adept at fitting both linear and non-linear functions to data sets. It employs non-linear least squares optimization, striving to uncover the best-fit parameters that minimize the sum of squared residuals. This involves comparing the predicted values of the function against the actual data points.

#### Fitting Non-linear Functions:
- When dealing with non-linear functions, `curve_fit` can still be utilized effectively.
- Crafting a function that treats non-linear squares in a linear fashion is a useful approach.

### `curve_fit`

- **`lm` :** *Levenberg-Marquardt algorithm*
   - A popular method for solving nonlinear least squares problems and is efficient for well-behaved problems.
   - It can handle both unconstrained and bounded problems.

- **`trf`(default) :** *Trust Region Reflective algorithm*
   - Uses a trust region approach combined with a reflection strategy to handle bounded constraints.
   - Suitable for problems with both bound constraints and general constraints.

- **`dogbox`:** *Dogleg algorithm*
   - Approximates the Gauss-Newton step within a trust region.
   - Suitable for unconstrained problems and can handle problems with singular or ill-conditioned Jacobian matrices.

### Differential Evolution (DE)

- DE is a population-based stochastic optimization algorithm commonly used for global optimization problems.
- Particularly useful when the objective function is non-linear, non-differentiable, or noisy.

**Steps:**
1. **Initialization:**
   - Create an initial population of candidate solutions represented as vectors in the search space.

2. **Mutation:**
   - Generate new trial vectors by perturbing the existing candidate solutions. 

3. **Crossover:**
   - Combine the mutant vectors with the target vectors to create trial vectors.

4. **Selection:**
   - Compare each trial vector with its corresponding target vector and select the better vector based on the objective function value.

5. **Replacement:**
   - Replace the old population with the selected trial vectors.

6. **Termination:**
   - Repeat steps 2-5 until a termination criterion is met, such as a maximum number of iterations or a target objective function value.

### Norm.logpdf function in SciPy

- Part of the `scipy.stats` module.
- Allows computation of the logarithm of the probability density function (PDF) of a normal (Gaussian) distribution.

### Explanation of norm.logpdf($y$, $y_{fit}$, $\sigma$) for MLE fits

The norm.logpdf(y, $y_{fit}$, $\sigma$) function computes the natural logarithm of the probability density function (PDF) of a normal distribution. 

- **$y$**:
  - These are the observed values, which could be a single value or a list/array of values.

- **$y_{fit}$**:
  - These are the model’s predicted or fitted values corresponding to the observed values in `y`.

- **$\sigma$**:
  - Represents the standard deviation, a measure of the dispersion of the residuals or errors in the data.

For each observed value $y_i$, the function calculates the log of the probability of observing that value, given a normal distribution with a mean at the corresponding predicted value $y_{fit_i}$ and a specified standard deviation sd. Mathematically, it can be represented as:


$$
\text{logpdf}_i = \log \left( \frac{1}{{\sigma} \sqrt{2\pi}} e^{ - \frac{(y_i - y_{fit_i})^2}{2 \cdot \sigma^2} } \right)
$$

log-likelihood is given by: 

$$
LL = \sum_{i=1}^{N} \text{logpdf}_i 
$$

Taking the sum of the logarithms of probabilities is mathematically equivalent to taking the logarithm of the product of the probabilities (see before). Since sums are computationally easier to manage than products, it is easier to use the log-likelihood versus likelihood.
    
**Practically most Maximum Likelihood Estimation is 'Minimum Negative Likelihood Estimation'**
    
Optimization algorithms are generally designed to minimize functions. So, instead of maximizing the log likelihood (LL), the routines actually minimizes the negative log likelihood (-LL or nLL). Mathematically, maximizing LL is the same as minimizing −LL, but minimization is standard practice in optimization theory and computational methods

# <center>Setting a seed</center>

Setting a seed in random number generation ensures that you get the same 'random' results every time you run the code. Essentially, it makes the randomness predictable and reproducible. Random number generators are based on algorithms that produce a sequence of numbers that seem random, but they are actually deterministic, meaning the sequence is fully determined by the initial conditions.
    By setting a seed, the sequence of random numbers generated by the function starts from the same point each time. This means that if you run the same code multiple times, it will produce the same output, ensuring reproducibility.
    
```python
np.random.seed(7)
```
The seed can be any number you like. You can always use the same number or change it every time.

**By running this code before every fitting analysis, you will ensure anyone can reproduce your results if provided will the original spreadsheet/csv/xlsx file**


# <center>Code guide</center>

### run this to import all necessary functions:
```python
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
# provided helper functions
from master_functions import *
```

### As an example, 4 responses with different characteristics were generated. The example is provided <span style="font-size: 80%;">(there is no need to repeat steps 1-8 detailed here)</span>:

1. **setting up:**
```python
dt = 0.05
x2 = np.arange(0, 100, dt) 
rms = 1
baseline = 25 # add a baseline period
x1 = np.arange(0, baseline, dt) 
```

2.  **alpha:**
```python
np.random.seed(42) # set a seed to make egs reproducible
params1 = [50, 10] 
y1 = alpha_alt(params1, x2) + np.random.normal(0,rms,len(x2))
bl1 = np.random.normal(0,rms,len(x1))
y1 = np.concatenate((bl1, y1))
```

3. **sum of 2 alphas:**
```python
params2 = [15, 5, 35, 10] 
y = alpha2(params, x) + np.random.normal(0,2,len(x))
y2 = alpha2_alt(params2, x2) + np.random.normal(0,rms,len(x2))
bl2 = np.random.normal(0,rms,len(x1))
y2 = np.concatenate((bl2, y2))
```
4. **product**:
```python
params3 = [50, 5, 20] 
y3 = product_alt(params3, x2) + np.random.normal(0,rms,len(x2))
bl3 = np.random.normal(0,rms,len(x1))
y3 = np.concatenate((bl3, y3))
```
5. **sum of 2 products:**
```python
params4 = [35, 5, 15, 15, 10, 30] 
y4 = product2_alt(params4, x2) + np.random.normal(0,rms,len(x2))
bl4 = np.random.normal(0,rms,len(x1))
y4 = np.concatenate((bl4, y4))
x = np.arange(0, 125, dt)
```
6. **create a dataframe:**
```python
df = pd.DataFrame({
    'x': x,
    'y1': y1,
    'y2': y2,
    'y3': y3,
    'y4': y4
})
```

7. **check if the directory exists, if not create it:**
```python
directory = "example data"
if not os.path.exists(directory):
    os.makedirs(directory)
```

8. **export the dataframe to a csv file with explicit comma delimiter:**
```python
df.to_csv(os.path.join(directory, 'eg_data.csv'), sep=',', index=False)
```


### Analysing provided example

1. **load example data:**
```python
x, df = load_data(folder='example data', filename='eg_data.csv', stim_time=25, baseline=True, time=True)
```

2. **choose a trace (in this case, the 4th in the dataframe, df):**
```python
y = df.iloc[:, 3].values
```
**nb** if data contains n traces then inputing ```y = df.iloc[:, 0].values``` retrieves the first trace and ```y = df.iloc[:, n-1].values ``` the last one

3. **fitting sum of 2 product functions to individual traces:**
```python
# set a seed
np.random.seed(7)
FITproduct2(x, y)
```

4. **batch fitting sum of 2 product functions to all traces:**
```python
# set a seed
np.random.seed(7)
results = FITproduct2_batch(x, y)
# to view the results:
results[0]
# also returns results as per the original method:
results[2]
# results[2] is an alternative form of the results[0]; results[2] can be converted to a (shortened) form of results[0] using:
product_conversion_df(results[2])
# nb. slight differences in areas are due to rounding errors
```
5. **using the widget to fit individual traces:**
```python
# set a seed
np.random.seed(7)
FITproduct2_widget(x, y)
```
