# <center>Code guide</center>

Follow instructions to open Jupyter Notebook
(assumes correct installation of Anaconda and fitting' environment; see README)

### To run:

Either:
**A. using Terminal:**

1. **Open Terminal**

2. **Navigate back to the your_directory and activate the 'fitting' environment**
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

**B. using Anaconda Navigator:**

1. **Open Anaconda Navigator**

2. **Select 'Environments' on left menu**

3. **Select 'fitting'**

4. **Press play and choose 'open with Jupyter Notebook'**

5. **Navigate to your_directory in Jupyter Notebook**


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
results = FITproduct2_batch(x, df)
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

### Analysing real data
**Example1**
1. **load data stored as csv file:**
```python
x, df = load_data(folder='example data', filename='ChAT-Cre X D2EGFP(iSPN)-voltage clamp.csv', stim_time=322.5, baseline=True, time=True)
```
2. **analyse individual traces:**
```python
y = df.iloc[:, 0].values
# set a seed
np.random.seed(7)
FITproduct2(x, y)
```

3. **analyse individual traces using widget:**
```python
# set a seed
y = df.iloc[:, 0].values
np.random.seed(7)
FITproduct2_widget(x, y)
```

4. **batch fitting sum of 2 product functions to all traces:**
```python
# set a seed
np.random.seed(7)
results = FITproduct2_batch(x, df)
# to view the results:
results[0]
# also returns results as per the original method:
results[2]
# results[2] is an alternative form of the results[0]; results[2] can be converted to a (shortened) form of results[0] using:
product_conversion_df(results[2])
# nb. slight differences in areas are due to rounding errors
```

**Example2**
1. **load data stored as xlsx file:**
```python
x, df = load_data(folder='example data', filename='data4_TPR.xlsx', stim_time=1349, baseline=True, time=True)
```

2. **batch fitting sum of 2 product functions to all traces:**
```python
# set a seed
np.random.seed(7)
results = FITproduct2_batch(x, df)
# to view the results:
results[0]
```

**Example3**
1. **load data stored as xlsx file:**
```python
x, df = load_data(folder='example data', filename='data5_TPR.xlsx', stim_time=1349, baseline=True, time=True)
```

2. **batch fitting sum of 2 product functions to all traces:**
```python
# set a seed
np.random.seed(7)
results = FITproduct2_batch(x, df)
# to view the results:
results[0]
```

