
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 1: Linear Regression

# Zach Goodenow<br>
# zachgood@rams.colostate.edu<br>
# CS 445<br>
# 1/30/18<br>

# ## Overview

# Describe the objective of this assignment, and very briefly how you accomplish it.  Say things like "linear model", "samples of inputs and known desired outputs" and "minimize the sum of squared errors". DELETE THIS TEXT AND INSERT YOUR OWN.

# ## Method

# Define in code cells the following functions as discussed in class.  Your functions' arguments and return types must be as shown here.
# 
#   * ```model = train(X, T)```
#   * ```predict = use(model, X)```
#   * ```error = rmse(predict, T)```
#   
# Let ```X``` be a two-dimensional matrix (```np.array```) with each row containing one data sample, and ```T``` be a two-dimensional matrix of one column containing the target values for each sample in ```X```.  So, ```X.shape[0]``` is equal to ```T.shape[0]```.   
# 
# Function ```train``` must standardize the input data in ```X``` and return a dictionary with  keys named ```means```, ```stds```, and ```w```.  
# 
# Function ```use``` must also standardize its input data X by using the means and standard deviations in the dictionary returned by ```train```.
# 
# Function ```rmse``` returns the square root of the mean of the squared error between ```predict``` and ```T```.
# 
# Also implement the function
# 
#    * ```model = trainSGD(X, T, learningRate, numberOfIterations)```
# 
# which performs the incremental training process described in class as stochastic gradient descent (SGC).  The result of this function is a dictionary with the same keys as the dictionary returned by the above ```train``` function.

# In this section, ilatex math formulas defining the formula that is being minimized, and the matrix calculation for finding the weights. 
# 
# In this section, include all necessary imports and the function definitions. Also include some math formulas using latex syntax that define the formula being minimized and the calculation of the weights using a matrix equation.  You do not need to include the math formulas showing the derivations.

# In[93]:

def standardize(X):
    return (X - X.mean(0)) / X.std(0)


def train(X, T):
    # Check that X.shape[0] is equal to T.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError('X and T have different shapes. X = {} & T = {}'.format(X.shape[0], T.shape[0]))

    # Standardize and add a column of 1s
    Xs1 = np.insert(standardize(X), 0, 1, 1)

    return {'means': X.mean(0), 'stds': X.std(0), 'w': np.linalg.lstsq(Xs1.T @ Xs1, Xs1.T @ T)[0]}


def use(model, X):
    return np.insert(standardize(X), 0, 1, 1) @ model.get('w')


def rmse(predict, T):
    return np.sqrt(((predict - T) ** 2).mean(0))[0]


def trainSGD(X, T, learningRate, numberOfIterations):
    # Check that X.shape[0] is equal to T.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError('X and T have different shapes. X = {} & T = {}'.format(X.shape[0], T.shape[0]))

    # Add column of 1s to front
    # X1 = np.insert(X, 0, 1, axis=1)

    # Standardize and add a column of 1s
    Xs1 = np.insert(standardize(X), 0, 1, 1)

    w = np.zeros((Xs1.shape[1], T.shape[1]))

    for iter in range(numberOfIterations):
        for n in range(Xs1.shape[0]):
            predicted = Xs1[n:n + 1, :] @ w  # n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
            # Update w using derivative of error for nth sample
            w += learningRate * Xs1[n:n + 1, :].T * (T[n:n + 1, :] - predicted)

    return {'means': X.mean(0), 'stds': X.std(0), 'w': w}


# ## Examples

# In[13]:

# from A1mysolution import *


# In[14]:

import numpy as np

X = np.arange(10).reshape((5,2))
T = X[:,0:1] + 2 * X[:,1:2] + np.random.uniform(-1, 1,(5, 1))
print('Inputs')
print(X)
print('Targets')
print(T)


# In[15]:

model = train(X, T)
model


# In[16]:

predicted = use(model, X)
predicted


# In[17]:

rmse(predicted, T)


# In[19]:

modelSGD = trainSGD(X, T, 0.01, 100)
modelSGD


# In[22]:

predicted = use(modelSGD, X)
predicted


# In[23]:

rmse(predicted, T)


# ## Data

# Download ```energydata_complete.csv``` from the [Appliances energy prediction Data Set ](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) at the UCI Machine Learning Repository. Ignore the first column (date and time), use the next two columns as target variables, and use all but the last two columns (named rv1 and rv2) as input variables. 
# 
# In this section include a summary of this data, including the number of samples, the number and kinds of input variables, and the number and kinds of target variables.  Also mention who recorded the data and how.  Some of this information can be found in the paper that is linked to at the UCI site for this data set.  Also show some plots of target variables versus some of the input variables to investigate whether or not linear relationships might exist.  Discuss your observations of these plots.

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Here, I am loading the data in using the *read_csv* function from pandas to preview the first 5 rows.  This results in 19735 rows and 29 columns.

# In[22]:

edat = pd.read_csv('energydata_complete.csv')
print(edat.shape)
edat[:5]


# In[61]:

edat.plot(kind='box', subplots=True, layout=(6,5), sharex=False, sharey=False, figsize=(10, 10))
plt.tight_layout()
plt.show()


# Since we want to ignore the first column and last 2 columns, I pass an array with values 1-26 to *usecols*.  I pass None to *header* so that the array will include the column names.  I do this so I can make an array for the names.  This matrix should have an additional row and 3 fewer columns than the previous cell.  The first row of the np array will be the names of each variable.  I slice the data into target names, target values, X names, and X values. 

# In[26]:

edata = np.array(pd.read_csv('energydata_complete.csv', usecols = range(1, 27), header = None))
edata.shape


# In[68]:

names_T = edata[0, :2]
edata_T = edata[1:, :2]
names_X = edata[0, 2:]
edata_X = edata[1:, 2:]
edata_T.shape, edata_X.shape


# In[86]:

# Single plot
plt.figure(figsize=(10,10))
plt.plot(edata_X[:, 20], edata_T[:, :1], '.', alpha=0.5)
plt.ylabel(names_T[0])
plt.xlabel(names_X[20])


# In[85]:

plt.figure(figsize=(10, 15))
for col in range(edata_X.shape[1]):
    plt.subplot(6,4, col+1)
    plt.plot(edata_X[:, col], edata_T[:, :1], '.', alpha=0.05);
    plt.ylabel(names_T[0])
    plt.xlabel(names_X[col])
plt.tight_layout()


# In[87]:

plt.figure(figsize=(10, 15))
for col in range(edata_X.shape[1]):
    plt.subplot(6,4, col+1)
    plt.plot(edata_X[:, col], edata_T[:, 1:], '.', alpha=0.05);
    plt.ylabel(names_T[1])
    plt.xlabel(names_X[col])
plt.tight_layout()


# ## Results

# Apply your functions to the data.  Compare the error you get as a result of both training functions.  Experiment with different learning rates for ```trainSGD``` and discuss the errors.
# 
# Make some plots of the predicted energy uses and the actual energy uses versus the sample index.  Also plot predicted energy use versus actual energy use.  Show the above plots for the appliances energy use and repeat them for the lights energy use. Discuss your observations of each graph.
# 
# Show the values of the resulting weights and discuss which ones might be least relevant for fitting your linear model.  Remove them, fit the linear model again, plot the results, and discuss what you see.

# ## Grading
# 
# Your notebook will be run and graded automatically.  Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A1grader.tar) and extract `A1grader.py` from it. Run the code in the following cell (after deleting the one containing A1mysolution) to demonstrate an example grading session.  You should see a perfect execution score of 70/70 if your functions are defined correctly. The remaining 30 points will be based on the results you obtain from the energy data and on your discussions.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook.  It will include additional tests.  You need not include code to test that the values passed in to your functions are the correct form.  

# In[94]:

get_ipython().magic('run -i "A1grader.py"')


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/41327).
# 
# Grading will be based on 
# 
#   * correct behavior of the required functions listed above,
#   * easy to understand plots in your notebook,
#   * readability of the notebook,
#   * effort in making interesting observations, and in formatting your notebook.

# ## Extra Credit

# Download a second data set and repeat all of the steps of this assignment on that data set.
