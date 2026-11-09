'''
Metrics functions

----

This file is a part of the Deep Echo State Network (DeepESN) Python Library

Luca Pedrelli
luca.pedrelli@santannapisa.it
lucapedrelli@gmail.com

The Telecommunications, Computer Engineering, and Photonics Institute (TeCIP)
Scuola Superiore Sant'Anna in Pisa
ICT-COISP (ICT for Complex Industrial Systems and Processes)
https://www.santannapisa.it/it/istituto/tecip

----
'''

import numpy as np

# Mean Absolute Percentage Error
def MAPE(X, Y): 
    return np.mean(np.abs((Y - X) / Y))

# Mean Squared Error
def MSE(X,Y):
    return  np.mean((X-Y)**2)

# Normalized Mean Squared Error
def NRMSE(X,Y):
    return  (np.mean((X-Y)**2) / (np.std(Y)**2))**0.5

# Mean Absolute Error
def MAE(X,Y):
    return  np.mean(np.abs(X-Y))
