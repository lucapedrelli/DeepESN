'''
Metrics functions

----

This file is a part of the DeepESN Python Library (DeepESNpy)

Luca Pedrelli
luca.pedrelli@di.unipi.it
lucapedrelli@gmail.com

Department of Computer Science - University of Pisa (Italy)
Computational Intelligence & Machine Learning (CIML) Group
http://www.di.unipi.it/groups/ciml/

----
'''

import numpy as np

# Accuracy function used to evaluate the prediction in polyphonic music tasks: true positive/(true positive + false positive + false negative)
def computeMusicAccuracy(threshold,X,Y):
    Y = np.concatenate(Y,1)
    Nsys = np.sum(X>threshold, axis=0)
    Nref = np.sum(Y>threshold, axis=0)
    Ncorr = np.sum((X>threshold) * (Y>threshold), axis=0)
        
    TP = np.sum(Ncorr)
    FP = np.sum(Nsys-Ncorr)
    FN = np.sum(Nref-Ncorr)
    ACCURACY = TP/float(TP + FP + FN)
    return ACCURACY

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
