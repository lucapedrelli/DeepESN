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

# Mean Squared Error
def MSE(threshold,X,Y):
    Y = np.concatenate(Y,1)
    return  np.mean((X-Y)**2)
