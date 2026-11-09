'''
SVD readout: ridge regression by singular value decomposition (batch)

    X = U diag(s) V^T,    Wout = Y V diag(s / (s^2 + lambda)) U^T

X: features (states and bias, one column per time step), Y: targets.

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

from .base import Readout


class SVDReadout(Readout):
    '''
    Ridge regression by singular value decomposition: batch, accurate (also
    with ill-conditioned states), slower than the normal equations. Each
    training (fit, train) solves on the given block and overwrites Wout.

    Parameters
    ----------
    regularization : ridge regularization lambda
    '''

    method = 'SVD'

    def _train(self, X, Y, lb):
        U, s, V = np.linalg.svd(X, full_matrices=False)
        s = s / (s**2 + lb)
        self.Wout = Y.dot(np.multiply(V.T, np.expand_dims(s, 0)).dot(U.T))
