'''
NE readout: ridge regression by normal equations (batch)

    primal (features <= time steps):  Wout = Y X^T (X X^T + lambda I)^-1
    dual   (features >  time steps):  Wout = Y (X^T X + lambda I)^-1 X^T

The two forms give the same solution; the dual one solves a system of the size
of the time steps instead of the features.

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


class NEReadout(Readout):
    '''
    Ridge regression by normal equations: batch, fast (the system has the size
    of the features, or of the time steps when they are fewer: dual form). Each training (fit, train)
    solves on the given block and overwrites Wout: offline readout, fixed
    after the training (no update()). Also Readout('NormalEquations').

    Parameters
    ----------
    regularization : ridge regularization lambda
    '''

    method = 'NE'
    aliases = ('normalequations', 'normal_equations')

    def _train(self, X, Y, lb):
        N, T = X.shape
        if N > T:                       # dual: system T x T
            K = X.T.dot(X) + np.eye(T) * lb
            alpha = np.linalg.solve(K, Y.T)
            self.Wout = alpha.T.dot(X.T)
        else:                           # primal: system N x N (DeepESN 1.x)
            B = Y.dot(X.T)
            A = X.dot(X.T)
            self.Wout = np.linalg.solve(A + np.eye(N) * lb, B.T).T
