'''
RLS readout: recursive least squares (online)

For each new sample (x, y), with V = (P + lambda I)^-1 (V = I / lambda at the start):

    k = V x / (1 + x^T V x),    Wout <- Wout + (y - Wout x) k^T,    V <- V - k (V x)^T

(Sherman-Morrison rank-1 update of the inverse).

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


class RLSReadout(Readout):
    '''
    Recursive least squares: online readout that updates the inverse
    V = (sum X X^T + lambda I)^-1 and Wout with one rank-1 (Sherman-Morrison)
    step per sample: at every step Wout is the ridge solution over all the data
    seen so far, without solving linear systems (cost per sample quadratic in
    the size of the features).

    Parameters
    ----------
    regularization : ridge regularization lambda (V = I / lambda at the start)
    '''

    method = 'RLS'
    online = True

    def _reset_state(self):
        self._V = None      # inverse of (sum X X^T + lambda I)

    def _train(self, X, Y, lb):
        if self._V is None:
            self._V = np.eye(X.shape[0]) / lb
            self.Wout = np.zeros((Y.shape[0], X.shape[0]))
        V = self._V
        W = self.Wout
        for t in range(X.shape[1]):
            x = X[:, t]
            y = Y[:, t]
            Vx = V.dot(x)
            k = Vx / (1.0 + x.dot(Vx))
            e = y - W.dot(x)
            W = W + np.outer(e, k)
            V = V - np.outer(k, Vx)
        self._V = V
        self.Wout = W
