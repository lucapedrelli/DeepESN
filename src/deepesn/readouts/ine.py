'''
iNE readout: incremental normal equations (online)

    P <- P + X X^T,    Q <- Q + Y X^T,    Wout = Q (P + lambda I)^-1

over all the samples seen so far (P = 0, Q = 0 at the start), solved after
each block of samples (one sample per step in online generative RC).

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


class iNEReadout(Readout):
    '''
    Incremental normal equations: online readout that accumulates the
    correlation matrices P = sum X X^T and Q = sum Y X^T of all the samples
    seen so far and, after each block, solves the ridge regression on them:
    at every step Wout is the ridge solution over all the data (the same as a
    batch training on them). Cost per update: one linear system of the size of
    the features.

    Parameters
    ----------
    regularization : ridge regularization lambda (added to P when solving)
    '''

    method = 'iNE'
    online = True

    def _reset_state(self):
        self._P = None      # sum of X X^T (without regularization)
        self._Q = None      # sum of Y X^T

    def _train(self, X, Y, lb):
        nf = X.shape[0]
        if self._P is None:
            self._P = np.zeros((nf, nf))
            self._Q = np.zeros((Y.shape[0], nf))
        self._P += X.dot(X.T)
        self._Q += Y.dot(X.T)
        self.Wout = np.linalg.solve(self._P + np.eye(nf) * lb, self._Q.T).T
