'''
Ridge readout: ridge regression by scikit-learn (batch)

    Wout = argmin ||Y - Wout X||^2 + lambda ||Wout||^2

X: features (states and bias, one column per time step), Y: targets. Requires
scikit-learn (pip install scikit-learn), which handles the primal/dual form.

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

from .base import Readout


class RidgeReadout(Readout):
    '''
    Ridge regression by scikit-learn (sklearn.linear_model.Ridge, without
    intercept: the bias is a feature): batch. Each training (fit, train)
    solves on the given block and overwrites Wout.

    Parameters
    ----------
    regularization : ridge regularization lambda (alpha of sklearn)
    '''

    method = 'Ridge'

    def _train(self, X, Y, lb):
        from sklearn.linear_model import Ridge
        clf = Ridge(alpha=lb, fit_intercept=False)
        clf.fit(X.T, Y.T)
        self.Wout = clf.coef_.reshape(Y.shape[0], X.shape[0])
