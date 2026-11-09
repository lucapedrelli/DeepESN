'''
Next-step prediction task

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

from .base import Task, as_series


class NextStepPrediction(Task):
    '''
    Next-step prediction: from the state x(t), the readout predicts the next
    value of the series u(t+1) (univariate or multivariate).

    The model is driven through the training and the evaluation segments
    (teacher forcing, from the zero state); the readout is trained on the
    pairs (x(t), u(t+1)) of the training segment after the washout and
    predicts every step of the evaluation segment from the previous state.
    If online (iNE, RLS), the readout is updated after each prediction with
    the true value (predict-then-update). online_update=False keeps an online
    readout fixed; None (default) updates only online readouts.

    Results: the metrics (default 'mse', 'mae'), 'score' (first metric),
    'pred' and 'true' (T_eval, F).
    '''

    def __init__(self, train, validation=None, test=None, washout=0, metrics=('mse', 'mae'),
                 online_update=None):
        super(NextStepPrediction, self).__init__(train, validation, test, washout, metrics)
        self.online_update = online_update

    @staticmethod
    def mackey_glass(n_train, n_validation=0, n_test=0, tau=17, beta=0.2, gamma=0.1, n=10,
                     dt=0.1, sampling=1.0, x0=1.2, skip=1000, seed=None):
        '''
        Mackey-Glass time series, dx/dt = beta x(t-tau) / (1 + x(t-tau)^n) - gamma x(t),
        integrated by the Euler method with step dt from the constant history x0
        (perturbed by uniform noise in (-0.1, 0.1) drawn by np.random.default_rng(seed)
        if seed is not None), sampled every sampling time units after discarding the
        first skip samples and squashed as tanh(x - 1): (train, validation, test)
        segments, arrays (T, 1).
        '''
        every = int(round(sampling / dt))
        lag = int(round(tau / dt))
        T = n_train + n_validation + n_test + skip
        steps = T * every
        x = np.empty(lag + steps + 1)
        x[:lag + 1] = x0
        if seed is not None:
            x[:lag + 1] += np.random.default_rng(seed).uniform(-0.1, 0.1, lag + 1)
        for i in range(lag, lag + steps):
            xd = x[i - lag]
            x[i + 1] = x[i] + dt * (beta * xd / (1.0 + xd**n) - gamma * x[i])
        sig = np.tanh(x[lag + every::every][skip:] - 1.0)[:, None]
        return sig[:n_train], sig[n_train:n_train + n_validation], sig[n_train + n_validation:]

    def run(self, model, train, evaluation):
        train, ev = as_series(train), as_series(evaluation)
        if train.shape[1] != ev.shape[1]:
            raise ValueError('training and evaluation segments with different numbers of features')
        U = np.vstack([train, ev])
        T_tr, T_ev = train.shape[0], ev.shape[0]

        states = model.compute_states(U.T)
        self._fit_readout(model, states, train)

        readout = model.readout
        update = self._online_update(model)
        pred = np.zeros_like(ev)
        for t in range(T_ev):
            x_t = states[:, T_tr - 1 + t:T_tr + t]          # state before the step to predict
            pred[t] = readout.predict(x_t).ravel()
            if update:
                readout.train([x_t], [ev[t:t+1].T])
        return self._result(pred, ev)
