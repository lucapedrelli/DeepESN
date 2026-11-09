'''
Memory capacity task

Reference paper:
H. Jaeger, "Short term memory in echo state networks", GMD Report 152,
German National Research Center for Information Technology, 2002

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


class MemoryCapacity(Task):
    '''
    Memory capacity (MC): from the state x(t), the readout reconstructs the
    delayed inputs u(t-k), k = 1, ..., max_delay (one output per delay), and

        MC = sum_k r_k^2,

    with r_k the correlation coefficient between the output for the delay k
    and u(t-k) on the evaluation segment (from its k-th step on, i.e. the
    first k-1 steps are discarded). The input is a scalar series, typically
    i.i.d. (see random_input).

    The model is driven through the training and the evaluation segments
    (teacher forcing, from the zero state); the readout is trained on the
    training segment after the washout and, if online (iNE, RLS), updated on
    the evaluation segment after each prediction (predict-then-update).
    online_update=False keeps an online readout fixed; None (default) updates
    only online readouts.

    Results: 'mc', 'mc_per_delay' (max_delay,), 'score' (= mc; greater is
    better), 'pred' and 'true' (T_eval, max_delay).
    '''

    greater_is_better = True

    def __init__(self, train, validation=None, test=None, washout=0, max_delay=200,
                 online_update=None):
        super(MemoryCapacity, self).__init__(train, validation, test, washout, metrics=())
        for x in (self.train, self.validation, self.test):
            if x is not None and x.shape[1] != 1:
                raise ValueError('the memory capacity needs a scalar input series')
        self.max_delay = int(max_delay)
        self.online_update = online_update

    def _repr_params(self):
        return ', max_delay=%d' % self.max_delay

    @staticmethod
    def random_input(n_train, n_validation=0, n_test=0, amplitude=0.8, seed=None, skip=0):
        '''
        i.i.d. input uniform in (-amplitude, amplitude), drawn by
        np.random.default_rng(seed) after discarding the first skip values:
        (train, validation, test) segments, arrays (T, 1).
        '''
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-amplitude, amplitude, size=(n_train + n_validation + n_test + skip, 1))
        sig = raw[skip:]
        return sig[:n_train], sig[n_train:n_train + n_validation], sig[n_train + n_validation:]

    @staticmethod
    def capacity(pred, true):
        '''Squared correlation r_k^2 per delay (column k-1: delay k, first k-1 steps discarded).'''
        K = true.shape[1]
        mc_per_delay = np.zeros(K)
        for k in range(K):
            p_k = pred[k:, k]
            q_k = true[k:, k]
            if len(p_k) < 2 or np.std(p_k) < 1e-12 or np.std(q_k) < 1e-12:
                continue
            rc = np.corrcoef(p_k, q_k)[0, 1]
            mc_per_delay[k] = float(rc**2) if np.isfinite(rc) else 0.0
        return mc_per_delay

    def run(self, model, train, evaluation):
        train, ev = as_series(train), as_series(evaluation)
        if train.shape[1] != 1 or ev.shape[1] != 1:
            raise ValueError('the memory capacity needs a scalar input series')
        K, w = self.max_delay, self.washout
        U = np.vstack([train, ev])
        T_full, T_tr, T_ev = U.shape[0], train.shape[0], ev.shape[0]
        if T_tr - w < 1:
            raise ValueError('the washout leaves no training samples')

        # targets: column k-1 is the input delayed by k steps (zero before the start)
        target = np.zeros((T_full, K))
        for k in range(1, K + 1):
            if k < T_full:
                target[k:, k - 1] = U[:T_full - k, 0]

        states = model.compute_states(U.T)
        readout = model.readout
        readout.fit([states[:, w:T_tr]], [target[w:T_tr].T])

        S_ev = states[:, T_tr:]
        Y_ev = target[T_tr:]
        update = self._online_update(model)
        pred = np.zeros((T_ev, K))
        for t in range(T_ev):
            a_t = S_ev[:, t:t+1]
            pred[t] = readout.predict(a_t).ravel()
            if update:
                readout.train([a_t], [Y_ev[t:t+1].T])

        mc_per_delay = self.capacity(pred, Y_ev)
        mc = float(mc_per_delay.sum())
        return {'mc': mc, 'mc_per_delay': mc_per_delay, 'score': mc, 'pred': pred, 'true': Y_ev}
