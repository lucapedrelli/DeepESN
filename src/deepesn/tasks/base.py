'''
Base class of the tasks

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

from ..metrics import MSE, MAE, NRMSE, MAPE

METRICS = {'mse': MSE, 'mae': MAE, 'nrmse': NRMSE, 'mape': MAPE}


def as_series(x):
    '''Time series as a (T, F) float array (time-major).'''
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError('a time series must be an array (T,) or (T, F), got shape %s' % (x.shape,))
    return x


class Task(object):
    '''
    Base class of the tasks on a time series.

    A task holds a (multivariate) time series split into training, validation
    and test segments, arrays (T, F) (time-major; (T,) for F = 1), and
    implements an evaluation protocol in run(model, train, evaluation):

    * training: the model is driven by the training segment from the zero
      state (teacher forcing) and the readout is trained from scratch, after
      the washout, with the regularization of model.readout;
    * evaluation on the evaluation segment, which follows the training
      segment (protocol of the subclass).

    evaluate(model, 'validation') runs the protocol on (training, validation)
    and evaluate(model, 'test') on (training + validation, test): for the test
    the readout is trained on training and validation together. Each run
    restarts the model from the zero state and the readout from scratch.

    Results: dict with the metrics, 'score' (first metric; greater_is_better
    tells its direction), 'pred' and 'true'. Metrics are names in METRICS or
    functions f(pred, true).
    '''

    #: True if a greater score is better (e.g. memory capacity), False for errors
    greater_is_better = False

    def __init__(self, train, validation=None, test=None, washout=0, metrics=('mse', 'mae')):
        self.train = as_series(train)
        self.validation = None if validation is None else as_series(validation)
        self.test = None if test is None else as_series(test)
        self.washout = int(washout)
        if isinstance(metrics, str) or callable(metrics):
            metrics = (metrics,)
        self.metrics = [(m, METRICS[m.lower()]) if isinstance(m, str) else (m.__name__, m) for m in metrics]

    def __repr__(self):
        sizes = [len(x) for x in (self.train, self.validation, self.test) if x is not None]
        return '%s(%s, washout=%d%s)' % (type(self).__name__, '/'.join(map(str, sizes)),
                                         self.washout, self._repr_params())

    def _repr_params(self):
        return ''

    def evaluate(self, model, split='validation'):
        '''Run the protocol on the validation ('validation') or test ('test') split.'''
        if split == 'validation':
            if self.validation is None:
                raise ValueError('the task has no validation segment')
            return self.run(model, self.train, self.validation)
        if split == 'test':
            if self.test is None:
                raise ValueError('the task has no test segment')
            train = self.train if self.validation is None else np.vstack([self.train, self.validation])
            return self.run(model, train, self.test)
        raise ValueError("split must be 'validation' or 'test', not %r" % (split,))

    def run(self, model, train, evaluation):
        '''Protocol on a training segment followed by an evaluation segment.'''
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _fit_readout(self, model, states, train):
        # one-step-ahead readout, trained on the pairs (x(t), u(t+1)), t = washout, ..., T_train - 2
        w = self.washout
        if train.shape[0] - w < 2:
            raise ValueError('the washout leaves no training samples')
        model.readout.fit([states[:, w:train.shape[0] - 1]], [train[w + 1:].T])

    def _online_update(self, model):
        update = getattr(self, 'online_update', None)
        return model.readout.online if update is None else bool(update)

    def _result(self, pred, true, **extra):
        out = {name: float(f(pred, true)) for name, f in self.metrics}
        out['score'] = out[self.metrics[0][0]]
        out['pred'] = pred
        out['true'] = true
        out.update(extra)
        return out
