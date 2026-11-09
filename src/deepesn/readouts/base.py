'''
Base class of the readouts of a DeepESN

    y(t) = Wout [x(t); 1]

The readout is trained in closed form (ridge regression) on the states of the
model; each training method is a subclass of Readout, in its own file.

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


class _ReadoutType(type):
    '''Metaclass of the readouts: Readout(method, ...) builds the readout of the
    method given by name (Readout('iNE', 1e-6) is an iNEReadout(1e-6)).'''

    def __call__(cls, *args, **kwargs):
        if cls is Readout:
            if args:
                method, args = args[0], args[1:]
            else:
                method = kwargs.pop('method', 'NE')
            cls = Readout.get(method)
        return type.__call__(cls, *args, **kwargs)


class Readout(metaclass=_ReadoutType):
    '''
    Linear readout trained in closed form (ridge regression), base class.

        Readout(method, regularization=0.0)

    builds the readout of a method given by name: 'SVD', 'NE' (batch), 'iNE',
    'RLS' (online), or any other registered method (case insensitive;
    'NormalEquations' is accepted for 'NE'). The readout of each method is also
    available as a class: SVDReadout, NEReadout, iNEReadout, RLSReadout.

    Parameters
    ----------
    regularization : ridge regularization lambda (default of fit, update and train)

    Arrays: states (N, T) and targets (Ny, T), one column per time step, or lists
    of such arrays (concatenated along time), as in DeepESN 1.x. The features are
    the states plus a constant 1 (bias): Wout is (Ny, N + 1).

    Training:
        fit(states, targets)      from scratch (clears the matrices of the online methods)
        update(states, targets)   online update with new samples (online methods only)
        train(states, targets)    one block, as DeepESN.trainReadout (1.x): the batch
                                  methods solve on the block, the online methods add it

    New training methods: subclass Readout, set the class attributes method (the
    name, for Readout(method)), aliases (other accepted names, lower case) and
    online (True if the method supports update()), and implement

        _train(X, Y, lb)    training on a block of samples: X (N + 1, T) features
                            (bias in the last row), Y (Ny, T) targets, lb
                            regularization; sets self.Wout
        _reset_state()      (optional) clears the matrices kept between the calls

    A subclass with a name is registered when it is defined.
    '''

    #: name of the training method, used by Readout(method) (None: not registered)
    method = None
    #: other names accepted by Readout(method), lower case
    aliases = ()
    #: True if the method supports online updates (update())
    online = False

    _registry = {}          # lower-case name -> class

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get('method'):
            for name in (cls.method,) + tuple(cls.aliases):
                Readout._registry[name.lower()] = cls

    @staticmethod
    def get(method):
        '''Class of the readout of a method given by name (or the class itself).'''
        if isinstance(method, type) and issubclass(method, Readout):
            return method
        cls = Readout._registry.get(str(method).lower())
        if cls is None:
            raise ValueError('unknown readout method %r: use one of %s'
                             % (method, ', '.join(Readout.methods())))
        return cls

    @staticmethod
    def methods():
        '''Names of the registered training methods.'''
        return tuple(dict.fromkeys(c.method for c in Readout._registry.values()))

    def __init__(self, regularization=0.0):
        self.regularization = regularization
        self.reset()

    def __repr__(self):
        return '%s(regularization=%r)' % (type(self).__name__, self.regularization)

    @property
    def trainMethod(self):
        # DeepESN 1.x name of the method
        return self.method

    @trainMethod.setter
    def trainMethod(self, value):
        # DeepESN 1.x: change the training method of the readout
        self.__class__ = Readout.get(value)
        self._reset_state()

    # ------------------------------------------------------------------
    # output weights, with cached views used by predict_step
    # ------------------------------------------------------------------
    @property
    def Wout(self):
        return self._Wout

    @Wout.setter
    def Wout(self, value):
        self._Wout = value
        if value is not None:
            self._Wout_x = value[:, :-1]
            self._Wout_b = value[:, -1]

    def reset(self):
        '''Clear the output weights and the matrices of the online methods.'''
        self.Wout = None
        self._reset_state()

    def _reset_state(self):
        pass

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(self, states, targets, regularization=None):
        '''Train from scratch on (states, targets) (clears the online matrices).'''
        self.reset()
        self.train(states, targets, regularization)
        return self

    def update(self, states, targets, regularization=None):
        '''Online update with new samples (online methods), e.g. one column.'''
        if not self.online:
            names = [m for m in Readout.methods() if Readout.get(m).online]
            raise ValueError('update() requires an online readout (%s), not %r'
                             % (', '.join(names), self.method))
        self.train(states, targets, regularization)
        return self

    def train(self, states, targets, regularization=None):
        '''
        Train on a block of samples, as DeepESN.trainReadout: the batch methods
        solve the ridge regression on the block (overwriting Wout), the online
        methods add the block to the data seen so far. The first call of the
        online methods (or after reset) initializes their matrices.
        '''
        lb = self.regularization if regularization is None else regularization
        X, Y = self._features(states, targets)
        self._train(X, Y, lb)
        return self

    @staticmethod
    def _features(states, targets):
        # concatenate the blocks and add the bias row: X (N + 1, T), Y (Ny, T)
        if not isinstance(states, (list, tuple)):
            states = [states]
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        trainStates = np.concatenate(states, 1)
        trainTargets = np.concatenate(targets, 1)
        X = np.ones((trainStates.shape[0] + 1, trainStates.shape[1]))
        X[:-1, :] = trainStates
        return X, trainTargets

    def _train(self, X, Y, lb):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def predict(self, states):
        '''Outputs (Ny, T) for states (N, T) or a list of them.'''
        if isinstance(states, (list, tuple)):
            states = np.concatenate(states, 1)
        return self._Wout[:, 0:-1].dot(states) + np.expand_dims(self._Wout[:, -1], 1)

    def predict_step(self, x):
        '''Output (Ny,) for one state x (N,).'''
        return self._Wout_x.dot(x) + self._Wout_b
