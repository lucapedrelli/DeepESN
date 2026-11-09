'''
Base class of the recurrent layers of a DeepESN

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

import random

import numpy as np


class RandomGenerators(object):
    '''
    Random number generators used to initialize the layers.

    seed=None uses the global generators (np.random and random), as in
    DeepESN 1.x: the initialization is then controlled by np.random.seed()
    and random.seed(). An int seed creates private generators
    (np.random.RandomState and random.Random) that draw exactly the same
    numbers as the global ones seeded with the same value, without changing
    the global state. A np.random.RandomState can also be given.
    '''

    def __init__(self, seed=None):
        if isinstance(seed, RandomGenerators):
            self.np, self.py = seed.np, seed.py
        elif seed is None:
            self.np, self.py = np.random, random
        elif isinstance(seed, np.random.RandomState):
            self.np, self.py = seed, random
        else:
            self.np, self.py = np.random.RandomState(seed), random.Random(seed)


class Layer(object):
    '''
    Base class of a recurrent (reservoir) layer.

    A layer owns its random weights (recurrent matrix W, input matrix Win
    whose last column is the bias), its state transition function and its
    auxiliary memory, if any. States are computed:

    * in batch, over a whole input sequence: compute_states(inputs) (as in
      DeepESN 1.x, e.g. teacher forcing on a known sequence);
    * online, one time step at a time: step(u), from the current context of
      the layer (last state and, for layers with memory, the past states).

    set_history(states) aligns the online context with the end of a sequence
    computed in batch, so that step() continues it; get_context() and
    set_context(context) save and restore the context (e.g. around a
    generative rollout); reset() clears it.

    Shapes: inputs (input_dim, T) and states (units, T), one column per time
    step as in DeepESN 1.x; step() takes and returns 1-D vectors. The arrays
    returned by step() and state must not be modified in place.

    Subclasses implement _rescale() (scaling of the recurrent matrix),
    compute_states() and step(), and optionally _init_extra() (additional
    random parameters) and the context methods (additional memory).
    '''

    #: True if the layer supports deep intrinsic plasticity pre-training
    supports_ip = False

    def __init__(self, units=100, spectral_radius=0.9, input_scaling=1.0, connectivity=1.0):
        self.units = int(units)                              # number of recurrent units
        self.spectral_radius = np.float64(spectral_radius)   # target spectral radius
        self.input_scaling = np.float64(input_scaling)       # scale of the input weights
        self.connectivity = connectivity                     # fraction of non-zero recurrent weights per row
        self.input_dim = None
        self.W = None                                        # recurrent weights (units, units)
        self._Win = None                                     # input weights + bias (units, input_dim+1)
        self._x = None                                       # current state (units,)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join(
            '%s=%s' % (k, v) for k, v in self._repr_params().items()))

    def _repr_params(self):
        return {'units': self.units, 'spectral_radius': float(self.spectral_radius),
                'input_scaling': float(self.input_scaling)}

    # ------------------------------------------------------------------
    # input weights: views on the input part and on the bias, used by step()
    # ------------------------------------------------------------------
    @property
    def Win(self):
        return self._Win

    @Win.setter
    def Win(self, value):
        self._Win = value
        if value is not None:
            self._Win_u = value[:, :-1]
            self._Win_b = value[:, -1]

    # ------------------------------------------------------------------
    # initialization
    #
    # DeepESN initializes all the layers in three phases, which reproduce the
    # sequence of random numbers of DeepESN 1.x:
    #   1. _init_recurrent: recurrent weights of all the layers;
    #   2. _init_input + _rescale: input weights and scaling, layer by layer;
    #   3. _init_extra: additional random parameters (e.g. RAM attention).
    # ------------------------------------------------------------------
    def build(self, input_dim, seed=None):
        '''Initialize a standalone layer for inputs of size input_dim.'''
        rng = RandomGenerators(seed)
        self._init_recurrent(rng)
        self._init_input(input_dim, rng)
        self._rescale()
        self._init_extra(rng)
        self.reset()
        return self

    @property
    def built(self):
        return self.W is not None and self._Win is not None

    def _init_recurrent(self, rng):
        Nr = self.units
        if self.connectivity < 1:
            # sparse recurrent weights: round(connectivity * Nr) random entries per row
            W = np.zeros((Nr, Nr))
            for row in range(Nr):
                number_row_elements = round(self.connectivity * Nr)
                row_elements = rng.py.sample(range(Nr), number_row_elements)
                W[row, row_elements] = rng.np.uniform(-1, +1, size=(1, number_row_elements))
        else:
            # fully-connected recurrent weights
            W = rng.np.uniform(-1, +1, size=(Nr, Nr))
        self.W = W

    def _init_input(self, input_dim, rng):
        self.input_dim = int(input_dim)
        s = self.input_scaling
        self.Win = rng.np.uniform(-s, s, size=(self.units, self.input_dim + 1))

    def _rescale(self):
        raise NotImplementedError

    def _init_extra(self, rng):
        pass

    @staticmethod
    def _max_abs_eig(M):
        eig_value, eig_vector = np.linalg.eig(M)
        return np.max(np.absolute(eig_value))

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def compute_states(self, inputs, initial_state=None):
        '''States (units, T) driven by the input sequence inputs (input_dim, T),
        starting from initial_state (zero by default). Does not change the
        online context (see set_history).'''
        raise NotImplementedError

    def step(self, u):
        '''Update the state with one input vector u (input_dim,) and return it.'''
        raise NotImplementedError

    def _initial_column(self, initial_state):
        if initial_state is None:
            return np.zeros((self.units, 1))
        return np.asarray(initial_state, dtype=np.float64).reshape(self.units, 1)

    # ------------------------------------------------------------------
    # online context
    # ------------------------------------------------------------------
    @property
    def state(self):
        '''Current state (units,).'''
        return self._x

    def reset(self, initial_state=None):
        '''Clear the context: state equal to initial_state (zero by default).'''
        self._x = self._initial_column(initial_state)[:, 0].copy()

    def set_history(self, states):
        '''Set the context to the end of the state sequence states (units, T),
        e.g. computed by compute_states, so that step() continues it.'''
        self._x = states[:, -1].copy()

    def get_context(self):
        '''Copy of the context (state and memory).'''
        return self._x.copy()

    def set_context(self, context):
        '''Restore a context returned by get_context().'''
        self._x = np.array(context, dtype=np.float64)
