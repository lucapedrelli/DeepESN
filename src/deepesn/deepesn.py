'''
DeepESN model

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99

Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
Neural Networks, 2018, vol. 108, pp. 33-47

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

import sys
import warnings

import numpy as np

from .layers import LiESNLayer, RandomGenerators
from .readouts import Readout


class Struct(object): pass


class _LayerParams(object):
    '''Dict-like view {layer index: parameter} on an attribute of the layers
    (DeepESN 1.x attributes W, Win, Gain and Bias).'''

    def __init__(self, layers, attr):
        self._layers = layers
        self._attr = attr

    def __getitem__(self, i):
        return getattr(self._layers[i], self._attr)

    def __setitem__(self, i, value):
        setattr(self._layers[i], self._attr, value)

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        return iter(range(len(self._layers)))

    def keys(self):
        return list(range(len(self._layers)))

    def values(self):
        return [self[i] for i in self]

    def items(self):
        return [(i, self[i]) for i in self]


class DeepESN(object):

    '''
    Deep Echo State Network (DeepESN) class:
    this class implement the DeepESN model suitable for
    time-serie prediction and sequence classification.

    A DeepESN is a stack of recurrent layers (see deepesn.layers) followed by
    a linear readout (see deepesn.readouts) on the concatenation of the states
    of all the layers (global state). Each layer is fed by the previous one.

        model = DeepESN(Nu, layers=[LiESNLayer(100, leaking_rate=0.5), ...],
                        readout=Readout('iNE', regularization=1e-6), seed=0)

    The DeepESN 1.x constructor DeepESN(Nu, Nr, Nl, configs) is still
    supported: it builds Nl LiESNLayer with Nr units from configs (rhos, lis,
    iss, IPconf, reservoirConf, readout) and gives the same weights and states
    as DeepESN 1.x.

    Initialization: seed=None uses the global generators np.random and random
    (as DeepESN 1.x); an int seed uses private generators that reproduce the
    same weights as np.random.seed(seed); random.seed(seed).
    '''

    def __init__(self, Nu, Nr=None, Nl=None, configs=None, verbose=0, layers=None, readout=None, seed=None):
        # initialize the DeepESN model

        if verbose:
            sys.stdout.write('init DeepESN...')
            sys.stdout.flush()

        if layers is None:
            # DeepESN 1.x constructor
            if Nr is None or Nl is None or configs is None:
                raise TypeError('DeepESN(Nu, layers=[...]) or DeepESN(Nu, Nr, Nl, configs)')
            layers, cfg_readout = self._layers_from_configs(Nr, Nl, configs)
            if readout is None:
                readout = cfg_readout
            self.IPconf = configs.IPconf  # configuration for Deep Intrinsic Plasticity
        else:
            if Nr is not None or Nl is not None or configs is not None:
                raise TypeError('give either layers or the DeepESN 1.x arguments (Nr, Nl, configs)')
            self.IPconf = Struct()
            self.IPconf.DeepIP = 0

        if len(layers) == 0:
            raise ValueError('a DeepESN needs at least one layer')
        if not isinstance(readout, Readout):
            readout = Readout('NE' if readout is None else readout)

        self.Nu = int(Nu)                # number of inputs
        self.layers = list(layers)       # recurrent layers
        self.readout = readout           # linear readout
        self.verbose = verbose

        self._offsets = np.cumsum([0] + [L.units for L in self.layers])

        self._init_layers(seed)

        if verbose:
            print('done.')
            sys.stdout.flush()

    @staticmethod
    def _layers_from_configs(Nr, Nl, configs):
        rhos = np.array(configs.rhos)    # spectral radius (maximum absolute eigenvalue)
        lis = np.array(configs.lis)      # leaky rate
        iss = np.array(configs.iss)      # input scale

        if len(rhos.shape) == 0:
            rhos = np.full(Nl, rhos)
        if len(lis.shape) == 0:
            lis = np.full(Nl, lis)
        if len(iss.shape) == 0:
            iss = np.full(Nl, iss)

        connectivity = configs.reservoirConf.connectivity
        layers = [LiESNLayer(Nr, spectral_radius=rhos[layer], input_scaling=iss[layer],
                             leaking_rate=lis[layer], connectivity=connectivity)
                  for layer in range(Nl)]

        # DeepESN 1.x: 'SVD' or normal equations for any other name
        method = getattr(configs.readout, 'trainMethod', 'NE')
        method = method if method in Readout.methods() else 'NE'
        readout = Readout(method)
        if hasattr(configs.readout, 'regularizations'):
            readout.regularizations = configs.readout.regularizations
        return layers, readout

    def _init_layers(self, seed):
        # same sequence of random numbers as DeepESN 1.x:
        # recurrent weights of all layers, then input weights layer by layer, then extras
        rng = RandomGenerators(seed)
        for L in self.layers:
            L._init_recurrent(rng)
        input_dim = self.Nu
        for L in self.layers:
            L._init_input(input_dim, rng)
            L._rescale()
            input_dim = L.units
        for L in self.layers:
            L._init_extra(rng)
        for L in self.layers:
            L.reset()

    def __repr__(self):
        layers = ',\n    '.join(repr(L) for L in self.layers)
        return 'DeepESN(Nu=%d, layers=[\n    %s],\n  readout=%r)' % (self.Nu, layers, self.readout)

    # ------------------------------------------------------------------
    # sizes and DeepESN 1.x attributes
    # ------------------------------------------------------------------
    @property
    def Nl(self):
        '''number of layers'''
        return len(self.layers)

    @property
    def Nr(self):
        '''number of units per layer (None if the layers have different sizes)'''
        sizes = set(L.units for L in self.layers)
        return sizes.pop() if len(sizes) == 1 else None

    @property
    def units(self):
        '''size of the global state (all the layers)'''
        return int(self._offsets[-1])

    def layer_slice(self, layer):
        '''rows of the global state of the given layer'''
        return slice(int(self._offsets[layer]), int(self._offsets[layer + 1]))

    @property
    def W(self):
        return _LayerParams(self.layers, 'W')        # recurrent weights

    @property
    def Win(self):
        return _LayerParams(self.layers, 'Win')      # input weights (+ bias)

    @property
    def Gain(self):
        return _LayerParams(self.layers, 'gain')     # activation function gain

    @property
    def Bias(self):
        return _LayerParams(self.layers, 'bias')     # activation function bias

    @property
    def rhos(self):
        return [float(L.spectral_radius) for L in self.layers]

    @property
    def lis(self):
        return np.array([getattr(L, 'leaking_rate', 1.0) for L in self.layers])

    @property
    def iss(self):
        return np.array([L.input_scaling for L in self.layers])

    @property
    def Wout(self):
        return self.readout.Wout

    @Wout.setter
    def Wout(self, value):
        self.readout.Wout = value

    # ------------------------------------------------------------------
    # states (DeepESN 1.x API)
    # ------------------------------------------------------------------
    def computeLayerState(self, input, layer, initialStatesLayer=None, DeepIP=0):
        # compute the state of a layer with pre-training if DeepIP == 1
        L = self.layers[layer]
        if DeepIP:
            return L.intrinsic_plasticity_states(input, self.IPconf)
        return L.compute_states(input, initialStatesLayer)

    def computeDeepIntrinsicPlasticity(self, inputs):
        # we incrementally perform the pre-training (deep intrinsic plasticity) over layers

        states = [np.zeros((self.units, inp.shape[1])) for inp in inputs]

        for layer in range(self.Nl):
            L = self.layers[layer]
            if not L.supports_ip:
                raise NotImplementedError('%s does not support intrinsic plasticity' % type(L).__name__)

            for epoch in range(self.IPconf.Nepochs):
                Gain_epoch = L.gain.copy()
                Bias_epoch = L.bias.copy()

                if len(inputs) == 1:
                    self.computeLayerState(inputs[0][:, self.IPconf.indexes], layer, DeepIP=1)
                else:
                    for i in self.IPconf.indexes:
                        self.computeLayerState(inputs[i], layer, DeepIP=1)

                if (np.linalg.norm(L.gain - Gain_epoch, 2) < self.IPconf.threshold) and \
                        (np.linalg.norm(L.bias - Bias_epoch, 2) < self.IPconf.threshold):
                    if self.verbose:
                        sys.stdout.write(str(epoch + 1) + '.')
                        sys.stdout.flush()
                    break

                if epoch + 1 == self.IPconf.Nepochs and self.verbose:
                    sys.stdout.write(str(epoch + 1) + '.')
                    sys.stdout.flush()

            inputs2 = [self.computeLayerState(inp, layer) for inp in inputs]
            for i in range(len(inputs)):
                states[i][self.layer_slice(layer), :] = inputs2[i]
            inputs = inputs2

        return states

    def computeState(self, inputs, DeepIP=0, initialStates=None, verbose=0):
        # compute the global state of DeepESN with pre-training if DeepIP == 1

        if self.IPconf.DeepIP and DeepIP:
            if verbose:
                sys.stdout.write('compute state with DeepIP...')
                sys.stdout.flush()
            states = self.computeDeepIntrinsicPlasticity(inputs)
        else:
            if verbose:
                sys.stdout.write('compute state...')
                sys.stdout.flush()
            states = [self.computeGlobalState(inp, initialStates) for inp in inputs]

        if verbose:
            print('done.')
            sys.stdout.flush()

        return states

    def computeGlobalState(self, input, initialStates):
        # compute the global state of DeepESN
        return self._global_states(input, initialStates, set_history=False)

    def _global_states(self, input, initialStates, set_history):
        if self.Nl == 1:
            L = self.layers[0]
            init = None if initialStates is None else initialStates[0:L.units, :]
            state = L.compute_states(input, init)
            if set_history:
                L.set_history(state)
            return state

        state = np.zeros((self.units, input.shape[1]))
        for layer, L in enumerate(self.layers):
            sl = self.layer_slice(layer)
            init = None if initialStates is None else initialStates[sl, :]
            layer_state = L.compute_states(input, init)
            if set_history:
                L.set_history(layer_state)
            state[sl, :] = layer_state
            input = state[sl, :]
        return state

    # ------------------------------------------------------------------
    # readout (DeepESN 1.x API)
    # ------------------------------------------------------------------
    def trainReadout(self, trainStates, trainTargets, lb, verbose=0):
        '''
        Deprecated (DeepESN 1.x): use fit_readout (training from scratch) or
        update_readout (online update of the iNE and RLS readouts).

        Trains the readout on a block without resetting it: the batch readouts
        (SVD, NE) solve on the block, overwriting the previous training; the
        online readouts (iNE, RLS) add the block to the data seen so far. Kept
        for the DeepESN 1.x scripts, with the same results as DeepESN 1.x.
        '''
        warnings.warn('DeepESN.trainReadout is deprecated: use fit_readout (training from scratch) '
                      'or update_readout (online update of iNE/RLS readouts)',
                      DeprecationWarning, stacklevel=2)
        if verbose:
            sys.stdout.write('train readout...')
            sys.stdout.flush()

        self.readout.train(trainStates, trainTargets, lb)

        if verbose:
            print('done.')
            sys.stdout.flush()

    def computeOutput(self, state):
        # compute a linear combination between the global state and the output weights
        return self.readout.predict(state)

    # ------------------------------------------------------------------
    # states: batch over a sequence and online (one step at a time)
    # ------------------------------------------------------------------
    def compute_states(self, inputs, initial_state=None):
        '''
        Global states (units, T) driven by the input sequence inputs (Nu, T)
        from initial_state (zero by default). The online context of the layers
        is set to the end of the sequence, so that step() continues it.
        '''
        if initial_state is not None:
            initial_state = np.asarray(initial_state, dtype=np.float64).reshape(self.units, 1)
        return self._global_states(inputs, initial_state, set_history=True)

    def reset(self, initial_state=None):
        '''Clear the online context (state equal to initial_state, zero by default).'''
        for layer, L in enumerate(self.layers):
            L.reset(None if initial_state is None else np.ravel(initial_state)[self.layer_slice(layer)])

    def step(self, u):
        '''Feed one input vector u (Nu,) and return the new global state (units,).'''
        if self.Nl == 1:
            return self.layers[0].step(u)
        x = u
        states = []
        for L in self.layers:
            x = L.step(x)
            states.append(x)
        return np.concatenate(states)

    @property
    def state(self):
        '''Current global state (units,).'''
        if self.Nl == 1:
            return self.layers[0].state
        return np.concatenate([L.state for L in self.layers])

    def get_context(self):
        '''Copy of the online context of all the layers (states and memories).'''
        return [L.get_context() for L in self.layers]

    def set_context(self, context):
        '''Restore a context returned by get_context().'''
        for L, c in zip(self.layers, context):
            L.set_context(c)

    # ------------------------------------------------------------------
    # readout and generation
    # ------------------------------------------------------------------
    def fit_readout(self, states, targets, regularization=None):
        '''Train the readout from scratch on states (units, T) and targets (Ny, T).'''
        self.readout.fit(states, targets, regularization)
        return self

    def update_readout(self, states, targets, regularization=None):
        '''Online update of the readout (iNE, RLS) with new samples.'''
        self.readout.update(states, targets, regularization)
        return self

    def predict(self, states):
        '''Readout outputs (Ny, T) for global states (units, T).'''
        return self.readout.predict(states)

    def generate(self, n_steps):
        '''
        Closed-loop (generative) run of n_steps from the current context: the
        output of the readout, trained for one-step-ahead prediction
        x(t) -> u(t+1), is fed back as next input. Returns the generated
        sequence (n_steps, Nu) and leaves the context after the last fed
        prediction (save and restore it with get_context/set_context).
        '''
        if self.readout.Wout is None:
            raise RuntimeError('the readout is not trained')
        if self.readout.Wout.shape[0] != self.Nu:
            raise ValueError('generate() needs a readout with Nu = %d outputs' % self.Nu)
        predict = self.readout.predict_step
        step = self.layers[0].step if self.Nl == 1 else self.step
        x = self.state
        y = predict(x)
        out = np.empty((n_steps, self.Nu))
        for s in range(n_steps):
            out[s] = y
            x = step(y)
            y = predict(x)
        return out
