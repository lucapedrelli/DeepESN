'''
ESN layer: standard (non-leaky) Echo State Network recurrence

    x(t) = tanh(W x(t-1) + Win [u(t); 1])

Reference paper for the ESN model:
H. Jaeger, H. Haas, "Harnessing nonlinearity: Predicting chaotic systems and
saving energy in wireless communication", Science, 2004, vol. 304, pp. 78-80

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

from .base import Layer


class ESNLayer(Layer):
    '''
    Echo State Network layer with the standard recurrence

        x(t) = tanh(W x(t-1) + Win [u(t); 1]).

    Initialization: W is drawn uniformly in (-1, 1) (dense, or sparse with the
    given connectivity) and rescaled so that its spectral radius is
    spectral_radius; Win is drawn uniformly in (-input_scaling, input_scaling)
    (last column: bias).

    The layer supports the deep intrinsic plasticity (IP) pre-training of
    DeepESN 1.x, which adapts a gain and a bias of the activation function,
    tanh(gain * (W x + Win u) + bias).

    Parameters
    ----------
    units : number of recurrent units
    spectral_radius : spectral radius of W
    input_scaling : scale of the input weights (and bias)
    connectivity : fraction of non-zero recurrent weights per row (1 = dense)
    '''

    supports_ip = True
    _leak = 1.0          # leaking rate used by the IP pre-training rule (none for the ESN)

    def __init__(self, units=100, spectral_radius=0.9, input_scaling=1.0, connectivity=1.0):
        super(ESNLayer, self).__init__(units, spectral_radius, input_scaling, connectivity)
        self._gain = None
        self._bias = None
        self._ip_active = False      # True once gain/bias differ from 1/0 (IP or user-set)

    # ------------------------------------------------------------------
    # activation gain and bias (intrinsic plasticity)
    # ------------------------------------------------------------------
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value
        self._ip_active = True

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value
        self._ip_active = True

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def _init_input(self, input_dim, rng):
        super(ESNLayer, self)._init_input(input_dim, rng)
        self._gain = np.ones((self.units, 1))
        self._bias = np.zeros((self.units, 1))
        self._ip_active = False

    def _rescale(self):
        # rho(W) = spectral_radius
        actual_rho = self._max_abs_eig(self.W)
        self.W = (self.W * self.spectral_radius) / actual_rho

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def compute_states(self, inputs, initial_state=None):
        x0 = self._initial_column(initial_state)
        input = self.Win[:, 0:-1].dot(inputs) + np.expand_dims(self.Win[:, -1], 1)
        T = input.shape[1]
        state = np.zeros((self.units, T))
        W = self.W
        if self._ip_active:
            G, B = self._gain, self._bias
            state[:, 0:1] = np.tanh(np.multiply(G, W.dot(x0) + input[:, 0:1]) + B)
            for t in range(1, T):
                state[:, t:t+1] = np.tanh(np.multiply(G, W.dot(state[:, t-1:t]) + input[:, t:t+1]) + B)
        else:
            state[:, 0:1] = np.tanh(W.dot(x0) + input[:, 0:1])
            for t in range(1, T):
                state[:, t:t+1] = np.tanh(W.dot(state[:, t-1:t]) + input[:, t:t+1])
        return state

    def step(self, u):
        net = self.W.dot(self._x) + (self._Win_u.dot(u) + self._Win_b)
        if self._ip_active:
            net = np.multiply(self._gain[:, 0], net) + self._bias[:, 0]
        x = np.tanh(net)
        self._x = x
        return x

    # ------------------------------------------------------------------
    # deep intrinsic plasticity (DeepESN 1.x)
    # ------------------------------------------------------------------
    def intrinsic_plasticity_states(self, inputs, IPconf):
        '''States (units, T) computed while adapting gain and bias with the IP
        rule of DeepESN 1.x (one pass over inputs (input_dim, T)). IPconf
        provides eta (learning rate), mu and sigma (target Gaussian).'''
        a = self._leak
        input = self.Win[:, 0:-1].dot(inputs) + np.expand_dims(self.Win[:, -1], 1)
        T = input.shape[1]
        state = np.zeros((self.units, T))
        state_net = np.zeros((self.units, T))
        state_net[:, 0:1] = input[:, 0:1]
        state[:, 0:1] = a * np.tanh(np.multiply(self._gain, state_net[:, 0:1]) + self._bias)
        for t in range(1, T):
            state_net[:, t:t+1] = self.W.dot(state[:, t-1:t]) + input[:, t:t+1]
            state[:, t:t+1] = (1 - a) * state[:, t-1:t] + a * np.tanh(
                np.multiply(self._gain, state_net[:, t:t+1]) + self._bias)

            eta = IPconf.eta
            mu = IPconf.mu
            sigma2 = IPconf.sigma**2

            # IP learning rule
            deltaBias = -eta * ((-mu / sigma2) + np.multiply(
                state[:, t:t+1], (2 * sigma2 + 1 - (state[:, t:t+1]**2) + mu * state[:, t:t+1]) / sigma2))
            deltaGain = eta / self._gain + deltaBias * state_net[:, t:t+1]

            # update gain and bias of activation function
            self._gain = self._gain + deltaGain
            self._bias = self._bias + deltaBias
        self._ip_active = True
        return state
