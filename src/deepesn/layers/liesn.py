'''
LiESN layer: leaky-integrator Echo State Network recurrence

    x(t) = (1 - a) x(t-1) + a tanh(W x(t-1) + Win [u(t); 1])

Reference paper for the leaky-integrator ESN:
H. Jaeger, M. Lukosevicius, D. Popovici, U. Siewert, "Optimization and
applications of echo state networks with leaky-integrator neurons",
Neural Networks, 2007, vol. 20, pp. 335-352

Reference paper for the initialization (spectral radius of the leaky dynamics):
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99

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

from .esn import ESNLayer


class LiESNLayer(ESNLayer):
    '''
    Leaky-integrator Echo State Network layer

        x(t) = (1 - a) x(t-1) + a tanh(W x(t-1) + Win [u(t); 1]),

    with leaking rate a in (0, 1] (a = 1 gives the standard ESN).

    Initialization: W is drawn uniformly in (-1, 1) and rescaled so that the
    matrix (1 - a) I + a W, which governs the leaky dynamics linearized around
    the origin, has spectral radius spectral_radius (as in DeepESN 1.x); for
    a = 1 this is rho(W) = spectral_radius. Win is drawn uniformly in
    (-input_scaling, input_scaling) (last column: bias).

    Parameters
    ----------
    units : number of recurrent units
    spectral_radius : spectral radius of (1 - a) I + a W
    input_scaling : scale of the input weights (and bias)
    leaking_rate : leaking rate a
    connectivity : fraction of non-zero recurrent weights per row (1 = dense)
    '''

    def __init__(self, units=100, spectral_radius=0.9, input_scaling=1.0, leaking_rate=1.0,
                 connectivity=1.0):
        super(LiESNLayer, self).__init__(units, spectral_radius, input_scaling, connectivity)
        self.leaking_rate = np.float64(leaking_rate)

    def _repr_params(self):
        p = super(LiESNLayer, self)._repr_params()
        p['leaking_rate'] = float(self.leaking_rate)
        return p

    @property
    def _leak(self):
        return self.leaking_rate

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def _rescale(self):
        # rho((1 - a) I + a W) = spectral_radius
        a = self.leaking_rate
        Nr = self.units
        Ws = (1 - a) * np.eye(Nr, Nr) + a * self.W
        actual_rho = self._max_abs_eig(Ws)
        Ws = (Ws * self.spectral_radius) / actual_rho
        self.W = (a**-1) * (Ws - (1. - a) * np.eye(Nr, Nr))

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def compute_states(self, inputs, initial_state=None):
        a = self.leaking_rate
        x0 = self._initial_column(initial_state)
        input = self.Win[:, 0:-1].dot(inputs) + np.expand_dims(self.Win[:, -1], 1)
        T = input.shape[1]
        state = np.zeros((self.units, T))
        W = self.W
        if self._ip_active:
            G, B = self._gain, self._bias
            state[:, 0:1] = (1 - a) * x0 + a * np.tanh(np.multiply(G, W.dot(x0) + input[:, 0:1]) + B)
            for t in range(1, T):
                state[:, t:t+1] = (1 - a) * state[:, t-1:t] + a * np.tanh(
                    np.multiply(G, W.dot(state[:, t-1:t]) + input[:, t:t+1]) + B)
        else:
            state[:, 0:1] = (1 - a) * x0 + a * np.tanh(W.dot(x0) + input[:, 0:1])
            for t in range(1, T):
                state[:, t:t+1] = (1 - a) * state[:, t-1:t] + a * np.tanh(
                    W.dot(state[:, t-1:t]) + input[:, t:t+1])
        return state

    def step(self, u):
        a = self.leaking_rate
        x_prev = self._x
        net = self.W.dot(x_prev) + (self._Win_u.dot(u) + self._Win_b)
        if self._ip_active:
            net = np.multiply(self._gain[:, 0], net) + self._bias[:, 0]
        x = (1.0 - a) * x_prev + a * np.tanh(net)
        self._x = x
        return x
