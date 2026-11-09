'''
Recurrent layers of a DeepESN

    ESNLayer      standard ESN recurrence, rho(W) initialization
    LiESNLayer    leaky-integrator ESN recurrence, rho((1-a) I + a W) initialization

New layer types derive from Layer (see layers/base.py).

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

from .base import Layer, RandomGenerators
from .esn import ESNLayer
from .liesn import LiESNLayer

__all__ = ['Layer', 'RandomGenerators', 'ESNLayer', 'LiESNLayer']
