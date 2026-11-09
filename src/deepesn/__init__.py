'''
Deep Echo State Network (DeepESN) Python Library

    DeepESN                   stack of recurrent layers + linear readout; also the
                              DeepESN 1.x API: DeepESN(Nu, Nr, Nl, configs),
                              computeState, trainReadout, computeOutput
    deepesn.layers            Layer (base class of new layers), ESNLayer, LiESNLayer
    deepesn.readouts          SVDReadout, NEReadout, iNEReadout, RLSReadout, RidgeReadout
                              (one file per method; Readout('iNE', ...) by name)
    deepesn.tasks             Task (base class of new tasks), MemoryCapacity

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

from .deepesn import DeepESN
from .readouts import Readout, SVDReadout, NEReadout, iNEReadout, RLSReadout, RidgeReadout
from .layers import Layer, ESNLayer, LiESNLayer
from .tasks import Task, MemoryCapacity

__version__ = '1.2.0'

__all__ = ['DeepESN',
           'Readout', 'SVDReadout', 'NEReadout', 'iNEReadout', 'RLSReadout', 'RidgeReadout',
           'Layer', 'ESNLayer', 'LiESNLayer',
           'Task', 'MemoryCapacity']
