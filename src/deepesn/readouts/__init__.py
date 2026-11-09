'''
Readouts of a DeepESN (one file per training method)

Linear readout y(t) = Wout [x(t); 1], trained in closed form by ridge
regression with regularization lambda:

    SVDReadout   'SVD'   batch, singular value decomposition (accurate)
    NEReadout    'NE'    batch, normal equations (fast)
    iNEReadout   'iNE'   online, incremental normal equations
    RLSReadout   'RLS'   online, recursive least squares
    RidgeReadout 'Ridge' batch, scikit-learn Ridge (requires scikit-learn)

iNE and RLS give at every step the ridge solution over all the data seen so
far, and are the online closed-form algorithms of online generative RC.

Readout(method, regularization) builds the readout of a method given by name:
Readout('iNE', 1e-6) is an iNEReadout(1e-6). New training methods derive from
Readout (see readouts/base.py) and, with a name, are also available by name.

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

from .base import Readout
from .svd import SVDReadout
from .ne import NEReadout
from .ine import iNEReadout
from .rls import RLSReadout
from .ridge import RidgeReadout

__all__ = ['Readout', 'SVDReadout', 'NEReadout', 'iNEReadout', 'RLSReadout', 'RidgeReadout']
