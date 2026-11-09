'''
Compatibility module: the readouts are in deepesn.readouts, one file per
training method (SVDReadout, NEReadout, iNEReadout, RLSReadout, RidgeReadout).
"from deepesn.readout import Readout" keeps working.

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

from .readouts import Readout, SVDReadout, NEReadout, iNEReadout, RLSReadout, RidgeReadout

__all__ = ['Readout', 'SVDReadout', 'NEReadout', 'iNEReadout', 'RLSReadout', 'RidgeReadout']
