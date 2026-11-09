'''
Task configuration file of DeepESN 1.x (dataset loaders).

The loaders are in deepesn.datasets and are importable from here for
compatibility; the tasks (e.g. MemoryCapacity) are in deepesn.tasks.

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

from .datasets import Struct, select_indexes, load_MG

__all__ = ['Struct', 'select_indexes', 'load_MG']
