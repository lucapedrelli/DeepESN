'''
Tasks of DeepESN

    MemoryCapacity               reconstruction of the delayed inputs, MC = sum_k r_k^2
    NextStepPrediction           prediction of the next value of the series u(t+1)

New tasks derive from Task (see tasks/base.py).

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

from .base import Task, METRICS, as_series
from .memory_capacity import MemoryCapacity
from .next_step import NextStepPrediction

__all__ = ['Task', 'METRICS', 'as_series', 'MemoryCapacity', 'NextStepPrediction']
