'''
Model configuration file

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis", 
Neurocomputing, 2017, vol. 268, pp. 87-99

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

import numpy as np

class Struct(object): pass

def config_MG(IP_indexes):

    configs = Struct()
    
    configs.rhos = 0.9 # set spectral radius 0.9 for all recurrent layers
    configs.lis = 1.0 # set li 1.0 for all recurrent layers
    configs.iss = 0.1 # set insput scale 0.1 for all recurrent layers
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 # deactivate pre-train

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1 # connectivity of recurrent matrix
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'SVD' # train with singular value decomposition (more accurate)
    configs.readout.regularizations = 10.0**np.array(range(-16,-1,1))
    
    return configs