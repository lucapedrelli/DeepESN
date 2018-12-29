'''
Model configuration file

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis", 
Neurocomputing, 2017, vol. 268, pp. 87-99

Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
Neural Networks, 2018, vol. 108, pp. 33-47 
    
----

This file is a part of the DeepESN Python Library (DeepESNpy)

Luca Pedrelli
luca.pedrelli@di.unipi.it
lucapedrelli@gmail.com

Department of Computer Science - University of Pisa (Italy)
Computational Intelligence & Machine Learning (CIML) Group
http://www.di.unipi.it/groups/ciml/

----
'''

import numpy as np

class Struct(object): pass

def config_pianomidi(IP_indexes):

    configs = Struct()
    
    configs.rhos = 0.1 # set spectral radius 0.1 for all recurrent layers
    configs.lis = 0.7 # set li 0.7 for all recurrent layers
    configs.iss = 0.1 # set insput scale 0.1 for all recurrent layers
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 1 # activate pre-train
    configs.IPconf.threshold = 0.1 # threshold for gradient descent in pre-train algorithm
    configs.IPconf.eta = 10**-5 # learning rate for IP rule
    configs.IPconf.mu = 0 # mean of target gaussian function
    configs.IPconf.sigma = 0.1 # std of target gaussian function
    configs.IPconf.Nepochs = 10 # maximum number of epochs
    configs.IPconf.indexes = IP_indexes # perform the pre-train on these indexes

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1 # connectivity of recurrent matrix
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'NormalEquations' # train with normal equations (faster)
    configs.readout.regularizations = 10.0**np.array(range(-4,-1,1))
    
    return configs


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