'''
Dataset loaders of the DeepESN 1.x experiments (moved from task.py, still
importable from deepesn.task)

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

import os

import numpy as np


class Struct(object): pass


def select_indexes(data, indexes, transient=0):

    if len(data) == 1:
        return [data[0][:, indexes][:, transient:]]

    return [data[i][:, transient:] for i in indexes]


def load_MG(path, metric_function):
    from scipy.io import loadmat

    data = loadmat(os.path.join(path, 'MG.mat'))  # load dataset

    dataset = Struct()
    dataset.name = data['dataset'][0][0][0][0]
    dataset.inputs = data['dataset'][0][0][1][0]
    dataset.targets = data['dataset'][0][0][2][0]

    # input dimension
    Nu = dataset.inputs[0].shape[0]

    # function used for model evaluation
    error_function = metric_function

    # select the model that achieves the minimum error on validation set
    optimization_problem = np.argmin

    TR_indexes = range(4000)  # indexes for training, validation and test set in Mackey-Glass task
    VL_indexes = range(4000, 5000)
    TS_indexes = range(5000, 9999)

    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes
