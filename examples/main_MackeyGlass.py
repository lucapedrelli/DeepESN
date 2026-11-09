'''
This is an example of the application of DeepESN model for next-step prediction on Mackey Glass time-series.

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

from deepesn import DeepESN, Readout, ESNLayer
from deepesn.tasks import NextStepPrediction


def main():

    # Mackey-Glass time series (tau = 17): 4000 steps for training, 1000 for validation, 5000 for test
    train, validation, test = NextStepPrediction.mackey_glass(4000, 1000, 5000)
    task = NextStepPrediction(train, validation, test, washout=100)

    # DeepESN with 5 recurrent layers of 100 units, readout trained by singular value decomposition
    Nr = 100  # number of recurrent units
    Nl = 5    # number of recurrent layers
    model = DeepESN(1, layers=[ESNLayer(Nr, spectral_radius=0.9, input_scaling=0.1) for _ in range(Nl)],
                    readout=Readout('SVD', regularization=0.0), seed=7)

    for split in ('validation', 'test'):
        result = task.evaluate(model, split)
        print('%s MSE: %.3e   MAE: %.3e' % (split.capitalize(), result['mse'], result['mae']))


if __name__ == "__main__":
    main()
