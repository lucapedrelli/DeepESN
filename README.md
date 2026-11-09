# Deep Echo State Network (DeepESN) Python Library

Deep Echo State Network (DeepESN) is an extension of the ESN model towards the deep learning paradigm (Deep Reservoir Computing). A DeepESN is a deep Recurrent Neural Network composed by a hierarchy of recurrent layers intrinsically able to develop hierarchical and distributed temporal features. Such characteristics make DeepESN suitable for time-series and sequences processing.

All details about DeepESN model are described in the reference paper (**CITATION REQUEST**): <br />
C. Gallicchio, A. Micheli, L. Pedrelli, <br />
["Deep Reservoir Computing: A Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99](https://www.sciencedirect.com/science/article/pii/S0925231217307567)

The design of DeepESN model in multivariate time-series prediction tasks is described in the following paper: <br />
C. Gallicchio, A. Micheli, L. Pedrelli, <br />
["Design of deep echo state networks", Neural Networks, 2018, vol. 108, pp. 33-47](https://www.sciencedirect.com/science/article/pii/S0893608018302223)


**AUTHOR INFORMATION**

Luca Pedrelli <br />
luca.pedrelli@santannapisa.it <br />
lucapedrelli@gmail.com

The Telecommunications, Computer Engineering, and Photonics Institute (TeCIP) <br />
Scuola Superiore Sant'Anna in Pisa <br />
ICT-COISP (ICT for Complex Industrial Systems and Processes) <br />
https://www.santannapisa.it/it/istituto/tecip

## Installation

```bash
pip install DeepESN
```

Development version (from this repository): `pip install -e .`
The `Ridge` readout needs scikit-learn (`pip install "DeepESN[ridge]"`).

## DeepESN 1.x API (unchanged)

The scripts written for DeepESN 1.x keep working and give the same results:

```python
from deepesn import DeepESN
from deepesn.configurations import config_MG

deepESN = DeepESN(Nu, Nr, Nl, configs)
states = deepESN.computeState(inputs, configs.IPconf.DeepIP)
deepESN.trainReadout(train_states, train_targets, regularization)
outputs = deepESN.computeOutput(states)
```

`configs.readout.trainMethod` selects the training of the readout: `'SVD'`, `'NE'` (also in dual form when the
features are more than the time steps), `'Ridge'` (scikit-learn), and the online `'iNE'` and `'RLS'`.

## New in 1.2: layers, readouts, tasks

A DeepESN can also be built as a stack of **layers** followed by a linear **readout**, evaluated on **tasks**.
This is the interface used by other repositories built on DeepESN (e.g. [RAM-ESN](https://github.com/lucapedrelli/RAM-ESN)),
which define their own layers and tasks.

```python
from deepesn import DeepESN, Readout, LiESNLayer
from deepesn.tasks import MemoryCapacity

train, validation, test = MemoryCapacity.random_input(5000, 1000, 1000, seed=0)
model = DeepESN(1, layers=[LiESNLayer(100, spectral_radius=0.9, input_scaling=0.1, leaking_rate=0.5)],
                readout=Readout('NE', regularization=1e-8), seed=0)
print(MemoryCapacity(train, validation, test, washout=1000, max_delay=200).evaluate(model, 'validation')['mc'])
```

**Layers** (`deepesn.layers`, one file per layer; each layer owns its recurrence, its initialization and its memory):

| Layer | State update | Initialization |
|---|---|---|
| `ESNLayer` | x(t) = tanh(W x(t-1) + Win u(t)) | rho(W) = spectral_radius |
| `LiESNLayer` | x(t) = (1-a) x(t-1) + a tanh(W x(t-1) + Win u(t)) | rho((1-a) I + a W) = spectral_radius |

Layers can be stacked (`layers=[...]`, also of different types and sizes); new layer types derive from `deepesn.layers.Layer`.
States are computed in batch over a sequence (`compute_states`) or online, one step at a time (`step`), with
context save/restore (`get_context`/`set_context`) and closed-loop generation (`generate`).
`seed=None` uses the global `np.random`/`random` generators (as DeepESN 1.x); an int seed gives the same weights as
`np.random.seed(seed); random.seed(seed)`.

**Readouts** (`deepesn.readouts`, one file per training method): linear readout y(t) = Wout [x(t); 1] on the
global state, trained in closed form by ridge regression (regularization lambda).

| Readout | Name | Training |
|---|---|---|
| `SVDReadout` | `'SVD'` | batch: singular value decomposition (accurate) |
| `NEReadout` | `'NE'` | batch: normal equations (fast); dual form when the features are more than the time steps |
| `RidgeReadout` | `'Ridge'` | batch: scikit-learn `Ridge` (requires scikit-learn) |
| `iNEReadout` | `'iNE'` | online: incremental normal equations, P += x x^T, Q += y x^T, Wout = Q (P + lambda I)^-1 |
| `RLSReadout` | `'RLS'` | online: recursive least squares (Sherman-Morrison rank-1 updates of the inverse) |

The online readouts (`update`) give at every step the ridge solution over all the data seen so far.
`Readout(name, regularization)` builds a readout by name (`Readout('iNE', 1e-6)` is an `iNEReadout(1e-6)`).
New training methods derive from `deepesn.readouts.Readout` (set `method` and `online`, implement `_train`)
and are then available also by name.

**Tasks** (`deepesn.tasks`): time series split into training, validation and test;
`evaluate(model, 'validation')` trains on the training split, `evaluate(model, 'test')` on training + validation.

| Task | Evaluation |
|---|---|
| `MemoryCapacity` | reconstruction of the delayed inputs u(t-k), k = 1..`max_delay`; MC = sum_k r_k^2 (`random_input` generates the i.i.d. input) |
| `NextStepPrediction` | prediction of the next value u(t+1) of a (multivariate) series; MSE, MAE (`mackey_glass` generates the Mackey-Glass series) |

New tasks derive from `deepesn.tasks.Task`.

See `examples/main_MackeyGlass.py`: next-step prediction of the Mackey-Glass series with a DeepESN of 5 layers.

**Readout training with the new interface.** `trainReadout` (1.x) still works, with a `DeprecationWarning`;
the new methods are `fit_readout` (training from scratch) and `update_readout` (online updates of iNE and RLS).

| Readout training | From scratch | SVD, NE, Ridge | iNE, RLS already trained |
|---|---|---|---|
| `fit_readout(states, targets[, regularization])` | yes | ridge on the block | restart from the block |
| `update_readout(states, targets[, regularization])` | no | error (offline readout) | add the block to the data seen so far |
| `trainReadout(states, targets, lb)` (1.x) | no | ridge on the block (previous training lost) | add the block to the data seen so far |
