# Methods

`type: Method` concept documents for concrete FL algorithms — one node per method,
cross-linked to the [papers](/papers/) that introduce or evaluate it.

_Scaffolded; not yet populated._ This layer is Phase 2 of the OKF restructure.
Each method node will capture the algorithm's mechanism, hyperparameters, and the
papers where it appears (e.g. FedAvg → [mcmahan-2017-fedavg](/papers/mcmahan-2017-fedavg.md)),
letting an agent traverse "which methods address non-IID under a communication
budget" without re-reading full papers. Candidate methods are the `baseline:`
values already recorded in the paper nodes (FedAvg, FedProx, SCAFFOLD, FedDyn,
MOON, FedProto, FedNova, QSGD, signSGD, FedPAQ, DAdaQuant, FedMD, FedDF, FedGen,
FedDistill, FedKD, CFD, DynFed, FedDT, AdaDQ-KD, …).
