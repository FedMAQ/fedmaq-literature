# Knowledge Graph Update Log

## 2026-07-09

- **Initialization**: Established the OKF bundle at `kg/` with the root [index](/index.md) (`okf_version: "0.1"`) and scaffolded sections `papers/`, `methods/`, `concepts/`, `findings/`, `gaps/`.
- **Creation**: Populated [papers/](/papers/) with 39 `type: Paper` nodes — the full FedMAQ thesis canon. 28 migrated from approved summaries; 11 hand-authored from source markdown (the 10 net-new baselines FedDyn, SCAFFOLD, MOON, FedProto, FedNova, QSGD, signSGD, Jeong FD+FAug, FedDF, FedGen, plus a re-authored English node for joseph-2026-air-quality whose prior summary was in Chinese).
- **Creation**: Generated [papers/index](/papers/index.md) — progressive-disclosure listing grouped by ingestion batch / theme.
- **Provenance**: Migration performed by `scripts/build_kg_papers.py`; hand-authored bodies retained under `scripts/kg_bodies/`.
