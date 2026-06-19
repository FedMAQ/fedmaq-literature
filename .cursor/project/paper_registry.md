# Paper Registry

Track conversion, indexing, and summary status. Run `fedmaq-lit ingest --slug <slug>` to process.

| Slug | PDF | Baseline | Conversion | Summary | Tags |
| ---- | --- | -------- | ---------- | ------- | ---- |
| mcmahan-2017-fedavg | McMahan et al. - 2017 - Communication-Efficient Learning... | FedAvg | none | none | vanilla |
| li-2020-fedprox | Li et al. - 2020 - Federated Optimization in Heterogeneous Networks | FedProx | none | none | vanilla |
| honig-2022-dadaquant | Hönig et al. - 2022 - DAdaQuant... | DAdaQuant | none | none | quantization |
| reisizadeh-2020-fedpaq | Reisizadeh et al. - 2020 - FedPAQ... | FedPAQ | none | none | quantization |
| song-2024-feddistill | Song et al. - 2024 - FedDistill... | FedDistill | none | none | kd |
| liu-2023-adagq | Liu et al. - 2023 - Communication-Efficient FL... Adaptive Gradient | AdaGQ | none | none | adaptive |
| cui-2026-laq-hc | Cui et al. - 2026 - Lightweight Adaptive Quantization... | LAQ-HC | none | none | adaptive |
| he-2025-dynfed | He et al. - 2025 - DynFed... | DynFed | ready | none | sota, kd |
| wang-2026-adadq-kd | Wang et al. - 2026 - AdaDQ-KD... | AdaDQ-KD | none | none | sota, kd |
| he-2025-feddt | He et al. - 2025 - FedDT... | FedDT | none | none | sota, kd |
| hinton-2015-distillation | Hinton et al. - 2015 - Distilling the Knowledge... | — | ready | none | kd, survey |
| jimenez-2024-non-iid-survey | Jimenez et al. - 2024 - Non-IID data in FL... | — | none | none | heterogeneity |
| qin-2025-kd-survey | Qin et al. - 2025 - Knowledge Distillation in FL... | — | none | none | kd, survey |

**Conversion:** `none` | `ready` | `failed`  
**Summary:** `none` | `draft` | `approved`

Add rows for remaining PDFs in `papers/` using `naming-conventions.mdc`.
