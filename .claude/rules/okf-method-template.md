---
description: OKF Method node template (kg/methods/{name}.md) — Phase 2
globs: kg/methods/**
alwaysApply: false
---

# OKF Method Node Template

A Method node describes one concrete FL algorithm — its mechanism, knobs, and the
papers that introduce or evaluate it — so an agent can traverse "which methods
address non-IID under a communication budget" without re-reading full papers.
One node per algorithm; the filename is the method's lowercase kebab-case name
(e.g. `scaffold.md`, `dadaquant.md`, `fedpaq.md`).

## Frontmatter

```yaml
---
type: Method
title: "<Method name>"                # e.g. "SCAFFOLD"
description: "<one-sentence mechanism>"
tags: [<family>, ...]                  # family: drift-correction | regularization |
                                       # quantization | distillation | joint-q-kd
introduced_by: /papers/{slug}.md       # the paper that proposes the method
timestamp: <ISO-8601 Z>
---
```

## Body sections

1. **Mechanism** — how the algorithm works; math inline with `\( … \)`.
2. **Key hyperparameters** — the tunable knobs (e.g. proximal \(\mu\), bit-width
   \(b\), local epochs \(E\)) and what they trade off.
3. **Communication & computation profile** — per-round payload, extra client
   state, uplink/downlink asymmetry, added compute.
4. **Papers** — where the method appears, as root-absolute links:
   introduces (`/papers/{slug}.md`), used-as-baseline-in, compared-against.
5. **FedMAQ relevance** — how it informs the thesis; whether it is a FedMAQ
   baseline; the tension or synergy with adaptive quantization + distillation.

## Closing sections

- `# Related` — root-absolute links to sibling methods, the `/concepts/` it
  instantiates, and its `/papers/` nodes.

Source from the curated paper nodes (`/papers/{slug}.md`); drop to
`markdown/{slug}/paper.md` only to verify an exact figure or equation. No emojis.
Declarative, factual prose.
