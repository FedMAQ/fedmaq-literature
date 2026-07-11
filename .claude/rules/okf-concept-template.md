# OKF Concept Node Template

A Concept node defines one cross-cutting idea once — quantization, knowledge
distillation, non-IID / client drift, communication efficiency — and links out to
the methods and papers that use it, so shared manuscript terminology resolves to a
single authoritative node. The filename is the concept's lowercase kebab-case
name (e.g. `quantization.md`, `non-iid-heterogeneity.md`).

## Frontmatter

```yaml
---
type: Concept
title: "<Concept>"                     # e.g. "Quantization"
description: "<one-sentence definition>"
tags: [<tag>, ...]
timestamp: <ISO-8601 Z>
---
```

## Body sections

1. **Definition** — what the idea is; math inline with `\( … \)` where it sharpens
   the definition.
2. **Why it matters for FedMAQ** — the role it plays in the thesis and its
   communication / heterogeneity implications.
3. **Variants & dimensions** — the axes along which the idea varies (e.g. uniform
   vs. adaptive bit-width; logit vs. feature distillation; proxy vs. data-free).
4. **Methods & papers** — root-absolute links to the `/methods/` that instantiate
   the concept and the `/papers/` that study it.

## Closing sections

- `# Related` — root-absolute links to adjacent `/concepts/`, `/methods/`, and
  `/papers/` nodes.

Source from the curated paper nodes (`/papers/{slug}.md`); drop to
`markdown/{slug}/paper.md` only to verify an exact figure. Declarative,
factual prose.
