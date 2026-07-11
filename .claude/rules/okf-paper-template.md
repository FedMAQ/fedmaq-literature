# OKF Paper Node Template

Each `kg/papers/{slug}.md` is an OKF node: YAML frontmatter, a five-section body,
then `# Related` and `# Citations`.

## Frontmatter

```yaml
---
type: Paper
title: "<canonical title>"          # quote free text (may contain colons)
description: "<one-sentence gist>"    # quoted
authors: "<Lead et al.>"
year: <YYYY>
bibkey: <slug>
baseline: <Name>                      # only if the paper is a baseline; omit otherwise
tags: [<tag>, ...]                    # batch tag first, then topic tags
resource: markdown/{slug}/paper.md    # repo-relative pointer to raw layer
timestamp: <ISO-8601 Z>
---
```

## Body sections

1. **Overview & Objectives** — core problem and aims.
2. **Methodology & Key Innovations** — the approach; math inline with `\( … \)`.
3. **Mathematical Formulation** — key objective/update equations.
4. **Limitations & Constraints** — stated or inferred gaps.
5. **FedMAQ Thesis Relevance** — how it informs the thesis; cross-link related
   nodes with root-absolute links, e.g. `[FedProx](/papers/li-2020-fedprox.md)`.

## Closing sections

- `# Related` — bulleted root-absolute links to sibling paper nodes.
- `# Citations` — `[1]` the raw conversion (`markdown/{slug}/paper.md`), `[2]`
  the source PDF path.

Declarative, factual prose. See `karimireddy-2020-scaffold.md` for a
reference node.
