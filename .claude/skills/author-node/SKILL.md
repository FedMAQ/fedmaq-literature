---
name: author-node
description: Author an OKF Paper node in kg/papers/{slug}.md
---

# Author OKF Node

1. Read `markdown/{slug}/paper.md` (not the PDF) for the source content.
2. Write `kg/papers/{slug}.md` following `okf-paper-template.md`: `type: Paper`
   frontmatter, five body sections, then `# Related` and `# Citations`.
3. Cross-link related nodes with root-absolute links (`/papers/{other}.md`) and
   list them under `# Related`.
4. Add a line to `kg/log.md` and confirm the paper appears in `kg/papers/index.md`.
5. There is no approval gate — commit directly; review is via `git diff`.

Verify OKF conformance: parseable frontmatter, non-empty `type`, and resolving
intra-bundle links (see `kg-conventions.md` and `SPEC.md`).
