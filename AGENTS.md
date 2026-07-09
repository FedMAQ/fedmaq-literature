# FedMAQ Literature — Agent Index

**New session:** [../fedmaq-experiments/HANDOFF.md](../fedmaq-experiments/HANDOFF.md)

This repo has two layers: a **raw** converted-text layer (`markdown/`) and a
**curated** OKF knowledge-graph layer (`kg/`). Reason over `kg/`; drop to
`markdown/` for exact quotes and equations. There is no vector store — use grep +
read over these directories.

| Resource                | Path                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| Knowledge graph (start) | `kg/index.md`                                                      |
| Paper nodes             | `kg/papers/{slug}.md` (39 `type: Paper` nodes)                     |
| Paper listing           | `kg/papers/index.md`                                               |
| Converted full text     | `markdown/{slug}/paper.md` (+ `meta.yaml`)                         |
| Paper status / registry | `.cursor/project/paper_registry.md`                                |
| OKF spec                | `SPEC.md`                                                          |
| Bundle conventions      | `.cursor/rules/kg-conventions.mdc`                                 |
| Node templates          | `.cursor/rules/okf-paper-template.mdc`, `okf-finding-template.mdc` |
| Source PDFs (pipeline)  | `papers/` (do not read in chat)                                    |
| Domain rules (sibling)  | `../fedmaq-experiments/.cursor/rules/`                             |
| CLI (conversion only)   | `uv run fedmaq-lit convert` / `list-slugs`                         |

## Conventions

- **Bundle root is `kg/`.** Intra-bundle links are root-absolute (`/papers/…`).
  References to the raw layer (`markdown/{slug}/paper.md`, `papers/…`) are
  repo-relative and point outside the bundle.
- **Do not parse `papers/*.pdf` in chat.** Use `markdown/{slug}/paper.md`.
- Curated knowledge is authored directly into `kg/` (no draft→approve gate);
  changes are reviewed via `git diff`.
