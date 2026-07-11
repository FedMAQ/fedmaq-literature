# FedMAQ Literature — Agent Index

**Session continuity:** `.claude/project/changelog.md`

This repo has two layers: a **raw** converted-text layer (`markdown/`) and a
**curated** OKF knowledge-graph layer (`kg/`). See `.claude/rules/kg-conventions.md`
for the full layer/bundle/PDF-handling rules.

| Resource                | Path                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Knowledge graph (start) | `kg/index.md`                                                                                                     |
| Paper nodes             | `kg/papers/{slug}.md`                                                                                             |
| Paper listing           | `kg/papers/index.md`                                                                                              |
| Converted full text     | `markdown/{slug}/paper.md` (+ `meta.yaml`)                                                                        |
| Paper status / registry | `.claude/project/paper_registry.md`                                                                               |
| OKF spec                | `docs/okf.md`                                                                                                     |
| Bundle conventions      | `.claude/rules/kg-conventions.md`                                                                                 |
| Agent memory policy     | `.claude/rules/agent-memory.md`                                                                                   |
| Node templates          | `.claude/rules/okf-concept-template.md`, `okf-finding-template.md`, `okf-method-template.md`, `okf-paper-template.md` |
| Source PDFs (pipeline)  | `papers/` (do not read in chat)                                                                                   |
| CLI (conversion only)   | `uv run fedmaq-lit convert` / `list-slugs`                                                                        |

## Conventions

- Curated knowledge is authored directly into `kg/` (no draft→approve gate);
  changes are reviewed via `git diff`.
