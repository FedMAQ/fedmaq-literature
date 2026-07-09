---
name: convert-paper
description: Convert a paper PDF to markdown/{slug}/paper.md via fedmaq-lit
---

# Convert Paper

1. Confirm the slug (and its PDF label) in `.cursor/project/paper_registry.md`.
   Add a row per `naming-conventions.mdc` if missing.
2. Install the converter: `uv sync --extra convert` (add `--extra marker` for the
   GPU fallback).
3. Convert: `uv run fedmaq-lit convert --slug {slug}` (or `--all` for every paper
   not yet `ready`).
4. Pipeline: Docling convert → QA (confidence grades + content checks) → Marker
   fallback if QA fails → write `markdown/{slug}/paper.md` + `meta.yaml`.
5. The registry `Conversion` column is updated automatically; verify `meta.yaml`
   for converter, QA confidence, and reasons.

Conversion is the only pipeline step — there is no indexing/embedding stage.
After conversion, author the OKF node (see the `author-node` skill).
