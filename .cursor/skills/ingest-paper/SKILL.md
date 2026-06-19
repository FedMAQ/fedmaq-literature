---
name: ingest-paper
description: Run fedmaq-lit ingest for a paper slug
---

# Ingest Paper

1. Confirm slug in `.cursor/project/paper_registry.md`.
2. Run `uv sync --extra convert` (and `--extra marker` for GPU fallback).
3. Convert: `uv run fedmaq-lit convert --slug {slug}`
   Or full ingest (convert only until P2 indexing):
   `uv run fedmaq-lit ingest --slug {slug} --convert-only`
4. Pipeline: Docling convert → QA (Docling confidence grades + content checks) → Marker fallback if needed → write `markdown/{slug}/paper.md` + `meta.yaml`.
5. Update registry: conversion status is written automatically; verify `meta.yaml` for converter, QA confidence, and reasons.
6. Serialize GPU: do not run conversion and Qwen3-Embedding-4B embedding concurrently.
