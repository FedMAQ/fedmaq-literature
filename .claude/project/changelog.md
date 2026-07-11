# Changelog

Milestones-only log for session continuity. Not a full commit history (see
`git log` for that) — entries here mark decisions and state changes an agent
picking up this repo cold would need to know about. claude-mem covers
session-level narrative; this file is the durable, git-tracked record.

**Policy (as of 2026-07-11):** add an entry when a milestone lands (kg/ bundle
restructure, phase completion, de-overclaim pass) — not per-commit.

## 2026-07-11 — De-overclaim pass on kg/ vis-a-vis polished manuscript
Updated kg/ node claims to match the manuscript's de-overclaimed,
communication-primary framing after its own grill-and-polish pass — including
retargeting the FedMAQ novelty claim to multi-signal combination rather than a
stronger originality claim.

## Earlier
Restructured around the OKF (Open Knowledge Format) bundle: added `kg/`
index/log, authored Paper nodes from source markdown, populated methods/
concepts/findings/gaps layers, added `docs/llm-wiki.md` and `docs/okf.md`
specs. Removed the local vector-RAG stack, kept the PDF-to-markdown conversion
pipeline (`convert-paper` skill). Migrated agent tooling from Cursor to Claude
Code (`.cursor/` to `.claude/rules/`); `AGENTS.md` retired in favor of
`CLAUDE.md`.
