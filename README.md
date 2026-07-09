# fedmaq-literature

PDF library, markdown conversions, and an **Open Knowledge Format (OKF)**
knowledge-graph bundle for the FedMAQ thesis
(_Communication-Efficient Federated Learning via Multi-Adaptive Quantization and
Knowledge Distillation_).

## Structure

```text
papers/               # Source PDFs, organized in ingestion batches (pipeline reads; gitignored)
markdown/{slug}/      # Converted full text (paper.md) + meta.yaml  — the raw, citable layer
kg/                   # OKF knowledge-graph bundle (the curated, agent-readable layer)
  index.md            #   bundle root (okf_version 0.1) + section map
  papers/{slug}.md    #   one `type: Paper` node per paper (39)
  methods/            #   `type: Method` nodes (scaffolded)
  concepts/           #   `type: Concept` nodes (scaffolded)
  findings/           #   `type: Finding` nodes (scaffolded)
  gaps/               #   `type: Gap` nodes (scaffolded)
src/fedmaq_literature/  # PDF conversion pipeline + registry + CLI
scripts/              # build_kg_papers.py (migration) + kg_bodies/ (hand-authored provenance)
SPEC.md               # the OKF v0.1 specification this bundle conforms to
```

Two layers, two purposes:

- **Raw layer** (`markdown/{slug}/paper.md`) — verbatim converted text. Agents read
  it directly for exact quotes/equations; it is the citation target. PDFs stay
  pipeline-only.
- **Knowledge layer** (`kg/`) — curated OKF concept documents with frontmatter,
  cross-links, and synthesis. This is what agents traverse to reason about the
  corpus and draft the manuscript. Start at [`kg/index.md`](kg/index.md).

There is no vector database: the corpus is small and finite, so file access
(grep + read over `markdown/` and `kg/`) replaces RAG retrieval.

## Conversion pipeline

- PDF → markdown: **Docling** primary, **Marker** GPU fallback when Docling QA fails.
- Writes `markdown/{slug}/paper.md` + `meta.yaml` and updates the conversion status
  in `.cursor/project/paper_registry.md`.

## Setup

```bash
uv sync --extra dev --extra convert     # Docling primary converter
# uv sync --extra marker                # optional GPU fallback

uv run fedmaq-lit list-slugs
uv run fedmaq-lit convert --slug he-2025-dynfed
uv run fedmaq-lit convert --all         # convert everything not yet ready
```

On Windows, if a HuggingFace model download fails on symlinks, the converter sets
`HF_HUB_DISABLE_SYMLINKS=1` automatically; enable Developer Mode for faster caching.

## Working with the knowledge graph

- Add a paper: place its PDF under `papers/<batch>/`, add a registry row, run
  `fedmaq-lit convert --slug {slug}`, then author `kg/papers/{slug}.md` following
  [`.cursor/rules/okf-paper-template.mdc`](.cursor/rules/okf-paper-template.mdc).
- Conventions for the bundle live in
  [`.cursor/rules/kg-conventions.mdc`](.cursor/rules/kg-conventions.mdc); the format
  itself is specified in [`SPEC.md`](SPEC.md).

## Agent onboarding

1. Read [../fedmaq-experiments/HANDOFF.md](../fedmaq-experiments/HANDOFF.md).
2. Read [AGENTS.md](AGENTS.md). Domain rules: `../fedmaq-experiments/.cursor/rules/`.
