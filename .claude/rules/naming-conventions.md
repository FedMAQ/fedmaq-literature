# Naming Conventions

**Slug:** lowercase `{leadauthor}-{year}-{short-title}` derived from PDF filename, e.g. `he-2025-dynfed` from `He et al. - 2025 - DynFed...pdf`.

**Paths:**
- PDF: `papers/<batch>/<original-filename>.pdf`
- Raw markdown: `markdown/{slug}/paper.md`, `meta.yaml`
- OKF Paper node: `kg/papers/{slug}.md`

The node's `bibkey` equals the slug; align `bib/references.bib` keys with slugs
where possible.
