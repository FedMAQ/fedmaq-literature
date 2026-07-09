---
description: OKF Finding node template (kg/findings/{topic}.md) — Phase 2+
globs: kg/findings/**
alwaysApply: false
---

# OKF Finding Node Template

A Finding node consolidates evidence across papers into one cross-cutting claim.
`findings/` is scaffolded but empty until Phase 2.

## Frontmatter

```yaml
---
type: Finding
title: "<the claim, stated as a finding>"
description: "<one-sentence summary>"
tags: [<topic>, ...]
timestamp: <ISO-8601 Z>
---
```

## Body sections

1. **Scope** — one paragraph on what the finding covers.
2. **Claim** — the consolidated statement.
3. **Evidence** — table of `paper | key result | link`, one row per supporting
   paper, linked root-absolute (`/papers/{slug}.md`).
4. **Open gaps** — what remains unresolved for FedMAQ; link `/gaps/` nodes.

## Closing sections

- `# Related` — root-absolute links to supporting paper/method/concept nodes.

Sources are the curated paper nodes (`/papers/{slug}.md`); drop to
`markdown/{slug}/paper.md` only to verify an exact figure. No emojis.
