---
name: approve-summary
description: Promote summary draft after human review
---

# Approve Summary

1. Human has reviewed `summaries/drafts/{slug}.md`.
2. Run `uv run fedmaq-lit approve --slug {slug}` (or move file manually per workflow).
3. Update registry to `summary: approved`.
