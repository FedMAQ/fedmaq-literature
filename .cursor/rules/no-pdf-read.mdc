---
description: Never parse papers/*.pdf in agent context
alwaysApply: true
---

# No PDF Read in Chat

- Do **not** read or parse `papers/*.pdf` in agent context.
- Reason over `kg/papers/{slug}.md`; drop to `markdown/{slug}/paper.md` for exact
  quotes and equations. Retrieval is grep + read — there is no vector store.
- PDF conversion runs only via `fedmaq-lit convert`.
