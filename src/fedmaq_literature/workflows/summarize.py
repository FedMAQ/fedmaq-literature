"""Literature paper summarization workflow using OpenRouter."""

from __future__ import annotations

from pathlib import Path

from fedmaq_literature.paths import markdown_dir, repo_root, registry_path
from fedmaq_literature.registry import parse_registry, update_registry_summary
from fedmaq_literature.workflows.llm import call_llm, DEFAULT_SUMMARIZE_MODEL


def run_summarize(
    slug: str | None = None,
    *,
    model: str | None = None,
    force: bool = False,
    root: Path | None = None,
) -> int:
    """Generate LLM paper summary draft for converted markdown files.

    If slug is specified, summarize only that paper.
    If slug is None, summarize all ready papers that do not have a summary yet.
    """
    root_path = repo_root(root)
    entries = parse_registry(registry_path(root_path))
    model_name = model or DEFAULT_SUMMARIZE_MODEL

    if slug:
        target_entries = [e for e in entries if e.slug == slug]
        if not target_entries:
            print(f"Error: Slug '{slug}' not found in registry.")
            return 1
        entry = target_entries[0]
        if entry.conversion != "ready":
            print(
                f"Error: Paper '{slug}' is not converted yet (status: {entry.conversion})."
            )
            return 1
    else:
        # Get all papers that are converted and ready
        target_entries = [e for e in entries if e.conversion == "ready"]

    # Filter out papers that already have summaries unless force is True
    if not force:
        target_entries = [e for e in target_entries if e.summary == "none"]

    if not target_entries:
        if slug:
            print(
                f"Paper '{slug}' already has a summary draft or approval. Use --force-summarize to override."
            )
        else:
            print("No pending papers to summarize.")
        return 0

    drafts_dir = root_path / "summaries" / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for entry in target_entries:
        s = entry.slug
        paper_file = markdown_dir(root_path) / s / "paper.md"
        if not paper_file.is_file():
            print(
                f"Warning: Converted paper markdown not found for '{s}' at {paper_file}. Skipping."
            )
            continue

        print(f"Generating summary draft for '{s}' using {model_name}...")
        try:
            paper_text = paper_file.read_text(encoding="utf-8")

            # Formulate the prompt
            system_prompt = (
                "You are an expert researcher in Federated Learning (FL), Model Compression, "
                "Quantization, and Knowledge Distillation (KD). Your job is to read an academic "
                "paper and produce a structured, high-quality, professional Markdown summary "
                "focusing on methodology, mathematical formulas, and relevance to communication efficiency."
            )

            prompt = (
                f"Below is the full text of the academic paper '{entry.pdf_label}' (slug: {s}).\n"
                f"Please analyze it and write a concise, highly-structured research summary in Markdown.\n\n"
                f"Include the following sections in your summary:\n"
                f"1. **Overview & Objectives**: What is the core problem being solved and the main objectives?\n"
                f"2. **Methodology & Key Innovations**: Explain the proposed method. Focus on the core system model.\n"
                f"3. **Mathematical Formulation**: Extract the exact formulas, algorithms, and updates (e.g., adaptive quantization, quantization step size, gradient compression, KD loss, client-server aggregation update rules).\n"
                f"4. **Limitations & Constraints**: What are the statistical or system assumptions, limitations, or communication bottleneck concerns?\n"
                f"5. **FedMAQ Thesis Relevance**: Map how this paper connects to FedMAQ (Communication-Efficient FL via Multi-Adaptive Quantization and Knowledge Distillation). Specifically mention if it can serve as a baseline (e.g., if it is {entry.baseline or 'none'}) or if its techniques can be integrated.\n\n"
                f"--- PAPER CONTENT START ---\n"
                f"{paper_text}\n"
                f"--- PAPER CONTENT END ---\n"
            )

            summary_draft = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_name,
                temperature=0.1,
            )

            draft_path = drafts_dir / f"{s}.md"
            draft_path.write_text(summary_draft, encoding="utf-8")
            print(f"Summary draft written to: {draft_path.relative_to(root_path)}")

            # Update registry summary status to "draft"
            update_registry_summary(s, "draft", root=root_path)

        except Exception as e:
            print(f"Error generating summary for '{s}': {e}")
            exit_code = 1

    return exit_code
