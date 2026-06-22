"""Literature summary approval and promotion workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

from fedmaq_literature.paths import repo_root, registry_path
from fedmaq_literature.registry import parse_registry, update_registry_summary


def run_approve(
    slug: str | None = None,
    *,
    root: Path | None = None,
) -> int:
    """Approve summary drafts and promote them to the main summaries/ folder.

    If slug is specified, approve only that summary.
    If slug is None, approve all summaries currently marked as 'draft'.
    """
    root_path = repo_root(root)
    entries = parse_registry(registry_path(root_path))

    if slug:
        target_entries = [e for e in entries if e.slug == slug]
        if not target_entries:
            print(f"Error: Slug '{slug}' not found in registry.")
            return 1
        entry = target_entries[0]
        if entry.summary != "draft":
            print(
                f"Warning: Summary for '{slug}' is in status '{entry.summary}', not 'draft'."
            )
            # We still check if the draft file exists to allow forced approval
    else:
        target_entries = [e for e in entries if e.summary == "draft"]

    if not target_entries:
        if slug:
            print(f"No summary draft exists to approve for slug '{slug}'.")
        else:
            print("No pending summary drafts to approve.")
        return 0

    drafts_dir = root_path / "summaries" / "drafts"
    approved_dir = root_path / "summaries"
    approved_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for entry in target_entries:
        s = entry.slug
        draft_file = drafts_dir / f"{s}.md"
        approved_file = approved_dir / f"{s}.md"

        if not draft_file.is_file():
            print(
                f"Error: Draft file for '{s}' not found at {draft_file}. Cannot approve."
            )
            exit_code = 1
            continue

        try:
            print(f"Approving summary for '{s}'...")
            # Copy to approved summaries folder
            shutil.copy2(draft_file, approved_file)
            print(f"Promoted to: {approved_file.relative_to(root_path)}")

            # Delete the draft
            draft_file.unlink()

            # Update registry summary status to "approved"
            update_registry_summary(s, "approved", root=root_path)

        except Exception as e:
            print(f"Error approving summary for '{s}': {e}")
            exit_code = 1

    return exit_code
