"""Paper registry parsing and slug-to-PDF resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fedmaq_literature.paths import papers_dir, registry_path, repo_root


@dataclass(frozen=True)
class PaperEntry:
    slug: str
    pdf_label: str
    baseline: str
    conversion: str
    indexing: str
    summary: str
    tags: str


TABLE_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
)


def parse_registry(path: Path | None = None) -> list[PaperEntry]:
    """Parse paper_registry.md table rows."""
    reg_file = registry_path() if path is None else path
    if not reg_file.is_file():
        return []

    entries: list[PaperEntry] = []
    for line in reg_file.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        slug, pdf_label, baseline, conversion, indexing, summary, tags = (
            field.strip() for field in match.groups()
        )
        if slug.lower() == "slug" or slug.startswith("-"):
            continue
        entries.append(
            PaperEntry(
                slug=slug,
                pdf_label=pdf_label,
                baseline=baseline,
                conversion=conversion,
                indexing=indexing,
                summary=summary,
                tags=tags,
            )
        )
    return entries


def _normalize_label(label: str) -> str:
    return label.strip().lower().rstrip(".")


def _pdf_matches_label(pdf_name: str, label: str) -> bool:
    pdf_stem = Path(pdf_name).stem.lower()
    label_norm = _normalize_label(label)
    pdf_norm = pdf_stem.rstrip(".")
    return pdf_norm.startswith(label_norm) or label_norm in pdf_norm


def resolve_pdf_path(slug: str, root: Path | None = None) -> Path:
    """Resolve slug to an existing PDF under papers/."""
    root_path = repo_root(root)
    papers = papers_dir(root_path)
    entries = parse_registry(registry_path(root_path))

    for entry in entries:
        if entry.slug != slug:
            continue
        for pdf_path in sorted(papers.glob("*.pdf")):
            if _pdf_matches_label(pdf_path.name, entry.pdf_label):
                return pdf_path
        raise FileNotFoundError(
            f"Registry slug '{slug}' maps to '{entry.pdf_label}' but no matching PDF in {papers}"
        )

    available = ", ".join(entry.slug for entry in entries) or "(registry empty)"
    raise KeyError(f"Unknown slug '{slug}'. Known slugs: {available}")


def update_registry_conversion(
    slug: str,
    status: str,
    *,
    root: Path | None = None,
) -> None:
    """Update Conversion column for a slug in paper_registry.md."""
    reg_file = registry_path(root)
    if not reg_file.is_file():
        return

    lines = reg_file.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    changed = False
    for line in lines:
        match = TABLE_ROW_RE.match(line.strip())
        if match and match.group(1).strip() == slug:
            parts = [part.strip() for part in match.groups()]
            parts[3] = status
            updated.append(f"| {' | '.join(parts)} |\n")
            changed = True
        else:
            updated.append(line if line.endswith("\n") else line + "\n")

    if changed:
        reg_file.write_text("".join(updated), encoding="utf-8")


def update_registry_indexing(
    slug: str,
    status: str,
    *,
    root: Path | None = None,
) -> None:
    """Update Indexing column for a slug in paper_registry.md."""
    reg_file = registry_path(root)
    if not reg_file.is_file():
        return

    lines = reg_file.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    changed = False
    for line in lines:
        match = TABLE_ROW_RE.match(line.strip())
        if match and match.group(1).strip() == slug:
            parts = [part.strip() for part in match.groups()]
            parts[4] = status
            updated.append(f"| {' | '.join(parts)} |\n")
            changed = True
        else:
            updated.append(line if line.endswith("\n") else line + "\n")

    if changed:
        reg_file.write_text("".join(updated), encoding="utf-8")


def update_registry_summary(
    slug: str,
    status: str,
    *,
    root: Path | None = None,
) -> None:
    """Update Summary column for a slug in paper_registry.md."""
    reg_file = registry_path(root)
    if not reg_file.is_file():
        return

    lines = reg_file.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    changed = False
    for line in lines:
        match = TABLE_ROW_RE.match(line.strip())
        if match and match.group(1).strip() == slug:
            parts = [part.strip() for part in match.groups()]
            parts[5] = status
            updated.append(f"| {' | '.join(parts)} |\n")
            changed = True
        else:
            updated.append(line if line.endswith("\n") else line + "\n")

    if changed:
        reg_file.write_text("".join(updated), encoding="utf-8")
