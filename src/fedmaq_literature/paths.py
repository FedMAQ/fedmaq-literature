"""Repository path helpers for fedmaq-literature."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return fedmaq-literature repo root (directory containing papers/)."""
    if start is not None:
        resolved = start.resolve()
        if (resolved / "papers").is_dir():
            return resolved

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "papers").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate fedmaq-literature repo root (expected pyproject.toml and papers/)."
    )


def papers_dir(root: Path | None = None) -> Path:
    return repo_root(root) / "papers"


def markdown_dir(root: Path | None = None) -> Path:
    return repo_root(root) / "markdown"


def registry_path(root: Path | None = None) -> Path:
    return repo_root(root) / ".cursor" / "project" / "paper_registry.md"
