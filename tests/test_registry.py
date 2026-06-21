"""Tests for paper registry parsing and updates."""

from pathlib import Path

from fedmaq_literature.registry import (
    PaperEntry,
    parse_registry,
    resolve_pdf_path,
    update_registry_conversion,
    update_registry_indexing,
)


REGISTRY = """# Paper Registry

| Slug | PDF | Baseline | Conversion | Indexing | Summary | Tags |
| ---- | --- | -------- | ---------- | -------- | ------- | ---- |
| he-2025-dynfed | He et al. - 2025 - DynFed | DynFed | none | none | none | sota |
"""


def test_parse_registry(tmp_path: Path) -> None:
    reg = tmp_path / "paper_registry.md"
    reg.write_text(REGISTRY, encoding="utf-8")
    entries = parse_registry(reg)
    assert entries == [
        PaperEntry(
            slug="he-2025-dynfed",
            pdf_label="He et al. - 2025 - DynFed",
            baseline="DynFed",
            conversion="none",
            indexing="none",
            summary="none",
            tags="sota",
        )
    ]


def test_resolve_pdf_path(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    pdf = papers / "He et al. - 2025 - DynFed Adaptive Federated Learning.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    reg_dir = tmp_path / ".cursor" / "project"
    reg_dir.mkdir(parents=True)
    (reg_dir / "paper_registry.md").write_text(REGISTRY, encoding="utf-8")

    resolved = resolve_pdf_path("he-2025-dynfed", root=tmp_path)
    assert resolved == pdf


def test_pdf_label_ellipsis_match() -> None:
    from fedmaq_literature.registry import _pdf_matches_label

    pdf = "Hinton et al. - 2015 - Distilling the Knowledge in a Neural Network.pdf"
    label = "Hinton et al. - 2015 - Distilling the Knowledge..."
    assert _pdf_matches_label(pdf, label)


def test_update_registry_conversion(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    reg_dir = tmp_path / ".cursor" / "project"
    reg_dir.mkdir(parents=True)
    reg = reg_dir / "paper_registry.md"
    reg.write_text(REGISTRY, encoding="utf-8")

    update_registry_conversion("he-2025-dynfed", "ready", root=tmp_path)
    text = reg.read_text(encoding="utf-8")
    assert (
        "| he-2025-dynfed | He et al. - 2025 - DynFed | DynFed | ready | none | none | sota |"
        in text
    )


def test_update_registry_indexing(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    reg_dir = tmp_path / ".cursor" / "project"
    reg_dir.mkdir(parents=True)
    reg = reg_dir / "paper_registry.md"
    reg.write_text(REGISTRY, encoding="utf-8")

    update_registry_indexing("he-2025-dynfed", "ready", root=tmp_path)
    text = reg.read_text(encoding="utf-8")
    assert (
        "| he-2025-dynfed | He et al. - 2025 - DynFed | DynFed | none | ready | none | sota |"
        in text
    )
