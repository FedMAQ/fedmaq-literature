"""Tests for conversion output writing."""

from pathlib import Path

from fedmaq_literature.convert.models import ConvertOutput, QAReport
from fedmaq_literature.convert.pipeline import write_conversion


def test_write_conversion(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    pdf = tmp_path / "papers" / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "markdown").mkdir()

    output = ConvertOutput(
        markdown="# Sample\n\nBody text.",
        converter="docling",
        pdf_path=pdf,
        page_count=3,
        qa=QAReport(
            passed=True,
            confidence=0.95,
            mean_grade="good",
            low_grade="fair",
            char_count=20,
            page_count=3,
        ),
        raw_confidence={"mean_grade": "good"},
    )

    out_dir = write_conversion("sample-slug", output, root=tmp_path)
    assert (out_dir / "paper.md").read_text(encoding="utf-8") == output.markdown
    meta = (out_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "slug: sample-slug" in meta
    assert "converter: docling" in meta
