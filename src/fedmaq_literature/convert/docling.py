"""Docling PDF-to-markdown adapter."""

from __future__ import annotations

import os
from pathlib import Path

from fedmaq_literature.convert.models import ConvertOutput, QAReport
from fedmaq_literature.convert.qa import assess_markdown

# Windows often lacks symlink privileges for the HF cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def _confidence_fields(result) -> tuple[str | None, str | None, float | None, dict]:
    confidence = getattr(result, "confidence", None)
    if confidence is None:
        return None, None, None, {}

    mean_grade = getattr(confidence, "mean_grade", None)
    low_grade = getattr(confidence, "low_grade", None)
    mean_score = getattr(confidence, "mean_score", None)

    if mean_grade is not None and hasattr(mean_grade, "value"):
        mean_grade = mean_grade.value
    if low_grade is not None and hasattr(low_grade, "value"):
        low_grade = low_grade.value
    if mean_grade is not None:
        mean_grade = str(mean_grade)
    if low_grade is not None:
        low_grade = str(low_grade)

    raw = {
        "mean_grade": mean_grade,
        "low_grade": low_grade,
        "mean_score": float(mean_score) if mean_score is not None else None,
    }
    return mean_grade, low_grade, raw["mean_score"], raw


def convert_pdf(pdf_path: Path) -> ConvertOutput:
    """Convert a PDF with Docling and run QA on the markdown output."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            ConversionStatus,
            DocumentConverter,
            PdfFormatOption,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Run: uv sync --extra convert"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = True
    pipeline_options.accelerator_options.num_threads = 1
    pipeline_options.ocr_batch_size = 1
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(
            f"Docling conversion failed for {pdf_path.name}: {exc}"
        ) from exc

    if result.status == ConversionStatus.FAILURE:
        raise RuntimeError(f"Docling conversion failed for {pdf_path.name}")

    markdown = result.document.export_to_markdown()
    page_count = len(list(result.document.pages)) if result.document.pages else None
    mean_grade, low_grade, mean_score, raw = _confidence_fields(result)

    qa: QAReport = assess_markdown(
        markdown,
        mean_grade=mean_grade,
        low_grade=low_grade,
        confidence=mean_score,
        page_count=page_count,
    )

    return ConvertOutput(
        markdown=markdown,
        converter="docling",
        pdf_path=pdf_path,
        page_count=page_count,
        qa=qa,
        raw_confidence=raw,
    )
