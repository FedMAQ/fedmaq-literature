"""Tests for conversion QA heuristics."""

from fedmaq_literature.convert.qa import assess_markdown, grade_at_least


def test_grade_at_least() -> None:
    assert grade_at_least("excellent", "good")
    assert grade_at_least("good", "good")
    assert not grade_at_least("fair", "good")


def test_assess_markdown_passes_with_good_grades() -> None:
    markdown = "# Title\n\n" + ("word " * 400)
    report = assess_markdown(
        markdown,
        mean_grade="good",
        low_grade="fair",
        confidence=0.9,
        page_count=10,
    )
    assert report.passed
    assert report.reasons == ()


def test_assess_markdown_fails_on_short_content() -> None:
    report = assess_markdown(
        "# Title\n\nshort", mean_grade="excellent", low_grade="good"
    )
    assert not report.passed
    assert any("char_count" in reason for reason in report.reasons)


def test_assess_markdown_fails_on_low_docling_grade() -> None:
    markdown = "# Title\n\n" + ("content " * 400)
    report = assess_markdown(markdown, mean_grade="poor", low_grade="poor")
    assert not report.passed
    assert any("mean_grade" in reason for reason in report.reasons)
