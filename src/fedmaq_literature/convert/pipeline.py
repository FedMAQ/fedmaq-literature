"""PDF conversion orchestration: Docling primary, Marker fallback."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from fedmaq_literature.convert.models import ConvertOutput
from fedmaq_literature.paths import markdown_dir, repo_root
from fedmaq_literature.registry import resolve_pdf_path, update_registry_conversion


def _qa_passed(output: ConvertOutput | None) -> bool:
    return bool(output and output.qa and output.qa.passed)


def convert_paper(
    slug: str,
    *,
    pdf_path: Path | None = None,
    force_marker: bool = False,
    skip_marker_fallback: bool = False,
    root: Path | None = None,
) -> ConvertOutput:
    """Convert a paper PDF to markdown/{slug}/paper.md with QA metadata."""
    root = root or repo_root()
    source = pdf_path or resolve_pdf_path(slug, root=root)
    docling_output: ConvertOutput | None = None
    marker_output: ConvertOutput | None = None

    if not force_marker:
        from fedmaq_literature.convert import docling as docling_backend

        try:
            docling_output = docling_backend.convert_pdf(source)
        except RuntimeError:
            docling_output = None

        if _qa_passed(docling_output):
            return _finalize(slug, docling_output, root=root)

        if skip_marker_fallback:
            if docling_output is None:
                update_registry_conversion(slug, "failed", root=root)
                raise RuntimeError(f"Docling conversion failed for slug '{slug}'")
            return _finalize(slug, docling_output, root=root, failed=True)

    if not force_marker and skip_marker_fallback:
        assert docling_output is not None
        return _finalize(slug, docling_output, root=root, failed=True)

    from fedmaq_literature.convert import marker as marker_backend

    try:
        marker_output = marker_backend.convert_pdf(source)
    except RuntimeError:
        marker_output = None

    if _qa_passed(marker_output):
        return _finalize(slug, marker_output, root=root)

    if _qa_passed(docling_output):
        return _finalize(slug, docling_output, root=root)

    chosen = _pick_best(docling_output, marker_output)
    if chosen is None:
        update_registry_conversion(slug, "failed", root=root)
        raise RuntimeError(f"All converters failed for slug '{slug}'")

    return _finalize(slug, chosen, root=root, failed=True)


def _pick_best(
    docling_output: ConvertOutput | None,
    marker_output: ConvertOutput | None,
) -> ConvertOutput | None:
    candidates = [
        output for output in (docling_output, marker_output) if output is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item.qa.char_count if item.qa else len(item.markdown),
    )


def _finalize(
    slug: str,
    output: ConvertOutput,
    *,
    root: Path,
    failed: bool = False,
) -> ConvertOutput:
    write_conversion(slug, output, root=root)
    if failed or not _qa_passed(output):
        update_registry_conversion(slug, "failed", root=root)
    else:
        update_registry_conversion(slug, "ready", root=root)
    return output


def _clean_math_block(math_content: str) -> str:
    import re

    # Match \intertext followed by curly braces
    intertext_regex = re.compile(r"\\intertext\s*\{([^}]+)\}")

    if intertext_regex.search(math_content):
        # Split equation on intertext and rebuild
        parts = []
        last_idx = 0
        for match in intertext_regex.finditer(math_content):
            start, end = match.span()
            text_content = match.group(1).strip()
            # Left part (before \intertext)
            left = math_content[last_idx:start].strip()
            if left:
                left_cleaned = _clean_math_block(left)
                if left_cleaned.strip().startswith("$$"):
                    parts.append(left_cleaned)
                else:
                    parts.append(f"$$\n{left_cleaned}\n$$")
            # The text itself
            parts.append(text_content)
            last_idx = end
        # Right part (after last \intertext)
        right = math_content[last_idx:].strip()
        if right:
            right_cleaned = _clean_math_block(right)
            if right_cleaned.strip().startswith("$$"):
                parts.append(right_cleaned)
            else:
                parts.append(f"$$\n{right_cleaned}\n$$")
        return "\n\n".join(parts)

    content = math_content.strip()

    # Strip \begin{equation} or \begin{equation*} wrappers (with optional \label)
    equation_start = re.match(
        r"^\\begin\{equation\*?\}(?:\s*\\label\{[^}]*\})?", content
    )
    if equation_start:
        start_len = equation_start.end()
        if content.endswith("\\end{equation}"):
            content = content[start_len : -len("\\end{equation}")].strip()
        elif content.endswith("\\end{equation*}"):
            content = content[start_len : -len("\\end{equation*}")].strip()

    # Deduplicate repetitive generation loop collapse patterns
    content = re.sub(r"(\\\s+){3,}", r"\\ ", content)
    content = re.sub(r"(\\quad\s*){3,}", r"\\quad ", content)
    content = re.sub(r"(\\text\s*\{\s*\}){3,}", r"", content)
    for _ in range(5):
        prev = content

        def repl(match):
            pattern = match.group(1)
            if len(pattern) >= 6:
                return pattern
            return match.group(0)

        content = re.sub(r"(.+?)(\s*\1){2,}", repl, content)
        if content == prev:
            break

    has_alignment = "&" in content or "\\\\" in content

    # Check if the content is wrapped in common alignment environments
    wrapped_envs = (
        "aligned",
        "align",
        "matrix",
        "cases",
        "split",
        "array",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
    )
    is_wrapped = any(content.startswith(f"\\begin{{{env}}}") for env in wrapped_envs)

    if has_alignment and not is_wrapped:
        if content.endswith("\\\\"):
            content = content[:-2].strip()
        return f"\\begin{{aligned}}\n{content}\n\\end{{aligned}}"

    return content


def _post_process_markdown(markdown_text: str) -> str:
    import re

    block_pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

    def replace_block(match):
        content = match.group(1)
        cleaned = _clean_math_block(content)
        if cleaned.strip().startswith("$$"):
            return cleaned
        return f"$$\n{cleaned.strip()}\n$$"

    processed = block_pattern.sub(replace_block, markdown_text)

    # Convert inline math containing & or \\ to block math wrapped in aligned
    inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")

    def replace_inline(match):
        content = match.group(1)
        if "&" in content or "\\\\" in content:
            cleaned = _clean_math_block(content)
            return f"\n$$\n{cleaned.strip()}\n$$\n"
        return f"${content}$"

    return inline_pattern.sub(replace_inline, processed)


def write_conversion(
    slug: str, output: ConvertOutput, *, root: Path | None = None
) -> Path:
    """Write paper.md and meta.yaml for a conversion result."""
    root = root or repo_root()
    out_dir = markdown_dir(root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_path = out_dir / "paper.md"
    cleaned_md = _post_process_markdown(output.markdown)
    paper_path.write_text(cleaned_md, encoding="utf-8")

    qa_dict = asdict(output.qa) if output.qa else None
    meta = {
        "slug": slug,
        "pdf": output.pdf_path.relative_to(root).as_posix(),
        "converter": output.converter,
        "converted_at": datetime.now(UTC).isoformat(),
        "page_count": output.page_count,
        "char_count": output.qa.char_count if output.qa else len(output.markdown),
        "qa": qa_dict,
        "confidence": output.raw_confidence or None,
    }
    meta_path = out_dir / "meta.yaml"
    meta_path.write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_dir
