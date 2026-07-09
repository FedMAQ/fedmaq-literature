"""One-shot migration: build kg/papers/ OKF Paper nodes from summaries + meta.

Reads the existing per-paper artifacts (paper_registry.md, markdown/{slug}/meta.yaml,
markdown/{slug}/paper.md, summaries/{slug}.md) and emits one OKF concept document per
paper into kg/papers/{slug}.md.

This is a *migration tool*, not a living regenerator: it reads summaries/, which the
OKF restructure deletes afterward. Kept in-repo as provenance for how the initial
paper nodes were assembled.

Usage:
    uv run python scripts/build_kg_papers.py            # write all summarized papers
    uv run python scripts/build_kg_papers.py --slug X   # single paper (preview)
    uv run python scripts/build_kg_papers.py --stdout --slug X   # print, don't write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from fedmaq_literature.paths import markdown_dir, papers_dir, repo_root
from fedmaq_literature.registry import parse_registry, resolve_pdf_path

# --- batch folder -> canonical tag -------------------------------------------
BATCH_TAGS = {
    "00 Surveys": ["survey"],
    "01 FL, Heterogeneity": ["fl-core"],
    "02 Quantization": ["quantization"],
    "03 KD": ["kd"],
    "04 Q+KD": ["quantization", "kd"],
    "05 Applications": ["application"],
}

# --- method/system names for cross-linking (NO surnames; corpus has li x3,
#     wang x2, he x2, so surname triggers would fabricate edges) --------------
METHOD_TRIGGERS: dict[str, list[str]] = {
    "mcmahan-2017-fedavg": ["FedAvg", "FederatedAveraging", "FedSGD"],
    "li-2020-fedprox": ["FedProx"],
    "karimireddy-2020-scaffold": ["SCAFFOLD"],
    "acar-2021-feddyn": ["FedDyn"],
    "li-2021-moon": ["MOON", "Model-Contrastive"],
    "tan-2022-fedproto": ["FedProto"],
    "wang-2020-fednova": ["FedNova"],
    "alistarh-2017-qsgd": ["QSGD"],
    "bernstein-2018-signsgd": ["signSGD", "SignSGD"],
    "honig-2022-dadaquant": ["DAdaQuant"],
    "reisizadeh-2020-fedpaq": ["FedPAQ"],
    "liu-2023-adagq": ["AdaGQ"],
    "cui-2026-laq-hc": ["LAQ-HC"],
    "li-2019-fedmd": ["FedMD"],
    "song-2024-feddistill": ["FedDistill"],
    "wu-2022-fedkd": ["FedKD"],
    "sattler-2022-cfd": ["CFD", "Compressed Federated Distillation"],
    "he-2025-dynfed": ["DynFed"],
    "wang-2026-adadq-kd": ["AdaDQ-KD", "AdaDQ"],
    "he-2025-feddt": ["FedDT"],
    "lin-2020-feddf": ["FedDF"],
    "zhu-2021-fedgen": ["FedGen"],
}


# --- canonical titles for papers whose PDF filename dropped a colon or was
#     filesystem-truncated mid-word (Windows 255-char limit) -----------------
TITLE_OVERRIDES: dict[str, str] = {
    "alterkawi-2025-smart-cities-review": "Federated Learning for Smart Cities: A Thematic Review of Challenges and Approaches",
    "cajas-ordonez-2025-edge-computing-survey": "Intelligent Edge Computing and Machine Learning: A Survey of Optimization and Applications",
    "he-2025-dynfed": "DynFed: Adaptive Federated Learning via Quantization-Aware Knowledge Distillation",
    "he-2025-feddt": "FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression",
    "honig-2022-dadaquant": "DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning",
    "jimenez-2024-non-iid-survey": "Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions",
    "joseph-2026-air-quality": "Air Quality Prediction Using Communication-Efficient Federated Learning with Compressed Deep Learning Models",
    "li-2019-fedmd": "FedMD: Heterogenous Federated Learning via Model Distillation",
    "liu-2023-adagq": "Communication-Efficient Federated Learning for Heterogeneous Edge Devices Based on Adaptive Gradient Quantization",
    "qin-2025-kd-survey": "Knowledge Distillation in Federated Learning: A Survey on Long Lasting Challenges and New Solutions",
    "reisizadeh-2020-fedpaq": "FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization",
    "richter-2024-electric-load": "Advancing Electric Load Forecasting: Leveraging Federated Learning for Distributed, Non-Stationary, and Discontinuous Time Series",
    "sattler-2022-cfd": "CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding",
    "song-2024-feddistill": "FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning",
    "sravanthi-2025-energy-management": "Federated Learning-Based Energy Management Systems for Privacy-Preserving Demand Forecasting in Smart Cities",
    "thangakrishnan-2025-spatiotemporal-fl": "Spatiotemporal Federated Learning for Privacy-Preserving Load Forecasting and Appliance Scheduling in Smart City Homes",
    "wang-2026-adadq-kd": "AdaDQ-KD: An Adaptive Dithering Quantization with Knowledge Distillation in Privacy-Preserving Federated Learning",
    # net-new papers (no summary; nodes hand-authored, but titles centralized here)
    "karimireddy-2020-scaffold": "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
    "li-2021-moon": "Model-Contrastive Federated Learning",
    "tan-2022-fedproto": "FedProto: Federated Prototype Learning across Heterogeneous Clients",
    "wang-2020-fednova": "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization",
    "alistarh-2017-qsgd": "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding",
    "bernstein-2018-signsgd": "signSGD: Compressed Optimisation for Non-Convex Problems",
    "jeong-2023-feddistill-aug": "Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data",
    "lin-2020-feddf": "Ensemble Distillation for Robust Model Fusion in Federated Learning",
    "zhu-2021-fedgen": "Data-Free Knowledge Distillation for Heterogeneous Federated Learning",
    "acar-2021-feddyn": "Federated Learning Based on Dynamic Regularization",
}


# Papers whose existing summary is unusable for a mechanical migration (e.g.
# written in the wrong language) — routed to hand-authoring from paper.md.
HAND_AUTHOR: set[str] = {"joseph-2026-air-quality"}  # summary is entirely Chinese


def _read_body_file(path: Path) -> tuple[str, str]:
    """Hand-authored body file: a leading 'description: ...' line, then the
    markdown body. Returns (description, body)."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^description:\s*(.+?)\s*\n", text)
    if not m:
        raise ValueError(f"{path} must start with a 'description:' line")
    description = m.group(1).strip().strip('"')
    body = text[m.end():].strip()
    return description, body


def _yaml_scalar(value: str) -> str:
    """Double-quote a free-text scalar so colons/specials stay valid YAML."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _load_meta(slug: str, root: Path) -> dict:
    meta_path = markdown_dir(root) / slug / "meta.yaml"
    if not meta_path.is_file():
        return {}
    return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}


_JOURNAL_JUNK = re.compile(
    r"^(article|review( article)?|journal|international journal|sustainable|proceedings)\b",
    re.IGNORECASE,
)


def _title_from_pdf(pdf: Path | None) -> str | None:
    """Human-curated title from the PDF filename: '<Author> - <year> - <Title>'
    or '<Author> - <Title>'. Windows filenames strip ':' so colons are absent,
    and very long titles may be filesystem-truncated — both fixed by hand later."""
    if pdf is None:
        return None
    parts = [p.strip() for p in pdf.stem.split(" - ")]
    if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[1]):
        return " - ".join(parts[2:]).strip()
    if len(parts) == 2:
        return parts[1].strip()
    return None


def _paper_title(slug: str, root: Path, pdf: Path | None, fallback: str) -> str:
    # Curated override wins (fixes dropped colons / filesystem truncation);
    # else prefer the PDF filename; fall back to paper.md's first heading only
    # when it is not obvious conversion junk (journal name / "ARTICLE" / "REVIEW").
    if slug in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[slug]
    from_pdf = _title_from_pdf(pdf)
    if from_pdf:
        return from_pdf
    paper_path = markdown_dir(root) / slug / "paper.md"
    if paper_path.is_file():
        for line in paper_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^#{1,3}\s+(.*\S)\s*$", line)
            if m and not _JOURNAL_JUNK.match(m.group(1)):
                return m.group(1).strip()
    return fallback


def _resolve_pdf(slug: str, root: Path) -> Path | None:
    """Actual current PDF path on disk (meta.yaml paths for older papers are
    stale — recorded before PDFs were reorganized into batch subfolders)."""
    try:
        return resolve_pdf_path(slug, root=root)
    except (FileNotFoundError, KeyError):
        return None


def _batch_from_pdf(pdf: Path | None, root: Path) -> str | None:
    # papers/01 FL, Heterogeneity/McMahan ...pdf -> "01 FL, Heterogeneity"
    if pdf is None:
        return None
    try:
        rel = pdf.relative_to(papers_dir(root))
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) >= 2 else None


def _authors_from_label(pdf_label: str) -> str:
    # "McMahan et al. - 2017 - Communication..." -> "McMahan et al."
    return pdf_label.split(" - ")[0].strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _clean_body(summary: str, title: str) -> str:
    """Strip LLM preamble before the first heading and trailing chatty notes."""
    lines = summary.splitlines()

    # Drop conversational opener lines before the first markdown heading or bold.
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s", line) or line.strip().startswith("**"):
            start = i
            break
    body = "\n".join(lines[start:]).strip()

    # Strip a leading horizontal rule if the summary opened with "---\n#".
    body = re.sub(r"^-{3,}\s*\n", "", body).strip()

    # Drop trailing italic "_This summary is intended..._" footer lines.
    body = re.sub(r"\n+_[^\n]*summary[^\n]*_\s*$", "", body, flags=re.IGNORECASE).strip()

    # Drop a trailing horizontal rule left dangling before the appended sections.
    body = re.sub(r"\n-{3,}\s*$", "", body).strip()

    # Drop a leading heading ONLY when it restates the paper title (or is an
    # explicit "Research Summary" banner) — never a real section heading like
    # "### 1. Overview". The frontmatter title already serves as the node H1.
    lines = body.splitlines()
    if lines:
        head_m = re.match(r"^#{1,3}\s+(.*\S)\s*$", lines[0])
        if head_m:
            htext = _norm(head_m.group(1))
            title_words = _norm(title).split()
            title_head = " ".join(title_words[:4])
            is_banner = bool(re.match(r"(research |paper )?summary", htext))
            is_title = bool(title_head) and title_head in htext
            if is_banner or is_title:
                body = "\n".join(lines[1:]).lstrip()
    body = re.sub(r"^#\s+", "## ", body, count=1)
    return body


def _description(body: str, title: str) -> str:
    """Best-effort one-sentence description from the summary body."""
    # Prefer the sentence following a "Core Problem" marker.
    m = re.search(r"Core Problem\*{0,2}:?\**\s*(.+)", body)
    text = m.group(1) if m else ""
    if not text:
        # First non-heading, non-bold-label paragraph sentence.
        for para in re.split(r"\n\s*\n", body):
            p = para.strip()
            if p.startswith("#") or p.startswith("|"):
                continue
            text = re.sub(r"^\*+|\*+$", "", p).strip()
            if text:
                break
    text = re.sub(r"\*\*|`|\$", "", text).strip()
    sentence = re.split(r"(?<=[.!?])\s", text)[0] if text else title
    sentence = sentence.strip()
    if len(sentence) > 220:
        sentence = sentence[:217].rsplit(" ", 1)[0] + "..."
    return sentence or title


def _related(slug: str, body: str) -> list[str]:
    hits = []
    for other, triggers in METHOD_TRIGGERS.items():
        if other == slug:
            continue
        for trig in triggers:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(trig)}(?![A-Za-z0-9])", body):
                hits.append(other)
                break
    return hits


def build_node(slug: str, root: Path, entries) -> str | None:
    entry = next((e for e in entries if e.slug == slug), None)
    if entry is None:
        return None
    meta = _load_meta(slug, root)
    pdf = _resolve_pdf(slug, root)
    pdf_repr = pdf.relative_to(root).as_posix() if pdf else entry.pdf_label
    title = _paper_title(slug, root, pdf, entry.pdf_label)
    authors = _authors_from_label(entry.pdf_label)
    year_m = re.search(r"-(\d{4})-", slug)
    year = year_m.group(1) if year_m else ""
    batch = _batch_from_pdf(pdf, root)
    tags: list[str] = list(BATCH_TAGS.get(batch or "", []))
    for t in re.split(r",\s*", entry.tags):
        t = t.strip()
        if t and t != "—" and t not in tags:
            tags.append(t)

    ts = meta.get("converted_at", "")
    ts = re.sub(r"\.\d+\+00:00$", "Z", ts) if isinstance(ts, str) else ""

    # Body source: a hand-authored body file (net-new papers + papers whose
    # existing summary is unusable) takes precedence; otherwise the approved
    # summary is cleaned and migrated mechanically.
    body_file = root / "scripts" / "kg_bodies" / f"{slug}.md"
    summary_path = root / "summaries" / f"{slug}.md"
    if body_file.is_file():
        description, body = _read_body_file(body_file)
    elif slug in HAND_AUTHOR or not summary_path.is_file():
        return None  # awaiting hand-authored body
    else:
        body = _clean_body(summary_path.read_text(encoding="utf-8"), title)
        description = _description(body, title)
    related = _related(slug, body)

    fm: list[str] = ["---", "type: Paper", f"title: {_yaml_scalar(title)}"]
    fm.append(f"description: {_yaml_scalar(description)}")
    fm.append(f"authors: {_yaml_scalar(authors)}")
    if year:
        fm.append(f"year: {year}")
    fm.append(f"bibkey: {slug}")
    if entry.baseline and entry.baseline != "—":
        fm.append(f"baseline: {entry.baseline}")
    fm.append(f"tags: [{', '.join(tags)}]")
    fm.append(f"resource: markdown/{slug}/paper.md")
    if ts:
        fm.append(f"timestamp: {ts}")
    fm.append("---")

    out = ["\n".join(fm), "", body]

    if related:
        out.append("\n# Related\n")
        rel_lines = []
        for r in related:
            r_meta = next((e for e in entries if e.slug == r), None)
            r_title = _paper_title(
                r, root, _resolve_pdf(r, root), r_meta.pdf_label if r_meta else r
            )
            rel_lines.append(f"- [{r_title}](/papers/{r}.md)")
        out.append("\n".join(rel_lines))

    out.append("\n# Citations\n")
    out.append(
        f"[1] Full-text conversion: [markdown/{slug}/paper.md](markdown/{slug}/paper.md)\n"
        f"[2] Source PDF: `{pdf_repr}`"
    )

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="single slug (default: all summarized)")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args(argv)

    root = repo_root()
    entries = parse_registry()
    out_dir = root / "kg" / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)

    slugs = [args.slug] if args.slug else [e.slug for e in entries]
    written, skipped = [], []
    for slug in slugs:
        node = build_node(slug, root, entries)
        if node is None:
            skipped.append(slug)
            continue
        if args.stdout:
            print(node)
        else:
            (out_dir / f"{slug}.md").write_text(node, encoding="utf-8")
        written.append(slug)

    if not args.stdout:
        print(f"wrote {len(written)} paper nodes to kg/papers/")
        if skipped:
            print(f"skipped {len(skipped)} (no summary; hand-author): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
