"""fedmaq-lit CLI — convert, ingest, query, summarize, approve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedmaq_literature.registry import parse_registry


def _cmd_convert(args: argparse.Namespace) -> int:
    from fedmaq_literature.convert import convert_paper
    from fedmaq_literature.registry import parse_registry
    from fedmaq_literature.paths import markdown_dir

    if not args.slug and not getattr(args, "all", False):
        print("error: either --slug or --all is required", file=sys.stderr)
        return 1

    force_convert = getattr(args, "force_convert", False)

    if args.slug:
        entries = parse_registry()
        entry = next((e for e in entries if e.slug == args.slug), None)
        if entry and entry.conversion == "ready" and not force_convert:
            if (markdown_dir() / args.slug / "paper.md").is_file():
                print(
                    f"Paper '{args.slug}' is already converted and ready. Skipping conversion."
                )
                return 0
        slugs = [args.slug]
    else:
        entries = parse_registry()
        if force_convert:
            slugs = [entry.slug for entry in entries]
        else:
            slugs = [entry.slug for entry in entries if entry.conversion != "ready"]

        if not slugs:
            print("All papers are already converted and ready.")
            return 0

    exit_code = 0
    for slug in slugs:
        print(f"Processing '{slug}'...")
        pdf_path = Path(args.pdf).resolve() if (args.pdf and args.slug) else None
        try:
            output = convert_paper(
                slug,
                pdf_path=pdf_path,
                force_marker=args.force_marker,
                skip_marker_fallback=args.no_marker_fallback,
            )
            qa = output.qa
            print(f"converted {slug} with {output.converter}")
            if qa:
                status = "passed" if qa.passed else "failed"
                print(f"qa: {status} ({qa.char_count} chars)")
                if qa.reasons:
                    print(f"qa notes: {', '.join(qa.reasons)}")
            if not qa or not qa.passed:
                exit_code = 2
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            print(f"error converting {slug}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


def _cmd_ingest(args: argparse.Namespace) -> int:
    code = _cmd_convert(args)
    if code != 0 or args.convert_only:
        if args.convert_only and code == 0:
            print("convert-only: skipping Chroma indexing.")
        return code

    from fedmaq_literature.ingest import run_ingest

    slug = args.slug if args.slug else None
    device = getattr(args, "device", None)
    return run_ingest(slug=slug, device=device)


def _cmd_list_slugs(_: argparse.Namespace) -> int:
    entries = parse_registry()
    if not entries:
        print("No slugs found in paper_registry.md")
        return 1
    for entry in entries:
        print(f"{entry.slug}\t{entry.conversion}\t{entry.pdf_label}")
    return 0


def _add_convert_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slug", help="Paper slug from paper_registry.md")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all pending papers in the registry",
    )
    parser.add_argument(
        "--pdf", help="Override PDF path (default: resolve from registry)"
    )
    parser.add_argument(
        "--force-marker",
        action="store_true",
        help="Skip Docling and convert with Marker only",
    )
    parser.add_argument(
        "--no-marker-fallback",
        action="store_true",
        help="Do not fall back to Marker when Docling QA fails",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force PDF conversion even if the paper is already marked ready",
    )


def _cmd_summarize(args: argparse.Namespace) -> int:
    from fedmaq_literature.workflows.summarize import run_summarize

    return run_summarize(
        slug=args.slug,
        model=args.model,
        force=args.force_summarize,
    )


def _cmd_approve(args: argparse.Namespace) -> int:
    from fedmaq_literature.workflows.approve import run_approve

    return run_approve(slug=args.slug)


def _cmd_query(args: argparse.Namespace) -> int:
    from fedmaq_literature.workflows.query import run_query

    return run_query(
        query_str=args.query,
        model=args.model,
        limit=args.limit,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedmaq-lit")
    sub = parser.add_subparsers(dest="command", required=True)

    convert_parser = sub.add_parser("convert", help="Convert PDF to markdown with QA")
    _add_convert_flags(convert_parser)
    convert_parser.set_defaults(handler=_cmd_convert)

    ingest_parser = sub.add_parser("ingest", help="Convert PDF, QA, index into Chroma")
    _add_convert_flags(ingest_parser)
    ingest_parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Run conversion only; skip embedding/indexing",
    )
    ingest_parser.add_argument(
        "--device",
        help="Device to run embeddings on (e.g., cuda, cpu)",
    )
    ingest_parser.set_defaults(handler=_cmd_ingest)

    list_parser = sub.add_parser("list-slugs", help="List slugs from paper_registry.md")
    list_parser.set_defaults(handler=_cmd_list_slugs)

    # Wire up query command
    query_parser = sub.add_parser(
        "query", help="Query the Chroma DB and synthesize an answer"
    )
    query_parser.add_argument(
        "--query", required=True, help="Research question or query string"
    )
    query_parser.add_argument(
        "--model", help="OpenRouter model name to use for synthesis"
    )
    query_parser.add_argument(
        "--limit", type=int, default=5, help="Number of passages to retrieve"
    )
    query_parser.add_argument(
        "--device", help="Device to run embeddings on (e.g., cuda, cpu)"
    )
    query_parser.set_defaults(handler=_cmd_query)

    # Wire up summarize command
    summarize_parser = sub.add_parser(
        "summarize", help="Generate paper summary draft via LLM"
    )
    summarize_parser.add_argument(
        "--slug", help="Paper slug (omit to summarize all pending)"
    )
    summarize_parser.add_argument(
        "--model", help="OpenRouter model name to use for summary"
    )
    summarize_parser.add_argument(
        "--force-summarize",
        action="store_true",
        help="Force summary generation even if draft/approval exists",
    )
    summarize_parser.set_defaults(handler=_cmd_summarize)

    # Wire up approve command
    approve_parser = sub.add_parser(
        "approve", help="Approve and promote a summary draft"
    )
    approve_parser.add_argument(
        "--slug", help="Paper slug (omit to approve all drafts)"
    )
    approve_parser.set_defaults(handler=_cmd_approve)

    # Keep remaining stubs
    for name in ("approve-synthesis",):
        stub = sub.add_parser(name, help=f"{name} (not yet implemented)")
        stub.set_defaults(handler=_stub_handler(name))

    args = parser.parse_args(argv)
    return args.handler(args)


def _stub_handler(command: str):
    def handler(_: argparse.Namespace) -> int:
        print(f"fedmaq-lit {command}: not yet implemented (see HANDOFF queue).")
        return 0

    return handler


if __name__ == "__main__":
    sys.exit(main())
