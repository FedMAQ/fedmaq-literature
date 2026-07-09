"""fedmaq-lit CLI — PDF conversion for the FedMAQ literature corpus.

Converts source PDFs into markdown/{slug}/paper.md (Docling primary, Marker
fallback) with QA. Knowledge is curated separately as OKF nodes under kg/;
see kg/index.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedmaq_literature.registry import parse_registry


def _cmd_convert(args: argparse.Namespace) -> int:
    from fedmaq_literature.convert import convert_paper
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedmaq-lit")
    sub = parser.add_subparsers(dest="command", required=True)

    convert_parser = sub.add_parser("convert", help="Convert PDF to markdown with QA")
    _add_convert_flags(convert_parser)
    convert_parser.set_defaults(handler=_cmd_convert)

    list_parser = sub.add_parser("list-slugs", help="List slugs from paper_registry.md")
    list_parser.set_defaults(handler=_cmd_list_slugs)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
