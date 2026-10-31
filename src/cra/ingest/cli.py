"""Command-line entrypoint for building the FAISS index.

Usage::

    python -m cra.ingest.cli --source data/sample
    python -m cra.ingest.cli --source data/sample --fake-embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cra.config import get_settings
from cra.ingest.pipeline import run_ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cra-ingest",
        description="Load documents, chunk them, embed, and build a FAISS index.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sample"),
        help="Directory of source documents (.md/.txt). Default: data/sample",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Where to write the index. Default: CRA_FAISS_INDEX_DIR setting.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (characters).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap (characters).",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use the deterministic hashing embedder (no model download).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.chunk_size is not None:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap = args.chunk_overlap

    source: Path = args.source
    if not source.exists():
        print(f"error: source directory does not exist: {source}", file=sys.stderr)
        return 2

    result = run_ingest(
        source,
        settings,
        fake_embeddings=args.fake_embeddings,
        index_dir=args.index_dir,
    )
    print(
        f"Ingested {result.num_documents} documents into "
        f"{result.num_chunks} chunks.\n"
        f"Index written to: {result.index_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
