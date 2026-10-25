"""Eval harness entry point: ``python -m cra.evals.runner``.

Runs every golden item through the full agent pipeline, scores the results,
writes a Markdown report, and exits non-zero if any metric is below its
threshold (so CI fails the PR).

By default it uses the fake LLM + hashing embeddings (no network), which is what
CI runs. Pass ``--real`` to run against real Bedrock + all-MiniLM locally.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from cra.config import Settings, get_settings
from cra.evals.dataset import load_golden_set
from cra.evals.metrics import aggregate, score_item
from cra.evals.report import check_thresholds, render_report, write_report
from cra.graph.orchestrator import Orchestrator
from cra.ingest.pipeline import run_ingest
from cra.observability.logging import configure_logging, get_logger
from cra.retrieval.service import RetrievalService

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "evals" / "golden_set.jsonl"
DEFAULT_THRESHOLDS = REPO_ROOT / "evals" / "thresholds.yaml"
DEFAULT_REPORTS = REPO_ROOT / "evals" / "reports"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


def _build_orchestrator(settings: Settings, *, fake: bool, index_dir: Path) -> Orchestrator:
    """Build (ingesting first if needed) an orchestrator for the eval run."""
    index_file = index_dir / "index.faiss"
    if not index_file.exists():
        run_ingest(SAMPLE_DIR, settings, fake_embeddings=fake, index_dir=index_dir)
    retrieval = RetrievalService.from_index_dir(index_dir, settings, fake_embeddings=fake)
    return Orchestrator(settings, retrieval)


def run_eval(
    golden_path: Path,
    thresholds_path: Path,
    reports_dir: Path,
    *,
    real: bool,
) -> bool:
    """Execute the eval and return True if all thresholds pass."""
    configure_logging("WARNING")
    logger = get_logger("cra.evals")

    get_settings.cache_clear()
    settings = get_settings()
    fake = not real
    if fake:
        # Ensure the fake path regardless of ambient env. The hashing embedder
        # produces lower cosine scores than a real model, so lower the
        # confidence gate to a value appropriate for it (answerable items
        # should answer rather than spuriously refuse).
        settings = settings.model_copy(
            update={
                "llm_provider": "fake",
                # The hashing embedder is a weak section-level ranker, so widen
                # the candidate set and lower the confidence gate to values
                # appropriate for it. Real embeddings use the configured
                # defaults and perform far better.
                "confidence_threshold": 0.09,
                "retrieval_top_k": 8,
            }
        )

    # Use a dedicated eval index dir so we don't clobber a user's main index.
    index_dir = settings.faiss_index_dir.parent / "eval_index"
    orchestrator = _build_orchestrator(settings, fake=fake, index_dir=index_dir)

    items = load_golden_set(golden_path)
    logger.warning("running %d golden items (provider=%s)", len(items), settings.llm_provider)

    results = []
    for item in items:
        start = time.perf_counter()
        res = orchestrator.answer(item.question, request_id=f"eval-{item.id}")
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(score_item(item, res.state, res.usage, latency_ms))

    metrics = aggregate(results)
    thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    gate_rows = check_thresholds(metrics, thresholds)

    report = render_report(
        results,
        metrics,
        gate_rows,
        provider=settings.llm_provider,
        num_chunks=orchestrator._retrieval.num_chunks,  # noqa: SLF001
    )
    out_path = write_report(report, reports_dir)

    all_passed = all(p for *_, p in gate_rows)
    print(report)
    print(f"\nReport written to: {out_path}")
    print(f"GATE: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cra-eval", description="Run the eval harness.")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument(
        "--real",
        action="store_true",
        help="Use real Bedrock + all-MiniLM instead of the fake LLM.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    passed = run_eval(args.golden, args.thresholds, args.reports_dir, real=args.real)
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
