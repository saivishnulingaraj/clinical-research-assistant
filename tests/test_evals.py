"""Tests for the eval harness: dataset, metrics, thresholds, end-to-end gate."""

from __future__ import annotations

from pathlib import Path

from cra.evals.dataset import load_golden_set
from cra.evals.report import check_thresholds
from cra.evals.runner import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evals" / "golden_set.jsonl"
THRESHOLDS = REPO_ROOT / "evals" / "thresholds.yaml"


def test_golden_set_composition() -> None:
    items = load_golden_set(GOLDEN)
    assert len(items) == 25
    by_type = dict.fromkeys(("answerable", "unanswerable", "injection"), 0)
    for it in items:
        by_type[it.type] += 1
    assert by_type["unanswerable"] == 5
    assert by_type["injection"] == 3
    assert by_type["answerable"] == 17
    # Injection items declare forbidden content to check for leakage.
    for it in items:
        if it.type == "injection":
            assert it.forbidden_substrings


def test_check_thresholds_direction() -> None:
    metrics = {
        "retrieval_hit_rate_at_k": 0.9,
        "citation_correctness": 0.4,
        "refusal_accuracy": 1.0,
        "injection_safety": 1.0,
        "avg_latency_ms": 10.0,
        "avg_cost_usd": 0.001,
    }
    thresholds = {
        "retrieval_hit_rate_at_k": 0.55,
        "citation_correctness": 0.25,
        "refusal_accuracy": 0.8,
        "injection_safety": 1.0,
        "max_avg_latency_ms": 2000.0,
        "max_avg_cost_usd": 0.05,
    }
    rows = check_thresholds(metrics, thresholds)
    assert all(passed for *_, passed in rows)

    # A latency over the ceiling must fail (max-direction).
    metrics["avg_latency_ms"] = 9999.0
    rows = check_thresholds(metrics, thresholds)
    failed = [name for name, _, _, passed in rows if not passed]
    assert "max_avg_latency_ms" in failed


def test_end_to_end_eval_gate_passes(tmp_path: Path) -> None:
    passed = run_eval(GOLDEN, THRESHOLDS, tmp_path / "reports", real=False)
    assert passed is True
    # A report (and a 'latest.md') must have been written.
    assert (tmp_path / "reports" / "latest.md").exists()
