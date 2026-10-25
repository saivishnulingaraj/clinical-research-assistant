"""Render a Markdown eval report and apply the CI threshold gate."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cra.evals.metrics import ItemResult

# Maps each metric to the comparison used against its threshold. "min" means the
# measured value must be >= threshold; "max" means it must be <= threshold.
_GATE_DIRECTION = {
    "retrieval_hit_rate_at_k": "min",
    "citation_correctness": "min",
    "refusal_accuracy": "min",
    "injection_safety": "min",
    "max_avg_latency_ms": "max",
    "max_avg_cost_usd": "max",
}

# Threshold key -> measured metric key.
_THRESHOLD_TO_METRIC = {
    "retrieval_hit_rate_at_k": "retrieval_hit_rate_at_k",
    "citation_correctness": "citation_correctness",
    "refusal_accuracy": "refusal_accuracy",
    "injection_safety": "injection_safety",
    "max_avg_latency_ms": "avg_latency_ms",
    "max_avg_cost_usd": "avg_cost_usd",
}


def check_thresholds(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> list[tuple[str, float, float, bool]]:
    """Return (name, measured, threshold, passed) for each gated metric."""
    rows: list[tuple[str, float, float, bool]] = []
    for tkey, limit in thresholds.items():
        metric_key = _THRESHOLD_TO_METRIC.get(tkey)
        if metric_key is None:
            continue
        measured = metrics.get(metric_key, 0.0)
        direction = _GATE_DIRECTION.get(tkey, "min")
        passed = measured >= limit if direction == "min" else measured <= limit
        rows.append((tkey, measured, float(limit), passed))
    return rows


def render_report(
    results: list[ItemResult],
    metrics: dict[str, float],
    gate_rows: list[tuple[str, float, float, bool]],
    *,
    provider: str,
    num_chunks: int,
) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_passed = all(p for *_, p in gate_rows)

    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"- Generated: {ts}")
    lines.append(f"- LLM provider: `{provider}`")
    lines.append(f"- Indexed chunks: {num_chunks}")
    lines.append(f"- Items evaluated: {len(results)}")
    lines.append(f"- Overall gate: {'✅ PASS' if all_passed else '❌ FAIL'}")
    lines.append("")

    lines.append("## Metrics vs thresholds")
    lines.append("")
    lines.append("| Metric | Measured | Threshold | Result |")
    lines.append("|---|---|---|---|")
    for name, measured, limit, passed in gate_rows:
        fmt = f"{measured:.0f}" if "latency" in name else f"{measured:.4f}"
        lim = f"{limit:.0f}" if "latency" in name else f"{limit:.4f}"
        lines.append(f"| {name} | {fmt} | {lim} | {'✅' if passed else '❌'} |")
    lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    lines.append("| ID | Type | Hit | Citation | Refused | Inj-safe | Latency ms | Cost USD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:

        def _b(v: bool) -> str:
            return "✅" if v else "—"

        hit = _b(r.hit) if r.item.expected_doc_ids else "n/a"
        cit = _b(r.citation_correct) if r.item.supporting_substrings else "n/a"
        if r.item.type == "unanswerable":
            refused = _b(r.refusal_correct)
        else:
            refused = "—" if not r.refused else "⚠️"
        inj = _b(r.injection_safe) if r.item.type == "injection" else "n/a"
        lines.append(
            f"| {r.item.id} | {r.item.type} | {hit} | {cit} | {refused} | {inj} | "
            f"{r.latency_ms:.0f} | {r.cost_usd:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(content: str, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"eval-{ts}.md"
    path.write_text(content, encoding="utf-8")
    # Also write/overwrite a stable "latest" file for easy linking.
    (reports_dir / "latest.md").write_text(content, encoding="utf-8")
    return path
