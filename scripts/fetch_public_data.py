#!/usr/bin/env python3
"""Download a sample of public ClinicalTrials.gov records into data/raw/.

This fetches real, public study records from the ClinicalTrials.gov v2 REST API
and writes each as a Markdown file using the same section layout as the
synthetic corpus, so the existing ingestion pipeline can index them unchanged.

The output directory (``data/raw/``) is gitignored — this data is NOT committed.
Run it locally::

    python scripts/fetch_public_data.py --max 300 --query "diabetes"

Notes
-----
* ClinicalTrials.gov data is public domain / U.S. Government work. Even so, this
  project only *commits* the synthetic corpus; fetched records stay local.
* Be polite to the API: it paginates 100 records per page and this script
  sleeps briefly between pages.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

API_URL = "https://clinicaltrials.gov/api/v2/studies"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"


def _get(d: dict, *keys: str, default: str = "") -> str:
    """Safely walk a nested dict, returning ``default`` if any key is missing."""
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, list):
        return "; ".join(str(x) for x in cur)
    return str(cur) if cur is not None else default


def study_to_markdown(study: dict) -> tuple[str, str]:
    """Convert one API study object into (nct_id, markdown_text)."""
    proto = study.get("protocolSection", {})
    nct = _get(proto, "identificationModule", "nctId", default="UNKNOWN")
    title = _get(proto, "identificationModule", "briefTitle", default=nct)
    summary = _get(proto, "descriptionModule", "briefSummary")
    conditions = _get(proto, "conditionsModule", "conditions")
    eligibility = _get(proto, "eligibilityModule", "eligibilityCriteria")
    status = _get(proto, "statusModule", "overallStatus")

    outcomes = proto.get("outcomesModule", {})
    primary = "; ".join(_get(o, "measure") for o in outcomes.get("primaryOutcomes", []) if o)
    secondary = "; ".join(_get(o, "measure") for o in outcomes.get("secondaryOutcomes", []) if o)

    url = f"https://clinicaltrials.gov/study/{nct}"
    md = f"""---
doc_id: {nct}
title: {json.dumps(title)}
source_url: {url}
synthetic: false
---

# {title}

## Summary
{summary or "Not provided."}

## Condition
{conditions or "Not provided."}

## Eligibility Criteria
{eligibility or "Not provided."}

## Primary Endpoints
{primary or "Not provided."}

## Secondary Endpoints
{secondary or "Not provided."}

## Status
{status or "Not provided."}
"""
    return nct, md


def fetch(max_records: int, query: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    page_token: str | None = None
    with httpx.Client(timeout=30.0) as client:
        while written < max_records:
            params: dict[str, object] = {
                "pageSize": min(100, max_records - written),
                "format": "json",
            }
            if query:
                params["query.term"] = query
            if page_token:
                params["pageToken"] = page_token

            resp = client.get(API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            studies = data.get("studies", [])
            if not studies:
                break
            for study in studies:
                nct, md = study_to_markdown(study)
                (out_dir / f"{nct}.md").write_text(md, encoding="utf-8")
                written += 1
                if written >= max_records:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.5)  # be polite
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=300, help="Max records to fetch.")
    parser.add_argument("--query", type=str, default="", help="Optional free-text search term.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dir.")
    args = parser.parse_args(argv)

    print(f"Fetching up to {args.max} public records into {args.out} ...")
    n = fetch(args.max, args.query, args.out)
    print(f"Wrote {n} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
