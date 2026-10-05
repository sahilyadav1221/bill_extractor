from __future__ import annotations

import asyncio
import ast
import json
import re
from dataclasses import dataclass
import os
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Local imports
from ..approaches.routing import extract as anthropic_extract
from ..metric import HeaderAndLineItemsAccuracy


# ---------- Paths ----------
EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR / "data"
REPORTS_DIR = EXP_DIR / "reports"
THIS_DIR = Path(__file__).resolve().parent
FLAVOR = os.getenv("FLAVOR", "accuracy_score").strip().lower()


@dataclass
class Pair:
    pdf: Path
    expected_json: Path


def discover_pairs(data_dir: Path, flavor: str = FLAVOR) -> List[Pair]:
    include_path = THIS_DIR / "batch_include.json"
    raw = json.loads(include_path.read_text(encoding="utf-8"))
    names: List[str] = raw[flavor] if isinstance(raw, dict) else list(raw)
    pairs = [
        Pair(
            pdf=data_dir / name,
            expected_json=(data_dir / f"{Path(name).stem}_result.json"),
        )
        for name in names
    ]
    missing: List[str] = []
    for p in pairs:
        if not p.pdf.exists():
            missing.append(f"PDF missing: {p.pdf.name}")
        if not p.expected_json.exists():
            missing.append(f"Expected JSON missing: {p.expected_json.name}")
    if missing:
        raise FileNotFoundError("Missing test files:\n" + "\n".join(missing))
    return pairs


async def extract_actual(pdf_path: Path) -> Dict:
    return await anthropic_extract(pdf_path, provider=os.getenv("BILL_EXTRACTOR_PROVIDER", "auto"))


def parse_mismatches_from_reason(reason: str) -> List[Tuple[str, object, object]]:
    """Parse metric.reason to extract list of (field_path, actual, expected).

    Expected format from HeaderAndLineItemsAccuracy:
      "Matched X/Y; first mismatches: [(path, actual, expected), ...]; ..."
    """
    try:
        anchor = "first mismatches:"
        if anchor not in reason:
            return []
        s = reason.split(anchor, 1)[1].strip()
        # Be robust to semicolons inside string values. Only trim the well-known
        # trailer that our metric appends ("; text sims: ...").
        trailer = "; text sims:"
        if trailer in s:
            s = s.split(trailer, 1)[0].strip()
        # If the trailer wasn't present but other text followed, try to locate the
        # closing bracket of the list conservatively (last ']' in the string).
        if "]" in s:
            s = s[: s.rfind("]") + 1]
        # s should now be something like: "[(...) , (...)]"
        lst = ast.literal_eval(s)
        rows: List[Tuple[str, object, object]] = []
        if isinstance(lst, list):
            for item in lst:
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and isinstance(item[0], str)
                ):
                    rows.append((item[0], item[1], item[2]))
        return rows
    except Exception:
        return []


def _normalize_field_path(field_path: str) -> str:
    """Normalize a mismatch field path so counts group logically.

    Examples:
    - "header.invoice_date" -> "header.invoice_date"
    - "line_items[0].amount" -> "line_items[].amount"
    - "taxes[12].rate.value" -> "taxes[].rate.value"
    """
    # Collapse any numeric indices inside [] to a generic [] to group counts
    return re.sub(r"\[\d+\]", "[]", field_path)


def summarize_mismatch_counts(
    results: List[Tuple["Pair", Optional[float], str, float]],
) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for _, score, reason, _ in results:
        if score is None or score >= 1.0:
            continue
        rows = parse_mismatches_from_reason(reason)
        for field, _, _ in rows:
            key = _normalize_field_path(field)
            counts[key] = counts.get(key, 0) + 1
    # Sort by count desc, then by key asc for stable output
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def write_report(
    out_dir: Path,
    results: List[Tuple[Pair, Optional[float], str, float]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = out_dir.name
    out_md = out_dir / f"deepeval_result_{ts}.md"

    scores_all = [s for _, s, _, _ in results if s is not None]
    total_scores = len(scores_all)
    mean_score = sum(scores_all) / total_scores if total_scores else 0.0
    median_score = statistics.median(scores_all) if total_scores else 0.0
    perfect = sum(1 for s in scores_all if s == 1.0)
    imperfect = total_scores - perfect
    perfect_ratio = (perfect / total_scores) if total_scores else 0.0
    imperfect_ratio = (imperfect / total_scores) if total_scores else 0.0

    latencies = [lat for _, _, _, lat in results]
    total = len(latencies)
    mean_latency = sum(latencies) / total if total else 0.0
    median_latency = statistics.median(latencies) if total else 0.0

    lines: List[str] = []
    lines.append(f"# Batch Evaluation Report ({ts})")
    lines.append("")
    lines.append(f"- Total files: {total}")
    if total_scores:
        lines.append(f"- Mean score: {mean_score:.4f}")
        lines.append(f"- Median score: {median_score:.4f}")
        lines.append(
            f"- Perfect: {perfect}/{total_scores} ({perfect_ratio:.0%}), Imperfect: {imperfect}/{total_scores} ({imperfect_ratio:.0%})"
        )
    lines.append(f"- Mean latency (s): {mean_latency:.3f}")
    lines.append(f"- Median latency (s): {median_latency:.3f}")
    # Mismatch summary table (aggregated across files)
    summary = summarize_mismatch_counts(results)
    if summary:
        lines.append("")
        lines.append("## Mismatch Summary")
        lines.append("")
        lines.append("| issue | count |")
        lines.append("|---|---:|")
        for issue, cnt in summary:
            lines.append(f"| `{issue}` | {cnt} |")
    lines.append("")
    lines.append("## Per-file Analysis")
    lines.append("")
    lines.append("| file | score | latency (s) |")
    lines.append("|---|---:|---:|")
    for pair, score, _, lat in results:
        score_str = "-" if score is None else f"{score:.4f}"
        lines.append(f"| {pair.pdf.name} | {score_str} | {lat:.3f} |")
    lines.append("")

    any_mismatch = False
    for pair, score, reason, _ in results:
        if score is None or score >= 1.0:
            continue
        any_mismatch = True
        lines.append(f"### {pair.pdf.name} — score={score:.4f}")
        rows = parse_mismatches_from_reason(reason)
        if not rows:
            lines.append("(No parsed mismatches; see raw reason below.)")
            lines.append("")
            lines.append("```")
            lines.append(reason)
            lines.append("```")
            lines.append("")
            continue

        # Simple Markdown table
        lines.append("| field | expected | actual |")
        lines.append("|---|---|---|")
        for field, actual, expected in rows:
            # Render as JSON-ish strings to keep formatting sane
            a = json.dumps(actual, ensure_ascii=False)
            e = json.dumps(expected, ensure_ascii=False)
            lines.append(f"| `{field}` | {e} | {a} |")
        lines.append("")

    if not any_mismatch and total_scores:
        lines.append("All files are perfect matches.")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

    # removed separate latency report; integrated into write_report


# YAML support and alternate include locations removed for simplicity.


def main() -> None:
    flavor = FLAVOR
    pairs = discover_pairs(DATA_DIR, flavor)
    if not pairs:
        print("No (PDF, expected JSON) pairs found in data/")
        return

    results: List[Tuple[Pair, Optional[float], str, float]] = []

    for pair in pairs:
        # Load expected; crash if missing/invalid as requested
        expected: Dict = json.loads(pair.expected_json.read_text(encoding="utf-8"))

        # Run extraction and measure latency
        try:
            t0 = time.perf_counter()
            actual = asyncio.run(extract_actual(pair.pdf))
            t1 = time.perf_counter()
            latency = t1 - t0
        except Exception as e:
            print(f"WARN: extraction failed for {pair.pdf.name}: {e}; skipping")
            continue

        # Build DeepEval test case and metric
        # The metric expects both actual and expected to have a top-level
        # "data" envelope that contains {"header", "line_items", ...}.
        # Keep the full "actual" (which already includes "data"), and wrap
        # the expected fixture accordingly so comparisons are not skipped.
        test_case = LLMTestCase(
            input=str(pair.pdf),
            actual_output=json.dumps(actual, ensure_ascii=False),
            expected_output=json.dumps({"data": expected}, ensure_ascii=False),
            # expected_output=json.dumps(expected, ensure_ascii=False),
        )
        metric = HeaderAndLineItemsAccuracy(threshold=1.0)
        # Compute score directly to avoid relying on DeepEval mutating the metric instance
        # (evaluate() may wrap/clone metrics, leaving our local instance unchanged).
        try:
            score: Optional[float] = float(metric.measure(test_case))
        except Exception:
            score = None
        reason = str(metric.reason or "")

        # Optionally run DeepEval's pretty printer for console output
        try:
            evaluate([test_case], [metric])
        except Exception:
            # If DeepEval's evaluate fails, we still have our measured score/reason
            pass

        results.append((pair, score, reason, latency))

    # Aggregate and write report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / ts
    out_md = write_report(out_dir, results)
    print(f"Wrote report: {out_md}")


if __name__ == "__main__":
    main()
