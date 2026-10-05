from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, cast
from decimal import Decimal, InvalidOperation

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _is_scalar(x: Any) -> bool:
    return not isinstance(x, (dict, list))


def _should_ignore_field(key: str) -> bool:
    """Ignore UUID-like fields anywhere in the structure."""
    k = (key or "").lower()
    # Ignore UUID fields and known non-deterministic/irrelevant fields
    # - uuid, *_uuid: identifiers that vary per run/document
    # - payment_terms: often free-form text not required for strict matching
    return (
        k == "uuid"
        or k.endswith("_uuid")
        or k == "payment_terms"
        or k == "exchange_rate"
        or k == "gst_treatment"
        or k == "is_rcm"
    )


def _to_decimal(val: Any) -> Optional[Decimal]:
    """Best-effort parse of numeric-like values to Decimal.

    - Accepts ints/floats/Decimal directly (via str conversion to avoid FP issues).
    - Accepts strings containing digits, optional commas and whitespace.
    - Returns None if not numeric-like.
    """
    if isinstance(val, (int, float, Decimal)):
        try:
            return Decimal(str(val))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(val, str):
        s = val.replace(",", "").replace(" ", "")
        # Short-circuit clearly non-numeric strings
        if not s:
            return None
        try:
            return Decimal(s)
        except (InvalidOperation, ValueError):
            return None
    return None


def _collect_header_pairs(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> List[Tuple[str, Any, Any]]:
    pairs: List[Tuple[str, Any, Any]] = []
    e_hdr = expected.get("data", {}).get("header", {}) or {}
    a_hdr = actual.get("data", {}).get("header", {}) or {}

    # Flat scalars
    for k, ve in e_hdr.items():
        if _is_scalar(ve) and not _should_ignore_field(k):
            va = a_hdr.get(k, None)
            pairs.append((f"header.{k}", va, ve))

    # Nested lists: activity, taxes (if present)
    def _canonical_value_for_sort(v: Any) -> str:
        d = _to_decimal(v)
        if d is not None:
            return str(d)
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return ""
        return str(v)

    def _filter_zero_line_amount(lst: List[Any]) -> List[Any]:
        """Remove dict entries with line_amount == 0 (numeric) for activity lists.

        If an element is a dict and contains a numeric-like `line_amount` that parses
        to Decimal(0), drop it. Elements without `line_amount` are retained.
        Non-dict elements are retained.
        """
        out: List[Any] = []
        for el in lst or []:
            if isinstance(el, dict):
                d = _to_decimal(el.get("line_amount"))
                if d is not None and d == 0:
                    continue
            out.append(el)
        return out

    def _sort_list_for_comparison(lst: List[Any]) -> List[Any]:
        if not isinstance(lst, list):
            return lst
        if lst and isinstance(lst[0], dict):

            def key_func(o: Dict[str, Any]):
                if not isinstance(o, dict):
                    return ()
                items = [
                    (k, _canonical_value_for_sort(v))
                    for k, v in sorted(o.items())
                    if _is_scalar(v) and not _should_ignore_field(k)
                ]
                return tuple(items)

            return sorted(lst, key=key_func)
        else:
            return sorted(lst, key=_canonical_value_for_sort)

    for list_key in ("activity", "taxes"):
        e_list = e_hdr.get(list_key, []) or []
        a_list = a_hdr.get(list_key, []) or []
        # For activities, ignore entries where line_amount == 0 before comparing
        if list_key == "activity":
            e_list = _filter_zero_line_amount(e_list)
            a_list = _filter_zero_line_amount(a_list)
        # Order-insensitive: sort both lists by canonicalized scalar content
        e_list_sorted = _sort_list_for_comparison(e_list)
        a_list_sorted = _sort_list_for_comparison(a_list)
        if isinstance(e_list_sorted, list) and isinstance(a_list_sorted, list):
            pairs.append(
                (f"header.{list_key}.length", len(a_list_sorted), len(e_list_sorted))
            )
            for j in range(min(len(e_list_sorted), len(a_list_sorted))):
                e_obj = e_list_sorted[j] or {}
                a_obj = a_list_sorted[j] or {}
                if isinstance(e_obj, dict) and isinstance(a_obj, dict):
                    for kk, vve in e_obj.items():
                        if _is_scalar(vve) and not _should_ignore_field(kk):
                            vva = a_obj.get(kk, None)
                            pairs.append((f"header.{list_key}[{j}].{kk}", vva, vve))
    return pairs


def _collect_line_item_pairs(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> List[Tuple[str, Any, Any]]:
    pairs: List[Tuple[str, Any, Any]] = []
    e_lines = expected.get("data", {}).get("line_items", []) or []
    a_lines = actual.get("data", {}).get("line_items", []) or []

    # Length comparison counts as a check
    pairs.append(("line_items.length", len(a_lines), len(e_lines)))

    for i in range(min(len(e_lines), len(a_lines))):
        e_li = e_lines[i] or {}
        a_li = a_lines[i] or {}
        # Flat scalar fields per line
        if isinstance(e_li, dict) and isinstance(a_li, dict):
            for k, ve in e_li.items():
                if _is_scalar(ve) and not _should_ignore_field(k):
                    va = a_li.get(k, None)
                    pairs.append((f"line_items[{i}].{k}", va, ve))

            # Nested lists on lines: item_activity (primary), activity, taxes, item_taxes
            # Sorting helpers from header scope
            def _canonical_value_for_sort(v: Any) -> str:
                d = _to_decimal(v)
                if d is not None:
                    return str(d)
                if isinstance(v, bool):
                    return "true" if v else "false"
                if v is None:
                    return ""
                return str(v)

            def _filter_zero_line_amount(lst: List[Any]) -> List[Any]:
                out: List[Any] = []
                for el in lst or []:
                    if isinstance(el, dict):
                        d = _to_decimal(el.get("line_amount"))
                        if d is not None and d == 0:
                            continue
                    out.append(el)
                return out

            def _sort_list_for_comparison(lst: List[Any]) -> List[Any]:
                if not isinstance(lst, list):
                    return lst
                if lst and isinstance(lst[0], dict):

                    def key_func(o: Dict[str, Any]):
                        if not isinstance(o, dict):
                            return ()
                        items = [
                            (k, _canonical_value_for_sort(v))
                            for k, v in sorted(o.items())
                            if _is_scalar(v) and not _should_ignore_field(k)
                        ]
                        return tuple(items)

                    return sorted(lst, key=key_func)
                else:
                    return sorted(lst, key=_canonical_value_for_sort)

            for list_key in ("item_activity", "activity", "taxes", "item_taxes"):
                e_list = e_li.get(list_key, []) or []
                a_list = a_li.get(list_key, []) or []
                # For activity lists, drop entries with line_amount == 0 before comparing
                if list_key in ("item_activity", "activity"):
                    e_list = _filter_zero_line_amount(e_list)
                    a_list = _filter_zero_line_amount(a_list)
                e_list_sorted = _sort_list_for_comparison(e_list)
                a_list_sorted = _sort_list_for_comparison(a_list)
                if isinstance(e_list_sorted, list) and isinstance(a_list_sorted, list):
                    pairs.append(
                        (
                            f"line_items[{i}].{list_key}.length",
                            len(a_list_sorted),
                            len(e_list_sorted),
                        )
                    )
                    for j in range(min(len(e_list_sorted), len(a_list_sorted))):
                        e_obj = e_list_sorted[j] or {}
                        a_obj = a_list_sorted[j] or {}
                        if isinstance(e_obj, dict) and isinstance(a_obj, dict):
                            for kk, vve in e_obj.items():
                                if _is_scalar(vve) and not _should_ignore_field(kk):
                                    vva = a_obj.get(kk, None)
                                    pairs.append(
                                        (
                                            f"line_items[{i}].{list_key}[{j}].{kk}",
                                            vva,
                                            vve,
                                        )
                                    )
    return pairs


class HeaderAndLineItemsAccuracy(BaseMetric):
    """Accuracy over bill header and line items (including nested activities/taxes).

    Compares scalar fields in `data.header` and each element of `data.line_items`.
    - Ignores UUID fields (keys equal to `uuid` or ending with `_uuid`).
    - Checks list lengths for `line_items`, header `activity`/`taxes`, and line-level
      `item_activity`/`activity`/`taxes`/`item_taxes`. For activity lists, entries
      with `line_amount` == 0 (after numeric parsing) are ignored before length
      and value comparison. Nested lists are compared
      order-insensitively by sorting both sides using a canonical scalar-content key.
    - Match rules:
      - Booleans: exact match.
      - Numbers: exact numeric equality after parsing (e.g., 55250, 55250.0, "55,250.00").
      - Text: raw-string similarity using difflib.
        - Default: match if ratio ≥ text_similarity_threshold.
        - For fields named `item_description` or `description`: match if ratio ≥ description_similarity_threshold
          (defaults to 0.8) without affecting other string fields.

    Accuracy = matches / total_checks.

    Example: If there are 12 checks and 11 match (bill_amount 55250 equals
    "55,250.00" numerically; vendor_name fails at text similarity threshold 1.0),
    accuracy = 11/12 = 0.9167. Test passes if accuracy ≥ threshold.
    """

    name = "HeaderAndLineItemsAccuracy"

    def __init__(
        self,
        text_similarity_threshold: float = 1.0,
        description_similarity_threshold: float = 0.8,
        threshold: float = 1.0,
    ):
        self.threshold = float(threshold)
        self.text_similarity_threshold = float(text_similarity_threshold)
        self.description_similarity_threshold = float(description_similarity_threshold)
        self.matches = 0
        self.total = 0
        self.score = 0.0
        self.reason = ""
        self.async_mode = False
        self._first_mismatches: List[Tuple[str, Any, Any]] = []

    def _text_similar(
        self, va: Any, ve: Any, *, threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[float]]:
        from difflib import SequenceMatcher

        if va is None or ve is None:
            return False, None
        try:
            ratio = SequenceMatcher(None, str(ve), str(va)).ratio()
        except Exception:
            return False, None
        thr = self.text_similarity_threshold if threshold is None else float(threshold)
        return ratio >= thr, ratio

    def _is_description_key(self, path: str) -> bool:
        """Return True if the last segment of the path is a description-like field.

        Examples of paths:
          - "line_items[0].item_description"
          - "header.activity[1].description"
        """
        try:
            last = path.split(".")[-1]
        except Exception:
            last = path
        return last in {"item_description", "description"}

    def _eq(self, path: str, va: Any, ve: Any) -> Tuple[bool, Optional[float]]:
        # Booleans: require exact
        if isinstance(ve, bool):
            return (isinstance(va, bool) and va is ve), None

        # Numbers: exact numeric equality (after parsing)
        de = _to_decimal(ve)
        da = _to_decimal(va)
        if de is not None and da is not None:
            return (da == de), None

        # Fallback to text similarity for all other scalar types
        if self._is_description_key(path):
            return self._text_similar(
                va, ve, threshold=self.description_similarity_threshold
            )
        return self._text_similar(va, ve)

    def measure(self, test_case: LLMTestCase) -> float:
        import json as _json

        actual = (
            _json.loads(test_case.actual_output)
            if isinstance(test_case.actual_output, str)
            else test_case.actual_output
        )
        expected = (
            _json.loads(test_case.expected_output)
            if isinstance(test_case.expected_output, str)
            else test_case.expected_output
        )

        pairs: List[Tuple[str, Any, Any]] = []
        pairs.extend(_collect_header_pairs(expected, actual))
        pairs.extend(_collect_line_item_pairs(expected, actual))

        total = 0
        matches = 0
        mismatches: List[Tuple[str, Any, Any]] = []
        mismatch_details: List[str] = []

        for path, va, ve in pairs:
            total += 1
            ok, ratio = self._eq(path, va, ve)
            if ok:
                matches += 1
            else:
                mismatches.append((path, va, ve))
                if ratio is not None:
                    mismatch_details.append(
                        f"{path}: sim={ratio:.2f} (actual={va}, expected={ve})"
                    )

        self.total = total
        self.matches = matches
        self.score = 1.0 if total == 0 else matches / total
        self._first_mismatches = mismatches[:5]

        extra = (
            f"; text sims: [{', '.join(mismatch_details[:3])}]"
            if mismatch_details
            else ""
        )
        self.reason = f"Matched {matches}/{total}; first mismatches: {self._first_mismatches}{extra}"
        return cast(float, self.score)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.measure(test_case)
        return cast(float, self.score)

    def is_successful(self) -> bool:
        return cast(float, self.score) >= self.threshold
