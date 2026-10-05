from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

from pypdf import PdfReader

from .openai_agent import extract as openai_extract
from .quality_router import analyze_document_quality

Provider = Literal["auto", "openai", "gemini"]

OPENAI_MODEL_ENV = "BILL_EXTRACTOR_OPENAI_MODEL"
GEMINI_MODEL_ENV = "BILL_EXTRACTOR_GEMINI_MODEL"
PROVIDER_ENV = "BILL_EXTRACTOR_PROVIDER"
SCAN_TEXT_THRESHOLD_ENV = "BILL_EXTRACTOR_SCAN_TEXT_THRESHOLD"
SCAN_PAGE_LIMIT_ENV = "BILL_EXTRACTOR_SCAN_PAGE_LIMIT"
QUALITY_THRESHOLD_ENV = "BILL_EXTRACTOR_QUALITY_THRESHOLD"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RouteDecision:
    provider: Literal["openai", "gemini"]
    model: str
    reason: str


def _default_openai_model() -> str:
    return os.environ.get(OPENAI_MODEL_ENV, os.environ.get("BILL_EXTRACTOR_MODEL", "gpt-5-mini"))


def _default_gemini_model() -> str:
    return os.environ.get(GEMINI_MODEL_ENV, "google/gemini-3.1-pro-preview")


def _quality_threshold() -> float:
    raw_value = os.environ.get(QUALITY_THRESHOLD_ENV, "0.58")
    try:
        return min(1.0, max(0.0, float(raw_value)))
    except ValueError:
        return 0.58


def _scan_text_threshold() -> int:
    raw_value = os.environ.get(SCAN_TEXT_THRESHOLD_ENV, "120")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 120


def _scan_page_limit() -> int:
    raw_value = os.environ.get(SCAN_PAGE_LIMIT_ENV, "3")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 3


def _pdf_text_char_count(path: Path) -> int:
    reader = PdfReader(str(path))
    pages_to_check = min(len(reader.pages), _scan_page_limit())
    total = 0

    for page_index in range(pages_to_check):
        try:
            page_text = reader.pages[page_index].extract_text() or ""
        except Exception:
            page_text = ""
        total += len(" ".join(page_text.split()))

    return total


def _emit_routing_block(title: str, fields: list[tuple[str, object]], *, suffix: str | None = None) -> None:
    print("", flush=True)
    print(title, flush=True)
    print("-" * 60, flush=True)
    for label, value in fields:
        print(f"{label:<22}: {value}", flush=True)
    if suffix:
        print(suffix, flush=True)
    print("", flush=True)


def choose_route(
    path: Path,
    *,
    provider: Provider = "auto",
    openai_model: Optional[str] = None,
    gemini_model: Optional[str] = None,
) -> RouteDecision:
    requested_provider = (provider or os.environ.get(PROVIDER_ENV, "auto")).strip().lower()
    selected_openai_model = openai_model or _default_openai_model()
    selected_gemini_model = gemini_model or _default_gemini_model()
    ext = path.suffix.lower()

    if requested_provider == "openai":
        try:
            analysis = analyze_document_quality(
                path,
                quality_threshold=_quality_threshold(),
            )
            reason = f"forced openai provider; {analysis.reason}"
            routing_mode = "force-selected"
        except Exception as exc:  # noqa: BLE001
            reason = f"forced openai provider; quality routing failed: {exc}"
            routing_mode = "force-selected"
        _emit_routing_block(
            "EXTRACTION PROCESS",
            [
                ("Selected Provider", "openai"),
                ("Selected Model", selected_openai_model),
                ("Routing Mode", routing_mode),
                ("Reason", reason),
            ],
            suffix="Extraction Started...",
        )
        return RouteDecision("openai", selected_openai_model, reason)
    if requested_provider == "gemini":
        try:
            analysis = analyze_document_quality(
                path,
                quality_threshold=_quality_threshold(),
            )
            reason = f"forced gemini provider; {analysis.reason}"
            routing_mode = "force-selected"
        except Exception as exc:  # noqa: BLE001
            reason = f"forced gemini provider; quality routing failed: {exc}"
            routing_mode = "force-selected"
        _emit_routing_block(
            "EXTRACTION PROCESS",
            [
                ("Selected Provider", "gemini"),
                ("Selected Model", selected_gemini_model),
                ("Routing Mode", routing_mode),
                ("Reason", reason),
            ],
            suffix="Extraction Started...",
        )
        return RouteDecision("gemini", selected_gemini_model, reason)

    if ext not in IMAGE_EXTENSIONS and ext != ".pdf":
        return RouteDecision("openai", selected_openai_model, f"unsupported extension {ext}")

    try:
        analysis = analyze_document_quality(
            path,
            quality_threshold=_quality_threshold(),
        )
        provider_name = analysis.recommended_provider
        selected_model = selected_openai_model if provider_name == "openai" else selected_gemini_model
        _emit_routing_block(
            "EXTRACTION PROCESS",
            [
                ("Selected Provider", provider_name),
                ("Selected Model", selected_model),
                ("Routing Mode", "auto-routed"),
                ("Reason", analysis.reason),
            ],
            suffix="Extraction Started...",
        )
        return RouteDecision(
            provider_name,
            selected_model,
            analysis.reason,
        )
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"quality routing failed: {exc}"
        if ext in IMAGE_EXTENSIONS:
            return RouteDecision("gemini", selected_gemini_model, f"image input routes to gemini ({fallback_reason})")
        if ext == ".pdf":
            text_chars = _pdf_text_char_count(path)
            if text_chars < _scan_text_threshold():
                return RouteDecision(
                    "gemini",
                    selected_gemini_model,
                    f"pdf text layer too small ({text_chars} chars); {fallback_reason}",
                )
            return RouteDecision(
                "openai",
                selected_openai_model,
                f"pdf text layer present ({text_chars} chars); {fallback_reason}",
            )
        return RouteDecision("openai", selected_openai_model, fallback_reason)


async def extract(
    path: Path,
    *,
    provider: Provider = "auto",
    openai_model: Optional[str] = None,
    gemini_model: Optional[str] = None,
) -> Dict[str, object]:
    decision = choose_route(
        path,
        provider=provider,
        openai_model=openai_model,
        gemini_model=gemini_model,
    )
    return await openai_extract(path, model=decision.model, provider=decision.provider)