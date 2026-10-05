from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import fitz
except ImportError as exc:  # pragma: no cover - required for PDF support
    raise ImportError("PyMuPDF (fitz) is required for bill quality routing") from exc

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


logger = logging.getLogger(__name__)

DocumentType = Literal["digital_pdf", "scanned_pdf", "image", "unsupported"]
Provider = Literal["openai", "gemini"]

QUALITY_WEIGHTS: Dict[str, float] = {
    "dpi_score": 0.25,
    "blur_score": 0.25,
    "ocr_confidence": 0.30,
    "noise_score": 0.10,
    "skew_score": 0.10,
}

DEFAULT_RENDER_DPI = 300
DEFAULT_MAX_PDF_PAGES = 3
DEFAULT_QUALITY_THRESHOLD = 0.58
DEFAULT_DIGITAL_TEXT_THRESHOLD = 120
DEFAULT_OCR_PAGE_LIMIT = 2

# Laplacian variance at which blur_score saturates to 1.0.
# At DEFAULT_RENDER_DPI (300 DPI) a clean rendered PDF typically produces
# variance in the 800–2000 range; values above this constant are clamped to 1.0.
# Increase this value to make blur a stronger discriminator for high-DPI inputs.
_BLUR_VARIANCE_SATURATION: float = 1200.0


@dataclass(frozen=True)
class QualityMetrics:
    dpi_estimate: float
    dpi_score: float
    blur_variance: float
    blur_score: float
    noise_score: float
    skew_angle: float
    skew_score: float
    contrast_score: float
    brightness_score: float
    ocr_confidence: float
    ocr_score: float
    text_density: float
    extracted_text_length: int
    word_count: int


@dataclass(frozen=True)
class QualityAnalysis:
    path: str
    file_type: Literal["pdf", "image", "unsupported"]
    document_type: DocumentType
    page_count: int
    analyzed_pages: int
    render_dpi: int
    metrics: QualityMetrics
    quality_score: float
    quality_threshold: float
    recommended_provider: Provider
    reason: str
    weights: Dict[str, float]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_ratio(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return _clamp(value / scale)


def _load_image_from_path(path: Path) -> np.ndarray:
    if cv2 is None:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return np.array(image)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _render_pdf_page(page: fitz.Page, render_dpi: int) -> np.ndarray:
    matrix = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    channels = pixmap.n
    image = samples.reshape(pixmap.height, pixmap.width, channels)
    if channels == 4:
        image = image[:, :, :3]
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.dot(image[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def _denoise(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    padded = np.pad(gray, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-1, -2)).astype(np.uint8)


def _threshold(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    threshold_value = float(np.mean(gray))
    return np.where(gray >= threshold_value, 255, 0).astype(np.uint8)


def _laplacian_variance(gray: np.ndarray) -> float:
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(gray.astype(np.float64), 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    lap = np.sum(windows * kernel, axis=(-1, -2))
    return float(np.var(lap))


def _estimate_skew_angle(binary: np.ndarray) -> float:
    dark_pixels = np.column_stack(np.where(binary < 128))
    if dark_pixels.shape[0] < 50:
        return 0.0

    coords = dark_pixels[:, ::-1].astype(np.float64)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vector = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(principal_vector[1], principal_vector[0]))
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(angle)


def _estimate_pdf_page_source_dpi(page: fitz.Page) -> float | None:
    try:
        raw_dict: Any = page.get_text("rawdict") or {}
    except Exception:  # noqa: BLE001
        raw_dict = {}

    candidates: list[float] = []
    for block in raw_dict.get("blocks", []):
        if not isinstance(block, dict) or block.get("type") != 1:
            continue

        width_px = float(block.get("width") or 0.0)
        height_px = float(block.get("height") or 0.0)
        bbox = block.get("bbox") or ()
        if width_px <= 0.0 or height_px <= 0.0 or len(bbox) != 4:
            continue

        bbox_width = max(float(bbox[2]) - float(bbox[0]), 0.01)
        bbox_height = max(float(bbox[3]) - float(bbox[1]), 0.01)
        dpi_x = width_px / (bbox_width / 72.0)
        dpi_y = height_px / (bbox_height / 72.0)
        candidates.append(float(min(dpi_x, dpi_y)))

    if candidates:
        return float(max(candidates))

    try:
        image_refs = page.get_images(full=True)
    except Exception:  # noqa: BLE001
        image_refs = []

    for image_ref in image_refs:
        if not image_ref or len(image_ref) < 3:
            continue
        xref = image_ref[0]
        width_px = float(image_ref[2] or 0.0)
        height_px = float(image_ref[3] or 0.0)
        if xref <= 0 or width_px <= 0.0 or height_px <= 0.0:
            continue

        try:
            rects = page.get_image_rects(xref)
        except Exception:  # noqa: BLE001
            rects = []

        for rect in rects:
            bbox_width = max(float(rect.width), 0.01)
            bbox_height = max(float(rect.height), 0.01)
            dpi_x = width_px / (bbox_width / 72.0)
            dpi_y = height_px / (bbox_height / 72.0)
            candidates.append(float(min(dpi_x, dpi_y)))

    return float(max(candidates)) if candidates else None


def _infer_dpi(
    image: np.ndarray,
    *,
    source_dpi_estimate: float | None,
    page_size_inches: tuple[float, float] | None,
    render_dpi: int,
) -> float:
    if source_dpi_estimate is not None and source_dpi_estimate > 0:
        return float(source_dpi_estimate)

    if page_size_inches is not None:
        page_width_inches = max(page_size_inches[0], 0.01)
        page_height_inches = max(page_size_inches[1], 0.01)
        dpi_x = image.shape[1] / page_width_inches
        dpi_y = image.shape[0] / page_height_inches
        return float(min(dpi_x, dpi_y))

    height, width = image.shape[:2]
    assumed_height_inches = 11.69 if width / max(height, 1) > 1.2 else 11.0
    assumed_width_inches = 8.27 if assumed_height_inches > 11.2 else 8.5
    return float(min(width / assumed_width_inches, height / assumed_height_inches))


def _confidence_to_quality(confidence: float) -> float:
    return _clamp(confidence)


def _emit_terminal_block(title: str, fields: Sequence[tuple[str, object]], *, prefix: str | None = None) -> None:
    print("", flush=True)
    if prefix:
        print(prefix, flush=True)
    print(title, flush=True)
    print("-" * 60, flush=True)
    for label, value in fields:
        print(f"{label:<22}: {value}", flush=True)
    print("", flush=True)


def _parse_orientation(osd_output: str) -> tuple[int, float]:
    rotate = 0
    for line in osd_output.splitlines():
        if "Rotate:" in line:
            try:
                rotate = int(line.split(":", 1)[1].strip())
            except ValueError:
                rotate = 0
            break
    orientation_score = 1.0 if rotate % 360 == 0 else 0.0
    return rotate, orientation_score


def _extract_words_and_confidence(
    image: np.ndarray,
    *,
    render_dpi: int,
) -> tuple[float, int, int, float, float, float, int, int]:
    if pytesseract is None:
        return 0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0

    try:
        osd_output = pytesseract.image_to_osd(image)
        orientation_degrees, orientation_score = _parse_orientation(osd_output)
    except Exception as exc:  # noqa: BLE001
        logger.warning("quality_router_ocr_fallback=%s", exc)
        return 0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        extracted_text = pytesseract.image_to_string(image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("quality_router_ocr_fallback=%s", exc)
        return 0.0, 0, 0, 0.0, 0.0, orientation_score, orientation_degrees, 0

    confidences: list[float] = []
    text_fragments: list[str] = []
    bbox_area = 0.0
    bbox_count = 0
    image_area = max(float(image.shape[0] * image.shape[1]), 1.0)

    for idx, text in enumerate(data.get("text", [])):
        if not text or not str(text).strip():
            continue
        try:
            conf_val = float(data.get("conf", [])[idx])
        except (TypeError, ValueError, IndexError):
            conf_val = -1.0
        if conf_val >= 0:
            confidences.append(conf_val)
        text_fragments.append(str(text).strip())
        try:
            width = float(data.get("width", [])[idx])
            height = float(data.get("height", [])[idx])
        except (TypeError, ValueError, IndexError):
            width = 0.0
            height = 0.0
        if width > 0 and height > 0:
            bbox_area += width * height
            bbox_count += 1

    extracted_text_length = len(extracted_text)
    word_count = len(text_fragments)
    avg_confidence = float(np.mean(confidences)) / 100.0 if confidences else 0.0
    text_density = _text_density_score(extracted_text_length, image.shape[:2], render_dpi)
    text_coverage = _clamp(bbox_area / image_area)
    return (
        avg_confidence,
        extracted_text_length,
        word_count,
        text_density,
        text_coverage,
        orientation_score,
        orientation_degrees,
        bbox_count,
    )


def _text_density_score(text_length: int, shape: Sequence[int], render_dpi: int) -> float:
    height = max(int(shape[0]), 1)
    width = max(int(shape[1]), 1)
    normalized_area = max((width * height) / max(render_dpi * render_dpi, 1), 1.0)
    density = text_length / normalized_area
    return _clamp(density / 600.0)


def _analyze_image(
    image: np.ndarray,
    *,
    source_dpi_estimate: float | None,
    page_size_inches: tuple[float, float] | None,
    render_dpi: int,
) -> dict[str, float | int]:
    gray = _to_gray(image)
    denoised = _denoise(gray)
    thresholded = _threshold(denoised)

    blur_variance = _laplacian_variance(gray)
    blur_score = _clamp(blur_variance / _BLUR_VARIANCE_SATURATION)

    residual = np.abs(gray.astype(np.float32) - denoised.astype(np.float32))
    noise_raw = float(np.mean(residual) / 255.0)
    noise_score = _clamp(1.0 - (noise_raw * 1.8))

    skew_angle = _estimate_skew_angle(thresholded)
    skew_score = _clamp(1.0 - min(abs(skew_angle), 15.0) / 15.0)

    contrast_score = _clamp(float(np.std(gray)) / 64.0)
    brightness_score = _clamp(1.0 - abs(float(np.mean(gray)) - 127.5) / 127.5)

    (
        ocr_confidence,
        extracted_text_length,
        word_count,
        density_score,
        text_coverage,
        orientation_score,
        orientation_degrees,
        bbox_count,
    ) = _extract_words_and_confidence(
        image, render_dpi=render_dpi
    )

    dpi_estimate = _infer_dpi(
        image,
        source_dpi_estimate=source_dpi_estimate,
        page_size_inches=page_size_inches,
        render_dpi=render_dpi,
    )
    dpi_score = _clamp(dpi_estimate / float(render_dpi))
    word_count_score = _normalize_ratio(word_count, 120)
    ocr_score = _clamp(
        (ocr_confidence * 0.35)
        + (density_score * 0.25)
        + (word_count_score * 0.15)
        + (text_coverage * 0.15)
        + (orientation_score * 0.10)
    )

    _emit_terminal_block(
        "BLUR ANALYSIS",
        [
            ("Status", "SUCCESS"),
            ("Blur Variance", f"{blur_variance:.2f}"),
            ("Blur Score", f"{blur_score:.3f}"),
        ],
    )
    _emit_terminal_block(
        "NOISE ANALYSIS",
        [
            ("Status", "SUCCESS"),
            ("Noise Raw", f"{noise_raw:.3f}"),
            ("Noise Score", f"{noise_score:.3f}"),
        ],
    )
    _emit_terminal_block(
        "SKEW ANALYSIS",
        [
            ("Status", "SUCCESS"),
            ("Skew Angle", f"{skew_angle:.2f}"),
            ("Skew Score", f"{skew_score:.3f}"),
        ],
    )
    _emit_terminal_block(
        "OCR ANALYSIS",
        [
            ("Status", "SUCCESS"),
            ("OCR Confidence", f"{ocr_confidence:.3f}"),
            ("Text Density", f"{density_score:.3f}"),
            ("Extracted Text Length", extracted_text_length),
            ("Word Count", word_count),
            ("OCR Score", f"{ocr_score:.3f}"),
            ("Text Coverage", f"{text_coverage:.3f}"),
            ("Orientation (deg)", orientation_degrees),
            ("Orientation Score", f"{orientation_score:.3f}"),
            ("BBoxes Detected", bbox_count),
        ],
    )
    _emit_terminal_block(
        "DPI ANALYSIS",
        [
            ("Status", "SUCCESS"),
            ("DPI Estimate", f"{dpi_estimate:.2f}"),
            ("DPI Score", f"{dpi_score:.3f}"),
        ],
    )

    return {
        "dpi_estimate": dpi_estimate,
        "dpi_score": dpi_score,
        "blur_variance": blur_variance,
        "blur_score": blur_score,
        "noise_score": noise_score,
        "skew_angle": skew_angle,
        "skew_score": skew_score,
        "contrast_score": contrast_score,
        "brightness_score": brightness_score,
        "ocr_confidence": _confidence_to_quality(ocr_confidence),
        "ocr_score": ocr_score,
        "text_density": density_score,
        "extracted_text_length": extracted_text_length,
        "word_count": word_count,
    }


def _combine_metric_values(values: Sequence[dict[str, float | int]]) -> QualityMetrics:
    if not values:
        return QualityMetrics(
            dpi_estimate=0.0,
            dpi_score=0.0,
            blur_variance=0.0,
            blur_score=0.0,
            noise_score=0.0,
            skew_angle=0.0,
            skew_score=0.0,
            contrast_score=0.0,
            brightness_score=0.0,
            ocr_confidence=0.0,
            ocr_score=0.0,
            text_density=0.0,
            extracted_text_length=0,
            word_count=0,
        )

    def average(key: str) -> float:
        numeric_values = [float(item[key]) for item in values if key in item]
        return float(np.mean(numeric_values)) if numeric_values else 0.0

    def _average_abs_skew_angle(rows: Sequence[dict[str, float | int]]) -> float:
        # Average the magnitude of each page's skew angle so that opposite-sign
        # angles (e.g. +10° on page 1, −10° on page 2) do not cancel each other
        # out and falsely report 0° (perfect alignment).
        magnitudes = [abs(float(row["skew_angle"])) for row in rows if "skew_angle" in row]
        return float(np.mean(magnitudes)) if magnitudes else 0.0

    return QualityMetrics(
        dpi_estimate=average("dpi_estimate"),
        dpi_score=average("dpi_score"),
        blur_variance=average("blur_variance"),
        blur_score=average("blur_score"),
        noise_score=average("noise_score"),
        skew_angle=_average_abs_skew_angle(values),
        skew_score=average("skew_score"),
        contrast_score=average("contrast_score"),
        brightness_score=average("brightness_score"),
        ocr_confidence=average("ocr_confidence"),
        ocr_score=average("ocr_score"),
        text_density=average("text_density"),
        extracted_text_length=int(round(average("extracted_text_length"))),
        word_count=int(round(average("word_count"))),
    )


def _weight_quality(metrics: QualityMetrics) -> float:
    return (
        metrics.dpi_score * QUALITY_WEIGHTS["dpi_score"]
        + metrics.blur_score * QUALITY_WEIGHTS["blur_score"]
        + metrics.ocr_score * QUALITY_WEIGHTS["ocr_confidence"]
        + metrics.noise_score * QUALITY_WEIGHTS["noise_score"]
        + metrics.skew_score * QUALITY_WEIGHTS["skew_score"]
    )


def _emit_quality_progress(analysis: QualityAnalysis) -> None:
    metrics = analysis.metrics
    _emit_terminal_block(
        "FINAL QUALITY SCORE",
        [
            ("Blur Score", f"{metrics.blur_score:.3f}"),
            ("Noise Score", f"{metrics.noise_score:.3f}"),
            ("Skew Score", f"{metrics.skew_score:.3f}"),
            ("OCR Score", f"{metrics.ocr_score:.3f}"),
            ("DPI Score", f"{metrics.dpi_score:.3f}"),
            ("Contrast Score", f"{metrics.contrast_score:.3f}"),
            ("Brightness Score", f"{metrics.brightness_score:.3f}"),
            ("Text Density", f"{metrics.text_density:.3f}"),
            ("Extracted Text Length", metrics.extracted_text_length),
            ("Word Count", metrics.word_count),
            ("Quality Score", f"{analysis.quality_score:.3f}"),
        ],
    )


def decide_provider(
    analysis: QualityAnalysis,
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> Provider:
    if analysis.document_type == "digital_pdf":
        return "openai" if analysis.quality_score >= quality_threshold else "gemini"

    if analysis.quality_score < quality_threshold:
        return "gemini"

    if analysis.metrics.ocr_confidence >= 0.72 and analysis.metrics.dpi_score >= 0.55:
        return "openai"

    return "gemini"


def _document_type_from_text(native_text_length: int, file_type: str) -> DocumentType:
    if file_type == "image":
        return "image"
    if native_text_length >= DEFAULT_DIGITAL_TEXT_THRESHOLD:
        return "digital_pdf"
    return "scanned_pdf"


def analyze_document_quality(
    path: Path,
    *,
    render_dpi: int = DEFAULT_RENDER_DPI,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ocr_page_limit: int = DEFAULT_OCR_PAGE_LIMIT,
) -> QualityAnalysis:
    _emit_terminal_block(
        "DOCUMENT QUALITY ANALYSIS",
        [
            ("File", path.name),
        ],
        prefix="=" * 60,
    )
    suffix = path.suffix.lower()
    file_type: Literal["pdf", "image", "unsupported"]
    if suffix == ".pdf":
        file_type = "pdf"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        file_type = "image"
    else:
        file_type = "unsupported"

    _emit_terminal_block(
        "DOCUMENT TYPE DETECTION",
        [
            ("Status", "SUCCESS"),
            ("File Type", file_type),
            ("Extension", suffix),
            ("Render DPI", render_dpi),
        ],
    )

    if file_type == "unsupported":
        metrics = QualityMetrics(
            dpi_estimate=0.0,
            dpi_score=0.0,
            blur_variance=0.0,
            blur_score=0.0,
            noise_score=0.0,
            skew_angle=0.0,
            skew_score=0.0,
            contrast_score=0.0,
            brightness_score=0.0,
            ocr_confidence=0.0,
            ocr_score=0.0,
            text_density=0.0,
            extracted_text_length=0,
            word_count=0,
        )
        return QualityAnalysis(
            path=str(path),
            file_type=file_type,
            document_type="unsupported",
            page_count=0,
            analyzed_pages=0,
            render_dpi=render_dpi,
            metrics=metrics,
            quality_score=0.0,
            quality_threshold=quality_threshold,
            recommended_provider="gemini",
            reason=f"unsupported file extension: {suffix}",
            weights=dict(QUALITY_WEIGHTS),
        )

    page_images: list[np.ndarray] = []
    page_sizes: list[tuple[float, float] | None] = []
    source_dpi_estimates: list[float | None] = []
    native_text_length = 0
    native_text_word_count = 0
    page_count = 1

    if file_type == "pdf":
        with fitz.open(str(path)) as pdf_document:
            page_count = len(pdf_document)
            pages_to_analyze = min(page_count, max_pdf_pages)
            for page_index in range(pages_to_analyze):
                page = pdf_document[page_index]
                native_text = str(page.get_text("text") or "")
                native_text_length += len(" ".join(native_text.split()))
                native_text_word_count += len(native_text.split())
                source_dpi_estimates.append(_estimate_pdf_page_source_dpi(page))
                page_images.append(_render_pdf_page(page, render_dpi))
                page_sizes.append((page.rect.width / 72.0, page.rect.height / 72.0))
    else:
        image = _load_image_from_path(path)
        page_images.append(image)
        page_sizes.append(None)
        source_dpi_estimates.append(None)
        page_count = 1

    document_type = _document_type_from_text(native_text_length, file_type)

    metric_rows: list[dict[str, float | int]] = []
    for image_index, image in enumerate(page_images[: max(1, ocr_page_limit if file_type == "pdf" else 1)]):
        page_size = page_sizes[min(image_index, len(page_sizes) - 1)] if page_sizes else None
        source_dpi_estimate = (
            source_dpi_estimates[min(image_index, len(source_dpi_estimates) - 1)]
            if source_dpi_estimates
            else None
        )
        metric_rows.append(
            _analyze_image(
                image,
                source_dpi_estimate=source_dpi_estimate,
                page_size_inches=page_size,
                render_dpi=render_dpi,
            )
        )

    metrics = _combine_metric_values(metric_rows)

    if document_type == "digital_pdf" and metrics.ocr_confidence <= 0.0:
        metrics = QualityMetrics(
            **{
                **asdict(metrics),
                "ocr_confidence": _clamp(native_text_length / 1800.0),
                "ocr_score": _clamp(native_text_length / 1800.0),
                "text_density": _clamp(native_text_length / 1200.0),
                "word_count": max(metrics.word_count, native_text_word_count),
            }
        )

    quality_score = _weight_quality(metrics)
    recommended_provider = decide_provider(
        QualityAnalysis(
            path=str(path),
            file_type=file_type,
            document_type=document_type,
            page_count=page_count,
            analyzed_pages=len(metric_rows),
            render_dpi=render_dpi,
            metrics=metrics,
            quality_score=quality_score,
            quality_threshold=quality_threshold,
            recommended_provider="openai",
            reason="",
            weights=dict(QUALITY_WEIGHTS),
        ),
        quality_threshold=quality_threshold,
    )

    reason = (
        f"document_type={document_type}, quality_score={quality_score:.3f}, "
        f"threshold={quality_threshold:.3f}, provider={recommended_provider}"
    )
    analysis = QualityAnalysis(
        path=str(path),
        file_type=file_type,
        document_type=document_type,
        page_count=page_count,
        analyzed_pages=len(metric_rows),
        render_dpi=render_dpi,
        metrics=metrics,
        quality_score=quality_score,
        quality_threshold=quality_threshold,
        recommended_provider=recommended_provider,
        reason=reason,
        weights=dict(QUALITY_WEIGHTS),
    )

    _emit_quality_progress(analysis)

    _emit_terminal_block(
        "ROUTING DECISION",
        [
            ("Status", "SUCCESS"),
            ("Selected Provider", recommended_provider),
            ("Routing Mode", "provider-recommended"),
            ("Reason", reason),
            ("Quality Score", f"{quality_score:.3f}"),
            ("Quality Threshold", f"{quality_threshold:.3f}"),
        ],
    )

    logger.info("quality_router_analysis=%s", json.dumps(asdict(analysis), ensure_ascii=False, indent=2))
    logger.info("quality_router_weights=%s", json.dumps(QUALITY_WEIGHTS, ensure_ascii=False))
    logger.info(
        "quality_router_score_breakdown=dpi_score=%.3f blur_score=%.3f ocr_confidence=%.3f noise_score=%.3f skew_score=%.3f quality_score=%.3f provider=%s",
        metrics.dpi_score,
        metrics.blur_score,
        metrics.ocr_confidence,
        metrics.noise_score,
        metrics.skew_score,
        quality_score,
        recommended_provider,
    )
    return analysis
