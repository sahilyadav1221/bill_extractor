from __future__ import annotations

from pathlib import Path

import pytest

from experiments.bill_extractor.approaches import routing
from experiments.bill_extractor.approaches import quality_router


class _FakePage:
	def __init__(self) -> None:
		self.calls = 0

	def get_text(self, mode: str):
		assert mode == "rawdict"
		self.calls += 1
		return {
			"blocks": [
				{
					"type": 1,
					"width": 2400,
					"height": 3300,
					"bbox": (0.0, 0.0, 595.2, 841.68),
				}
			]
		}

	def get_images(self, full: bool = False):
		return []


def test_forced_provider_still_runs_quality_analysis(monkeypatch, capsys):
	calls: list[Path] = []

	def fake_analyze(path: Path, **kwargs):
		calls.append(path)
		return quality_router.QualityAnalysis(
			path=str(path),
			file_type="pdf",
			document_type="digital_pdf",
			page_count=1,
			analyzed_pages=1,
			render_dpi=quality_router.DEFAULT_RENDER_DPI,
			metrics=quality_router.QualityMetrics(
				dpi_estimate=300.0,
				dpi_score=1.0,
				blur_variance=1.0,
				blur_score=1.0,
				noise_score=1.0,
				skew_angle=0.0,
				skew_score=1.0,
				contrast_score=1.0,
				brightness_score=1.0,
				ocr_confidence=1.0,
				ocr_score=1.0,
				text_density=1.0,
				extracted_text_length=10,
				word_count=2,
			),
			quality_score=0.99,
			quality_threshold=0.58,
			recommended_provider="openai",
			reason="mocked quality analysis",
			weights=dict(quality_router.QUALITY_WEIGHTS),
		)

	monkeypatch.setattr(routing, "analyze_document_quality", fake_analyze)

	decision = routing.choose_route(Path("experiments/bill_extractor/data/12.pdf"), provider="gemini")

	assert decision.provider == "gemini"
	assert calls == [Path("experiments/bill_extractor/data/12.pdf")]
	assert "forced gemini provider" in decision.reason

	output = capsys.readouterr().out
	assert "quality_router_progress=" not in output


def test_estimate_pdf_page_source_dpi_uses_embedded_image_metadata():
	page = _FakePage()

	dpi = quality_router._estimate_pdf_page_source_dpi(page)

	assert dpi == pytest.approx(282.7, rel=0.02)
	assert page.calls == 1
