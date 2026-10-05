import asyncio
import json
import os
from pathlib import Path

from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase

from .approaches.routing import extract as anthropic_extract
from .metric import HeaderAndLineItemsAccuracy


# ---------- Paths ----------
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
INPUT_PDF = DATA_DIR / "101.pdf"
EXPECTED_JSON = DATA_DIR / "101_result.json"


# ---------- Helper: load expected JSON ----------
def _load_expected() -> dict:
    return json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))


async def _extract_actual_from_pdf(pdf_path: Path) -> dict:
    """Run the Anthropic-backed extractor via local approaches wrapper."""
    return await anthropic_extract(pdf_path, provider=os.getenv("BILL_EXTRACTOR_PROVIDER", "auto"))


# ---------- DeepEval test ----------
def test_bill_extraction_accuracy():
    assert INPUT_PDF.exists(), f"Missing input file: {INPUT_PDF}"
    expected = _load_expected()

    # Run extraction (async)
    actual = asyncio.run(_extract_actual_from_pdf(INPUT_PDF))

    # Ensure both actual and expected include the top-level "data" key
    # because the metric only reads under that envelope.
    test_case = LLMTestCase(
        input=str(INPUT_PDF),
        actual_output=json.dumps(actual, ensure_ascii=False),
        expected_output=json.dumps({"data": expected}, ensure_ascii=False),
    )

    metric = HeaderAndLineItemsAccuracy(threshold=1.0)
    assert_test(test_case, [metric])
    evaluate([test_case], [metric])


if __name__ == "__main__":
    test_bill_extraction_accuracy()
