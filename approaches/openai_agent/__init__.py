#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

LLM extraction for invoices/bills using the OpenAI Agents SDK + direct PDF upload.
- Inputs: a single file path (PDF or PNG/JPG)
- Output: strict, validated JSON (Pydantic schema) for "best possible" bill coverage
- Async test hook: extract(Path) -> {"status": "success", "data": {...}}
- CLI: one Typer command that runs extraction for one file

Install
-------
pip install -U openai-agents openai pydantic pypdf typer[all]

Environment
-----------
export OPENAI_API_KEY=sk-...

Models
------
Defaults to "gpt-5-mini" for OpenAI. Gemini uses OpenRouter with
"google/gemini-3.1-pro-preview" unless overridden.
(We keep one extraction pipeline and switch only the provider config.)

Why direct PDF upload?
----------------------
We upload the PDF via Files API and pass it to the model as an `input_file`
content item—no rasterization step needed. For images, the Responses API
accepts `input_image` parts; we feed them as base64 data URIs for portability.
Refs: input_file (PDF) and file uploads + Responses items.  # see citations in the README block.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# --- OpenAI clients & Agents SDK ---
from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings  # OpenAI Agents SDK
from agents.models.interface import Model
from agents.models.openai_provider import OpenAIProvider


_REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_REPO_ROOT / ".env.txt", override=False)
load_dotenv(_REPO_ROOT / ".env", override=False)


# ============================================================
# 1) Target schema (matches data/104_result.json)
# ============================================================


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_type: Optional[str] = None  # e.g., GST, CGST, SGST
    line_amount: Optional[float] = None
    is_percent_value: Optional[bool] = None


class Header(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor_name: Optional[str] = None
    invoice_no: Optional[str] = None
    activity: Optional[List[Activity]] = None
    bill_header_uuid: Optional[str] = None
    bill_currency: Optional[str] = None
    bill_date: Optional[str] = None  # YYYY-MM-DD
    exchange_rate: Optional[str] = None
    payment_terms: Optional[str] = None
    pan: Optional[str] = None
    due_date: Optional[str] = None
    bill_amount: Optional[float] = None
    source_of_supply: Optional[str] = None
    destination_of_supply: Optional[str] = None
    gst_in: Optional[str] = None
    gst_treatment: Optional[str] = None
    is_rcm: Optional[bool] = None
    customer_gstin: Optional[str] = None


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line_item_uuid: Optional[str] = None
    item_name: Optional[str] = None
    item_description: Optional[str] = None
    item_quantity: Optional[float] = None
    item_unit_price: Optional[float] = None
    item_amount: Optional[float] = None
    item_product_code: Optional[str] = None
    item_activity: Optional[List[Activity]] = None
    item_ledger: Optional[str] = None


class BillSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: Optional[Header] = None
    line_items: Optional[List[LineItem]] = None
    third_party_product: Optional[str] = None
    extraction_confidence_score: Optional[float] = None
    total_bills: Optional[int] = None
    current_bill: Optional[int] = None
    failed_row_number: Optional[int] = None


# ============================================================
# 2) Agent config and prompt
# ============================================================

SYSTEM_INSTRUCTIONS = """\
You are a meticulous extraction agent for invoices/bills.
Return ONLY a JSON object matching this schema (keys and types):
{
  "header": {
    "vendor_name": str|null,
    "invoice_no": str|null,
    "activity": [ {"activity_type": str|null, "line_amount": number|null, "is_percent_value": bool|null} ]|null,
    "bill_header_uuid": str|null,
    "bill_currency": str|null,
    "bill_date": "YYYY-MM-DD"|null,
    "exchange_rate": str|null,
    "payment_terms": str|null,
    "pan": str|null,
    "due_date": "YYYY-MM-DD"|null,
    "bill_amount": number|null,
    "source_of_supply": str|null,
    "destination_of_supply": str|null,
    "gst_in": str|null,
    "gst_treatment": str|null,
    "is_rcm": bool|null,
    "customer_gstin": str|null,
  },
  "line_items": [
    {
      "line_item_uuid": str|null,
      "item_name": str|null,
      "item_description": str|null,
      "item_quantity": number|null,
      "item_unit_price": number|null,
      "item_amount": number|null,
      "item_product_code": str|null,
      "item_activity": [ {"activity_type": str|null, "line_amount": number|null, "is_percent_value": bool|null} ]|null,
      "item_ledger": str|null
    }
  ]|null,
  "third_party_product": str|null,
  "extraction_confidence_score": number|null,
  "total_bills": number|null,
  "current_bill": number|null,
  "failed_row_number": number|null
}
Rules:
- Parse ONLY what is visible; do not invent values.
- Dates are ISO 8601 (YYYY-MM-DD) when confidently read; if a date field is unknown or missing, output an empty string "" (not null).
- Numbers are JSON numbers (not strings).
- If totals disagree with arithmetic, keep the printed amounts.
- If the bill is large or contains many charges but lacks clear per-item rows, ensure `line_items` includes at least one entry summarizing the charges. When summarizing, `item_name` and `item_description` must be representative of the summarized line's details and grounded in the document content (no fabrication).
- Apply the above summarization rule only for multi-page documents with more than 3 pages; do not force summarized `line_items` for single-page bills.
- State fields normalization:
  - header.source_of_supply MUST be only the state name (e.g., "Karnataka"), no city/country/prefix like "State:".
  - header.destination_of_supply should be inferred from the "Bill To" section's state when not explicitly printed; use only the state name. If no clear Bill To state, leave null.
  - Prefer explicitly printed fields like "Place of Supply"/"State of Supply" when present; otherwise infer as above.
  - If "Source of Supply" is not explicitly printed, derive header.source_of_supply from the Vendor/Supplier address state (the state in the "Vendor", "Seller", or "Supplier" address block).
  - Treat "Place of Supply" as the same concept as destination_of_supply. If a "Place of Supply" value is present, set header.destination_of_supply to that value's state. If absent, fall back to the "Bill To" state as above.
- Activity normalization:
  - header.activity[].activity_type and line_items[].item_activity[].activity_type MUST be one of:
    ["CGST", "IGST", "SGST", "GST", "Tax", "TDS", "Roundoff", "Adjustment", "Discount"]. Use exactly these spellings; if unsure, leave null.
  - For every activity, set `is_percent_value` based on what `line_amount` represents:
    - If recording a percentage (e.g., 18%), set `line_amount` to the numeric percent without the symbol (e.g., 18) and `is_percent_value` = true.
    - If recording a currency amount, set `line_amount` to that amount and `is_percent_value` = false.
    - When both a percent rate and a computed amount are present, prefer the currency amount; set `line_amount` to the amount and `is_percent_value` = false.
    - Do not mix representations; `is_percent_value` must always match the representation stored in `line_amount`.
- Empty-string rule for string fields: When a string-like field is not present on the document or cannot be confidently extracted, output an empty string "" rather than null. This applies to fields such as `header.pan`, `header.bill_header_uuid`, `header.invoice_no`, `header.bill_currency`, `header.exchange_rate`, `header.payment_terms`, `header.due_date`, `header.source_of_supply`, `header.destination_of_supply`, `header.gst_in`, `header.gst_treatment`,`header.customer_gstin`, `third_party_product`, and, within `line_items`, `line_item_uuid`, `item_name`, `item_description`, `item_product_code`, and `item_ledger`, as well as `activity_type` inside any `activity` lists.
- Description formatting:
  - Ensure `line_items[].item_description` is a single continuous line with no newline characters. If the source wraps across lines, join with single spaces, collapse repeated spaces, and trim leading/trailing whitespace. Do not insert line breaks.
- Quantity/price defaults (services or no units):
  - When a line item shows a currency amount but does not print a quantity and/or unit price (common in service invoices), set `item_quantity = 1` and `item_unit_price = item_amount`. Do not emit null for these fields in this case.
  - If a unit price is printed but quantity is not, and an item amount is also printed, prefer `item_quantity = 1` and `item_unit_price = item_amount` rather than back-computing fractional quantities.
  - Only apply these defaults when a reliable item currency amount is visible; if no amount is printed, leave missing numeric fields as null.
- Currency detection:
  - Set `header.bill_currency = "INR"` when the document clearly indicates Indian Rupees via any of: the rupee symbol `₹`, labels like `INR` or `Rs.`/`Rs`. If no explicit currency marking is present but the vendor address clearly indicates India (e.g., address includes "India" or Indian state names) and there is no conflicting currency evidence, infer `INR`.
  - Set `header.bill_currency = "USD"` when the dollar symbol `$` appears near totals/amounts and there is no explicit cue for another dollar currency (e.g., `CAD`, `AUD`, `NZD`, `SGD`, `HKD`, `MXN`, prefixes like `C$`, `A$`). Treat `$` as USD even if the vendor address is in India unless a different dollar code is printed.
  - Precedence for conflicting cues: explicit ISO code (e.g., `INR`, `USD`, `EUR`) near totals/headers > currency symbol (e.g., `₹`, `$`) > address cues. If still ambiguous after applying precedence, leave `bill_currency` empty.
 - GSTIN formatting:
  - Output only the 15-character GSTIN in `header.gst_in` and `header.customer_gstin` — strip labels/prefixes like "GSTIN:", "GST No.", "GSTIN/UIN:", remove spaces/punctuation, and use uppercase. Avoid including any surrounding text.
   - Validate shape loosely: 15 characters, alphanumeric, following the standard structure (state code 2 digits + PAN 10 chars + entity code 1 char + 'Z' + checksum 1 char). If the captured text does not conform, leave `gst_in` empty rather than emitting a malformed value.
   - `header.gst_in` belongs to vendor's GSTIN
   - `header.customer_gstin` belongs to customer's GSTIN
 - HSN/SAC propagation:
   - If an HSN/SAC code is printed outside the items table (e.g., in the header or summary area) and there is exactly one line item, set that line’s `item_product_code` to the detected HSN/SAC value when the line does not already have a product code. Use the exact printed code (trimmed), and do not infer or fabricate.
   - If multiple HSN/SAC codes are present or there are multiple line items, do not propagate; only fill per-line when the code appears alongside that specific line.
 - Freight handling:
   - Treat any freight/shipping/transport/delivery/courier charge (e.g., labels such as "Freight", "Freight Charge(s)", "Shipping Charges") as a regular line item, NOT as an activity.
   - When only a lump-sum freight amount is printed in totals or a summary box (with no dedicated row), create a separate line item capturing it. Use the printed label for `item_name` (or "Freight Charges" if unlabeled) and set `item_amount` to the printed currency amount. Do not set a quantity or unit price unless explicitly printed. Do not record these charges in `header.activity`.
"""

USER_TASK_TEXT = (
    "Extract the bill/invoice into the output schema carefully and completely."
)

DEFAULT_MODEL = os.environ.get("BILL_EXTRACTOR_MODEL", "gpt-5-mini")
DEFAULT_GEMINI_MODEL = os.environ.get(
    "BILL_EXTRACTOR_GEMINI_MODEL", "google/gemini-3.1-pro-preview"
)
MAX_PAGES = 12


def _load_bill_openai_key() -> Optional[str]:
    return os.getenv("BILL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _load_bill_openrouter_api_key() -> Optional[str]:
    return os.getenv("BILL_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _load_bill_openrouter_base_url() -> Optional[str]:
    return os.getenv("BILL_OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")


def _default_model_for_provider(provider: str) -> str:
    provider_name = provider.strip().lower()
    if provider_name == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_MODEL


def _build_openai_client(provider: str, *, timeout: float = 60.0) -> AsyncOpenAI:
    provider_name = provider.strip().lower()
    openrouter_api_key = _load_bill_openrouter_api_key()
    openrouter_base_url = _load_bill_openrouter_base_url()
    openai_api_key = _load_bill_openai_key()

    if provider_name == "gemini":
        if not openrouter_api_key or not openrouter_base_url:
            raise ValueError(
                "Gemini extraction requires OPENROUTER_API_KEY and OPENROUTER_BASE_URL"
            )
        return AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
            default_headers={
                "X-OpenRouter-Title": "Bill Extractor POC",
                "HTTP-Referer": "http://localhost",
            },
            timeout=timeout,
        )

    if provider_name == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI extraction requires OPENAI_API_KEY")
        return AsyncOpenAI(api_key=openai_api_key, timeout=timeout)

    if not openai_api_key:
        raise ValueError("OpenAI extraction requires OPENAI_API_KEY")
    return AsyncOpenAI(api_key=openai_api_key, timeout=timeout)


def _build_agent_model(provider: str, model_name: str, *, timeout: float = 60.0) -> Model:
    client = _build_openai_client(provider, timeout=timeout)
    return OpenAIProvider(openai_client=client).get_model(model_name)


def _build_model_settings(model_name: str) -> ModelSettings:
    """
    Configure model settings.
    - For GPT‑5 family, explicitly request low reasoning effort.
    - For non-reasoning models, keep deterministic settings.
    """
    if model_name.lower().startswith("gpt-5"):
        # Responses supports `reasoning={"effort": "low"}` for reasoning models.
        # The Agents SDK forwards this via ModelSettings.
        return ModelSettings(reasoning={"effort": "low"})
    return ModelSettings(temperature=0.0)


def _build_agent(
    model: str,
    *,
    provider: str = "openai",
    instructions: str = None,
) -> Agent:
    return Agent(
        name="Bill Extractor",
        instructions=instructions or SYSTEM_INSTRUCTIONS,
        output_type=BillSchema,  # structured outputs (Pydantic) via Agents SDK
        model=_build_agent_model(provider, model),
        model_settings=_build_model_settings(model),
    )


# ============================================================
# 3) PDF & image helpers (no PDF->image conversion)
# ============================================================


def _pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return len(reader.pages)


async def _upload_pdf_and_get_file_id(path: Path) -> str:
    """
    Uploads a PDF using OpenAI Files API and returns its file_id.
    We use purpose='user_data' to feed the file as model input later.  # canonical pattern
    """
    client = AsyncOpenAI()
    # PyPI docs: file param accepts a Path directly; the async client reads it for you.
    # https://pypi.org/project/openai/1.68.2/
    up = await client.files.create(file=path, purpose="user_data")
    return up.id


def _pdf_path_to_data_uri(path: Path | str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def _image_path_to_data_uri(path: Path) -> str:
    """
    Build a data URI for PNG/JPG without resizing (simple & robust).
    The Responses API accepts image inputs as data URLs on `input_image` items.
    """
    mime, _ = mimetypes.guess_type(path.as_posix())
    if not mime:
        # default to PNG if unknown
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_items_for_pdf(file_id: str) -> List[dict]:
    return [
        {"type": "input_text", "text": USER_TASK_TEXT},
        {"type": "input_file", "file_id": file_id},  # PDF direct input
    ]


def _build_items_for_pdf_data(path: Path | str) -> List[dict]:
    return [
        {"type": "input_text", "text": USER_TASK_TEXT},
        {
            "type": "input_file",
            "filename": Path(path).name,
            "file_data": _pdf_path_to_data_uri(path),
        },
    ]


def _build_items_for_image(data_uri: str) -> List[dict]:
    return [
        {"type": "input_text", "text": USER_TASK_TEXT},
        {"type": "input_image", "image_url": data_uri},  # image via data-URI
    ]


async def _run_agent(
    items: List[dict], *, model: str, provider: str = "openai", instructions: str = None
) -> BillSchema:
    agent = _build_agent(model, provider=provider, instructions=instructions)
    # Responses API now expects top-level "input" items such as a
    # single "message". Wrap our content parts into one user message.
    message = {
        "type": "message",
        "role": "user",
        "content": items,
    }
    result = await Runner.run(agent, [message])  # Agents SDK (Responses under the hood)
    bill: BillSchema = result.final_output_as(BillSchema)  # typed validation
    return bill


# ============================================================
# 4) Post-processing helpers
# ============================================================


def _coalesce_empty_strings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert None to "" for known string fields so downstreams see empty
    strings instead of null. Leaves numeric/bool fields as-is (may stay None
    and be dropped later). Works in-place and returns the payload.

    Targeted fields include:
      - header: vendor_name, invoice_no, bill_header_uuid, bill_currency,
        bill_date, exchange_rate, payment_terms, pan, due_date,
        source_of_supply, destination_of_supply, gst_in, gst_treatment
      - line_items[].: line_item_uuid, item_name, item_description,
        item_product_code, item_ledger
      - activity: activity_type (both header.activity[] and item_activity[])
      - top-level: third_party_product
    """

    header_string_fields = {
        "vendor_name",
        "invoice_no",
        "bill_header_uuid",
        "bill_currency",
        "bill_date",
        "exchange_rate",
        "payment_terms",
        "pan",
        "due_date",
        "source_of_supply",
        "destination_of_supply",
        "gst_in",
        "gst_treatment",
        "customer_gstin",
    }

    line_item_string_fields = {
        "line_item_uuid",
        "item_name",
        "item_description",
        "item_product_code",
        "item_ledger",
    }

    def _fix_activity_list(act_list: Any) -> Any:
        if isinstance(act_list, list):
            for a in act_list:
                if isinstance(a, dict) and a.get("activity_type") is None:
                    a["activity_type"] = ""
        return act_list

    # top-level
    if payload.get("third_party_product") is None:
        payload["third_party_product"] = ""

    # header-level
    header = payload.get("header")
    if isinstance(header, dict):
        for k in header_string_fields:
            if header.get(k) is None:
                header[k] = ""
        # header.activity
        if "activity" in header:
            header["activity"] = _fix_activity_list(header.get("activity"))

    # line_items
    line_items = payload.get("line_items")
    if isinstance(line_items, list):
        for li in line_items:
            if not isinstance(li, dict):
                continue
            for k in line_item_string_fields:
                if li.get(k) is None:
                    li[k] = ""
            if "item_activity" in li:
                li["item_activity"] = _fix_activity_list(li.get("item_activity"))

    return payload


def _strip_nones(payload: Any) -> Any:
    """Recursively remove keys with value None to keep JSON compact."""
    if isinstance(payload, dict):
        return {k: _strip_nones(v) for k, v in payload.items() if v is not None}
    if isinstance(payload, list):
        return [_strip_nones(v) for v in payload]
    return payload


# ============================================================
# 4) Public async API (test hook)
# ============================================================


@dataclass
class ExtractResult:
    data: BillSchema
    model: str
    source: str


async def extract(
    pdf_path: Path,
    model: Optional[str] = None,
    *,
    provider: str = "openai",
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async test hook exactly as requested.
    Accepts a Path to a single PDF or PNG/JPG file.
    For PDFs: enforce ≤ 3 pages, else raise.
    Returns: {"status": "success", "data": bill_dict}
    """
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    ext = pdf_path.suffix.lower()
    provider_name = provider.strip().lower()
    selected_model = model or _default_model_for_provider(provider_name)
    cleanup_file_id: Optional[str] = None

    if ext == ".pdf":
        pages = _pdf_page_count(pdf_path)
        if pages > MAX_PAGES:
            raise ValueError(f"PDF has {pages} pages; limit is {MAX_PAGES}.")
        if provider_name == "gemini":
            items = _build_items_for_pdf_data(pdf_path)
        else:
            file_id = await _upload_pdf_and_get_file_id(pdf_path)
            cleanup_file_id = file_id
            items = _build_items_for_pdf(file_id)
    elif ext in {".png", ".jpg", ".jpeg"}:
        data_uri = _image_path_to_data_uri(pdf_path)
        items = _build_items_for_image(data_uri)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or PNG/JPG.")

    try:
        bill = await _run_agent(
            items,
            model=selected_model,
            provider=provider_name,
            instructions=instructions,
        )
    finally:
        if cleanup_file_id:
            try:
                client = _build_openai_client(provider_name)
                await client.files.delete(cleanup_file_id)
            except Exception:
                pass

    # Start with a full dump so we can normalize string fields set to None.
    data = bill.model_dump(exclude_none=False)
    data = _coalesce_empty_strings(data)
    # Drop remaining None (for numeric/bool fields we keep previous behavior of omission).
    data = _strip_nones(data)
    # Ensure required envelope-compatible fields are present with sensible defaults
    data.setdefault("third_party_product", "tally")
    data.setdefault("extraction_confidence_score", 0.0)
    data.setdefault("total_bills", 1)
    data.setdefault("current_bill", 1)
    data.setdefault("failed_row_number", -1)
    return {
        "status": "success",
        "data": data,
        "message": "Bill data extracted successfully",
    }


# ============================================================
# 5) Simple Typer CLI (single command)
# ============================================================


def _write_json(path: Optional[Path], payload: Dict[str, Any], *, also_print: bool):
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        typer.echo(f"✅ Wrote: {path}")
    if also_print or not path:
        typer.echo(text)


def main(
    path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input file (PDF ≤3 pages, or PNG/JPG)"
    ),
    provider: str = typer.Option(
        os.environ.get("BILL_EXTRACTOR_PROVIDER", "auto"),
        "--provider",
        "-p",
        help="Provider to use: auto, openai, or gemini",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the model string for the selected provider",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write JSON to this path"
    ),
    print_json: bool = typer.Option(False, "--print", help="Also print JSON to stdout"),
):
    async def _run():
        from ..routing import extract as routed_extract

        res = await routed_extract(
            path,
            provider=provider,
            openai_model=model,
            gemini_model=model,
        )
        _write_json(out, res["data"], also_print=print_json)

    asyncio.run(_run())


if __name__ == "__main__":
    typer.run(main)
