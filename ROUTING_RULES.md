# Bill Extractor Quality and Routing Rules

## Document Type Detection
- The input is classified as `pdf`, `image`, or `unsupported`.
- For PDFs, native text is checked:
  - If the PDF has enough native text, it is classified as `digital_pdf`.
  - If native text is low, it is classified as `scanned_pdf`.
- Images are treated as `image`.

## Quality Analysis
- Always runs regardless of document type.
- Quality scores are computed for providers: **auto**, **openai**, and **gemini**.
- Even when a provider is forced, the quality score is still calculated and printed.
- Forced provider selection does not override quality scoring.

## The 5 Main Quality Parameters
| Parameter       | Description |
|-----------------|--------------|
| dpi_score       | Measures resolution / clarity of the source document |
| blur_score      | Measures how sharp the image is |
| ocr_confidence  | Measures OCR confidence in extracted text |
| noise_score     | Measures how clean the image is after denoising |
| skew_score      | Measures how aligned or rotated the text is |

## Final Quality Score Formula
- The final score is a weighted sum of the 5 parameters.
- Current weights:
  - dpi_score = 0.25
  - blur_score = 0.25
  - ocr_confidence = 0.30
  - noise_score = 0.10
  - skew_score = 0.10

## Routing Rules
### For digital_pdf:
- If `document_type == "digital_pdf"`
  - If `quality_score >= threshold` → route to **openai**
  - Else → route to **gemini**
- OCR confidence and skew rules are not used here as extra gates.

### For scanned_pdf and image:
- If `document_type != "digital_pdf"`
  - If `quality_score < threshold` → route to **gemini**
  - Else if `quality_score >= threshold`, then:
    - If `ocr_confidence >= 0.72` AND `dpi_score >= 0.55`, route to **openai**
    - Otherwise → route to **gemini**

## Forced Provider Rule
- If `provider="openai"`: always route to OpenAI, but still compute and print full quality score.
- If `provider="gemini"`: always route to Gemini, but still compute and print full quality score.

## Selected Model Rule
- When OpenAI is selected: default model is **gpt-5-mini** unless overridden.
- When Gemini is selected: default model is **google/gemini-3.1-pro-preview** unless overridden.

## Logging Rule
The pipeline prints structured terminal blocks in execution order showing:
document type detection, blur analysis, noise analysis, skew analysis, OCR analysis, DPI analysis, final quality score, routing decision, extraction process.

## Very Important Summary
- For digital_pdf: main decision based on whether *quality_score* vs *threshold* meets criteria.
- For scanned_pdf / image: high quality alone isn't enough; OCR confidence and DPI must also be acceptable.
- Forced providers always win over routing logic; however, quality analysis still runs.


=================================================================================================================

# Decision Path and Score Composition Analysis

I’ve got the exact decision path and score composition, so I’m checking only the local conditions that explain why a score above threshold still routes to Gemini.

## Review of Code and Results
- Reviewed `quality_router.py` and `routing.py`, analyzed OCR results.
- The reason Gemini was chosen is not the threshold alone.

### Routing Logic in `quality_router.py:565`
For scanned PDFs:
- If `quality_score < threshold`, use Gemini.
- If `quality_score >= threshold`, still use Gemini unless both:
  - `ocr_confidence >= 0.72`
  - `dpi_score >= 0.55`

Your run had:
- `quality_score = 0.589`, which is above `0.580`, but
- `ocr_confidence = 0.000`, so the second condition fails.

This explains why it still routed to Gemini. The threshold only makes the PDF eligible for routing; it does not automatically select OpenAI for scanned PDFs.

## Score Calculation Details
The score is exactly the weighted sum from `quality_router.py:375`:



y_qualty_score = dpi_score * 0.25 + blur_score * 0.25 + ocr_confidence * 0.30 + noise_score * 0.10 + skew_score * 0.10

Using your printed values:
- **dpi_score** = 0.959 → `0.959 × 0.25 = 0.23975`
- **blur_score** = 1.000 → `1.000 × 0.25 = 0.25000`
- **ocr_confidence** = 0.000 → `0.000 × 0.30 = 0.00000`
- **noise_score** = 0.991 → `0.991 × 0.10 = 0.09910`
- **skew_score** = 0.000 → `0.000 × 0.10 = 0.00000`

### Total Score Calculation:

total ≈ ० .58885 ≈ **~0 .589**

This demonstrates that even with OCR and skew at zero, the overall score can be above 0.58 because the other signals are strong enough.

## OCR Handling Explanation
There is no surfaced exception in your output for OCR failures because `_extract_words_and_confidence()` catches these failures and returns zeros instead of breaking the pipeline.
Possible reasons include:
- PaddleOCR returned no usable result.
- OCR output structure did not produce confidence/text fragments.
- Internal OCR backend failure that was swallowed by error handling.

Since this function is designed to fail softly, the pipeline continues running, printing zero values instead of throwing errors.

## Skew Calculation Details
In `_analyze_image()`, skew score is computed as:

```python
skew_score = clamp(1 - min(abs(skew_angle), 15) / 15)

def clamp(x):
    return max(0, min(x, 1))
```

### What This Means

The skew score measures how tilted the scanned document is.

- If the document angle is close to `0°`, the document is considered straight.
  - The `skew_score` becomes close to `1`.

- As the skew angle increases toward `±15°`, the score gradually decreases.

- If the skew angle reaches `15°` or more, the value is clipped to the maximum penalty.
  - The `skew_score` becomes close to `0`.

### Example

Your document had a skew angle of approximately `-89.20°`.

Since this exceeds the allowed `±15°` range:

- the value is clipped,
- maximum penalty is applied,
- and the final skew score becomes `0`.

### Effect On Final Quality Score

Skew contributes only `10%` weight to the total quality score.

Because of this:

- skew alone cannot dominate provider routing,
- it only slightly affects the final decision.

### Current Routing Logic

The intended routing behavior is:

- Good quality + strong OCR confidence → OpenAI
- Low quality or weak OCR confidence → Gemini

### Your Current Case

This scanned PDF has:

- strong DPI,
- good blur score,
- but zero OCR confidence.

Because OCR confidence is extremely weak, the router prefers Gemini.

### Possible Future Tuning

The routing logic can be adjusted in future versions.

For example:

- high DPI,
- good blur quality,
- and otherwise clean scans

could be allowed to outweigh weak OCR confidence when deciding whether to use OpenAI.
