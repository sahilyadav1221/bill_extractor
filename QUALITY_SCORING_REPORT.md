# Bill Extractor Quality Scoring Report

## Purpose

This report explains how the bill_extractor quality scoring pipeline works, which parameters are measured, how each parameter contributes to the final quality score, and why those parameters are used.

The implementation lives in `experiments/bill_extractor/approaches/quality_router.py` and is used by `experiments/bill_extractor/approaches/routing.py` before extraction begins.

## High-Level Flow

The quality pipeline works in this order:

1. Detect whether the input is a PDF, image, or unsupported file.
2. For PDFs, render pages for analysis and also inspect the source PDF metadata when possible.
3. Compute per-page image quality metrics.
4. Aggregate metrics across analyzed pages.
5. Convert those metrics into normalized scores.
6. Combine the normalized scores using weighted scoring.
7. Produce a final `quality_score` and a routing recommendation.
8. Print structured checkpoint output in the terminal.

## Inputs Used By The Scoring Pipeline

The pipeline uses different signals depending on document type:

- PDF file structure and page count
- Page text extracted from the PDF native text layer
- Page image metadata when available
- Rendered page bitmap
- OCR output when PaddleOCR is available
- Basic image statistics such as variance, contrast, and brightness

## Document Type Detection

### What It Does

The pipeline first checks the file extension and classifies the document as:

- `pdf`
- `image`
- `unsupported`

Then it uses the native text layer in PDFs to estimate whether the document is:

- `digital_pdf`
- `scanned_pdf`
- `image`
- `unsupported`

### Why It Matters

Document type changes how the scoring behaves:

- A digital PDF usually has a clean text layer and should often route to the higher-confidence extraction path.
- A scanned PDF usually depends on OCR and image quality.
- A standalone image has no PDF text layer, so the OCR/image quality path is more important.

### Important Threshold

The current implementation uses `DEFAULT_DIGITAL_TEXT_THRESHOLD = 120`.

If the PDF text layer has at least 120 non-whitespace characters across analyzed pages, the document is treated as `digital_pdf`.

## Quality Dimensions

The final score is built from these weighted dimensions:

- `dpi_score`
- `blur_score`
- `ocr_confidence`
- `noise_score`
- `skew_score`

Additional metrics are also measured and printed, even though they do not directly contribute to the weighted score:

- `dpi_estimate`
- `blur_variance`
- `contrast_score`
- `brightness_score`
- `ocr_score`
- `text_density`
- `extracted_text_length`
- `word_count`
- `skew_angle`

## Metric Definitions And Reasons

### 1. DPI Estimate

#### Metric

- `dpi_estimate`
- `dpi_score`

#### What It Measures

This estimates how dense the source document is in pixels per inch.

#### How It Is Calculated

The pipeline tries to recover original source DPI from the PDF page metadata first:

- It inspects `page.get_text("rawdict")`
- It looks for image blocks and their bounding boxes
- It also checks `page.get_images(full=True)` and `page.get_image_rects(...)`

If source metadata is available, that estimate is preferred.

If source metadata is not available, the pipeline falls back to a heuristic using:

- rendered image width and height
- page size in inches
- the current render DPI

Then it converts the estimate into a normalized score:

```text
dpi_score = clamp(dpi_estimate / render_dpi)
```

#### Why It Is Included

DPI is a strong signal for scan quality:

- Higher DPI usually means sharper text and better OCR reliability.
- Low DPI usually means blurry glyphs and weaker extraction.

#### Notes

This is not always a perfect “original scan DPI” measurement.

- For scanned PDFs with embedded raster images, it can estimate the original image density more realistically.
- For vector PDFs or PDFs without useful image metadata, the fallback is only an approximation.

### 2. Blur Score

#### Metric

- `blur_variance`
- `blur_score`

#### What It Measures

This measures sharpness using the variance of the Laplacian of the grayscale image.

#### How It Is Calculated

- Convert image to grayscale.
- Compute Laplacian variance.
- Normalize it by dividing by a constant (`1200.0`).
- Clamp it to the range `[0, 1]`.

```text
blur_score = clamp(blur_variance / 1200.0)
```

#### Why It Is Included

Blur directly affects OCR and human readability:

- Sharp documents are easier to extract.
- Blurry scans usually produce incorrect words, missing digits, and broken line items.

### 3. OCR Confidence

#### Metric

- `ocr_confidence`
- `ocr_score`

#### What It Measures

This reflects how confident OCR is on the analyzed image.

#### How It Is Calculated

The pipeline uses PaddleOCR when available.

- OCR is run on the analyzed image.
- Text fragments and confidences are collected.
- Average OCR confidence is computed.
- Confidence is clamped to `[0, 1]`.

If PaddleOCR is not installed or OCR fails, the pipeline falls back to `0.0`.

The OCR score is derived from:

- OCR confidence
- text density
- word count

#### Why It Is Included

OCR confidence is one of the best practical indicators of whether scanned content is readable.

- High OCR confidence usually means clean text and strong extraction potential.
- Low OCR confidence usually means poor scan quality, noise, blur, or skew.

### 4. Noise Score

#### Metric

- `noise_score`

#### What It Measures

This estimates how much noise remains after denoising.

#### How It Is Calculated

- Convert image to grayscale.
- Denoise the grayscale image.
- Compare the original grayscale image with the denoised image.
- Measure residual noise.
- Convert that into a score where less residual noise means a higher score.

```text
noise_score = clamp(1.0 - (noise_raw * 1.8))
```

#### Why It Is Included

Noise affects OCR by introducing false edges, broken characters, and visual artifacts.

- Clean documents produce stable OCR.
- Noisy documents often degrade extraction accuracy.

### 5. Skew Score

#### Metric

- `skew_angle`
- `skew_score`

#### What It Measures

This measures whether the text is rotated or misaligned.

#### How It Is Calculated

- Threshold the image to isolate dark pixels.
- Estimate dominant orientation using covariance and principal component analysis.
- Convert the angle to a normalized score.

```text
skew_score = clamp(1.0 - min(abs(skew_angle), 15.0) / 15.0)
```

#### Why It Is Included

Skew is a major OCR failure mode:

- Slight rotation can reduce OCR stability.
- Strong rotation often breaks line detection and field extraction.

### 6. Contrast Score

#### Metric

- `contrast_score`

#### What It Measures

This measures intensity spread in the grayscale image.

#### How It Is Calculated

- Compute standard deviation of grayscale values.
- Normalize it against a constant.
- Clamp to `[0, 1]`.

```text
contrast_score = clamp(std(gray) / 64.0)
```

#### Why It Is Included

Contrast helps identify whether text stands out from the background.

- Better contrast usually improves readability.
- Low contrast often means faded scans or poor lighting.

#### Why It Is Not Weighted Directly

It is tracked for visibility and diagnostics, but the final score prioritizes the strongest routing signals:

- DPI
- Blur
- OCR confidence
- Noise
- Skew

### 7. Brightness Score

#### Metric

- `brightness_score`

#### What It Measures

This measures how close the image brightness is to a balanced midpoint.

#### How It Is Calculated

- Compute the mean grayscale brightness.
- Measure distance from the midpoint `127.5`.
- Convert that into a normalized score.

#### Why It Is Included

Very dark or very bright pages often indicate poor scan conditions.

#### Why It Is Not Weighted Directly

Brightness is useful as a supporting diagnostic signal, but not as strong as the main extraction-quality signals.

### 8. Text Density

#### Metric

- `text_density`

#### What It Measures

This estimates how much text was found relative to the effective page area.

#### How It Is Calculated

It uses:

- OCR text length
- image dimensions
- render DPI

The idea is to measure how text-rich the page appears.

#### Why It Is Included

Text density helps distinguish:

- dense invoice pages
- sparse cover pages
- weak OCR output

It is especially useful as a supporting signal for OCR quality.

### 9. Extracted Text Length

#### Metric

- `extracted_text_length`

#### What It Measures

This is the total length of OCR-extracted text.

#### Why It Is Included

It helps identify whether the OCR engine found real content or almost nothing.

- Larger extracted text length usually indicates stronger OCR coverage.
- Near-zero text length can indicate failure or blank pages.

### 10. Word Count

#### Metric

- `word_count`

#### What It Measures

This is the number of OCR-extracted words.

#### Why It Is Included

It gives another practical signal that OCR is seeing actual bill content.

- More words usually imply richer text capture.
- Very low word count may mean the image is unreadable or sparse.

## Weightage Used In Final Score

The current `QUALITY_WEIGHTS` are:

```text
DPI score        = 0.25
Blur score       = 0.25
OCR confidence   = 0.30
Noise score      = 0.10
Skew score       = 0.10
```

## Why These Weights Were Chosen

### OCR Confidence = 0.30

This has the highest weight because OCR confidence is the most direct signal of whether the document is readable by the extraction flow.

### DPI Score = 0.25

DPI strongly affects scan clarity and OCR success.

### Blur Score = 0.25

Blur is another primary determinant of scan usability.

### Noise Score = 0.10

Noise matters, but it is usually secondary to blur, DPI, and OCR confidence.

### Skew Score = 0.10

Skew affects extraction quality, but usually as a supporting signal rather than the only deciding factor.

## Final Quality Score Formula

The final score is a weighted sum of the main normalized metrics:

```text
quality_score =
    dpi_score * 0.25 +
    blur_score * 0.25 +
    ocr_confidence * 0.30 +
    noise_score * 0.10 +
    skew_score * 0.10
```

## How The Final Score Is Used

### Auto Routing

When provider is `auto`, the quality score helps decide whether to route to `openai` or `gemini`.

- Digital PDFs use the quality threshold directly.
- Scanned PDFs and images use quality plus extra OCR/DPI conditions.

### Forced Routing

If provider is explicitly forced to `openai` or `gemini`:

- The full quality score is still computed.
- All metrics are still printed.
- The requested provider is still used.

This is useful for monitoring document quality without changing routing behavior.

## Runtime Checkpoints Printed To Terminal

The pipeline now prints structured checkpoints such as:

- document detection completed
- blur score calculated
- noise score calculated
- skew score calculated
- OCR score/confidence calculated
- DPI score calculated
- quality score calculation started
- final quality score generated
- provider routing decision completed
- extraction provider selected
- before extraction handoff

These checkpoints are intended to show the scoring workflow step by step in the terminal.

## Summary

The quality scoring system is designed to estimate whether a bill or document is suitable for reliable extraction.

The final score is not based on a single signal. It combines multiple document-quality dimensions because each one captures a different failure mode:

- low resolution
- blur
- OCR uncertainty
- noise
- skew

This gives the router a practical quality signal before extraction begins, and it also provides terminal-visible progress checkpoints for debugging and monitoring.
