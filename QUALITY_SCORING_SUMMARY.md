# Bill Extractor Quality Score Summary

## What the Quality Score Is

The quality score is a single number used to estimate whether a bill or document is likely to be readable and safe to send into the extraction agent.

It is not the extraction result itself. It is a pre-check that summarizes document quality before routing.

## Why We Calculate It

We calculate quality score to answer one practical question:

**Is this document good enough for reliable extraction, or should we route it differently?**

In the current bill_extractor flow, this score helps us:

- decide provider when `provider="auto"`
- still inspect document quality when provider is forced to `openai` or `gemini`
- print progress checkpoints in terminal during runtime
- understand why a document may fail or produce weak extraction results

## Current Use In The Pipeline

The quality score is currently used before extraction starts.

- If provider is `auto`, the score influences provider routing.
- If provider is forced, the score is still calculated and printed, but the forced provider remains selected.

So the score is both a routing signal and a runtime quality trace.

## The 5 Main Parameters

The final quality score is built from these five parameters:

- `dpi_score`
- `blur_score`
- `ocr_confidence`
- `noise_score`
- `skew_score`

These five are used because they cover the main document failure modes:

- low resolution
- blur
- weak OCR readability
- image noise
- rotated or misaligned text

## Why These 5 Parameters Matter

### 1. DPI Score

**What it tells us:** how strong the document resolution appears.

**Why we use it:** low-resolution documents are harder for OCR and often lose digits, symbols, and small text.

**Use cases:**

- scanned bills with compressed images
- low-quality camera captures
- blurry text caused by small source resolution

**Other signals considered for this parameter:**

- source PDF image metadata when available
- page size in inches
- rendered image dimensions as fallback

**Weightage:** `0.25`

### 2. Blur Score

**What it tells us:** how sharp the text and edges appear.

**Why we use it:** blur directly hurts OCR accuracy and makes character boundaries harder to detect.

**Use cases:**

- out-of-focus scans
- motion blur from camera-captured bills
- soft or washed-out page images

**Other signals considered for this parameter:**

- Laplacian variance on grayscale image
- sharpened edge response indirectly through the variance measure

**Weightage:** `0.25`

### 3. OCR Confidence

**What it tells us:** how confident OCR is in the detected text.

**Why we use it:** this is one of the clearest indicators of whether the document is readable in practice.

**Use cases:**

- documents with clean printed text
- pages where OCR can extract vendor names, numbers, and line items reliably
- scanned bills where OCR output quality is the main extraction risk

**Other signals considered for this parameter:**

- PaddleOCR confidence values
- OCR text length
- word count
- text density

**Weightage:** `0.30`

### 4. Noise Score

**What it tells us:** how clean the page is after denoising.

**Why we use it:** noisy images can introduce false edges, broken characters, and OCR mistakes.

**Use cases:**

- grainy photos
- background artifacts
- scanned documents with compression noise

**Other signals considered for this parameter:**

- difference between original grayscale and denoised image
- residual pixel noise after cleanup

**Weightage:** `0.10`

### 5. Skew Score

**What it tells us:** how well aligned the text is on the page.

**Why we use it:** rotated text is harder for OCR to segment and can reduce extraction quality.

**Use cases:**

- slightly tilted phone photos
- scanned pages placed at an angle
- documents with visible page rotation

**Other signals considered for this parameter:**

- detected skew angle
- thresholded text alignment on the page

**Weightage:** `0.10`

## Final Score Formula

The final quality score is a weighted combination of the five parameters:

```text
quality_score =
    dpi_score * 0.25 +
    blur_score * 0.25 +
    ocr_confidence * 0.30 +
    noise_score * 0.10 +
    skew_score * 0.10
```

## How To Read The Final Score

- A higher score means the document is likely easier to extract.
- A lower score means the document may need a different provider or may be lower confidence.
- The score does not replace extraction; it only helps route and monitor the extraction flow.

## Practical Meaning In The Current System

Right now the score is useful for three things:

1. **Provider routing** when provider is `auto`
2. **Quality visibility** when provider is forced
3. **Debugging and progress tracking** because every important score stage is printed in the terminal

## Short Conclusion

The quality score is a compact document-readability signal. We compute it because it helps us decide routing, understand document health, and explain extraction risk before the agent runs.

The five parameters were chosen because they capture the main document quality problems that affect bill extraction: resolution, blur, OCR reliability, noise, and skew.
