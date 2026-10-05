# Bill Extractor Experiments

This folder hosts a minimal DeepEval experiment for the bill extractor.

- Data: `data/102.pdf` and `data/102_result.json` (from the bill extractor tests)
- Runner: `test_runner.py` compares actual vs expected using a rule-based Accuracy metric and calls the Anthropic-backed extractor.

Notes:
- The actual extractor is implemented in `approaches/anthropic_ai/__init__.py` and uses `AnthropicBulkBillExtractor` under the hood. Ensure Anthropic credentials and any required services are configured.
