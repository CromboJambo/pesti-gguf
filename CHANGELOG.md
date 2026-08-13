# Changelog

All notable changes to this project will be documented in this file.

The versioning follows `0.x.y` while we're establishing the API, with:
- **x** = major structural shifts (format inference added, breaking parser changes)
- **y** = minor additions or fixes (new features, bug fixes, docs)

## [0.2.3] - 2026-08-12

### Added
- **Format inference engine** (`src/format_inference.rs`)
  - `infer_tensor_format()`: Detects actual quantization from raw data size
  - `validate_tensor_metadata()`: Flags dtype mismatches and suspicious layers
  - Supports Q4_0, Q4K, Q4K_M, Q5_0, Q5K, Q5K_M, Q6K, Q8_0
  - Confidence scoring (0.0–1.0) based on size match quality
- **New module exports**: `format_inference::InferredFormat`, `format_inference::Warning`

### Documentation
- README updated with "honest truth" about GGUF metadata inconsistencies
- Added "The Entropy Problem" section explaining Rust's type-driven approach to format detection
- Format inference usage examples and migration guide

### Testing
- 8 new unit tests for format inference (100% pass rate)
- Full test suite: 32 passed, 5 ignored (missing corpus files)

---

## [0.2.2] - 2026-08-07
- Initial crates.io release
- Full v1/v2/v3 GGUF parsing
- Conformance testing with Qwen2.5 models
