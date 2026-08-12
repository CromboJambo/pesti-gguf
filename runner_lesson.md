 📋 Specification: pesti-gguf Format Inference Engine

    Problem Statement

    GGUF files from various quantization tools have inconsistent metadata:

    • Tensors claim one dtype (e.g., Q4_K_M) but store data in another format (e.g., Q4_0)
    • Shape claims may not match actual data size
    • Every consumer needs to add ad-hoc fallback logic

    Goal

    Add a format inference engine to pesti-gguf that:

    1. Detects actual quantization format from raw data size
    2. Validates metadata consistency
    3. Provides warnings for suspicious assignments
    4. Makes the parser robust by default

    ────────────────────────────────────

    🎯 API Design

    Core Functions

    1. infer_tensor_format(claimed_dtype, claimed_elements, raw_data) -> Vec<InferredFormat>

      ─ rust
      /// Infer actual quantization format from GGUF tensor data
      ///
      /// # Arguments
      /// - `claimed_dtype`: The dtype claimed in GGUF header
      /// - `claimed_elements`: Element count from GGUF header
      /// - `raw_data`: Raw quantized bytes from file
      ///
      /// # Returns
      /// Sorted list of inferred formats by confidence (best match first)
      pub fn infer_tensor_format(
          claimed_dtype: GgufDtype,
          claimed_elements: usize,
          raw_data: &[u8],
      ) -> Result<Vec<InferredFormat>>
      #[derive(Debug)]
      pub struct InferredFormat {
          pub dtype: GgufDtype,
          pub bytes_per_element: f32,
          pub elements_per_block: usize,
          pub bytes_per_block: usize,
          pub confidence: f32,  // 0.0-1.0
      }

    2. validate_tensor_metadata(name, dtype, shape, raw_data) -> Vec<Warning>

      ─ rust
      /// Validate tensor metadata and return warnings for inconsistencies
      pub fn validate_tensor_metadata(
          name: &str,
          dtype: GgufDtype,
          shape: &[u64],
          raw_data: &[u8],
      ) -> Result<Vec<Warning>>
      #[derive(Debug)]
      pub enum Warning {
          DtypeMismatch {
              tensor_name: String,
              claimed: GgufDtype,
              inferred: GgufDtype,
              confidence: f32,
          },
          SuspiciousEmbeddingFormat {
              tensor_name: String,
              suggested: GgufDtype,
          },
      }

    ────────────────────────────────────

    📊 Known Format Specifications

      Format  BPE*    E/PB**  B/PB*  Notes
      ──────  ──────  ──────  ─────  ────────────────────
      Q4_0    0.5     32      16     No scales per block
      Q4_K    0.5625  16      9      Scales + h[2]
      Q4_K_M  1.6875  32      54     d + h[4] + qs
      Q5_0    0.625   32      20     5-bit without scales
      Q5_K    0.75    16      12     5-bit with scales
      Q5_K_M  1.0625  32      34     5-bit hybrid
      Q6_K    1.0     32      40     6-bit quantization
      Q8_0    1.0     32      34     8-bit, 2B scale

    *BPE = Bytes Per Element
    **E/PB = Elements Per Block
    *B/PB = Bytes Per Block

    ────────────────────────────────────

    🔧 Implementation Plan

    Phase 1: Core Inference Engine

    ☐ Add format_inference.rs module
    ☐ Implement infer_tensor_format() with size-based matching
    ☐ Support Q4_0, Q4_K, Q4_K_M (minimum for Llama 3.1)
    ☐ Add confidence scoring based on size match quality

    Phase 2: Metadata Validation

    ☐ Add validate_tensor_metadata() function
    ☐ Detect dtype mismatches (>80% confidence threshold)
    ☐ Flag suspicious embedding/output layer formats
    ☐ Add warning types for different inconsistency categories

    Phase 3: Integration & Testing

    ☐ Add unit tests with mock data (Q4_0, Q4_K, Q4_K_M samples)
    ☐ Add integration tests with real GGUF files (llama3.1-8b-q4_k_m.gguf)
    ☐ Benchmark inference performance (<1μs per tensor target)
    ☐ Update documentation with usage examples

    Phase 4: Consumer Migration

    ☐ Update pesti-runner to use new API instead of ad-hoc logic
    ☐ Add migration guide for other consumers
    ☐ Deprecate old fallback patterns in favor of inference engine

    ────────────────────────────────────

    🧪 Test Cases

    Case 1: Q4_K_M Tensor (Correct)

      ─ rust
      let raw_data = vec![0u8; 54]; // 1 block of Q4_K_M
      let inferred = infer_tensor_format(GgufDtype::Q4_K_M, 32, &raw_data);
      assert!(inferred[0].dtype == GgufDtype::Q4_K_M);
      assert!(inferred[0].confidence > 0.95);

    Case 2: Q4_0 Data Claimed as Q4_K_M

      ─ rust
      let raw_data = vec![0u8; 16]; // Q4_0 size (32 elements)
      let inferred = infer_tensor_format(GgufDtype::Q4_K_M, 32, &raw_data);
      assert!(inferred.iter().any(|f| f.dtype == GgufDtype::Q4_0));

    Case 3: token_embd.weight from llama3.1-8b-q4_k_m.gguf

      ─ rust
      let raw_data = vec![0u8; 295_501_824]; // Actual data size
      let inferred = infer_tensor_format(GgufDtype::Q4_K_M, 394_002_432, &raw_data);
      // Should detect Q4_0 format (BPE=0.5) vs claimed Q4_K_M (BPE=1.6875)
      assert!(inferred.iter().any(|f| f.dtype == GgufDtype::Q4_0));

    ────────────────────────────────────

    📚 Documentation

    README Addition

    Format Inference

    GGUF files often have inconsistent metadata. This crate provides tools to detect the actual quantization format    from raw data:

      ─ rust
      ─ rust
      ─ rust
      ─ rust
    tensor.n_elements as usize,
    &tensor.data
)?;
for hint in &inferred {
    println!("Possible format: {} (confidence: {:.2})",
        hint.dtype, hint.confidence);
}

Migration Guide

Old approach (ad-hoc fallback):

─ rust
match dtype {
    GgufDtype::Q4_K_M => dequantize_q4_k_m(data),
    _ => panic!("Unknown format"),
}

New approach (robust inference):

─ rust
let hints = infer_tensor_format(dtype, n_elements, data)?;
let best_match = hints.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()).unwrap();
dequantize(best_match.dtype, data)?;

────────────────────────────────────

🚀 Success Criteria

• ✅ Handles all Llama 3.1 8B Q4_K_M tensors correctly (including token_embd.weight)
• ✅ <5% performance overhead vs raw parsing
• ✅ Backward compatible (doesn't break existing API)
• ✅ Documented with examples and migration guide
• ✅ Test coverage >90% for inference logic

────────────────────────────────────

💡 Key Design Decisions

1. Return multiple candidates instead of single best match → allows consumers to choose based on context
2. Confidence scoring → lets consumers decide when to trust vs warn
3. Non-breaking addition → new functions alongside existing API
4. Format-agnostic → works with any dtype, not just Q4_K variants

────────────────────────────────────
