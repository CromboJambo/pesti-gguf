# pesti-gguf

[![Crates.io](https://img.shields.io/crates/v/pesti-gguf.svg)](https://crates.io/crates/pesti-gguf)
[![Docs.rs](https://docs.rs/pesti-gguf/badge.svg)](https://docs.rs/pesti-gguf)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

**Memory-safe, minimal-dependency GGUF parser for Rust — with format inference**

## What is this?

A production-ready parser for [GGUF](https://github.com/ggml-org/llama.cpp/blob/master/docs) model weight files (used by llama.cpp, Ollama, etc.). Written in pure Rust with just three minimal dependencies: serde, serde_json, and thiserror.

**The honest truth**: GGUF files from various quantization tools have **inconsistent metadata**. Tensors claim one dtype (e.g., Q4_K_M) but store data in another format (e.g., Q4_0). Every consumer needs fallback logic — pesti-gguf provides it by default.

## Why use it?

- **Memory safety**: No buffer overflows, no undefined behavior
- **Zero FFI overhead**: Pure Rust, no C++ bindings needed  
- **Type-safe**: Structured error handling instead of panic-prone Option chains
- **WASM-ready**: Can run in browsers without C++ WASM overhead
- **Format inference**: Detects actual quantization from raw data size (see [Inference Engine](#format-inference-engine))

## Quick Start

```bash
cargo add pesti-gguf
```

```rust
use pesti_gguf::parse_gguf;
use std::path::Path;

fn main() -> Result<(), pesti_gguf::GgufError> {
    let header = parse_gguf(Path::new("model.gguf"))?;
    
    println!("Version: {}", header.version);
    println!("KV pairs: {}", header.kv_pairs.len());
    println!("Tensors: {}", header.tensors.len());
    
    Ok(())
}
```

## Features

+ Full v1/v2/v3 GGUF format support
+ Version-aware parsing (auto-detects format)
+ Comprehensive error types (InvalidMagic, UnsupportedVersion, etc.)
+ Alignment validation (`general.alignment` KV pair)
+ String length limits (1 MiB max per string)
+ **Format inference engine** for detecting actual quantization from data size

## Format Inference Engine

GGUF files often have metadata that doesn't match the actual stored data. The `format_inference` module provides tools to detect the real format:

```rust
use pesti_gguf::{parse_gguf, format_inference::{infer_tensor_format, GgufDtype}};
use std::path::Path;

let header = parse_gguf(Path::new("model.gguf"))?;

for tensor in &header.tensors {
    let n_elements = tensor.element_count() as usize;
    let raw_data = /* extract from file */;
    
    let inferred = infer_tensor_format(
        GgufDtype::from_u32(tensor.dtype),
        n_elements,
        &raw_data,
    )?;
    
    for hint in &inferred {
        println!("Possible format: {} (confidence: {:.2})", 
            hint.dtype, hint.confidence);
    }
}
```

**Key behaviors**:
- Returns multiple candidates sorted by confidence (0.0–1.0)
- Supports Q4_0, Q4K, Q4K_M, Q5_0, Q5K, Q5K_M, Q6K, Q8_0
- Confidence based on size match quality (exact match = 1.0)
- Warnings for suspicious layer formats (embeddings/output with unusual quantization)

## Conformance Testing

Tests validate against real GGUF files from the [Qwen2.5 conformance corpus](https://huggingface.co/Qwen):

### ✅ Tested (files present in repo)
+ `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MB) - *runs automatically*

### ⏸️ Auto-skipped (requires download)
+ `qwen2.5-3b-instruct-q4_k_m.gguf` (2.0 GB) - *tests ignored until file detected*

**Get started with a model:**  
Download any GGUF model from [HuggingFace](https://huggingface.com/models?library=gguf) and place it in `conformance-corpus/`. The tests will automatically validate your chosen model.

📥 **Example**: [Qwen2.5-3B-Instruct (4K context)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/qwen2.5-3b-instruct-q4_k_m.gguf)

## Comparison with llama.cpp

| Feature        | pesti-gguf              | llama.cpp          |
|----------------|-------------------------|--------------------|
| Language       | Pure Rust               | C++                |
| Dependencies   | 3 crates: serde, serde_json, thiserror | CUDA libs, OpenBLAS |
| Memory Safety  | Compile-time guarantees | Runtime checks     |
| FFI Required   | No                      | N/A (native)       |
| WASM Ready     | Yes                     | Requires Emscripten|

## The Entropy Problem: Headers vs Data Shape

Here's the uncomfortable truth about GGUF parsing in Rust:

**GGUF headers lie**. They claim tensor dtypes that don't match the actual stored data. This isn't a bug — it's a feature of how quantization tools work. Some exporters write dtype metadata before knowing the final format; others use fallback logic that breaks type consistency.

In C++, you can read raw bytes and "just work" with them. In Rust, the compiler demands you **bake the entropy decision into the code**. Either:
1. Trust the header (and crash when dequantization fails)
2. Add fallback logic for every possible dtype combination
3. Infer the real format from data size (what pesti-gguf does)

There's no escaping it — Rust makes you confront the mismatch between metadata and reality. The inference engine is our answer: detect the actual format, not the claimed one.

## License

Dual-licensed: **AGPL-3.0** or **Apache-2.0**

---

*Built by [@crombojambo](https://github.com/crombojambo) for the Rust LLM ecosystem — where we admit GGUF files are messy and build tools that work anyway.*
