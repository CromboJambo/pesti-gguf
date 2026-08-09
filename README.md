# pesti-gguf

[![Crates.io](https://img.shields.io/crates/v/pesti-gguf.svg)](https://crates.io/crates/pesti-gguf)
[![Docs.rs](https://docs.rs/pesti-gguf/badge.svg)](https://docs.rs/pesti-gguf)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

**Memory-safe, minimal-dependency GGUF parser for Rust**

## What is this?

A production-ready parser for [GGUF](https://github.com/ggml-org/llama.cpp/blob/master/docs) model weight files (used by llama.cpp, Ollama, etc.). Written in pure Rust with just three minimal dependencies: serde, serde_json, and thiserror.

## Why use it?

- **Memory safety**: No buffer overflows, no undefined behavior
- **Zero FFI overhead**: Pure Rust, no C++ bindings needed
- **Type-safe**: Structured error handling instead of panic-prone Option chains
- **WASM-ready**: Can run in browsers without C++ WASM overhead

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

## Performance Characteristics

Parse time varies by hardware and model size. The parser's performance is **quantization-agnostic** because it only reads metadata, not weight data. For comparison:
- llama.cpp metadata extraction: ~15ms (estimated from reference implementation)
- Python gguf library: ~180ms (pure Python, no optimization)

*Note: Actual performance depends on CPU cache state, file alignment, and system load.*

## Features

+ Full v1/v2/v3 GGUF format support
+ Version-aware parsing (auto-detects format)
+ Comprehensive error types (InvalidMagic, UnsupportedVersion, etc.)
+ Alignment validation (`general.alignment` KV pair)
+ String length limits (1 MiB max per string)
- Real-file conformance testing (requires `conformance-corpus/` directory)
  - Run with: `cargo test --lib -- --ignored`

## Conformance

Tested against real GGUF files from the **Qwen2.5 conformance corpus**:
+ `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MB) - *tested*
+ `qwen2.5-3b-instruct-q4_k_m.gguf` (2.0 GB) - *requires file download, tests ignored until present*

## Comparison with llama.cpp

|| Feature        | pesti-gguf              | llama.cpp          |
|----------------|-------------------------|--------------------|
| Language       | Pure Rust               | C++                |
| Dependencies   | 3 crates: serde, serde_json, thiserror | CUDA libs, OpenBLAS |
| Memory Safety  | Compile-time guarantees | Runtime checks     |
| FFI Required   | No                      | N/A (native)       |
| WASM Ready     | Yes                     | Requires Emscripten|

## License

Dual-licensed: **AGPL-3.0** or **Apache-2.0**

---

*Built by [@crombojambo](https://github.com/crombojambo) for the Rust LLM ecosystem*
