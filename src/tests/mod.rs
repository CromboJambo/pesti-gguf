//! Test utilities for conformance tests

use std::path::Path;

/// Helper to get the path to a conformance corpus file relative to the workspace root.
/// 
/// This avoids hardcoding `/home/crombo/projects/llm-workspace` and makes tests portable.
pub fn conformance_corpus_path(filename: &str) -> std::path::PathBuf {
    // Start from CARGO_MANIFEST_DIR (pesti-gguf crate directory)
    let manifest_dir = std::env!("CARGO_MANIFEST_DIR");
    
    // Navigate up to find conformance-corpus relative to crate location
    // Try common patterns: ../conformance-corpus, ../../conformance-corpus, ../../../conformance-corpus
    let path = Path::new(manifest_dir);
    
    // Try 3 levels up (most common for standalone crates)
    if let Some(parent) = path.parent() {
        let corpus_path = parent.join("conformance-corpus").join(filename);
        if corpus_path.exists() {
            return corpus_path;
        }
    }
    
    // Try 2 levels up
    if let Some(grandparent) = path.parent().and_then(|p| p.parent()) {
        let corpus_path = grandparent.join("conformance-corpus").join(filename);
        if corpus_path.exists() {
            return corpus_path;
        }
    }
    
    // Try 1 level up
    if let Some(great_grandparent) = path.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
        let corpus_path = great_grandparent.join("conformance-corpus").join(filename);
        if corpus_path.exists() {
            return corpus_path;
        }
    }
    
    // Fallback: return path relative to manifest dir (will fail gracefully with error message)
    Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("conformance-corpus").join(filename))
        .expect("Failed to compute conformance corpus path - corpus not found in common locations")
}
