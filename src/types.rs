use serde::{Deserialize, Serialize};
use crate::error::GgufError;

/// GGUF file format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    V1,
    V2,
    V3,
}

impl GgufVersion {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    /// Returns the wire format for this GGUF version.
    pub fn wire_format(self) -> GgufWireFormat {
        match self {
            Self::V1 => GgufWireFormat::V1,
            Self::V2 => GgufWireFormat::V2,
            Self::V3 => GgufWireFormat::V3,
        }
    }
}

/// Integer width for wire-format fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntWidth {
    /// 4-byte little-endian integer.
    U32,
    /// 8-byte little-endian integer.
    U64,
}

impl IntWidth {
    pub const fn is_u32(self) -> bool {
        matches!(self, Self::U32)
    }

    pub const fn is_u64(self) -> bool {
        matches!(self, Self::U64)
    }
}

/// Describes the wire-format encoding differences between GGUF versions.
///
/// Instead of branching on version numbers, the parser and writer
/// are parameterized by this struct. The version-specific code becomes
/// trivial: select the right `GgufWireFormat` constant, then call the
/// generic parser/writer.
///
/// The key insight: GGUF v1/v2/v3 differ **only** in integer widths
/// for specific fields. The grammar (what fields exist, in what order)
/// is the same across all versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufWireFormat {
    /// Width of tensor_count and kv_count in the file header.
    pub header_count_width: IntWidth,
    /// Width of KV pair key length prefix.
    pub key_width: IntWidth,
    /// Width of string value length prefix (top-level and in arrays).
    pub string_width: IntWidth,
    /// Width of array element count.
    pub array_count_width: IntWidth,
    /// Width of tensor info name length prefix.
    pub tensor_name_width: IntWidth,
}

impl GgufWireFormat {
    /// V1: u32 counts, u32 keys, u32 strings, u32 array counts, u32 tensor names.
    pub const V1: Self = Self {
        header_count_width: IntWidth::U32,
        key_width: IntWidth::U32,
        string_width: IntWidth::U32,
        array_count_width: IntWidth::U32,
        tensor_name_width: IntWidth::U32,
    };

    /// V2: u64 counts, u32 keys, u32 strings, u64 array counts, u32 tensor names.
    pub const V2: Self = Self {
        header_count_width: IntWidth::U64,
        key_width: IntWidth::U32,
        string_width: IntWidth::U32,
        array_count_width: IntWidth::U64,
        tensor_name_width: IntWidth::U32,
    };

    /// V3: u64 counts, u64 keys, u64 strings, u64 array counts, u64 tensor names.
    pub const V3: Self = Self {
        header_count_width: IntWidth::U64,
        key_width: IntWidth::U64,
        string_width: IntWidth::U64,
        array_count_width: IntWidth::U64,
        tensor_name_width: IntWidth::U64,
    };
}

impl From<GgufVersion> for GgufWireFormat {
    fn from(v: GgufVersion) -> Self {
        v.wire_format()
    }
}

/// GGUF key-value value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Uint64,
    Int64,
    Bool,
    String,
    Array,
    Int8Array,
    Uint8Array,
    Float64,
    Bfloat16,
    Float16,
}

/// GGUF key-value pair value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GgufKvValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Bool(bool),
    String(String),
    Array(Vec<GgufKvValue>),
    Int8Array(Vec<i8>),
    Uint8Array(Vec<u8>),
    Float64(f64),
    Bfloat16(f32), // Stored as f32, but represents bfloat16 bit pattern
    Float16(u16),
}

impl GgufKvValue {
    /// Compute the serialized byte size of this value, given the string length prefix width in bytes.
    /// String and array string elements use `str_len_bytes` for their length prefix.
    pub fn raw_byte_size_with_str_width(&self, str_len_bytes: usize) -> usize {
        match self {
            Self::Uint8(..) | Self::Int8(..) | Self::Bool(..) => 1,
            Self::Uint16(..) | Self::Int16(..) | Self::Bfloat16(..) | Self::Float16(..) => 2,
            Self::Uint32(..) | Self::Int32(..) | Self::Float32(..) => 4,
            Self::Uint64(..) | Self::Int64(..) | Self::Float64(..) => 8,
            Self::String(s) => str_len_bytes + s.len(),
            Self::Int8Array(arr) => 8 + arr.len(),
            Self::Uint8Array(arr) => 8 + arr.len(),
            Self::Array(arr) => {
                let elem_size = match arr.first().map(|v| v.value_type()) {
                    Some(GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool
                    | GgufValueType::Int8Array | GgufValueType::Uint8Array) => 1,
                    Some(GgufValueType::Uint16 | GgufValueType::Int16
                    | GgufValueType::Float16) => 2,
                    Some(GgufValueType::Uint32 | GgufValueType::Int32
                    | GgufValueType::Float32) => 4,
                    Some(GgufValueType::Uint64 | GgufValueType::Int64) => 8,
                    Some(GgufValueType::String) => {
                        return arr.iter().map(|v| match v {
                            Self::String(s) => str_len_bytes + s.len(),
                            _ => 0,
                        }).sum::<usize>() + 1 + 8;
                    }
                    _ => 4,
                };
                1 + 8 + arr.len() * elem_size
            }
        }
    }

    pub fn value_type(&self) -> GgufValueType {
        match self {
            Self::Uint8(_) => GgufValueType::Uint8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::Uint16(_) => GgufValueType::Uint16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::Uint32(_) => GgufValueType::Uint32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Uint64(_) => GgufValueType::Uint64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::Int8Array(_) => GgufValueType::Int8Array,
            Self::Uint8Array(_) => GgufValueType::Uint8Array,
            Self::Float64(_) => GgufValueType::Float64,
            Self::Bfloat16(_) => GgufValueType::Bfloat16,
            Self::Float16(_) => GgufValueType::Float16,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(*v as u32),
            Self::Int8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Int16(v) => Some(*v as u32),
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Bfloat16(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

impl GgufValueType {
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            13 => Some(Self::Int8Array),
            14 => Some(Self::Uint8Array),
            15 => Some(Self::Bfloat16),
            16 => Some(Self::Float16),
            _ => None,
        }
    }

    pub const fn to_u32(self) -> u32 {
        match self {
            Self::Uint8 => 0,
            Self::Int8 => 1,
            Self::Uint16 => 2,
            Self::Int16 => 3,
            Self::Uint32 => 4,
            Self::Int32 => 5,
            Self::Float32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::Uint64 => 10,
            Self::Int64 => 11,
            Self::Float64 => 12,
            Self::Int8Array => 13,
            Self::Uint8Array => 14,
            Self::Bfloat16 => 15,
            Self::Float16 => 16,
        }
    }

    pub const fn element_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool | Self::Int8Array | Self::Uint8Array => Some(1),
            Self::Uint16 | Self::Int16 | Self::Bfloat16 | Self::Float16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// A single key-value pair from the GGUF header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufKvPair {
    pub key: String,
    pub value_type: GgufValueType,
    pub value: GgufKvValue,
}

impl GgufKvPair {
    /// Compute serialized byte size using the wire format to determine key/string widths.
    pub fn raw_byte_size_for_format(&self, format: &GgufWireFormat) -> usize {
        let key_bytes = self.key.len();
        let key_len_bytes: usize = match format.key_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        let str_len_bytes: usize = match format.string_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        let value_bytes = self.value.raw_byte_size_with_str_width(str_len_bytes);
        key_len_bytes + key_bytes + 4 + value_bytes
    }
}

/// GGUF tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
    I8, I16, I32, I64, F64, BF16, Q1_K, Q4_K_M, Q5_K_M, Q6_K_S, Q8_K_M, Q2_K_S, Q3_K_S, Q4_K_S, Q5_K_S, Q2_K_M,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, TQ1_0, TQ2_0,
    IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8, MXFP4, NVFP4, Q1_0, Q2_0, Unknown(u32),
}

impl GgufDtype {
    pub const fn from_u32(v: u32) -> Self {
        // IDs 0-42 follow official ggml.h / GGUF conventions.
        // IDs 43+ are pesti-specific custom types (non-colliding with standard GGML).
        match v {
            0 => Self::F32, 1 => Self::F16, 2 => Self::Q4_0, 3 => Self::Q4_1,
            6 => Self::Q5_0, 7 => Self::Q5_1, 8 => Self::Q8_0, 9 => Self::Q8_1,
            10 => Self::Q2_K, 11 => Self::Q3_K, 12 => Self::Q4_K, 13 => Self::Q5_K,
            14 => Self::Q6_K, 15 => Self::Q8_K, 16 => Self::IQ2_XXS, 17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS, 19 => Self::IQ1_S, 20 => Self::Q1_K, 21 => Self::Q4_K_M,
            22 => Self::Q5_K_M, 24 => Self::I8, 25 => Self::I16,
            26 => Self::I32, 27 => Self::I64, 28 => Self::F64, 29 => Self::Q2_K_M,
            30 => Self::BF16, 31 => Self::Q4_0_4_4, 32 => Self::Q4_0_4_8, 33 => Self::Q4_0_8_8,
            34 => Self::TQ1_0, 35 => Self::TQ2_0, 36 => Self::IQ4_NL_4_4, 37 => Self::IQ4_NL_4_8,
            38 => Self::IQ4_NL_8_8, 39 => Self::MXFP4, 40 => Self::NVFP4, 41 => Self::Q1_0,
            42 => Self::Q2_0,
            // pesti custom: _S and _M variants (non-standard, no ggml.h equivalent)
            43 => Self::Q6_K_S, 44 => Self::Q8_K_M,
            45 => Self::Q2_K_S, 46 => Self::Q3_K_S, 47 => Self::Q4_K_S,
            48 => Self::Q5_K_S,
            _ => Self::Unknown(v),
        }
    }

    pub const fn to_u32(self) -> u32 {
        // Must match from_u32() exactly. IDs 43+ are pesti custom types.
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::BF16 => 30,
            Self::Q1_K => 20,
            Self::Q4_K_M => 21,
            Self::Q5_K_M => 22,
            Self::Q2_K_M => 29,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::Q4_0_4_4 => 31,
            Self::Q4_0_4_8 => 32,
            Self::Q4_0_8_8 => 33,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            Self::IQ4_NL_4_4 => 36,
            Self::IQ4_NL_4_8 => 37,
            Self::IQ4_NL_8_8 => 38,
            Self::MXFP4 => 39,
            Self::NVFP4 => 40,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
            // pesti custom: _S and _M variants (non-standard, no ggml.h equivalent)
            Self::Q6_K_S => 43,
            Self::Q8_K_M => 44,
            Self::Q2_K_S => 45,
            Self::Q3_K_S => 46,
            Self::Q4_K_S => 47,
            Self::Q5_K_S => 48,
            Self::Unknown(v) => v,
        }
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1
            | Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K
            | Self::Q1_K | Self::Q4_K_M | Self::Q5_K_M | Self::Q6_K_S | Self::Q8_K_M
            | Self::Q2_K_S | Self::Q3_K_S | Self::Q4_K_S | Self::Q5_K_S | Self::Q2_K_M)
    }

    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4, Self::F16 => 2, Self::Q8_0 | Self::Q8_1 => 2,
            Self::I8 => 1, Self::I16 => 2, Self::I32 => 4, Self::I64 => 8,
            Self::F64 => 8, Self::BF16 => 2, _ => 0,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32", Self::F16 => "F16", Self::Q4_0 => "Q4_0", Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0", Self::Q5_1 => "Q5_1", Self::Q8_0 => "Q8_0", Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K", Self::Q3_K => "Q3_K", Self::Q4_K => "Q4_K", Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K", Self::Q8_K => "Q8_K", Self::I8 => "I8", Self::I16 => "I16",
            Self::I32 => "I32", Self::I64 => "I64", Self::F64 => "F64", Self::BF16 => "BF16",
            Self::Q1_K => "Q1_K", Self::Q4_K_M => "Q4_K_M", Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K_S => "Q6_K_S", Self::Q8_K_M => "Q8_K_M", Self::Q2_K_S => "Q2_K_S",
            Self::Q3_K_S => "Q3_K_S", Self::Q4_K_S => "Q4_K_S", Self::Q5_K_S => "Q5_K_S",
            Self::Q2_K_M => "Q2_K_M", Self::IQ2_XXS => "IQ2_XXS", Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS", Self::IQ1_S => "IQ1_S", Self::Q4_0_4_4 => "Q4_0_4_4",
            Self::Q4_0_4_8 => "Q4_0_4_8", Self::Q4_0_8_8 => "Q4_0_8_8", Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0", Self::IQ4_NL_4_4 => "IQ4_NL_4_4", Self::IQ4_NL_4_8 => "IQ4_NL_4_8",
            Self::IQ4_NL_8_8 => "IQ4_NL_8_8", Self::MXFP4 => "MXFP4", Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0", Self::Q2_0 => "Q2_0", Self::Unknown(_) => "unknown",
        }
    }
}

/// A single tensor's metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub dtype: u32,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn ndims(&self) -> u32 {
        self.shape.len() as u32
    }

    /// Compute the actual stored byte size on disk with overflow checking.
    pub fn stored_size(&self) -> Result<u64, GgufError> {
        let n = self.element_count();
        
        if n > u64::MAX / 8 {
            return Err(GgufError::InvalidTensor(
                "Tensor element count too large (would overflow)".to_string(),
            ));
        }

        let dtype = GgufDtype::from_u32(self.dtype);
        match dtype {
            GgufDtype::F32 => Ok(n.checked_mul(4).ok_or_else(|| GgufError::InvalidTensor("F32 size overflow".to_string()))?),
            GgufDtype::F16 | GgufDtype::BF16 => Ok(n.checked_mul(2).ok_or_else(|| GgufError::InvalidTensor("F16/BF16 size overflow".to_string()))?),
            GgufDtype::Q8_0 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                let base = full_blocks.checked_mul(34).ok_or_else(|| GgufError::InvalidTensor("Q8_0 size overflow".to_string()))?;
                let tail = if remaining > 0 { 2 + remaining } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q8_0 size overflow".to_string()))?)
            }
            GgufDtype::Q8_1 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                let base = full_blocks.checked_mul(36).ok_or_else(|| GgufError::InvalidTensor("Q8_1 size overflow".to_string()))?;
                let tail = if remaining > 0 { 4 + remaining } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q8_1 size overflow".to_string()))?)
            }
            GgufDtype::Q4_0 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                let base = full_blocks.checked_mul(18).ok_or_else(|| GgufError::InvalidTensor("Q4_0 size overflow".to_string()))?;
                let tail = if remaining > 0 { 2 + remaining.div_ceil(2) } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q4_0 size overflow".to_string()))?)
            }
            GgufDtype::Q4_1 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                let base = full_blocks.checked_mul(20).ok_or_else(|| GgufError::InvalidTensor("Q4_1 size overflow".to_string()))?;
                let tail = if remaining > 0 { 4 + remaining.div_ceil(2) } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q4_1 size overflow".to_string()))?)
            }
            GgufDtype::Q5_0 => Ok(n.checked_div(2).ok_or_else(|| GgufError::InvalidTensor("Q5_0 size overflow".to_string()))? + 32 + 16),
            GgufDtype::Q5_1 => Ok(n.checked_div(2).ok_or_else(|| GgufError::InvalidTensor("Q5_1 size overflow".to_string()))? + 64 + 16),
            GgufDtype::Q2_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                let base = full_blocks.checked_mul(84).ok_or_else(|| GgufError::InvalidTensor("Q2_K size overflow".to_string()))?;
                let tail = if remaining > 0 { remaining / 4 + remaining / 16 + 4 } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q2_K size overflow".to_string()))?)
            }
            GgufDtype::Q3_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                let base = full_blocks.checked_mul(110).ok_or_else(|| GgufError::InvalidTensor("Q3_K size overflow".to_string()))?;
                let tail = if remaining > 0 { remaining / 4 + remaining / 8 + 12 } else { 0 };
                Ok(base.checked_add(tail).ok_or_else(|| GgufError::InvalidTensor("Q3_K size overflow".to_string()))?)
            }
            GgufDtype::Q4_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                Ok((full_blocks * 144 + if remaining > 0 { remaining / 2 + 16 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q4_K size overflow".to_string()))?)
            }
            GgufDtype::Q5_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                Ok((full_blocks * 176 + if remaining > 0 { remaining / 2 + remaining / 8 + 16 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q5_K size overflow".to_string()))?)
            }
            GgufDtype::Q6_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                Ok((full_blocks * 210 + if remaining > 0 { remaining / 16 + 3 * remaining / 4 + 2 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q6_K size overflow".to_string()))?)
            }
            GgufDtype::Q8_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                Ok((full_blocks * 292 + if remaining > 0 { remaining + remaining / 16 * 2 + 4 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q8_K size overflow".to_string()))?)
            }
            GgufDtype::Q1_K => {
                let full_blocks = n / 256;
                let remaining = n % 256;
                Ok((full_blocks * 64 + if remaining > 0 { remaining / 8 + remaining / 64 + 96 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q1_K size overflow".to_string()))?)
            }
            GgufDtype::Q4_K_M => Ok((n / 256 * 144 + if n % 256 > 0 { (n % 256) / 2 + 16 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q4_K_M size overflow".to_string()))?),
            GgufDtype::Q5_K_M => Ok((n / 256 * 176 + if n % 256 > 0 { (n % 256) / 2 + (n % 256) / 8 + 16 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q5_K_M size overflow".to_string()))?),
            GgufDtype::Q8_K_M => Ok((n / 256 * 292 + if n % 256 > 0 { (n % 256) + (n % 256) / 16 * 2 + 4 } else { 0 }).try_into().map_err(|_| GgufError::InvalidTensor("Q8_K_M size overflow".to_string()))?),
            GgufDtype::Q2_K_S | GgufDtype::Q3_K_S | GgufDtype::Q4_K_S | GgufDtype::Q5_K_S | GgufDtype::Q6_K_S | GgufDtype::Q2_K_M => Ok(n / 4 + 24),
            GgufDtype::I8 => Ok(n),
            GgufDtype::I16 => Ok(n.checked_mul(2).ok_or_else(|| GgufError::InvalidTensor("I16 size overflow".to_string()))?),
            GgufDtype::I32 => Ok(n.checked_mul(4).ok_or_else(|| GgufError::InvalidTensor("I32 size overflow".to_string()))?),
            GgufDtype::I64 => Ok(n.checked_mul(8).ok_or_else(|| GgufError::InvalidTensor("I64 size overflow".to_string()))?),
            GgufDtype::F64 => Ok(n.checked_mul(8).ok_or_else(|| GgufError::InvalidTensor("F64 size overflow".to_string()))?),
            GgufDtype::IQ2_XXS | GgufDtype::IQ2_XS | GgufDtype::IQ3_XXS | GgufDtype::IQ1_S => Ok(n / 4 + 256),
            GgufDtype::Q4_0_4_4 | GgufDtype::Q4_0_4_8 | GgufDtype::Q4_0_8_8 => Ok(n / 2 + 16),
            GgufDtype::TQ1_0 | GgufDtype::TQ2_0 => Ok(n / 3 + 128),
            GgufDtype::IQ4_NL_4_4 | GgufDtype::IQ4_NL_4_8 | GgufDtype::IQ4_NL_8_8 => Ok(n / 2 + 32),
            GgufDtype::MXFP4 | GgufDtype::NVFP4 => Ok(n / 4 + 64),
            GgufDtype::Q1_0 | GgufDtype::Q2_0 => Ok(n / 4 + 128),
            GgufDtype::Unknown(_) => Ok(n.checked_mul(2).ok_or_else(|| GgufError::InvalidTensor("Unknown dtype size overflow".to_string()))?),
        }
    }

    pub fn raw_byte_size(&self) -> usize {
        8 + self.name.len() + 4 + (self.shape.len() * 8) + 4 + 8
    }
}

/// Parsed GGUF header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeader {
    pub version: u32,
    pub kv_pairs: Vec<GgufKvPair>,
    pub tensors: Vec<GgufTensorInfo>,
    #[serde(default = "default_data_alignment")]
    pub data_alignment: Option<u64>,
    pub data_section_start: u64,
}

fn default_data_alignment() -> Option<u64> { Some(32) }

impl GgufHeader {
    pub fn get_kv<T: From<GgufKvValue>>(&self, key: &str) -> Option<T> {
        self.kv_pairs.iter().find(|p| p.key == key).map(|p| T::from(p.value.clone()))
    }

    pub fn get_kv_str(&self, key: &str) -> Option<&str> {
        self.kv_pairs.iter().find(|p| p.key == key).and_then(|p| p.value.as_str())
    }

    pub fn get_kv_u32(&self, key: &str) -> Option<u32> {
        self.kv_pairs.iter().find(|p| p.key == key).and_then(|p| p.value.as_u32())
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get_kv_str("general.architecture").or_else(|| self.get_kv_str("arch"))
    }

    pub fn context_length(&self) -> Option<u32> {
        self.get_kv_u32("llama.context_length").or_else(|| self.get_kv_u32("n_ctx"))
    }

    pub fn embedding_length(&self) -> Option<u32> {
        self.get_kv_u32("llama.embedding_length").or_else(|| self.get_kv_u32("n_embd"))
    }

    pub fn block_count(&self) -> Option<u32> {
        self.get_kv_u32("llama.block_count").or_else(|| self.get_kv_u32("n_layer"))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    pub fn total_tensor_bytes_f32(&self) -> u64 {
        // Return actual byte count for F32 tensors (element_count * 4 bytes per F32)
        self.tensors.iter().map(|t| t.element_count() * 4).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_roundtrip_all() {
        // All valid dtype IDs: 0-42 (official) + 43-48 (pesti custom)
        for v in [
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48,
        ] {
            let dt = GgufDtype::from_u32(v);
            assert_eq!(dt.to_u32(), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_dtype_integer_types() {
        // dtype 24-28 are I8/I16/I32/I64/F64 per official ggml.h
        assert_eq!(GgufDtype::from_u32(24), GgufDtype::I8);
        assert_eq!(GgufDtype::from_u32(25), GgufDtype::I16);
        assert_eq!(GgufDtype::from_u32(26), GgufDtype::I32);
        assert_eq!(GgufDtype::from_u32(27), GgufDtype::I64);
        assert_eq!(GgufDtype::from_u32(28), GgufDtype::F64);
    }

    #[test]
    fn test_stored_size_i8() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 24 };
        assert_eq!(info.stored_size().unwrap(), 100);
    }

    #[test]
    fn test_stored_size_i16() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 25 };
        assert_eq!(info.stored_size().unwrap(), 200);
    }

    #[test]
    fn test_stored_size_i32() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 26 };
        assert_eq!(info.stored_size().unwrap(), 400);
    }

    #[test]
    fn test_stored_size_i64() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 27 };
        assert_eq!(info.stored_size().unwrap(), 800);
    }

    #[test]
    fn test_stored_size_f64() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 28 };
        assert_eq!(info.stored_size().unwrap(), 800);
    }

    #[test]
    fn test_stored_size_overflow() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![u64::MAX / 4 + 1], offset: 0, dtype: 0 };
        assert!(info.stored_size().is_err());
    }

    #[test]
    fn test_total_tensor_bytes_f32() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![
                GgufTensorInfo { name: "a".into(), shape: vec![100], offset: 0, dtype: 0 }, // F32
                GgufTensorInfo { name: "b".into(), shape: vec![50], offset: 0, dtype: 0 },   // F32
            ],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        // (100 + 50) elements * 4 bytes per F32 = 600 bytes
        assert_eq!(header.total_tensor_bytes_f32(), 600);
    }

    #[test]
    fn test_v2_alignment_computation() {
        use crate::parser::compute_data_section_start;
        
        // Test that alignment is applied for v2 files with custom alignment
        let kv_pairs: Vec<GgufKvPair> = vec![];
        let tensors: Vec<GgufTensorInfo> = vec![];
        
        // Compute with 256-byte alignment - should align header size to 256
        let data_start = compute_data_section_start(2, &kv_pairs, &tensors, Some(256));
        assert_eq!(data_start % 256, 0, "Data section start should be aligned to 256 bytes");
        
        // With default alignment (32)
        let data_start_32 = compute_data_section_start(2, &kv_pairs, &tensors, Some(32));
        assert_eq!(data_start_32 % 32, 0, "Data section start should be aligned to 32 bytes");
    }
}
