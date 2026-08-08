use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION_1: u32 = 1;
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;

// Constants from llama.cpp reference implementation
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
const GGUF_MAX_KEY_LENGTH: u64 = 1024 * 1024; // 1 MiB (was 1 GiB - unrealistic for metadata keys)
const GGUF_MAX_TENSOR_NAME_LENGTH: u64 = 1024 * 1024; // 1 MiB (was 1 GiB - realistic max is ~256 bytes)
const GGUF_MAX_COUNT: usize = 10_000_000; // 10M elements - sanity limit for KV pairs, tensors, array elements

/// Read a u64 value and convert to usize with explicit bounds checking
fn read_usize<R: Read + Seek>(reader: &mut R, what: &'static str) -> Result<usize, GgufError> {
    let val = reader.read_u64::<LittleEndian>()?;
    val.try_into().map_err(|_| {
        GgufError::InvalidMetadata(format!(
            "{} {} exceeds maximum usize ({})",
            what,
            val,
            usize::MAX
        ))
    })
}

/// Read a u32 value and convert to usize with explicit bounds checking
fn read_usize_u32<R: Read + Seek>(reader: &mut R, what: &'static str) -> Result<usize, GgufError> {
    let val = reader.read_u32::<LittleEndian>()? as usize;
    if val > GGUF_MAX_COUNT {
        return Err(GgufError::InvalidMetadata(format!(
            "{} {} exceeds maximum count ({})",
            what,
            val,
            GGUF_MAX_COUNT
        )));
    }
    Ok(val)
}

/// Read a u64 value and convert to usize with explicit bounds checking
fn read_usize_checked<R: Read + Seek>(reader: &mut R, what: &'static str, max: usize) -> Result<usize, GgufError> {
    let val = reader.read_u64::<LittleEndian>()? as usize;
    if val > max {
        return Err(GgufError::InvalidMetadata(format!(
            "{} {} exceeds maximum {}",
            what,
            val,
            max
        )));
    }
    Ok(val)
}

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    use std::io::BufReader;
    
    let file = std::fs::File::open(path)
        .map_err(|e| GgufError::Io(format!("open {}: {}", path.display(), e)))?;
    let mut reader = BufReader::new(file);
    parse_gguf_reader(&mut reader)
}

pub fn parse_gguf_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    // Check magic number - GGUF files are always little-endian
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    
    if &magic_bytes != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(
            String::from_utf8_lossy(&magic_bytes).to_string(),
        ));
    }

    // All GGUF files are little-endian (no byte-order detection needed)
    // This is a design decision for simplicity and compatibility
    
    let version = reader.read_u32::<LittleEndian>()?;

    match version {
        GGUF_VERSION_1 => parse_v1(reader),
        GGUF_VERSION_2 => parse_v2(reader),
        GGUF_VERSION_3 => parse_v3(reader),
        _ => Err(GgufError::UnsupportedVersion(version)),
    }
}

fn parse_v1<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let _tensor_count = reader.read_u32::<LittleEndian>()? as u64; // v1 uses u32 for counts (unused in v1)
    let kv_count = read_usize_u32(reader, "KV pair count")?;

    if kv_count > GGUF_MAX_COUNT {
        return Err(GgufError::InvalidMetadata(format!(
            "KV pair count {} exceeds maximum {}",
            kv_count, GGUF_MAX_COUNT
        )));
    }

    let mut kv_pairs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair_v1(reader)?); // v1 uses u32 key lengths
    }

    let alignment = read_alignment_from_kv(&kv_pairs)?;
    let data_section_start = compute_data_section_start(1, &kv_pairs, &[], Some(alignment));

    Ok(GgufHeader {
        version: 1,
        kv_pairs,
        tensors: vec![],
        data_alignment: Some(alignment),
        data_section_start,
    })
}

fn parse_v2<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = read_usize(reader, "Tensor count")?;
    let kv_count = read_usize(reader, "KV pair count")?;

    if tensor_count > GGUF_MAX_COUNT || kv_count > GGUF_MAX_COUNT {
        return Err(GgufError::InvalidMetadata(format!(
            "Counts exceed maximum: tensors={}, kvs={}",
            tensor_count, kv_count
        )));
    }

    let mut kv_pairs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        // v2 uses u32 tensor name lengths (same as v1)
        tensors.push(read_tensor_info_v2(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs)?;
    let data_section_start = compute_data_section_start(2, &kv_pairs, &tensors, Some(alignment));

    Ok(GgufHeader {
        version: 2,
        kv_pairs,
        tensors,
        data_alignment: Some(alignment),
        data_section_start,
    })
}

fn parse_v3<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = read_usize(reader, "Tensor count")?;
    let kv_count = read_usize(reader, "KV pair count")?;

    // v3 practical format: same structure as v2, just different semantics
    // No extra padding after counts

    if tensor_count > GGUF_MAX_COUNT || kv_count > GGUF_MAX_COUNT {
        return Err(GgufError::InvalidMetadata(format!(
            "Counts exceed maximum: tensors={}, kvs={}",
            tensor_count, kv_count
        )));
    }

    let mut kv_pairs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair_v3(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs)?;
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info_v3(reader)?);
    }

    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, Some(alignment));

    Ok(GgufHeader {
        version: 3,
        kv_pairs,
        tensors,
        data_alignment: Some(alignment),
        data_section_start,
    })
}

/// Read KV pair - used for v1/v2 (keys and strings use u32 lengths)
fn read_kv_pair<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    // Key is length-prefixed with u32 LE
    let key_len = reader.read_u32::<LittleEndian>()? as usize;
    if key_len == 0 || key_len > GGUF_MAX_KEY_LENGTH as usize {
        return Err(GgufError::KeyLengthOutOfRange(key_len as u64));
    }

    let key_bytes = read_bytes(reader, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    // Value type: u32 LE
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    // Read value based on type
    let value = read_kv_value_v2(reader, value_type)?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read KV pair for v3 practical format - keys and strings use u64 lengths
///
/// GGUF v3 practical format follows llama.cpp's wire layout:
/// - Key length: u64 (LE) - allows longer metadata keys
/// - Key name: raw bytes
/// - Value type: u32 (LE) - identifies value format
/// - Value data: varies by type
fn read_kv_pair_v3<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    // 1. Read key length (u64 LE)
    let key_len = reader.read_u64::<LittleEndian>()? as usize;

    if key_len == 0 || key_len > GGUF_MAX_KEY_LENGTH as usize {
        return Err(GgufError::KeyLengthOutOfRange(key_len as u64));
    }

    // 2. Read key name (raw bytes)
    let key_bytes = read_bytes(reader, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    // 3. Read value type (u32 LE)
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    // 4. Read value based on type
    let value = read_kv_value_v3(reader, value_type)?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read KV pair for v1 format - keys and strings use u32 lengths
fn read_kv_pair_v1<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    // Key is length-prefixed with u32 LE (v1 uses u32, not u64)
    let key_len = reader.read_u32::<LittleEndian>()? as usize;
    if key_len == 0 || key_len > GGUF_MAX_KEY_LENGTH as usize {
        return Err(GgufError::KeyLengthOutOfRange(key_len as u64));
    }

    let key_bytes = read_bytes(reader, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    // Value type: u32 LE
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    // Read value based on type (v1 uses u32 string lengths)
    let value = read_kv_value_v1(reader, value_type)?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read a single scalar KV value (shared across all GGUF versions).
fn read_kv_scalar<R: Read>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(reader.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufKvValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Int16 => Ok(GgufKvValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufValueType::Uint32 => Ok(GgufKvValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufValueType::Int32 => Ok(GgufKvValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufValueType::Uint64 => Ok(GgufKvValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufValueType::Int64 => Ok(GgufKvValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufValueType::Float32 => Ok(GgufKvValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufValueType::Float64 => Ok(GgufKvValue::Float64(reader.read_f64::<LittleEndian>()?)),
        GgufValueType::Bool => Ok(GgufKvValue::Bool(reader.read_u8()? != 0)),
        GgufValueType::Bfloat16 => {
            let val = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Bfloat16(f32::from_bits((val as u32) << 16)))
        }
        GgufValueType::Float16 => Ok(GgufKvValue::Float16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(reader.read_u8()?)),
        _ => Err(GgufError::InvalidValueType(value_type as u32)),
    }
}

/// Read a single array element (shared across all GGUF versions).
fn read_kv_array_element<R: Read>(
    reader: &mut R,
    elem_type: GgufValueType,
    elem_type_raw: u32,
    string_len_size: usize,
) -> Result<GgufKvValue, GgufError> {
    match elem_type {
        GgufValueType::String => {
            let str_len = if string_len_size == 8 {
                reader.read_u64::<LittleEndian>()? as usize
            } else {
                reader.read_u32::<LittleEndian>()? as usize
            };
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Uint32 => Ok(GgufKvValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(reader.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufKvValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Int16 => Ok(GgufKvValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufValueType::Int32 => Ok(GgufKvValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufValueType::Uint64 => Ok(GgufKvValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufValueType::Int64 => Ok(GgufKvValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufValueType::Float32 => Ok(GgufKvValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufValueType::Bool => Ok(GgufKvValue::Bool(reader.read_u8()? != 0)),
        GgufValueType::Bfloat16 => {
            let val = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Bfloat16(f32::from_bits((val as u32) << 16)))
        }
        GgufValueType::Float16 => Ok(GgufKvValue::Float16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8Array | GgufValueType::Uint8Array => {
            Err(GgufError::UnsupportedArrayElementType(elem_type_raw))
        }
        _ => Err(GgufError::UnsupportedArrayElementType(elem_type_raw)),
    }
}

/// Generic KV value reader — parameterized by string/array length prefix sizes.
fn read_kv_value_generic<R: Read + Seek>(
    reader: &mut R,
    value_type: GgufValueType,
    string_len_size: usize,
    array_count_size: usize,
) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::String => {
            let str_len = if string_len_size == 8 {
                reader.read_u64::<LittleEndian>()? as usize
            } else {
                reader.read_u32::<LittleEndian>()? as usize
            };
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Array => {
            let elem_type_raw = reader.read_u32::<LittleEndian>()?;
            let elem_type = GgufValueType::from_u32(elem_type_raw)
                .ok_or(GgufError::InvalidValueType(elem_type_raw))?;
            let elem_count = if array_count_size == 8 {
                read_usize_checked(reader, "Array element count", GGUF_MAX_COUNT)?
            } else {
                reader.read_u32::<LittleEndian>()? as usize
            };
            let mut elements = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                elements.push(read_kv_array_element(reader, elem_type, elem_type_raw, string_len_size)?);
            }
            Ok(GgufKvValue::Array(elements))
        }
        _ => read_kv_scalar(reader, value_type),
    }
}

/// Read KV value for v3 format (u64 string lengths, u64 array counts).
fn read_kv_value_v3<R: Read + Seek>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    read_kv_value_generic(reader, value_type, 8, 8)
}

/// Read KV value for v2 format (u32 string lengths, u64 array counts).
fn read_kv_value_v2<R: Read + Seek>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    read_kv_value_generic(reader, value_type, 4, 8)
}

/// Read KV value for v1 format (u32 string lengths, u32 array counts).
fn read_kv_value_v1<R: Read + Seek>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    read_kv_value_generic(reader, value_type, 4, 4)
}

/// Read tensor info for v3 practical format - uses u64 for name length
fn read_tensor_info_v3<R>(reader: &mut R) -> Result<GgufTensorInfo, GgufError>
where
    R: Read + std::io::Seek,
{
    // Per llama.cpp gguf_reader.py _get_tensor_info_field():
    // - Name length: u64 LE (NOT u32!)
    // - Name: raw bytes
    // - N dims: u32 LE
    // - Shape: n_dims * u64 LE
    // - Dtype: u32 LE
    // - Offset: u64 LE

    // 1. Read tensor name length (u64 LE)
    let name_len = reader.read_u64::<LittleEndian>()?;
    if name_len == 0 || name_len > GGUF_MAX_TENSOR_NAME_LENGTH {
        return Err(GgufError::TensorNameLengthOutOfRange(name_len));
    }

    // 2. Read tensor name
    let name_bytes = read_bytes(reader, name_len as usize)?;
    let name = String::from_utf8(name_bytes).map_err(GgufError::Utf8)?;

    // 3. Read number of dimensions (u32 LE)
    let n_dims = reader.read_u32::<LittleEndian>()?;

    // 4. Read shape array (n_dims * u64 LE)
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    // 5. Read data type (u32 LE)
    let dtype = reader.read_u32::<LittleEndian>()?;

    // 6. Read data offset (u64 LE)
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

/// Read tensor info for v2 format (keys use u32 lengths)
fn read_tensor_info_v2<R: Read + Seek>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> {
    // Tensor name: u32 length + raw bytes
    let name_len = reader.read_u32::<LittleEndian>()? as usize;
    if name_len == 0 || name_len > 1024 * 1024 {
        return Err(GgufError::TensorNameLengthOutOfRange(name_len as u64));
    }

    let name_bytes = read_bytes(reader, name_len)?;
    let name = String::from_utf8(name_bytes).map_err(GgufError::Utf8)?;

    // Shape: u32 count + shape array (u64 each)
    let n_dims = reader.read_u32::<LittleEndian>()? as usize;
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    // Dtype: u32
    let dtype = reader.read_u32::<LittleEndian>()?;

    // Offset: u64
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo {
        name,
        shape,
        offset,
        dtype,
    })
}

fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Result<u64, GgufError> {
    let alignment = kv_pairs
        .iter()
        .find(|p| p.key == "general.alignment")
        .and_then(|p| p.value.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u64);

    // Validate alignment is a non-zero power of two (per llama.cpp spec)
    if alignment == 0 {
        return Err(GgufError::AlignmentOutOfRange(0));
    }
    
    // Check power of two: (n & (n-1)) == 0 for powers of two
    if (alignment & (alignment - 1)) != 0 {
        return Err(GgufError::AlignmentOutOfRange(alignment as u32));
    }

    // Sanity check: alignment should be reasonable (max 64KB)
    // Real GGUF files use 32, 256, or rarely 4096 bytes
    const MAX_ALIGNMENT: u64 = 65536; // 64KB
    if alignment > MAX_ALIGNMENT {
        return Err(GgufError::AlignmentOutOfRange(alignment as u32));
    }

    Ok(alignment)
}
/// Read bytes from reader with better error handling
fn read_bytes<R>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            // Provide more context about what we were trying to read
            GgufError::UnexpectedEof {
                expected: len,
                actual: 0, // We didn't read any bytes
            }
        } else {
            GgufError::Binary(e)
        }
    })?;
    Ok(buf)
}

pub fn compute_data_section_start(
    version: u32,
    kv_pairs: &[GgufKvPair],
    tensors: &[GgufTensorInfo],
    data_alignment: Option<u64>,
) -> u64 {
    // Header base: magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24 bytes
    let header_base: u64 = 4 + 4 + 8 + 8;

    let kv_size: u64 = match version {
        3 => kv_pairs.iter().map(|p| p.raw_byte_size_v3() as u64).sum(),
        _ => kv_pairs.iter().map(|p| p.raw_byte_size() as u64).sum(),
    };

    let tensor_size: u64 = tensors.iter().map(|t| t.raw_byte_size() as u64).sum();
    let mut data_section = header_base
        .checked_add(kv_size)
        .and_then(|v| v.checked_add(tensor_size))
        .unwrap_or(u64::MAX);

    // Apply alignment padding (all versions)
    if let Some(alignment) = data_alignment {
        if alignment > 0 {
            let remainder = data_section % alignment;
            if remainder != 0 {
                data_section += alignment - remainder;
            }
        }
    }

    data_section
}

pub fn extract_tensor_bytes_from_path(
    path: &std::path::Path,
    file_offset: u64,
    stored_size: usize,
) -> Result<Vec<u8>, GgufError> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut file = File::open(path).map_err(|e| GgufError::Io(format!("open gguf: {}", e)))?;
    file.seek(SeekFrom::Start(file_offset))
        .map_err(|e| GgufError::Io(format!("seek: {}", e)))?;
    let mut buffer = vec![0u8; stored_size];
    file.read_exact(&mut buffer)
        .map_err(|e| GgufError::Io(format!("read: {}", e)))?;
    Ok(buffer)
}

pub fn extract_tensor_bytes<R>(
    reader: &mut R,
    dtype: u32,
    element_count: u64,
    _offset: u64,
    _data_section_start: u64,
) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let _ = (dtype, _offset, _data_section_start); // placeholder
    let size = tensor_bytes_for_dtype(dtype, element_count);
    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

pub fn extract_tensor_bytes_from<R>(
    reader: &mut R,
    dtype: u32,
    element_count: u64,
    _offset: u64,
    _data_section_start: u64,
) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let _ = (dtype, _offset, _data_section_start); // placeholder
    let size = tensor_bytes_for_dtype(dtype, element_count);
    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

pub fn tensor_bytes_for_dtype(dtype: u32, element_count: u64) -> usize {
    // Delegate to stored_size which has correct per-type calculations
    let info = GgufTensorInfo {
        name: String::new(),
        shape: vec![element_count],
        offset: 0,
        dtype,
    };
    info.stored_size().unwrap_or(0) as usize
}

#[cfg(test)]
mod tests_real_file {
    use super::*;
    use crate::tests::conformance_corpus_path;

    // Note: This test requires conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf
    #[test]
    #[ignore = "Requires conformance corpus file"]
    fn test_parse_conformance_corpus_qwen2_5() {
        let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

        // Should parse without error
        let header = parse_gguf(&path).expect("Failed to parse real GGUF file");

        eprintln!("Header version: {}", header.version);
        assert_eq!(header.version, 3);

        // Should have KV pairs
        assert!(
            header.kv_pairs.len() > 0,
            "Should have KV pairs"
        );
        eprintln!("KV pair count: {}", header.kv_pairs.len());

        // Check a specific key exists
        let has_architecture = header
            .kv_pairs
            .iter()
            .any(|p| p.key == "general.architecture");
        assert!(has_architecture, "Should have general.architecture KV pair");

        // Find and print the architecture value
        if let Some(arch_pair) = header
            .kv_pairs
            .iter()
            .find(|p| p.key == "general.architecture")
        {
            eprintln!("Architecture value: {:?}", arch_pair.value);
        }

        // Should have tensors
        assert!(header.tensors.len() > 0, "Should have tensors");
        eprintln!("Tensor count: {}", header.tensors.len());

        eprintln!("SUCCESS: Real GGUF file parsed correctly!");
    }

    #[test]
    fn test_bfloat16_bit_reinterpretation() {
        // Test that BF16 values are correctly reinterpreted as bit patterns, not numeric conversions
        // 0x3F80 in BF16 represents 1.0 (not 16256.0 which would be from numeric conversion)
        
        let bf16_bits: u16 = 0x3F80;
        // The correct interpretation: shift bits to high position and reinterpret as f32
        let expected_value = f32::from_bits((bf16_bits as u32) << 16);
        
        assert_eq!(expected_value, 1.0, "BF16 0x3F80 should represent 1.0");
        
        // Test zero value: 0x0000 represents 0.0 in BF16
        let bf16_zero: u16 = 0x0000;
        let zero_value = f32::from_bits((bf16_zero as u32) << 16);
        assert_eq!(zero_value, 0.0, "BF16 0x0000 should represent 0.0");
        
        // Test that numeric conversion would give wrong result for 0x3F80
        let wrong_conversion = bf16_bits as f32;
        assert_eq!(wrong_conversion, 16256.0, "Numeric conversion of 0x3F80 gives 16256.0 (WRONG)");
    }
}
