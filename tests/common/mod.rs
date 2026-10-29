#![allow(dead_code)]
use std::path::{Path, PathBuf};

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Reads a C-order NumPy array with little-endian 32-bit floating-point values.
pub fn read_npy_f32(path: impl AsRef<Path>) -> Result<(Vec<usize>, Vec<f32>), String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let (descr, shape, data_offset) = parse_npy_header(&bytes)?;
    if descr != "<f4" {
        return Err(format!("expected NPY dtype '<f4', found {descr:?}"));
    }

    let count = element_count(&shape)?;
    let payload_len = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "NPY payload size overflows usize".to_owned())?;
    let payload = bytes
        .get(data_offset..)
        .ok_or_else(|| "NPY header extends beyond end of file".to_owned())?;
    if payload.len() != payload_len {
        return Err(format!(
            "NPY payload has {} bytes; expected {payload_len} for shape {shape:?}",
            payload.len()
        ));
    }

    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok((shape, values))
}

/// Reads a C-order NumPy array with unsigned 8-bit integer values.
pub fn read_npy_u8(path: impl AsRef<Path>) -> Result<(Vec<usize>, Vec<u8>), String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let (descr, shape, data_offset) = parse_npy_header(&bytes)?;
    if descr != "|u1" && descr != "u1" {
        return Err(format!("expected NPY dtype '|u1' or 'u1', found {descr:?}"));
    }

    let count = element_count(&shape)?;
    let payload = bytes
        .get(data_offset..)
        .ok_or_else(|| "NPY header extends beyond end of file".to_owned())?;
    if payload.len() != count {
        return Err(format!(
            "NPY payload has {} bytes; expected {count} for shape {shape:?}",
            payload.len()
        ));
    }

    Ok((shape, payload.to_vec()))
}

/// Writes a C-order NumPy array with little-endian 32-bit floating-point values.
pub fn write_npy_f32(path: impl AsRef<Path>, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let count = element_count(shape)?;
    if data.len() != count {
        return Err(format!(
            "data has {} values; expected {count} for shape {shape:?}",
            data.len()
        ));
    }

    let mut bytes = make_npy_prefix("<f4", shape)?;
    let payload_len = data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "NPY payload size overflows usize".to_owned())?;
    bytes.reserve(payload_len);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path.as_ref(), bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.as_ref().display()))
}

/// Writes a C-order NumPy array with unsigned 8-bit integer values.
pub fn write_npy_u8(path: impl AsRef<Path>, shape: &[usize], data: &[u8]) -> Result<(), String> {
    let count = element_count(shape)?;
    if data.len() != count {
        return Err(format!(
            "data has {} values; expected {count} for shape {shape:?}",
            data.len()
        ));
    }

    let mut bytes = make_npy_prefix("|u1", shape)?;
    bytes.extend_from_slice(data);
    std::fs::write(path.as_ref(), bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.as_ref().display()))
}

/// Returns the configured model package path, or skips model-package tests when unset.
pub fn model_package() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("PRELUDIO_MODEL_PACKAGE") else {
        eprintln!("[SKIP] PRELUDIO_MODEL_PACKAGE not set; skipping model package test");
        return None;
    };
    if value.is_empty() {
        eprintln!("[SKIP] PRELUDIO_MODEL_PACKAGE not set; skipping model package test");
        return None;
    }

    let path = PathBuf::from(value);
    assert!(
        path.exists(),
        "PRELUDIO_MODEL_PACKAGE does not exist: {}",
        path.display()
    );
    Some(path)
}

fn parse_npy_header(bytes: &[u8]) -> Result<(String, Vec<usize>, usize), String> {
    if bytes.len() < 10 || !bytes.starts_with(NPY_MAGIC) {
        return Err("invalid NPY file: missing magic header".to_owned());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (prefix_len, header_len): (usize, usize) = match (major, minor) {
        (1, 0) => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        (2, 0) | (3, 0) => {
            if bytes.len() < 12 {
                return Err("invalid NPY file: truncated version 2/3 header".to_owned());
            }
            (
                12,
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            )
        }
        _ => return Err(format!("unsupported NPY format version {major}.{minor}")),
    };
    let data_offset = prefix_len
        .checked_add(header_len)
        .ok_or_else(|| "NPY header length overflows usize".to_owned())?;
    if data_offset > bytes.len() {
        return Err("invalid NPY file: header extends beyond end of file".to_owned());
    }
    let header = std::str::from_utf8(&bytes[prefix_len..data_offset])
        .map_err(|error| format!("invalid NPY header encoding: {error}"))?;

    let descr_value = header_field(header, "descr")?;
    let descr = parse_quoted_value(descr_value)?;
    let fortran_order = header_field(header, "fortran_order")?
        .split(|character: char| character == ',' || character == '}' || character.is_whitespace())
        .next()
        .unwrap_or("");
    if fortran_order != "False" {
        return Err("NPY array must use C order (fortran_order: False)".to_owned());
    }
    let shape = parse_shape(header_field(header, "shape")?)?;

    Ok((descr, shape, data_offset))
}

fn header_field<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    for quote in ['\'', '"'] {
        let needle = format!("{quote}{key}{quote}");
        if let Some(index) = header.find(&needle) {
            let value = header[index + needle.len()..].trim_start();
            if let Some(value) = value.strip_prefix(':') {
                return Ok(value.trim_start());
            }
        }
    }
    Err(format!("invalid NPY header: missing {key:?} field"))
}

fn parse_quoted_value(value: &str) -> Result<String, String> {
    let mut characters = value.chars();
    let quote = characters
        .next()
        .filter(|character| *character == '\'' || *character == '"')
        .ok_or_else(|| "invalid NPY header: expected quoted dtype".to_owned())?;
    let end = value[1..]
        .find(quote)
        .map(|index| index + 1)
        .ok_or_else(|| "invalid NPY header: unterminated dtype".to_owned())?;
    Ok(value[1..end].to_owned())
}

fn parse_shape(value: &str) -> Result<Vec<usize>, String> {
    let value = value.trim_start();
    let tuple = value
        .strip_prefix('(')
        .ok_or_else(|| "invalid NPY header: expected shape tuple".to_owned())?;
    let end = tuple
        .find(')')
        .ok_or_else(|| "invalid NPY header: unterminated shape tuple".to_owned())?;
    let dimensions = &tuple[..end];
    let mut shape = Vec::new();
    for dimension in dimensions
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let size = dimension
            .parse::<usize>()
            .map_err(|error| format!("invalid NPY shape dimension {dimension:?}: {error}"))?;
        shape.push(size);
    }
    Ok(shape)
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "NPY element count overflows usize".to_owned())
    })
}

fn make_npy_prefix(descr: &str, shape: &[usize]) -> Result<Vec<u8>, String> {
    let shape = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let shape = match shape.is_empty() {
        true => "()".to_owned(),
        false if shape.contains(", ") || shape.parse::<usize>().is_err() => {
            format!("({shape})")
        }
        false => format!("({shape},)"),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len())
        .map_err(|_| "NPY header is too long for version 1 format".to_owned())?;

    let mut bytes = Vec::with_capacity(10 + header.len());
    bytes.extend_from_slice(NPY_MAGIC);
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    Ok(bytes)
}
