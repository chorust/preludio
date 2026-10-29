#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use std::{fs, path::Path};

pub struct NpyF32 {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub fn read_npy_f32(path: impl AsRef<Path>) -> Result<NpyF32> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("{} is not a NumPy .npy file", path.display());
    }

    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated NPY v{major} header in {}", path.display());
            }
            (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12)
        }
        _ => bail!("unsupported NPY version {major} in {}", path.display()),
    };
    let data_start = header_start + header_len;
    if data_start > bytes.len() {
        bail!("truncated NPY header in {}", path.display());
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .with_context(|| format!("NPY header in {} is not UTF-8", path.display()))?;

    let descr_key = header.find("descr").context("NPY header has no descr")?;
    let descr_tail = &header[descr_key + "descr".len()..];
    let descr_colon = descr_tail.find(':').context("malformed NPY descr field")?;
    let descr_value = descr_tail[descr_colon + 1..].trim_start();
    let quote = descr_value.as_bytes().first().copied().context("empty NPY descr")?;
    if quote != b'\'' && quote != b'"' {
        bail!("unsupported NPY descr format in {}", path.display());
    }
    let descr_end = descr_value[1..]
        .find(quote as char)
        .context("unterminated NPY descr")? + 1;
    let descr = &descr_value[1..descr_end];
    if descr != "<f4" && descr != "=f4" {
        bail!("expected little-endian float32 NPY data, got {descr:?} in {}", path.display());
    }

    let shape_key = header.find("shape").context("NPY header has no shape")?;
    let shape_tail = &header[shape_key + "shape".len()..];
    let open = shape_tail.find('(').context("malformed NPY shape field")?;
    let close = shape_tail[open + 1..]
        .find(')')
        .context("unterminated NPY shape")? + open + 1;
    let shape = shape_tail[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.parse::<usize>().context("invalid NPY dimension"))
        .collect::<Result<Vec<_>>>()?;
    let expected = shape.iter().product::<usize>();
    let data = &bytes[data_start..];
    if data.len() != expected * 4 {
        bail!(
            "NPY element count/size mismatch in {}: shape {shape:?}, data bytes {}",
            path.display(),
            data.len()
        );
    }
    let values = data
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect();
    Ok(NpyF32 { shape, values })
}

pub fn compare(got: &[f32], reference: &[f32]) -> Result<(f64, f64, usize)> {
    if got.len() != reference.len() {
        bail!("output length {} differs from reference length {}", got.len(), reference.len());
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut failures = 0usize;
    for (&actual, &expected) in got.iter().zip(reference) {
        let abs = (actual as f64 - expected as f64).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        let tolerance = 1.0e-5 + 1.0e-4 * (expected as f64).abs();
        if !actual.is_finite() || abs > tolerance {
            failures += 1;
        }
    }
    Ok((max_abs, sum_abs / got.len() as f64, failures))
}
