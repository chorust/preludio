use crate::error::PreludioError;
use crate::manifest::{Manifest, PreprocessingStep};
use crate::observation::{FrameValues, ObservationSequence};

#[derive(Clone)]
struct AxisWeights {
    first: usize,
    count: usize,
    weights: Vec<f32>,
}

fn axis_weights(input_size: usize, output_size: usize) -> Vec<AxisWeights> {
    let scale = input_size as f32 / output_size as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut table = Vec::with_capacity(output_size);

    for out_index in 0..output_size {
        let center = scale * (out_index as f32 + 0.5);
        let first_unclamped = (center - support + 0.5) as i64;
        let end_unclamped = (center + support + 0.5) as i64;
        let first = first_unclamped.max(0) as usize;
        let end = std::cmp::min(end_unclamped.max(0) as usize, input_size);
        let count = end.saturating_sub(first);
        let mut weights = vec![0.0f32; count];
        let mut total = 0.0f32;

        for (tap, weight) in weights.iter_mut().enumerate() {
            let source_index = first + tap;
            let x = (source_index as f32 - center + 0.5) * inverse_scale;
            let distance = x.abs();
            let value = if distance < 1.0 { 1.0 - distance } else { 0.0 };
            *weight = value;
            total += value;
        }

        if total != 0.0 {
            for weight in weights.iter_mut() {
                *weight /= total;
            }
        }
        table.push(AxisWeights {
            first,
            count,
            weights,
        });
    }
    table
}

#[inline]
pub fn to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7fffff;

    if exponent == 0xff {
        return sign | 0x7c00 | if fraction != 0 { 0x0200 } else { 0 };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let full_fraction = fraction | 0x800000;
        let shift = 14 - half_exponent;
        let truncated = full_fraction >> shift;
        let remainder = full_fraction & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let round_up = remainder > halfway || (remainder == halfway && (truncated & 1) != 0);
        let mut half = sign | (truncated as u16);
        if round_up {
            half = half.wrapping_add(1);
        }
        return half;
    }

    let truncated_fraction = fraction >> 13;
    let remainder = fraction & 0x1fff;
    let round_up = remainder > 0x1000 || (remainder == 0x1000 && (truncated_fraction & 1) != 0);
    let mut half = sign | ((half_exponent as u16) << 10) | truncated_fraction as u16;
    if round_up {
        half = half.wrapping_add(1);
    }
    half
}

#[inline]
pub fn from_half_bits(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let fraction = (half & 0x03ff) as u32;

    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f800000 | (fraction << 13));
    }
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let mut norm_fraction = fraction;
        let mut shift = 0;
        while (norm_fraction & 0x0400) == 0 {
            norm_fraction <<= 1;
            shift += 1;
        }
        norm_fraction &= 0x03ff;
        let f32_exponent = 127 - 15 + 1 - shift;
        return f32::from_bits(sign | (f32_exponent << 23) | (norm_fraction << 13));
    }

    let f32_exponent = exponent + 127 - 15;
    f32::from_bits(sign | (f32_exponent << 23) | (fraction << 13))
}

#[inline]
pub fn round_f16(value: f32) -> f32 {
    from_half_bits(to_half_bits(value))
}

pub fn resize_bilinear_aa_f32(
    input: &[f32],
    frames: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
) -> Result<Vec<f32>, PreludioError> {
    if frames == 0 || in_height == 0 || in_width == 0 || out_height == 0 || out_width == 0 {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::InvalidShape {
                name: "resize".to_string(),
                shape: vec![frames, in_height, in_width, out_height, out_width],
            },
        ]));
    }

    if [frames, in_height, in_width, out_height, out_width]
        .iter()
        .any(|&dimension| dimension > i64::MAX as usize)
    {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize".to_string(),
            },
        ]));
    }

    let Some(input_len) = frames
        .checked_mul(in_height)
        .and_then(|length| length.checked_mul(in_width))
    else {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize input".to_string(),
            },
        ]));
    };
    if input.len() != input_len {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ValueLength {
                index: 0,
                expected: input_len,
                actual: input.len(),
            },
        ]));
    }

    let Some(horizontal_len) = frames
        .checked_mul(in_height)
        .and_then(|length| length.checked_mul(out_width))
    else {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize intermediate".to_string(),
            },
        ]));
    };
    let Some(output_len) = frames
        .checked_mul(out_height)
        .and_then(|length| length.checked_mul(out_width))
    else {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize output".to_string(),
            },
        ]));
    };
    let max_elements = isize::MAX as usize / std::mem::size_of::<f32>();
    if horizontal_len > max_elements || output_len > max_elements {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize buffer".to_string(),
            },
        ]));
    }
    let max_axis_entries = isize::MAX as usize / std::mem::size_of::<AxisWeights>();
    if out_height > max_axis_entries || out_width > max_axis_entries {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "resize weights".to_string(),
            },
        ]));
    }

    let x_weights = axis_weights(in_width, out_width);
    let y_weights = axis_weights(in_height, out_height);

    let mut horizontal = vec![0.0f32; horizontal_len];
    for frame in 0..frames {
        let input_frame = frame * in_height * in_width;
        let horizontal_frame = frame * in_height * out_width;
        for y in 0..in_height {
            let input_row = input_frame + y * in_width;
            let output_row = horizontal_frame + y * out_width;
            for (out_x, coefficients) in x_weights.iter().enumerate() {
                if coefficients.count == 0 {
                    continue;
                }
                let mut value = input[input_row + coefficients.first] * coefficients.weights[0];
                for tap in 1..coefficients.count {
                    value = input[input_row + coefficients.first + tap]
                        .mul_add(coefficients.weights[tap], value);
                }
                horizontal[output_row + out_x] = value;
            }
        }
    }

    let mut output = vec![0.0f32; output_len];
    for frame in 0..frames {
        let horizontal_frame = frame * in_height * out_width;
        let output_frame = frame * out_height * out_width;
        for (out_y, coefficients) in y_weights.iter().enumerate() {
            if coefficients.count == 0 {
                continue;
            }
            let output_row = output_frame + out_y * out_width;
            let first_row = horizontal_frame + coefficients.first * out_width;
            for x in 0..out_width {
                let mut value = horizontal[first_row + x] * coefficients.weights[0];
                for tap in 1..coefficients.count {
                    value = horizontal[first_row + tap * out_width + x]
                        .mul_add(coefficients.weights[tap], value);
                }
                output[output_row + x] = value;
            }
        }
    }
    Ok(output)
}

fn invalid_manifest(message: impl Into<String>) -> PreludioError {
    PreludioError::ModelUnavailable(crate::error::ModelUnavailable::ManifestInvalid {
        message: message.into(),
    })
}

fn model_input_dimensions(manifest: &Manifest) -> Result<(usize, usize, usize), PreludioError> {
    let shape = &manifest.tensor_input.shape;
    if shape.len() != 4 {
        return Err(invalid_manifest(
            "tensor_input.shape must have four dimensions",
        ));
    }
    let [batch, frames, height, width] = [shape[0], shape[1], shape[2], shape[3]];
    if batch != 1 || frames != manifest.observation.frames {
        return Err(invalid_manifest(
            "tensor_input.shape batch and time dimensions do not match the model contract",
        ));
    }
    if frames == 0 || height == 0 || width == 0 {
        return Err(invalid_manifest(
            "tensor_input time and spatial dimensions must be positive",
        ));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension));
    let Some(expected_len) = expected_len else {
        return Err(invalid_manifest("tensor_input.shape arithmetic overflow"));
    };
    if expected_len > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(invalid_manifest(
            "tensor_input.shape exceeds maximum allocation size",
        ));
    }
    Ok((height, width, expected_len))
}

pub fn preprocess(
    manifest: &Manifest,
    seq: &ObservationSequence,
) -> Result<Vec<f32>, PreludioError> {
    crate::validate::validate(manifest, seq)?;
    let (target_h, target_w, expected_output_len) = model_input_dimensions(manifest)?;
    let frames = seq.frames.len();
    let mut current_h = seq.grid.shape[0];
    let mut current_w = seq.grid.shape[1];
    let capacity = frames
        .checked_mul(current_h)
        .and_then(|size| size.checked_mul(current_w))
        .ok_or_else(|| {
            PreludioError::IncompatibleInput(vec![crate::error::InputViolation::ShapeOverflow {
                name: "observation tensor".to_string(),
            }])
        })?;

    if capacity > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::ShapeOverflow {
                name: "observation tensor".to_string(),
            },
        ]));
    }
    let mut buf: Vec<f32> = Vec::with_capacity(capacity);
    let threshold = seq.zero_threshold;
    for (index, frame) in seq.frames.iter().enumerate() {
        let values = match &frame.values {
            FrameValues::U8(values) => values,
            FrameValues::F32(_) => {
                return Err(PreludioError::IncompatibleInput(vec![
                    crate::error::InputViolation::UnsupportedValueType {
                        index,
                        expected: "u8 grayscale values".to_string(),
                        actual: "f32".to_string(),
                    },
                ]));
            }
        };
        if let Some(boundary) = threshold {
            buf.extend(
                values
                    .iter()
                    .map(|&value| if value < boundary { 0.0 } else { value as f32 }),
            );
        } else {
            buf.extend(values.iter().map(|&value| value as f32));
        }
    }

    let mut normalized = false;
    for step in &manifest.preprocessing {
        match step {
            PreprocessingStep::U8Div255 => {
                if !normalized {
                    for value in &mut buf {
                        *value /= 255.0;
                    }
                    normalized = true;
                }
            }
            PreprocessingStep::ResizeBilinearAa { .. } => {
                if (current_h, current_w) != (target_h, target_w) {
                    buf = resize_bilinear_aa_f32(
                        &buf, frames, current_h, current_w, target_h, target_w,
                    )?;
                    current_h = target_h;
                    current_w = target_w;
                }
            }
            PreprocessingStep::RoundF16 => {
                if (current_h, current_w) != (target_h, target_w) {
                    buf = resize_bilinear_aa_f32(
                        &buf, frames, current_h, current_w, target_h, target_w,
                    )?;
                    current_h = target_h;
                    current_w = target_w;
                }
                for value in &mut buf {
                    *value = round_f16(*value);
                }
            }
        }
    }

    if (current_h, current_w) != (target_h, target_w) {
        buf = resize_bilinear_aa_f32(&buf, frames, current_h, current_w, target_h, target_w)?;
    }
    if buf.len() != expected_output_len {
        return Err(invalid_manifest(format!(
            "preprocessing produced {} values for tensor_input.shape requiring {expected_output_len}",
            buf.len()
        )));
    }
    Ok(buf)
}
