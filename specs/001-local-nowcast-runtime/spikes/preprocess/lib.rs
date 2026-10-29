#![forbid(unsafe_code)]
use std::cmp::min;

pub const FRAMES: usize = 4;
pub const INPUT_HEIGHT: usize = 732;
pub const INPUT_WIDTH: usize = 732;
pub const OUTPUT_HEIGHT: usize = 448;
pub const OUTPUT_WIDTH: usize = 448;
pub const INPUT_LEN: usize = FRAMES * INPUT_HEIGHT * INPUT_WIDTH;
pub const OUTPUT_LEN: usize = FRAMES * OUTPUT_HEIGHT * OUTPUT_WIDTH;
const MAX_TAPS: usize = 5;

#[derive(Clone, Copy)]
struct AxisWeights {
    first: usize,
    count: usize,
    weights: [f32; MAX_TAPS],
}

fn axis_weights(input_size: usize, output_size: usize) -> Vec<AxisWeights> {
    let scale = input_size as f32 / output_size as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_taps = ((support.ceil() as usize) * 2 + 1).min(MAX_TAPS);
    let mut table = Vec::with_capacity(output_size);

    for out_index in 0..output_size {
        let center = scale * (out_index as f32 + 0.5);
        let first_unclamped = (center - support + 0.5) as i64;
        let end_unclamped = (center + support + 0.5) as i64;
        let first = first_unclamped.max(0) as usize;
        let end = min(end_unclamped.max(0) as usize, input_size);
        let count = end.saturating_sub(first).min(max_taps);
        let mut weights = [0.0f32; MAX_TAPS];
        let mut total = 0.0f32;

        for (tap, weight) in weights.iter_mut().take(count).enumerate() {
            let source_index = first + tap;
            let x = (source_index as f32 - center + 0.5) * inverse_scale;
            let distance = x.abs();
            let value = if distance < 1.0 { 1.0 - distance } else { 0.0 };
            *weight = value;
            total += value;
        }

        if total != 0.0 {
            for weight in weights.iter_mut().take(count) {
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
fn to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7fffff;

    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        let payload = (fraction >> 13) as u16;
        return sign | 0x7c00 | payload | u16::from(payload == 0);
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = fraction | 0x800000;
        let shift = (14 - half_exponent) as u32;
        let truncated = significand >> shift;
        let remainder_mask = (1u32 << shift) - 1;
        let remainder = significand & remainder_mask;
        let halfway = 1u32 << (shift - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && (truncated & 1) != 0));
        return sign | rounded as u16;
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
fn from_half_bits(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let fraction = (half & 0x03ff) as u32;
    if exponent == 0 && fraction != 0 {
        let value = (fraction as f32) * (1.0 / 16_777_216.0);
        return if sign == 0 { value } else { -value };
    }
    let bits = if exponent == 0 {
        sign
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + (127 - 15)) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

pub fn preprocess_u8(input: &[u8]) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
    if input.len() != INPUT_LEN {
        return Err("input must contain exactly four 732x732 uint8 frames");
    }

    let x_weights = axis_weights(INPUT_WIDTH, OUTPUT_WIDTH);
    let y_weights = axis_weights(INPUT_HEIGHT, OUTPUT_HEIGHT);
    let mut normalized = Vec::with_capacity(INPUT_LEN);
    normalized.extend(input.iter().map(|&value| value as f32 / 255.0));

    let mut horizontal = vec![0.0f32; FRAMES * INPUT_HEIGHT * OUTPUT_WIDTH];
    for frame in 0..FRAMES {
        let input_frame = frame * INPUT_HEIGHT * INPUT_WIDTH;
        let horizontal_frame = frame * INPUT_HEIGHT * OUTPUT_WIDTH;
        for y in 0..INPUT_HEIGHT {
            let input_row = input_frame + y * INPUT_WIDTH;
            let output_row = horizontal_frame + y * OUTPUT_WIDTH;
            for (out_x, coefficients) in x_weights.iter().enumerate() {
                let mut value = normalized[input_row + coefficients.first] * coefficients.weights[0];
                for tap in 1..coefficients.count {
                    value = normalized[input_row + coefficients.first + tap]
                        .mul_add(coefficients.weights[tap], value);
                }
                horizontal[output_row + out_x] = value;
            }
        }
    }

    let mut prehalf = vec![0.0f32; OUTPUT_LEN];
    for frame in 0..FRAMES {
        let horizontal_frame = frame * INPUT_HEIGHT * OUTPUT_WIDTH;
        let output_frame = frame * OUTPUT_HEIGHT * OUTPUT_WIDTH;
        for (out_y, coefficients) in y_weights.iter().enumerate() {
            let output_row = output_frame + out_y * OUTPUT_WIDTH;
            let first_row = horizontal_frame + coefficients.first * OUTPUT_WIDTH;
            for x in 0..OUTPUT_WIDTH {
                let mut value = horizontal[first_row + x] * coefficients.weights[0];
                for tap in 1..coefficients.count {
                    value = horizontal[first_row + tap * OUTPUT_WIDTH + x]
                        .mul_add(coefficients.weights[tap], value);
                }
                prehalf[output_row + x] = value;
            }
        }
    }

    let rounded = prehalf
        .iter()
        .map(|&value| from_half_bits(to_half_bits(value)))
        .collect();
    Ok((prehalf, rounded))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn preprocess_wasm(input: &[u8]) -> Result<Vec<f32>, wasm_bindgen::JsValue> {
    let (prehalf, rounded) = preprocess_u8(input)
        .map_err(wasm_bindgen::JsValue::from_str)?;
    let mut output = Vec::with_capacity(OUTPUT_LEN * 2);
    output.extend(prehalf);
    output.extend(rounded);
    Ok(output)
}
