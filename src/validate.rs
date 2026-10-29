use crate::error::{InputViolation, PreludioError};
use crate::manifest::Manifest;
use crate::observation::{FrameMeta, FrameValues, ObservationSequence, PixelQuality};

fn collect_time_violations(
    manifest: &Manifest,
    frame_count: usize,
    times: impl Iterator<Item = i64>,
) -> Vec<InputViolation> {
    let mut violations = Vec::new();
    if frame_count != manifest.observation.frames {
        violations.push(InputViolation::FrameCount {
            expected: manifest.observation.frames,
            actual: frame_count,
        });
    }

    let mut previous = None;
    for (index, current) in times.enumerate() {
        if let Some(previous_time) = previous {
            if current <= previous_time {
                violations.push(InputViolation::NotIncreasing { index });
            }
        }
        previous = Some(current);
    }
    violations
}

pub fn validate_times(manifest: &Manifest, frames: &[FrameMeta]) -> Result<(), PreludioError> {
    let mut violations = collect_time_violations(
        manifest,
        frames.len(),
        frames.iter().map(|frame| frame.valid_time),
    );
    if frames.len() < 2 {
        violations.push(InputViolation::InsufficientFrames {
            minimum: 2,
            actual: frames.len(),
        });
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(PreludioError::IncompatibleInput(violations))
    }
}

pub fn validate(manifest: &Manifest, seq: &ObservationSequence) -> Result<(), PreludioError> {
    let mut violations = collect_time_violations(
        manifest,
        seq.frames.len(),
        seq.frames.iter().map(|frame| frame.valid_time),
    );

    let [height, width] = seq.grid.shape;
    let expected_len = if height == 0 || width == 0 {
        violations.push(InputViolation::InvalidShape {
            name: "observation grid".to_string(),
            shape: vec![height, width],
        });
        None
    } else {
        match height.checked_mul(width) {
            Some(length) => Some(length),
            None => {
                violations.push(InputViolation::ShapeOverflow {
                    name: "observation grid".to_string(),
                });
                None
            }
        }
    };

    if let Some(length) = expected_len {
        if seq.frames.len().checked_mul(length).is_none() {
            violations.push(InputViolation::ShapeOverflow {
                name: "observation tensor".to_string(),
            });
        }
    }

    if seq.encoding != manifest.observation.encoding {
        violations.push(InputViolation::EncodingMismatch {
            expected: manifest.observation.encoding.clone(),
            actual: seq.encoding.clone(),
        });
    }

    for (index, frame) in seq.frames.iter().enumerate() {
        if let Some(expected_len) = expected_len {
            let actual_len = frame.values.len();
            if actual_len != expected_len {
                violations.push(InputViolation::ValueLength {
                    index,
                    expected: expected_len,
                    actual: actual_len,
                });
            }
        }

        if matches!(&frame.values, FrameValues::F32(_)) {
            violations.push(InputViolation::UnsupportedValueType {
                index,
                expected: "u8 grayscale values".to_string(),
                actual: "f32".to_string(),
            });
        }

        if let Some(quality) = &frame.quality {
            if let Some(expected_len) = expected_len {
                if quality.len() != expected_len {
                    violations.push(InputViolation::ValueLength {
                        index,
                        expected: expected_len,
                        actual: quality.len(),
                    });
                }
            }

            let mut missing = 0;
            let mut outside_coverage = 0;
            for &pixel_quality in quality {
                match pixel_quality {
                    PixelQuality::Missing => missing += 1,
                    PixelQuality::OutsideCoverage => outside_coverage += 1,
                    PixelQuality::Valid => {}
                }
            }
            if missing > 0 || outside_coverage > 0 {
                violations.push(InputViolation::UnsupportedQuality {
                    index,
                    missing,
                    outside_coverage,
                });
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(PreludioError::IncompatibleInput(violations))
    }
}
