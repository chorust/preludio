use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationSequence {
    pub grid: GridDescriptor,
    pub encoding: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_threshold: Option<u8>,
    pub frames: Vec<ObservationFrame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationFrame {
    pub id: String,
    pub source: String,
    pub valid_time: i64,
    pub values: FrameValues,
    pub quality: Option<Vec<PixelQuality>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameMeta {
    pub id: String,
    pub source: String,
    pub valid_time: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FrameValues {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl FrameValues {
    pub fn len(&self) -> usize {
        match self {
            FrameValues::U8(v) => v.len(),
            FrameValues::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelQuality {
    Valid = 0,
    Missing = 1,
    OutsideCoverage = 2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridDescriptor {
    pub id: String,
    pub shape: [usize; 2],
    pub georeference: Option<GeoReference>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoReference {
    pub crs: String,
    pub affine: [f64; 6],
}
