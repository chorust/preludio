#![forbid(unsafe_code)]

pub mod error;
pub mod forecast;
pub mod manifest;
pub mod observation;
pub mod preprocess;
pub mod time;
pub mod validate;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod session;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use error::PreludioError;
pub use forecast::{finish_forecast, ExecutionInfo, Forecast, ForecastMetadata};
pub use manifest::ModelPackage;
pub use observation::{
    FrameValues, GeoReference, GridDescriptor, ObservationFrame, ObservationSequence, PixelQuality,
};
pub use preprocess::preprocess;
pub use validate::validate;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use session::{FrameMeta, Session, SessionOptions, TargetPolicy};

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert_eq!(super::VERSION, "0.1.0");
    }
}
