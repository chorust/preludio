//! Preludio — on-device radar nowcasting.
//!
//! Placeholder crate reserving the name. Runtime APIs will land in later versions.

#![forbid(unsafe_code)]

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::VERSION.is_empty());
    }
}
