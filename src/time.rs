use crate::error::PreludioError;

/// Parses an RFC 3339 string (with `Z` or numeric offset) to UTC Unix seconds (i64).
/// Strings without a timezone offset are rejected.
pub fn parse_rfc3339_utc(s: &str) -> Result<i64, PreludioError> {
    // RFC 3339 requires timezone offset. Verify that time::OffsetDateTime parses it.
    let odt = time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map_err(|_| {
            PreludioError::IncompatibleInput(vec![crate::error::InputViolation::NotIncreasing {
                index: 0,
            }])
        })?;
    Ok(odt.unix_timestamp())
}

/// Formats UTC Unix seconds as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn format_rfc3339_utc(timestamp: i64) -> Result<String, PreludioError> {
    let odt = time::OffsetDateTime::from_unix_timestamp(timestamp)
        .map_err(|_| PreludioError::IncompatibleInput(vec![]))?;
    let utc = odt.to_offset(time::UtcOffset::UTC);
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        utc.year(),
        u8::from(utc.month()),
        utc.day(),
        utc.hour(),
        utc.minute(),
        utc.second()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_format() {
        let ts = 1701337320; // 2023-11-30T09:42:00Z
        let formatted = format_rfc3339_utc(ts).unwrap();
        assert_eq!(formatted, "2023-11-30T09:42:00Z");

        assert_eq!(parse_rfc3339_utc("2023-11-30T09:42:00Z").unwrap(), ts);
        assert_eq!(parse_rfc3339_utc("2023-11-30T17:42:00+08:00").unwrap(), ts);
        // Missing timezone
        assert!(parse_rfc3339_utc("2023-11-30T09:42:00").is_err());
    }
}
