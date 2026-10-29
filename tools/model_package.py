"""Build and validate Preludio model package manifests."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema


_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "001-local-nowcast-runtime"
    / "contracts"
    / "model-package.schema.json"
)

_DEFAULT_MODEL: dict[str, Any] = {
    "file": "model.onnx",
    "sha256": "7c5a6ed7865c0c85054ce45b2fb0b2f66694453491f7e6acabba14ee51533de8",
    "format": "onnx",
    "opset": 18,
}
_DEFAULT_ORIGIN: dict[str, Any] = {
    "checkpoint": {
        "file": "step_22000.ckpt",
        "step": 22000,
        "sha256": "c5d8b457ad603ff53502616e97cc048aee0b22ded033b91acc5742b88fbc4a12",
        "run": "ghost_unet3d_wmse/2026-10-23_23-54-23",
    },
    "source_sha256": "366e590ec938e391c6a43f796254c8836915f1817e3250ba6eee3011254ff07d",
    "config_sha256": "3b8b6dc349d78e840f086d4e78d3e374e31059a25a582525242008b040d09599",
    "exporter": "tools/export_ghost_unet3d.py",
}
_DEFAULT_OBSERVATION: dict[str, Any] = {
    "frames": 4,
    "cadence_seconds": 360,
    "cadence_tolerance_seconds": 0,
    "encoding": "training-grayscale-u8",
    "grid": {
        "id": "onix-station-png-732",
        "shape": [732, 732],
        "georeferenced": False,
    },
    "missing_policy": "reject",
}
_DEFAULT_TENSOR_INPUT: dict[str, Any] = {
    "name": "radar",
    "dtype": "float32",
    "shape": [1, 4, 448, 448],
    "axes": ["batch", "time", "y", "x"],
    "range": [0.0, 1.0],
}
_DEFAULT_TENSOR_OUTPUT: dict[str, Any] = {
    "name": "forecast",
    "dtype": "float32",
    "shape": [1, 20, 448, 448],
    "axes": ["batch", "lead_time", "y", "x"],
    "range": [0.0, 1.0],
}
_DEFAULT_PREPROCESSING: list[dict[str, Any]] = [
    {"op": "u8_div_255"},
    {
        "op": "resize_bilinear_aa",
        "height": 448,
        "width": 448,
        "align_corners": False,
    },
    {"op": "round_f16"},
]
_DEFAULT_FORECAST: dict[str, Any] = {
    "lead_minutes": list(range(6, 126, 6)),
    "encoding": "training-grayscale-normalized",
    "grid": {
        "id": "onix-station-png-732/448",
        "shape": [448, 448],
        "georeferenced": False,
    },
}


def validate_manifest(path_or_dict: str | Path | Mapping[str, Any]) -> None:
    """Validate a manifest file or dictionary against the checked-in schema."""
    if isinstance(path_or_dict, Mapping):
        manifest = path_or_dict
    else:
        with Path(path_or_dict).open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    jsonschema.validate(instance=manifest, schema=schema)


def build_manifest(
    *,
    model_id: str = "ghost-unet3d-step-22000",
    model: Mapping[str, Any] | None = None,
    origin: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    tensor_input: Mapping[str, Any] | None = None,
    tensor_output: Mapping[str, Any] | None = None,
    preprocessing: Sequence[Mapping[str, Any]] | None = None,
    forecast: Mapping[str, Any] | None = None,
    supported_targets: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a v1 manifest, defaulting to the current step-22000 model."""
    manifest = {
        "schema": "preludio.model-package/1",
        "id": model_id,
        "model": copy.deepcopy(_DEFAULT_MODEL if model is None else model),
        "origin": copy.deepcopy(_DEFAULT_ORIGIN if origin is None else origin),
        "observation": copy.deepcopy(
            _DEFAULT_OBSERVATION if observation is None else observation
        ),
        "tensor_input": copy.deepcopy(
            _DEFAULT_TENSOR_INPUT if tensor_input is None else tensor_input
        ),
        "tensor_output": copy.deepcopy(
            _DEFAULT_TENSOR_OUTPUT if tensor_output is None else tensor_output
        ),
        "preprocessing": copy.deepcopy(
            _DEFAULT_PREPROCESSING if preprocessing is None else preprocessing
        ),
        "forecast": copy.deepcopy(
            _DEFAULT_FORECAST if forecast is None else forecast
        ),
        "supported_targets": copy.deepcopy(
            [] if supported_targets is None else supported_targets
        ),
    }
    validate_manifest(manifest)
    return manifest


def _assert_record_target_matches(
    record: Mapping[str, Any],
    record_path: Path,
    *,
    platform: str,
    engine: str,
    engine_version: str,
    provider: str,
    threads: int | str,
    requires_cross_origin_isolation: bool,
    target_schema_properties: Mapping[str, Any],
) -> int:
    """Require a record's reported target to support the registered target."""
    target = record["target"]
    expected_values = {
        "platform": platform,
        "engine": engine,
        "engine_version": engine_version,
        "provider": provider,
    }
    for field, expected in expected_values.items():
        actual = target[field]
        if actual != expected:
            raise ValueError(
                f"{record_path} target {field} mismatch: {actual!r} vs {expected!r}"
            )

    actual_threads = target["threads"]
    if threads != "default" and actual_threads != threads:
        raise ValueError(
            f"{record_path} target threads mismatch: {actual_threads!r} vs {threads!r}"
        )

    isolation_field = "cross_origin_isolated"
    if isolation_field in target_schema_properties:
        if isolation_field in target:
            actual_isolation = target[isolation_field]
            if actual_isolation != requires_cross_origin_isolation:
                raise ValueError(
                    f"{record_path} target {isolation_field} mismatch: "
                    f"{actual_isolation!r} vs {requires_cross_origin_isolation!r}"
                )
        elif requires_cross_origin_isolation:
            raise ValueError(
                f"{record_path} target is missing {isolation_field} required by registration"
            )

    return actual_threads


def add_target(
    package_dir: str | Path,
    target_id: str,
    platform: str,
    engine: str,
    engine_version: str,
    provider: str,
    threads: int | str,
    requires_cross_origin_isolation: bool,
    validation_rel: str,
    measurement_rel: str,
) -> None:
    """Append a supported_targets entry only when referenced records validate and pass."""
    pkg_path = Path(package_dir)
    manifest_path = pkg_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val_path = pkg_path / validation_rel
    if not val_path.is_file():
        raise FileNotFoundError(f"validation record not found: {val_path}")
    with open(val_path, "r", encoding="utf-8") as f:
        val_record = json.load(f)

    meas_path = pkg_path / measurement_rel
    if not meas_path.is_file():
        raise FileNotFoundError(f"measurement record not found: {meas_path}")
    with open(meas_path, "r", encoding="utf-8") as f:
        meas_record = json.load(f)

    # Validate records against validation-record.schema.json
    record_schema_path = _SCHEMA_PATH.parent / "validation-record.schema.json"
    with open(record_schema_path, "r", encoding="utf-8") as f:
        record_schema = json.load(f)

    # Schema store for resolving model-package.schema.json and forecast-metadata.schema.json
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        pkg_schema = json.load(f)
    with open(_SCHEMA_PATH.parent / "forecast-metadata.schema.json", "r", encoding="utf-8") as f:
        fc_schema = json.load(f)

    schema_store = {
        "model-package.schema.json": pkg_schema,
        "https://github.com/chorust/preludio/schemas/model-package.schema.json": pkg_schema,
        "forecast-metadata.schema.json": fc_schema,
        "https://github.com/chorust/preludio/schemas/forecast-metadata.schema.json": fc_schema,
    }
    resolver = jsonschema.RefResolver.from_schema(record_schema, store=schema_store)
    val_validator = jsonschema.Draft202012Validator(record_schema, resolver=resolver)
    val_validator.validate(val_record)

    meas_subschema = record_schema["$defs"]["measurement"]
    meas_subschema_full = copy.deepcopy(record_schema)
    meas_subschema_full.update({
        "required": meas_subschema["required"],
        "properties": meas_subschema["properties"],
    })
    meas_validator = jsonschema.Draft202012Validator(meas_subschema_full, resolver=resolver)
    meas_validator.validate(meas_record)

    # Check pass status
    if not val_record.get("result", {}).get("pass"):
        raise ValueError(f"validation record {val_path} did not pass")
    if not meas_record.get("pass"):
        raise ValueError(f"measurement record {meas_path} did not pass")

    # Check model sha256 match
    expected_sha = manifest["model"]["sha256"]
    if val_record.get("model_sha256") != expected_sha:
        raise ValueError(f"validation record model_sha256 mismatch: {val_record.get('model_sha256')} vs {expected_sha}")
    if meas_record.get("model_sha256") != expected_sha:
        raise ValueError(f"measurement record model_sha256 mismatch: {meas_record.get('model_sha256')} vs {expected_sha}")

    target_schema_properties = record_schema["$defs"]["target"]["properties"]
    validation_threads = _assert_record_target_matches(
        val_record,
        val_path,
        platform=platform,
        engine=engine,
        engine_version=engine_version,
        provider=provider,
        threads=threads,
        requires_cross_origin_isolation=requires_cross_origin_isolation,
        target_schema_properties=target_schema_properties,
    )
    measurement_threads = _assert_record_target_matches(
        meas_record,
        meas_path,
        platform=platform,
        engine=engine,
        engine_version=engine_version,
        provider=provider,
        threads=threads,
        requires_cross_origin_isolation=requires_cross_origin_isolation,
        target_schema_properties=target_schema_properties,
    )
    # Record schemas store effective positive thread counts; a manifest may
    # declare its runtime default. Both records must agree on the actual count.
    if threads == "default" and validation_threads != measurement_threads:
        raise ValueError(
            "validation and measurement record target threads mismatch for default: "
            f"{validation_threads!r} vs {measurement_threads!r}"
        )

    new_entry = {
        "id": target_id,
        "platform": platform,
        "engine": engine,
        "engine_version": engine_version,
        "provider": provider,
        "threads": threads,
        "requires_cross_origin_isolation": requires_cross_origin_isolation,
        "validation": validation_rel,
        "measurement": measurement_rel,
    }

    targets = manifest.get("supported_targets", [])
    # Replace if id already exists, otherwise append
    targets = [t for t in targets if t.get("id") != target_id]
    targets.append(new_entry)
    manifest["supported_targets"] = targets

    validate_manifest(manifest)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _parse_threads(value: str) -> int | str:
    if value == "default":
        return value
    try:
        threads = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threads must be a positive integer or 'default'"
        ) from exc
    if threads < 1:
        raise argparse.ArgumentTypeError(
            "threads must be a positive integer or 'default'"
        )
    return threads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a model package manifest"
    )
    validate_parser.add_argument("manifest", type=Path)

    add_target_parser = subparsers.add_parser(
        "add-target", help="append a supported target to a model package manifest"
    )
    add_target_parser.add_argument("package_dir", type=Path)
    add_target_parser.add_argument("--id", required=True)
    add_target_parser.add_argument("--platform", required=True)
    add_target_parser.add_argument("--engine", required=True)
    add_target_parser.add_argument("--engine-version", required=True)
    add_target_parser.add_argument("--provider", required=True)
    add_target_parser.add_argument("--threads", type=_parse_threads, required=True)
    add_target_parser.add_argument("--requires-cross-origin-isolation", action="store_true")
    add_target_parser.add_argument("--validation", required=True)
    add_target_parser.add_argument("--measurement", required=True)

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            validate_manifest(args.manifest)
        except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
            parser.exit(1, f"{parser.prog}: invalid manifest: {exc}\n")
        print(f"valid: {args.manifest}")
    elif args.command == "add-target":
        try:
            add_target(
                args.package_dir,
                target_id=args.id,
                platform=args.platform,
                engine=args.engine,
                engine_version=args.engine_version,
                provider=args.provider,
                threads=args.threads,
                requires_cross_origin_isolation=args.requires_cross_origin_isolation,
                validation_rel=args.validation,
                measurement_rel=args.measurement,
            )
        except Exception as exc:
            parser.exit(1, f"{parser.prog}: failed to add target: {exc}\n")
        print(f"target added: {args.id} in {args.package_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
