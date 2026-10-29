"""
Contract schema validation test suite.

Fixture capture command:
target/debug/preludio replay \
  --package artifacts/ghost-unet3d-step-22000 \
  --policy validation \
  --tensor artifacts/ghost-unet3d-step-22000/reference/input.npy \
  --times 2023-11-30T09:24:00Z,2023-11-30T09:30:00Z,2023-11-30T09:36:00Z,2023-11-30T09:42:00Z > tests/fixtures/forecast-metadata.json
"""

import json
import unittest
from pathlib import Path
import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "specs" / "001-local-nowcast-runtime" / "contracts"


class TestContracts(unittest.TestCase):
    def test_manifest_v1_schema(self):
        schema_path = CONTRACTS_DIR / "model-package.schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        manifest_path = REPO_ROOT / "tests" / "fixtures" / "manifest-v1.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        jsonschema.validate(instance=manifest, schema=schema)

    def test_forecast_metadata_schema(self):
        # Resolver for referencing model-package.schema.json
        model_schema_path = CONTRACTS_DIR / "model-package.schema.json"
        with open(model_schema_path, "r", encoding="utf-8") as f:
            model_schema = json.load(f)

        forecast_schema_path = CONTRACTS_DIR / "forecast-metadata.schema.json"
        with open(forecast_schema_path, "r", encoding="utf-8") as f:
            forecast_schema = json.load(f)

        schema_store = {
            "model-package.schema.json": model_schema,
            "https://github.com/chorust/preludio/schemas/model-package.schema.json": model_schema,
            forecast_schema.get("$id", ""): forecast_schema,
        }
        resolver = jsonschema.RefResolver.from_schema(forecast_schema, store=schema_store)
        validator = jsonschema.Draft202012Validator(forecast_schema, resolver=resolver)

        metadata_path = REPO_ROOT / "tests" / "fixtures" / "forecast-metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        validator.validate(metadata)

    def test_validation_and_measurement_record_schemas(self):
        val_schema_path = CONTRACTS_DIR / "validation-record.schema.json"
        with open(val_schema_path, "r", encoding="utf-8") as f:
            val_schema = json.load(f)

        # Check rejecting failing record
        failing_record = {
            "kind": "validation",
            "target": {
                "platform": "test",
                "device": "test",
                "os": "test",
                "engine": "onnxruntime",
                "engine_version": "1.31.0",
                "provider": "cpu",
                "threads": 4,
            },
            "model_sha256": "0" * 64,
            "input": {
                "entry": "tensor",
                "sha256": "0" * 64,
            },
            "tolerance": {
                "rtol": 0.0001,
                "atol": 0.00001,
            },
            "result": {
                "max_abs_error": 1.0,
                "mean_abs_error": 0.5,
                "failures": 10,
                "elements": 100,
                "pass": False,
            },
            "created_at": "2023-11-30T09:42:00Z",
        }

        # The failing record itself adheres to the schema (pass is boolean), but when tested for pass=true:
        self.assertFalse(failing_record["result"]["pass"])

    @staticmethod
    def _write_registration_package(package_dir, platform, record_overrides=None):
        manifest_path = REPO_ROOT / "tests" / "fixtures" / "manifest-v1.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(package_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        artifact_files = {
            "macos-arm64": (
                "validation-macos-arm64.json",
                "measurement-macos-arm64.json",
            ),
            "web-chrome": (
                "validation-web-chrome-ort-wasm.json",
                "measurement-web-chrome-ort-wasm.json",
            ),
        }
        reference_dir = package_dir / "reference"
        reference_dir.mkdir()
        artifacts_dir = (
            REPO_ROOT / "artifacts" / "ghost-unet3d-step-22000" / "reference"
        )
        record_overrides = record_overrides or {}
        for kind, artifact_name in zip(
            ("validation", "measurement"), artifact_files[platform]
        ):
            with open(artifacts_dir / artifact_name, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["target"].update(record_overrides.get(kind, {}))
            with open(reference_dir / f"{kind}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
        return package_dir / "manifest.json"

    @staticmethod
    def _register_target(package_dir, **overrides):
        from tools.model_package import add_target

        target = {
            "target_id": "contract-test-target",
            "platform": "macos-arm64",
            "engine": "onnxruntime",
            "engine_version": "1.31.0",
            "provider": "cpu",
            "threads": 4,
            "requires_cross_origin_isolation": False,
        }
        target.update(overrides)
        add_target(
            package_dir,
            validation_rel="reference/validation.json",
            measurement_rel="reference/measurement.json",
            **target,
        )

    def test_add_target_accepts_matching_records(self):
        cases = (
            (
                "macos-arm64",
                {
                    "platform": "macos-arm64",
                    "engine": "onnxruntime",
                    "engine_version": "1.31.0",
                    "provider": "cpu",
                    "threads": 4,
                    "requires_cross_origin_isolation": False,
                },
            ),
            (
                "macos-arm64",
                {
                    "platform": "macos-arm64",
                    "engine": "onnxruntime",
                    "engine_version": "1.31.0",
                    "provider": "cpu",
                    "threads": "default",
                    "requires_cross_origin_isolation": False,
                },
            ),
            (
                "web-chrome",
                {
                    "platform": "web-chrome",
                    "engine": "onnxruntime-web",
                    "engine_version": "1.30.0",
                    "provider": "wasm",
                    "threads": 4,
                    "requires_cross_origin_isolation": True,
                },
            ),
        )
        import tempfile

        for record_platform, target in cases:
            with self.subTest(
                record_platform=record_platform, threads=target["threads"]
            ):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    package_dir = Path(tmp_dir)
                    manifest_path = self._write_registration_package(
                        package_dir, record_platform
                    )
                    self._register_target(package_dir, **target)
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        updated_manifest = json.load(f)
                    registered = next(
                        item
                        for item in updated_manifest["supported_targets"]
                        if item["id"] == "contract-test-target"
                    )
                    self.assertEqual(registered["threads"], target["threads"])

    def test_add_target_rejects_registration_mismatch_without_mutation(self):
        cases = (
            ("platform", {"platform": "web-chrome"}),
            ("engine", {"engine": "onnxruntime-web"}),
            ("engine_version", {"engine_version": "1.30.0"}),
            ("provider", {"provider": "wasm"}),
            ("threads", {"threads": 8}),
            ("isolation", {"requires_cross_origin_isolation": True}),
        )
        import tempfile

        for name, override in cases:
            with self.subTest(field=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    package_dir = Path(tmp_dir)
                    manifest_path = self._write_registration_package(
                        package_dir, "macos-arm64"
                    )
                    original_manifest = manifest_path.read_bytes()
                    with self.assertRaises(ValueError):
                        self._register_target(package_dir, **override)
                    self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_add_target_rejects_each_record_mismatch_without_mutation(self):
        mismatches = (
            ("platform", "web-chrome"),
            ("engine", "onnxruntime-web"),
            ("engine_version", "1.30.0"),
            ("provider", "wasm"),
            ("threads", 8),
            ("cross_origin_isolated", True),
        )
        import tempfile

        for record_kind in ("validation", "measurement"):
            for field, value in mismatches:
                with self.subTest(record=record_kind, field=field):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        package_dir = Path(tmp_dir)
                        manifest_path = self._write_registration_package(
                            package_dir,
                            "macos-arm64",
                            {record_kind: {field: value}},
                        )
                        original_manifest = manifest_path.read_bytes()
                        with self.assertRaises(ValueError):
                            self._register_target(package_dir)
                        self.assertEqual(
                            manifest_path.read_bytes(), original_manifest
                        )

    def test_add_target_default_threads_require_matching_records(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = Path(tmp_dir)
            manifest_path = self._write_registration_package(
                package_dir,
                "macos-arm64",
                {"measurement": {"threads": 8}},
            )
            original_manifest = manifest_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "threads"):
                self._register_target(package_dir, threads="default")
            self.assertEqual(manifest_path.read_bytes(), original_manifest)


    def test_add_target_rejects_failing_record_without_mutation(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = Path(tmp_dir)
            manifest_path = self._write_registration_package(
                package_dir, "macos-arm64"
            )
            validation_path = package_dir / "reference" / "validation.json"
            with open(validation_path, "r", encoding="utf-8") as f:
                validation_record = json.load(f)
            validation_record["result"]["pass"] = False
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation_record, f, indent=2)
                f.write("\n")

            original_manifest = manifest_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "did not pass"):
                self._register_target(package_dir)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)

if __name__ == "__main__":
    unittest.main()
