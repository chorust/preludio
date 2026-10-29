#!/usr/bin/env python3
"""Upgrade an experimental export to the v1 Preludio model package layout."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from model_package import build_manifest, validate_manifest


_REFERENCE_FILES = ("input.npy", "reference.npy", "forecast.npy", "validation.json")
_DEFAULT_CHECKPOINT_RUN = "ghost_unet3d_wmse/2026-10-23_23-54-23"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_reference_file(package_dir: Path, filename: str) -> None:
    source = package_dir / filename
    if not source.exists():
        return
    if not source.is_file():
        raise ValueError(f"Expected a file at {source}")

    reference_dir = package_dir / "reference"
    reference_dir.mkdir(exist_ok=True)
    destination = reference_dir / filename
    if destination.exists():
        if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
            raise FileExistsError(f"Conflicting reference files: {source} and {destination}")
        source.unlink()
        return
    shutil.move(str(source), str(destination))


def _origin_from_experimental_manifest(old_manifest: Mapping[str, Any]) -> dict[str, Any]:
    old_origin = old_manifest.get("origin", {})
    old_checkpoint = old_manifest.get("checkpoint")
    if old_checkpoint is None and isinstance(old_origin, Mapping):
        old_checkpoint = old_origin.get("checkpoint")
    if not isinstance(old_checkpoint, Mapping):
        raise ValueError("Experimental manifest is missing checkpoint information")

    return {
        "checkpoint": {
            "file": old_checkpoint.get("file", "step_22000.ckpt"),
            "step": old_checkpoint.get("step", 22000),
            "sha256": old_checkpoint["sha256"],
            "run": old_checkpoint.get("run", _DEFAULT_CHECKPOINT_RUN),
        },
        "source_sha256": old_manifest.get(
            "source_sha256", old_origin.get("source_sha256") if isinstance(old_origin, Mapping) else None
        ),
        "config_sha256": old_manifest.get(
            "config_sha256", old_origin.get("config_sha256") if isinstance(old_origin, Mapping) else None
        ),
        "exporter": old_origin.get("exporter", "tools/export_ghost_unet3d.py")
        if isinstance(old_origin, Mapping)
        else "tools/export_ghost_unet3d.py",
    }


def upgrade(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(old_manifest, Mapping):
        raise ValueError(f"Expected an object in {manifest_path}")

    model_path = package_dir / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file is missing: {model_path}")

    model = {
        "file": "model.onnx",
        "sha256": _sha256(model_path),
        "format": "onnx",
        "opset": 18,
    }
    manifest = build_manifest(
        model_id=old_manifest["id"],
        model=model,
        origin=_origin_from_experimental_manifest(old_manifest),
        supported_targets=[],
    )
    validate_manifest(manifest)

    for filename in _REFERENCE_FILES:
        _move_reference_file(package_dir, filename)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="experimental model package directory")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    manifest = upgrade(args.directory)
    print(f"Upgraded {args.directory / 'manifest.json'} ({manifest['model']['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
