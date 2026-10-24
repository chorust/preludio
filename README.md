# Preludio

**On-device radar nowcasting.**

Forecast the rain before it begins. On your device.

This early `0.0.1` release reserves the crate name. The runtime (model load → preprocess → on-device inference → forecast output) will land in later versions.

The first deployment experiment exports an Onix GhostUNet3D checkpoint to ONNX
and checks CPU inference against PyTorch. See the
[model contract and export instructions](docs/ghost-unet3d.md).

## Documentation

The project documents are in Chinese and distinguish confirmed direction,
measured results, and proposed work.

| Document | Contents |
| --- | --- |
| [Spec](docs/spec.md) | Product scope, data contract, requirements, and acceptance criteria |
| [Roadmap](docs/roadmap.md) | Milestones, dependencies, and evidence needed to proceed |
| [Architecture](docs/architecture.md) | Current tooling, target runtime, integration boundaries, and decisions |
| [GhostUNet3D spike](docs/ghost-unet3d.md) | Reproduction, numerical validation, and deployment limits |

## License

Apache-2.0
