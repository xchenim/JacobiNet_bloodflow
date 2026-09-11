# JacobiNet Blood Flow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2508.02537-b31b1b.svg)](https://arxiv.org/abs/2508.02537)

Official repository for the blood-flow experiments accompanying **Solved in Unit
Domain: JacobiNet for Differentiable Coordinate Transformations**.

> **Release status:** the public repository structure is ready. Source code,
> reproducible configurations, and example assets will be added with the first
> code release.

## Overview

JacobiNet learns continuous, differentiable coordinate transformations from
irregular physical domains to a shared unit domain. The blood-flow experiments
couple the learned mapping with physics-informed neural networks (PINNs) for the
two-dimensional incompressible steady Navier--Stokes equations in vessel-like
geometries.

The first complete release is planned to include:

- geometry preprocessing and coordinate-pair generation;
- JacobiNet training and inference;
- PINN-based blood-flow prediction;
- configurations for stenosis and aneurysm experiments;
- evaluation and visualization scripts; and
- small examples or download instructions for larger datasets and checkpoints.

## Repository layout

```text
JacobiNet_bloodflow/
|-- src/          # Model, physics, data, and utility modules
|-- configs/      # Reproducible experiment configurations
|-- examples/     # Small end-to-end examples
|-- tests/        # Automated tests
|-- data/         # Data documentation and small tracked assets
|-- docs/         # Extended documentation and release checklist
`-- .github/      # Contribution templates
```

## Getting started

Installation and reproduction commands will be finalized together with the
source release so that they match the tested environment exactly. Until then,
please watch the repository for the first tagged release.

## Reproducibility policy

Each published experiment should include its configuration, random seed, input
data provenance, checkpoint or checkpoint-download instructions, and an
evaluation command. Large generated files and private or restricted datasets
must not be committed to Git history.

## Citation

If you use this work, please cite:

```bibtex
@article{chen2025jacobinet,
  title   = {Solved in Unit Domain: JacobiNet for Differentiable Coordinate Transformations},
  author  = {Chen, Xi and Yang, Jianchuan and Zhang, Junjie and Yang, Runnan and Liu, Xu and Wang, Hong and Ren, Ziyu and Hu, Wenqi},
  journal = {arXiv preprint arXiv:2508.02537},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.02537}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Contributing and support

Bug reports and focused contributions are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For security
issues, follow [SECURITY.md](SECURITY.md) instead of filing a public issue.

## License

This project is released under the [MIT License](LICENSE).
