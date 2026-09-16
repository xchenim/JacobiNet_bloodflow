# JacobiNet blood-flow modeling

Research code for three-dimensional coronary reconstruction and physics-informed
blood-flow modeling.

## Code packages and data

| Resource | Purpose |
|---|---|
| [AttentionCNN_3Dreconstruction](AttentionCNN_3Dreconstruction/README.md) | Reconstruct vessel coordinates and radii from paired projection images. |
| [JacobiNetPINN_flowsolver](JacobiNetPINN_flowsolver/README.md) | Train and evaluate JacobiNet coordinates and the physics-informed flow model. |
| [RCA_generator](RCA_generator/README.md) | Generate synthetic single-RCA geometry and paired projection images. |
| [synthetic_100 dataset](https://huggingface.co/datasets/Xi-UST/JacobiNet_bloodflow/tree/main) | 100 synthetic cases with paired projections, geometry, CFD results, and evaluation references. Hosted on Hugging Face; see [data setup](data/README.md). |

The model packages include their reference weights and reproduction entrypoints.
See each package README for the tested environment, installation, inputs, outputs,
and training commands.

## Download

The AttentionCNN checkpoint uses Git LFS:

```bash
git lfs install
git clone https://github.com/xchenim/JacobiNet_bloodflow.git
cd JacobiNet_bloodflow
git lfs pull
```

## Reproduce the supplied cohort

Download the [`synthetic_100` dataset](https://huggingface.co/datasets/Xi-UST/JacobiNet_bloodflow/tree/main)
from Hugging Face and place its `synthetic_100/` directory beside the three code
packages (see [data setup](data/README.md)). Install the dependencies in the
corresponding package README, then run:

```bash
python AttentionCNN_3Dreconstruction/reproduce/evaluate.py --output-root outputs/reconstruction_100
python JacobiNetPINN_flowsolver/reproduce/evaluate.py --output-root outputs/flow_100
```

Use new output directories outside the code and input data. These entrypoints
evaluate the supplied cohort; they do not generate meshes or solve new CFD cases.

## License scope

The [MIT license](LICENSE), attributed to **Xi Chen, et al.**, applies to authorized
project-owned code. Dependencies retain their licenses.

`RCA_generator` adapts the upstream vessel generator accompanying Iyer et al.,
*A multi-stage neural network approach for coronary 3D reconstruction from
uncalibrated X-ray angiography images*. Its inherited code and control-point
assets retain academic/non-commercial terms. See its [license scope](RCA_generator/LICENSE)
and [third-party notices](RCA_generator/NOTICE.md). The repository must not be
treated as entirely MIT-licensed.

## Related methodology

[CITATION.cff](CITATION.cff) records the JacobiNet coordinate-transform methodology,
*Solved in Unit Domain: JacobiNet for Differentiable Coordinate Transformations*.
For the generator's upstream attribution, see the [RCA package README](RCA_generator/README.md).
