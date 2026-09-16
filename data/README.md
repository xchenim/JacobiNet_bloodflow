# Synthetic 100-case dataset

Download the dataset from **[Hugging Face: Xi-UST/JacobiNet_bloodflow](https://huggingface.co/datasets/Xi-UST/JacobiNet_bloodflow/tree/main)**.

The dataset contains the `synthetic_100/` folder with paired projection images,
geometry and reconstruction targets, CFD files, sampling workbooks, and numerical
evaluation references for 100 synthetic cases.

Place the complete `synthetic_100/` folder at the repository root:

```text
JacobiNet_bloodflow/
|-- AttentionCNN_3Dreconstruction/
|-- JacobiNetPINN_flowsolver/
|-- RCA_generator/
|-- synthetic_100/
|   |-- cases/
|   |-- checksums.csv
|   |-- manifest.json
|   `-- README.md
`-- data/
    `-- README.md
```

This `data/` directory contains the download instructions. The reproduction
entrypoints expect `synthetic_100/` beside the code packages.

For installation and evaluation commands, see the
[reconstruction README](../AttentionCNN_3Dreconstruction/README.md) and
[flow-solver README](../JacobiNetPINN_flowsolver/README.md). The dataset's own
README describes its files, units, and CFD criteria.
