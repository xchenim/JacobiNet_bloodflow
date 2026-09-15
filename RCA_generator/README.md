# RCA generator

Generate unbranched right coronary artery geometry with one Gaussian stenosis and two projection views.

Adapted from Iyer et al., [A multi-stage neural network approach for coronary 3D reconstruction from uncalibrated X-ray angiography images](https://doi.org/10.1038/s41598-023-44633-2), and its [vessel_tree_generator code](https://github.com/kritiyer/vessel_tree_generator).

## Run

Python 3.12, CPU. Run from the package's parent directory:

```bash
python -m pip install -r RCA_generator/requirements.txt
python RCA_generator/main.py --output-root outputs/generated_rca --count 10 --seed 99 --split train
```

Use `--config settings.json` to override [configs.py](configs.py). Outputs require a new or empty directory outside the code and frozen dataset.

Each `<output>/<split>/<case_id>/` contains three NPY files and six PNGs. Geometry stores `(x, y, z, radius)` in metres: full shape `(1, 140, 4)`, local shape `(N, 4)`. Full images are 512 x 512; cropped and distance-transform inputs are 128 x 128. Manifests record dataset layout, configuration, seed, and hashes.


## License

Authorized project-owned contributions: [MIT](LICENSE), **Xi Chen, et al.** Inherited code and control-point assets retain academic/non-commercial terms; see [license.md](license.md) and [NOTICE.md](NOTICE.md).
