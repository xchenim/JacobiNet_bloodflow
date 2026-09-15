# AttentionCNN 3D reconstruction

Reconstruct 12 ordered `(x, y, z, radius)` points in metres from paired 128 x 128 relative distance-transform images. MSCBAM combines CBAM with multiscale radius pooling.

## Reproduce

Python 3.12. Keep this package beside `synthetic_100`; run from their parent directory:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r AttentionCNN_3Dreconstruction/requirements.txt
python AttentionCNN_3Dreconstruction/reproduce/evaluate.py --output-root outputs/reconstruction_100
```

Reads case `projections/` and `weights/attentioncnn.pt`; writes predictions, per-case metrics, and summary/confusion files. Use a new output directory outside code/data. For CPU, use `--device cpu` and the `cpu` wheel index.

Evaluation uses float64 parameters, inputs, XYZ, labels, and metrics; attention pooling and radius decoding use float32, with radii exported as float64. Autocast/TF32 are disabled; no radius smoothing. Architecture and checkpoint SHA-256 are checked.

This evaluates 100 supplied cases, not the original 5,000-case test set.

## Train

[train.py](train.py) provides two-stage training and validation-based selection. Separate train/validation data and normalization statistics are required; `synthetic_100` is for evaluation. Use `python AttentionCNN_3Dreconstruction/train.py --help`.

## License

Project-owned code: [MIT](LICENSE), **Xi Chen, et al.** Dependencies retain their licenses.
