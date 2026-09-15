# JacobiNetPINN flow solver

Steady incompressible flow prediction using JacobiNet vessel coordinates and a Fourier-feature PINN. Spatial derivatives are computed explicitly; autograd computes training parameter gradients.

## Reproduce

Python 3.12. Keep this package beside `synthetic_100` and run from their parent directory:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r JacobiNetPINN_flowsolver/requirements.txt
python JacobiNetPINN_flowsolver/reproduce/evaluate.py --output-root outputs/flow_100
```

The command evaluates all 100 cases using `weights/<case_id>/{jacobinet,pinn}.pth`, checked against the dataset manifest. It writes per-case arrays/metrics, `summary.json`, and `table_comparison.json`. Output must be a new directory outside the code and inputs.

Default `--device reference` follows recorded inference devices and requires CUDA for pressure quadrature. For CPU use `--device cpu` and install from the `cpu` wheel index instead of `cu124`. Use `--metrics fields pressure wss` to select metrics or repeat `--case-id` for a subset.

## Train

Run [train_jacobinet.py](train_jacobinet.py), then [train_pinn.py](train_pinn.py); use `--help` for arguments. Both require CUDA and a new/empty `--output-root` outside the inputs and code; PINN also requires `--jacobinet-checkpoint`.

Seed: 99. JacobiNet stops at hard-boundary RMSE `< 1e-3`, capped at 100,000 steps. PINN uses the case's manifest `horizon_steps` and selects the lowest fixed-validation PDE score without CFD labels. Resume with `--resume-checkpoint` and `--resume-history` into another empty directory.

## License

Project-owned code: [MIT](LICENSE), **Xi Chen, et al.** Dependencies retain their licenses.
