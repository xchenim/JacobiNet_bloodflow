# First-release checklist

## Repository hygiene

- [ ] Confirm every file may be distributed publicly.
- [ ] Remove secrets, tokens, private paths, and personal or restricted data.
- [ ] Scan the full Git history, not only the latest working tree.
- [ ] Keep large checkpoints and datasets out of Git; publish them as release assets
      or in a research-data archive and record checksums.
- [ ] Verify third-party code and datasets have compatible licenses and attribution.

## Reproducibility

- [ ] Pin a tested Python and dependency environment.
- [ ] Document CPU/GPU, CUDA, memory, and expected runtime requirements.
- [ ] Include a small smoke-test example that runs without private data.
- [ ] Provide commands for preprocessing, training, inference, and evaluation.
- [ ] Record random seeds and expected metrics with reasonable tolerances.
- [ ] Test setup from a fresh clone on at least one clean environment.

## Publication

- [ ] Replace the release-status note in the root README.
- [ ] Confirm author names and citation metadata in `CITATION.cff`.
- [ ] Create a semantic version tag such as `v0.1.0`.
- [ ] Write release notes listing supported experiments and known limitations.
- [ ] Archive the tagged release with Zenodo if a DOI is desired.
- [ ] Enable private vulnerability reporting and branch protection on `main`.
