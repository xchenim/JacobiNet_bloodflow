# License scope and third-party sources

The MIT license in `LICENSE`, with attribution to **Xi Chen, et al.**, applies
to contributions for which the project authors hold the necessary rights.
It does not replace the academic/non-commercial terms in `license.md` for
inherited geometry/projection code or the supplied control-point assets.
This package must not be described as entirely MIT-licensed.

This package is adapted from the vessel-generation code accompanying **[A multi-stage neural network approach for coronary 3D reconstruction from uncalibrated X-ray angiography images](https://doi.org/10.1038/s41598-023-44633-2)** by Kritika Iyer, Brahmajee K. Nallamothu, C. Alberto Figueroa, and Raj R. Nadakuditi, *Scientific Reports* 13, 17603 (2023). The upstream repository is [kritiyer/vessel_tree_generator](https://github.com/kritiyer/vessel_tree_generator).

The inherited centerline augmentation, tube construction, forward projection, and RCA control-point assets originate from that generator. Its [license.md](https://github.com/kritiyer/vessel_tree_generator/blob/main/license.md) limits use to academic, non-commercial research. The paper's publication license does not replace the code's license. This attribution identifies the upstream project; it does not assert additional commercial or sublicensing permissions.

Source references retained in the numerical helpers:

- Equal-scale 3D axes: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
- Tube surface construction: https://www.mathworks.com/matlabcentral/fileexchange/5562-tubeplot
- Tube extrusion reference: https://www.mathworks.com/matlabcentral/fileexchange/25086-extrude-a-ribbon-tube-and-fly-through-it

These links identify source references; they do not establish additional
permissions or transfer copyright to the project authors.
