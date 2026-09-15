if __package__:
    from . import configs
else:
    import configs
from geomdl import BSpline, utilities, operations
import numpy as np

import random

if __package__:
    from .augmentation import shear_centerlines, warp1
    from .configs import (
        ds_min,
        ds_max,
        start_id,
        end_id,
        stenosis_len_min_mm,
        stenosis_len_max_mm,
        L_RCA_max,
        L_RCA_min,
    )
else:
    from augmentation import shear_centerlines, warp1
    from configs import (
        ds_min,
        ds_max,
        start_id,
        end_id,
        stenosis_len_min_mm,
        stenosis_len_max_mm,
        L_RCA_max,
        L_RCA_min,
    )
from scipy.interpolate import interp1d


def resample_centerline_equal_spacing(C, num_points):
    """Interpolate an (N, 3) centerline at uniform cumulative arc-length positions."""

    diff = np.diff(C, axis=0)
    seg_len = np.linalg.norm(diff, axis=1)
    s = np.insert(np.cumsum(seg_len), 0, 0)

    new_s = np.linspace(0, s[-1], num_points)

    fx = interp1d(s, C[:, 0], kind="linear")
    fy = interp1d(s, C[:, 1], kind="linear")
    fz = interp1d(s, C[:, 2], kind="linear")

    C_new = np.stack([fx(new_s), fy(new_s), fz(new_s)], axis=1)
    return C_new


def RCA_vessel_curve(
    sample_size,
    mean_ctrl_pts,
    stdev_ctrl_pts,
    length,
    rng,
    is_main=True,
    shear=False,
    warp=False,
):
    """Sample an RCA B-spline of the requested arc length in metres. Return (N, 3) points and tangents."""

    random_ctrl_points = rng.uniform(
        mean_ctrl_pts - configs.control_std_lower * stdev_ctrl_pts,
        mean_ctrl_pts
        + stdev_ctrl_pts
        + (configs.control_std_upper - 1) * stdev_ctrl_pts,
    )
    if is_main:
        # Prevent the first sampled segment from reversing its axial direction.
        if random_ctrl_points[0, -1] - random_ctrl_points[1, -1] > 0.0001:
            alpha = rng.uniform(0.5, 1)
            random_ctrl_points[1, -1] = random_ctrl_points[0, -1] + alpha * 0.0015

    new_ctrl_points = random_ctrl_points.copy()

    if shear:
        new_ctrl_points = shear_centerlines(new_ctrl_points, configs.shear_strength)

    if warp:
        new_ctrl_points = warp1(new_ctrl_points, configs.warp_strength)

    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = new_ctrl_points.tolist()
    # generates uniform knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size
    scaling = length / operations.length_curve(curve)
    curve = operations.scale(curve, scaling)

    C = np.array(curve.evalpts)

    ct1 = operations.tangent(
        curve, np.linspace(0, 1, curve.sample_size).tolist(), normalize=True
    )
    curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
    dC = curvetan[:, 1, :3]

    return C, dC


def gaussian(mu, sigma, num_points):
    """Return the sampled Gaussian stenosis profile; sigma controls lesion shape, not smoothing."""
    x = np.linspace(-2, 2, num_points)
    bell_curve_vector = 1 / (sigma * 2 * np.pi) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return bell_curve_vector


def stenosis_generator(
    num_stenoses,
    radius_vector,
    branch_points,
    is_main=True,
    stenosis_severity=None,
    stenosis_position=None,
    stenosis_length=None,
    stenosis_type="gaussian",
):
    """Apply one Gaussian lesion. Severity is a fractional local diameter reduction; lengths are point counts."""
    if num_stenoses != 1 or not is_main or any(p is not None for p in branch_points):
        raise ValueError("Only one lesion on an unbranched RCA is supported")
    if stenosis_type != "gaussian":
        raise ValueError("Only the Gaussian lesion profile is supported")
    # Fractional local diameter reduction equals fractional local radius reduction.
    if stenosis_severity is None:
        stenosis_severity = [
            random.uniform(ds_min, ds_max) for _ in range(num_stenoses)
        ]

    # Sample the lesion center within the configured interior interval.
    if stenosis_position is None:
        num_centerline_points = len(radius_vector)
        start = int(start_id * num_centerline_points)
        end = int(end_id * num_centerline_points)
        possible_stenosis_positions = np.arange(start, end)

        stenosis_position = [np.random.choice(possible_stenosis_positions)]

    new_radius_vector = radius_vector.copy()

    if stenosis_length is not None:
        len_stenosis = stenosis_length
        if len(len_stenosis) < num_stenoses:
            len_stenosis = len_stenosis * num_stenoses
    else:
        num_centerline_points = len(radius_vector)

        # Conversion uses the mean configured RCA length (mm).
        ds_mm = (L_RCA_min + L_RCA_max) * 1000 / 2 / num_centerline_points

        len_stenosis = []

        for _ in range(num_stenoses):

            # Uniform length sampling in nominal mm.
            L_mm = random.uniform(stenosis_len_min_mm, stenosis_len_max_mm)

            n_pts = int(round(L_mm / ds_mm))

            if n_pts % 2 != 0:
                n_pts += 1

            n_pts = max(n_pts, 6)

            len_stenosis.append(n_pts)

    for i in range(num_stenoses):
        pos = stenosis_position[i]
        half = int(len_stenosis[i] / 2)
        if pos - half < 0 or pos + half > len(radius_vector):
            raise ValueError("stenosis extends beyond the supersampled centerline")
        mu = 0
        sigma = configs.stenosis_profile_sigma
        stenosis_vec = gaussian(mu, sigma, len_stenosis[i])

        scaled_vec = stenosis_vec / np.max(stenosis_vec) * stenosis_severity[i]
        new_radius_vector[
            pos - int(len_stenosis[i] / 2) : pos + int(len_stenosis[i] / 2)
        ] = new_radius_vector[
            pos - int(len_stenosis[i] / 2) : pos + int(len_stenosis[i] / 2)
        ] - np.multiply(
            scaled_vec,
            radius_vector[
                pos - int(len_stenosis[i] / 2) : pos + int(len_stenosis[i] / 2)
            ],
        )
        vessel_stenosis_positions = stenosis_position
    return new_radius_vector, stenosis_severity, vessel_stenosis_positions, len_stenosis


def get_vessel_surface(
    curve,
    derivatives,
    branch_points,
    num_centerline_points,
    num_circle_points,
    radius,
    num_stenoses=0,
    is_main_branch=True,
    constant_radius=True,
    stenosis_severity=None,
    stenosis_position=None,
    stenosis_length=None,
    stenosis_type="gaussian",
    return_surface=False,
):
    """Build cross-sectional rings around an (N, 3) curve with (N, 3) tangents. Radii and coordinates are metres."""
    # based on https://www.mathworks.com/matlabcentral/fileexchange/5562-tubeplot and
    # https://www.mathworks.com/matlabcentral/fileexchange/25086-extrude-a-ribbon-tube-and-fly-through-it
    if not is_main_branch or any(p is not None for p in branch_points):
        raise ValueError("Only unbranched RCA surfaces are supported")
    if len(radius) == 1 and constant_radius:
        r = np.tile(radius, num_centerline_points)
    elif len(radius) == 1 and not constant_radius:
        # Combine linear taper with independent Gaussian radius perturbations.
        taper = random.uniform(configs.main_taper_min, configs.main_taper_max)
        r = np.flip(
            np.multiply(
                np.tile(radius, num_centerline_points),
                np.linspace(taper, 1, num_centerline_points),
            )
            + np.array(
                [
                    random.gauss(0, configs.radius_noise_std_m)
                    for i in range(num_centerline_points)
                ]
            )
        )
    else:
        r = radius  # vector containing user-specified radii along centerline

    # create stenoses
    new_r = r.copy()
    percent_stenosis = None
    stenosis_pos = None
    num_stenosis_points = 0
    if num_stenoses > 0:
        new_r, percent_stenosis, stenosis_pos, num_stenosis_points = stenosis_generator(
            num_stenoses,
            r,
            branch_points,
            is_main=is_main_branch,
            stenosis_severity=stenosis_severity,
            stenosis_position=stenosis_position,
            stenosis_length=stenosis_length,
            stenosis_type=stenosis_type,
        )

    if not return_surface:
        return new_r, percent_stenosis, stenosis_pos, num_stenosis_points

    t = np.linspace(0, 2 * np.pi, num_circle_points)
    C = curve
    dC = derivatives

    keep_inds = np.squeeze(np.argwhere(np.sum(abs(dC), 1) != 0))
    dC = dC[keep_inds]
    C = C[keep_inds]

    normal_vector = np.zeros((3))
    idx = np.argmin(np.abs(C[1, :]))
    normal_vector[idx] = 1

    surface = []

    cfact = np.tile(np.cos(t), (3, 1))
    sfact = np.tile(np.sin(t), (3, 1))

    for k in range(C.shape[0]):
        convec = np.cross(normal_vector, dC[k, :])
        convec = convec / np.linalg.norm(convec)
        normal_vector = np.cross(dC[k, :], convec)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # add endcaps to vessel surface for projections
        if k == 0:
            surface_r = np.linspace(0, new_r[k], 50)[1:]

        elif k == C.shape[0] - 1:
            surface_r = np.flip(np.linspace(0, new_r[k], 50)[1:])
        else:
            surface_r = [new_r[k]]

        for R in surface_r:
            points = (
                np.tile(C[k, :], (num_circle_points, 1))
                + np.multiply(
                    cfact.T, np.tile(R * normal_vector, (num_circle_points, 1))
                )
                + np.multiply(sfact.T, np.tile(R * convec, (num_circle_points, 1)))
            )
            surface.append(points)

    surface = np.array(surface)

    X = np.squeeze(surface[:, :, 0])
    Y = np.squeeze(surface[:, :, 1])
    Z = np.squeeze(surface[:, :, 2])

    return X, Y, Z, new_r, percent_stenosis, stenosis_pos, num_stenosis_points
