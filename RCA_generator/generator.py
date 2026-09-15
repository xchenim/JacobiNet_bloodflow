"""Seeded single-RCA geometry, paired views, and stenosis labels."""

from pathlib import Path
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from . import configs
    from .tube_functions import (
        RCA_vessel_curve,
        get_vessel_surface,
        resample_centerline_equal_spacing,
    )
    from .fwd_projection_functions import generate_projection_images, set_axes_equal
    from .stenosis_projections import stenosis_cut, crop_around_mask
else:
    import configs
    from tube_functions import (
        RCA_vessel_curve,
        get_vessel_surface,
        resample_centerline_equal_spacing,
    )
    from fwd_projection_functions import generate_projection_images, set_axes_equal
    from stenosis_projections import stenosis_cut, crop_around_mask


def _sample_vessel(rng):
    """Sample a supersampled centerline and its radius profile in metres."""
    count = configs.centerline_supersampling * configs.num_centerline_points
    length = random.uniform(configs.L_RCA_min, configs.L_RCA_max)
    control_points = (
        np.load(Path(configs.control_point_path) / "RCA_ctrl_points.npy") / 1000
    )
    centerline, tangents = RCA_vessel_curve(
        count,
        np.mean(control_points, axis=0),
        np.std(control_points, axis=0),
        length,
        rng,
        shear=configs.shear,
        warp=configs.warp,
    )
    centerline = resample_centerline_equal_spacing(centerline, count)
    radius = [random.uniform(configs.D_RCA_min, configs.D_RCA_max) / 2]
    surface = get_vessel_surface(
        centerline,
        tangents,
        [None],
        count,
        configs.projection_circle_points,
        radius,
        is_main_branch=True,
        num_stenoses=1,
        constant_radius=configs.constant_radius,
        stenosis_severity=configs.stenosis_severity,
        stenosis_position=configs.stenosis_position,
        stenosis_length=configs.stenosis_length,
        stenosis_type="gaussian",
        return_surface=True,
    )
    return centerline, surface


def _save_visualization(surface, index, output):
    if not configs.save_visualization or index >= 10:
        return
    x, y, z = surface[:3]
    coordinates = np.stack((x.T, y.T, z.T)).T
    fig = plt.figure(figsize=(2, 2), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=20.0, azim=-70)
    ax.plot_surface(
        coordinates[:, :, 0],
        coordinates[:, :, 1],
        coordinates[:, :, 2],
        alpha=0.5,
        color="blue",
    )
    set_axes_equal(ax)
    plt.axis("off")
    plt.savefig(output / f"{index:05d}_3Dsurface", bbox_inches="tight")
    plt.close(fig)


def _save_stenosis(
    index,
    geometry,
    lesion,
    point_lengths,
    images,
    center,
    theta,
    phi,
    labels_dir,
    images_dir,
):
    """Export the local geometry and the two fixed-width projected ROI images."""
    lesion_index = int(np.argmax(lesion["stenosis_severity"]))
    position = int(lesion["stenosis_position"][lesion_index])
    length = int(point_lengths[lesion_index] / configs.centerline_supersampling)
    half = int(np.ceil(0.5 * length))
    half = np.clip(half, 5, len(geometry) // 4)
    start = max(0, position - half)
    end = min(len(geometry), position + half)
    raw = geometry[start:end, :]
    centerline, radius = raw[:, :3], raw[:, 3]
    roi_images = stenosis_cut(
        full_img=images,
        C_stenosis=centerline,
        coords_center=center,
        theta_array=theta,
        phi_array=phi,
        img_dim=configs.img_dim,
        ImagerPixelSpacing=configs.ImagerPixelSpacing,
        SID=configs.SID,
        SOD=configs.SOD,
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    for view, img in zip("ab", roi_images):
        mask = (img > 0).astype(np.uint8)
        cropped = crop_around_mask(img.astype(float), mask, configs.crop_size)
        plt.imsave(
            images_dir / f"image{index:05d}{view}_stenosis_centercrop.png",
            cropped,
            cmap="gray",
        )
        mask = (cropped > 0).astype(np.uint8)
        distance = cv2.distanceTransform(
            mask * 255, cv2.DIST_L2, configs.distance_transform_mask_size
        )
        normalized = (
            (distance / distance.max()) * 255 if distance.max() > 0 else distance
        )
        cv2.imwrite(
            str(images_dir / f"image{index:05d}{view}_stenosis.png"),
            normalized.astype(np.uint8),
        )
    np.save(labels_dir / f"{index:05d}_stenosis_raw", raw)
    centered = centerline - np.mean(centerline, axis=0)
    np.save(
        labels_dir / f"{index:05d}_stenosis",
        np.concatenate((centered, radius[:, None]), axis=1),
    )


def _generate_case(index, rng, output):
    flat_layout = configs.output_layout == "dataset"
    case_output = output / configs.split / f"{index:05d}" if flat_layout else output
    labels_dir = case_output if flat_layout else output / "labels"
    images_dir = case_output if flat_layout else output / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    try:
        centerline, surface = _sample_vessel(rng)
    except ValueError as error:
        raise ValueError(f"Invalid sampling for case {index:05d}: {error}") from error
    x, y, z, radius, severity, positions, lengths = surface
    supersampling = configs.centerline_supersampling
    geometry = np.concatenate((centerline, np.expand_dims(radius, axis=-1)), axis=1)[
        ::supersampling, :
    ]
    coordinates = np.ascontiguousarray(
        np.stack((x.flatten(), y.flatten(), z.flatten())).T
    )
    center = np.mean(coordinates, axis=0)
    images, theta, phi = generate_projection_images(
        coordinates - center,
        index,
        configs.num_projections,
        configs.img_dim,
        str(case_output),
        configs.ImagerPixelSpacing,
        configs.SID,
        RCA=True,
        prefix="",
        flat_layout=flat_layout,
    )
    lesion = {
        "num_stenoses": 1,
        "stenosis_severity": [float(value) for value in severity],
        "stenosis_position": [int(value / supersampling) for value in positions],
        "num_stenosis_points": [int(value / supersampling) for value in lengths],
        "inlet_radius_mm": float(radius[0] * 1000),
        "outlet_radius_mm": float(radius[-1] * 1000),
    }
    metadata = {
        "spline_index": index,
        "tree_type": ["RCA"],
        "num_centerline_points": configs.num_centerline_points,
        "theta_array": [float(value) for value in theta.tolist()],
        "phi_array": [float(value) for value in phi.tolist()],
        "main_vessel": lesion,
        "ImagerPixelSpacing": configs.ImagerPixelSpacing,
        "SID": configs.SID,
    }
    np.save(labels_dir / f"{index:05d}", np.array([geometry]))
    info_dir = output / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / f"{index:05d}.info.0").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _save_stenosis(
        index,
        geometry,
        lesion,
        lengths,
        images,
        center,
        theta,
        phi,
        labels_dir,
        images_dir,
    )
    _save_visualization(surface, index, output)


def generate(seed: int) -> None:
    """Write the validated single-RCA protocol without changing seeded draw order."""
    configs.validate()
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    output = Path(configs.save_path)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(configs.num_trees):
        _generate_case(index, rng, output)
        if (index + 1) % 10 == 0:
            print(f"Completed {index + 1}/{configs.num_trees} vessels")
