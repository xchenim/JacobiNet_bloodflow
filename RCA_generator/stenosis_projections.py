"""Fixed-width projected centerline bands and paired stenosis crops."""

import numpy as np

if __package__:
    from . import configs
    from .fwd_projection_functions import (
        get_local_params,
        ray_image_intersection,
        convert3D_to_pixels,
    )
else:
    import configs
    from fwd_projection_functions import (
        get_local_params,
        ray_image_intersection,
        convert3D_to_pixels,
    )


def crop_around_mask(img, mask, crop_size=128):
    """Crop around the foreground centroid, clamping the crop to image bounds."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((crop_size, crop_size), dtype=img.dtype)

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    h, w = img.shape
    half = crop_size // 2

    y0 = np.clip(cy - half, 0, h - crop_size)
    x0 = np.clip(cx - half, 0, w - crop_size)

    return img[y0 : y0 + crop_size, x0 : x0 + crop_size]


def build_centerline_band_roi(img_shape, uv):
    """Rasterize a fixed pixel-width band, excluding two segments at each end."""

    H, W = img_shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    cl = uv.astype(float)
    seg_vec = cl[1:] - cl[:-1]
    seg_len2 = np.sum(seg_vec**2, axis=1)

    mask = np.zeros(len(pts), dtype=bool)

    for i in range(2, len(seg_vec) - 2):
        p0 = cl[i]
        v = seg_vec[i]
        l2 = seg_len2[i] + 1e-8

        t = np.sum((pts - p0) * v, axis=1) / l2
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + t[:, None] * v

        dist = np.linalg.norm(pts - proj, axis=1)

        mask |= dist <= configs.roi_band_radius_px

    return mask.reshape(H, W).astype(np.uint8)


def stenosis_cut(
    full_img,
    C_stenosis,
    coords_center,
    theta_array,
    phi_array,
    img_dim,
    ImagerPixelSpacing,
    SID,
    SOD,
):
    """Project the local centerline and mask each full view with its fixed-width band."""
    num_projections = len(theta_array)

    SID_arr = np.ones(num_projections) * SID
    OID_arr = SID_arr - SOD

    V_sensor, V_source, localX, localY = get_local_params(
        theta_array, phi_array, num_projections, OID_arr, SOD, coord_system_change=True
    )
    sensorWidth = ImagerPixelSpacing * img_dim / 1000.0

    C_proj = C_stenosis - coords_center

    stenosis_imgs = []

    for k in range(num_projections):

        pts_on_plane = ray_image_intersection(
            C_proj, V_source[k], localX[k], localY[k], V_sensor[k]
        )

        uv = convert3D_to_pixels(
            pts_on_plane, k, img_dim, V_sensor, sensorWidth, localX, localY
        )
        uv = uv.astype(int)
        uv[:, 1] = img_dim - 1 - uv[:, 1]

        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < img_dim)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < img_dim)
        )
        uv = uv[valid]

        if len(uv) < 5:
            raise ValueError(
                f"Projected stenosis has fewer than five detector points in view {k}"
            )
        roi_mask = build_centerline_band_roi(
            img_shape=full_img[k].shape,
            uv=uv,
        )

        stenosis_img = full_img[k] * roi_mask
        stenosis_imgs.append(stenosis_img)

    return stenosis_imgs
