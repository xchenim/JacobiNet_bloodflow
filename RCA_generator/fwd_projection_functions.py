import numpy as np
import random
from skimage import morphology as morph
from skimage import filters
import matplotlib.pyplot as plt
import os

if __package__:
    from . import configs
else:
    import configs

SOD = configs.SOD


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def ray_image_intersection(voxel, V_source, localX, localY, image_point):
    """Intersect source rays with the detector plane, omitting parallel rays."""
    eps = 1e-6
    n = np.expand_dims(np.cross(localX, localY), axis=0)
    u = V_source - voxel
    proj_u = np.einsum(
        "ij,ij->i", n, u
    )  # multiplies n_ij*u_ij and sums along j axis = row-wise dot product

    valid_projection = np.abs(proj_u) > eps
    valid_voxels = voxel[valid_projection, :]
    valid_u = u[valid_projection, :]

    w = valid_voxels - image_point
    scaling = np.expand_dims(
        np.divide(-np.einsum("ij,ij->i", n, w), proj_u[valid_projection]), axis=-1
    )
    scaled_u = np.multiply(scaling, valid_u)
    points = valid_voxels + scaled_u
    return points


def get_local_params(
    theta_array, phi_array, numImg, distanceDetectortoISO, ISO, coord_system_change=True
):
    theta = theta_array / 180 * np.pi
    phi = phi_array / 180 * np.pi
    if coord_system_change == True:

        coordSystemChanger = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    else:
        coordSystemChanger = np.eye(3)
    Rotation_AP1 = np.zeros((3, 3, numImg))
    Rotation_AP2 = np.zeros((3, 3, numImg))
    Rotation_AP3 = np.zeros((3, 3, numImg))

    for jj in range(numImg):
        AP1 = theta[jj]
        # z rotation converts to -x
        Rotation_AP1[:, :, jj] = (
            coordSystemChanger
            @ np.array(
                [
                    [np.cos(AP1), -np.sin(AP1), 0],
                    [np.sin(AP1), np.cos(AP1), 0],
                    [0, 0, 1],
                ]
            )
            @ np.linalg.inv(coordSystemChanger)
        )

        AP2 = phi[jj]  # x rotation converts to y
        Rotation_AP2[:, :, jj] = (
            coordSystemChanger
            @ np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(AP2), np.sin(AP2)],
                    [0, -np.sin(AP2), np.cos(AP2)],
                ]
            )
            @ np.linalg.inv(coordSystemChanger)
        )

        AP3 = 0  # y rotation converts to -z
        Rotation_AP3[:, :, jj] = (
            coordSystemChanger
            @ np.array(
                [
                    [np.cos(AP3), 0, np.sin(AP3)],
                    [0, 1, 0],
                    [-np.sin(AP3), 0, np.cos(AP3)],
                ]
            )
            @ np.linalg.inv(coordSystemChanger)
        )

    V_sensor = np.zeros((3, numImg))
    V_source = np.zeros((3, numImg))
    localX = np.zeros((3, numImg))
    localY = np.zeros((3, numImg))

    for jj in range(numImg):
        radiusOfImagingSphere = distanceDetectortoISO[jj]

        V_sensor[:, jj] = np.squeeze(
            Rotation_AP1[:, :, jj]
            @ Rotation_AP2[:, :, jj]
            @ Rotation_AP3[:, :, jj]
            @ (np.array([[0], [0], [1]]) * radiusOfImagingSphere)
        )
        V_source[:, jj] = -V_sensor[:, jj] / radiusOfImagingSphere * ISO

        localX[:, jj] = np.squeeze(
            Rotation_AP1[:, :, jj]
            @ Rotation_AP2[:, :, jj]
            @ Rotation_AP3[:, :, jj]
            @ np.array([[0], [1], [0]])
        )
        localY[:, jj] = np.squeeze(
            Rotation_AP1[:, :, jj]
            @ Rotation_AP2[:, :, jj]
            @ Rotation_AP3[:, :, jj]
            @ np.array([[-1], [0], [0]])
        )

    localX = localX.T
    localY = localY.T
    V_sensor = V_sensor.T
    V_source = V_source.T
    return V_sensor, V_source, localX, localY


def rotate_volume(alpha, beta, gamma, volume_coords):

    AP1 = alpha / 180 * np.pi
    Rotation_AP1 = np.array(
        [[1, 0, 0], [0, np.cos(AP1), -np.sin(AP1)], [0, np.sin(AP1), np.cos(AP1)]]
    )

    AP2 = beta / 180 * np.pi
    Rotation_AP2 = np.array(
        [[np.cos(AP2), 0, np.sin(AP2)], [0, 1, 0], [-np.sin(AP2), 0, np.cos(AP2)]]
    )

    AP3 = gamma / 180 * np.pi
    Rotation_AP3 = np.array(
        [[np.cos(AP3), -np.sin(AP3), 0], [np.sin(AP3), np.cos(AP3), 0], [0, 0, 1]]
    )

    rotation_matrix = np.squeeze(Rotation_AP1 @ Rotation_AP2 @ Rotation_AP3)

    rotated_volume = np.dot(volume_coords, rotation_matrix.T)
    return rotated_volume


def convert3D_to_pixels(
    projected_points, plane_index, img_dim, V_sensor, sensorWidth, localX, localY
):
    i = plane_index
    X = np.expand_dims(localX[i, :], axis=0)
    Y = np.expand_dims(localY[i, :], axis=0)

    # Preserve the detector's one-based pixel-center convention.
    local_origin = V_sensor[i, :] + sensorWidth * (
        (1 - img_dim / 2) / img_dim * X + (1 - img_dim / 2) / img_dim * Y
    )
    local_Xmax = V_sensor[i, :] + sensorWidth * (
        (img_dim - img_dim / 2) / img_dim * X + (1 - img_dim / 2) / img_dim * Y
    )
    local_Ymax = V_sensor[i, :] + sensorWidth * (
        (1 - img_dim / 2) / img_dim * X + (img_dim - img_dim / 2) / img_dim * Y
    )

    # vector defining the ray intersection point (point3D) on the image plane
    v = projected_points - local_origin
    vx_projected = np.multiply(
        np.expand_dims(
            np.einsum("ij,ij->i", v, X) / np.einsum("ij,ij->i", X, X), axis=-1
        ),
        X,
    )
    vy_projected = np.multiply(
        np.expand_dims(
            np.einsum("ij,ij->i", v, Y) / np.einsum("ij,ij->i", Y, Y), axis=-1
        ),
        Y,
    )

    a = np.sign(np.einsum("ij,ij->i", local_Xmax - local_origin, vx_projected))
    b = np.sign(np.einsum("ij,ij->i", local_Ymax - local_origin, vy_projected))

    x = (
        a
        * img_dim
        * np.linalg.norm(vx_projected, axis=1)
        / np.linalg.norm(local_Xmax - local_origin)
    )
    y = (
        b
        * img_dim
        * np.linalg.norm(vy_projected, axis=1)
        / np.linalg.norm(local_Ymax - local_origin)
    )
    y = img_dim - y
    return np.array([x, y]).T  # reverse coords to get value of matrix at [row, col]


def convert_projection_to_standard_angles(projection_angles):
    """
    projection_angles: A list of strings; each string must have the form "LAO/RAO X, CRA/CAU Y".
    For example, ['LAO 35, CAU 10', 'RAO 20, CRA 0']

    Projection angles use an object coordinate frame with the xz plane horizontal.
    Standard angles refer to azimuthal and elevation angle representation of spherical coordinates.
    """
    projection_key = {"RAO": 1, "LAO": -1, "CRA": 1, "CAU": -1}
    theta_array = []
    phi_array = []
    for angle_pair in projection_angles:
        theta_string = angle_pair.split(",")[0]
        phi_string = angle_pair.split(",")[1]

        theta_projection = (
            float(theta_string.split()[1]) * projection_key[theta_string.split()[0]]
        )
        phi_projection = (
            float(phi_string.split()[1]) * projection_key[phi_string.split()[0]]
        )

        theta = theta_projection - 90
        phi = phi_projection

        theta_array.append(theta)
        phi_array.append(phi)

    return theta_array, phi_array


def generate_projection_images(
    surface_coords,
    spline_index,
    num_projections,
    image_dim,
    save_path,
    imagerpxspacing,
    sid,
    RCA=False,
    prefix="",
    flat_layout=False,
):
    """Rasterize the two seeded RCA views; lengths are metres and pixel spacing is mm/pixel."""
    image_list = []
    SID = np.ones(num_projections) * sid
    OID = SID - SOD

    img_dim = image_dim
    ImagerPixelSpacing = imagerpxspacing  # conversion from px to mm
    sensorWidth = ImagerPixelSpacing * img_dim / 1000

    spatial_coords_3D = surface_coords
    alpha, beta, gamma = 0, 0, 0
    rotated_spatial_coords_3D = rotate_volume(alpha, beta, gamma, spatial_coords_3D)

    numImg = num_projections

    if not RCA or num_projections != 2:
        raise ValueError("Projection generation requires the two-view RCA protocol")
    # Reserve the six uniform draws in the published seeded sampling sequence.
    for lower, upper in (
        (-75, -55),
        (-40, -20),
        (30, 60),
        (45, 65),
        (15, 35),
        (80, 110),
    ):
        random.uniform(lower, upper)

    # two canonical RCA views

    def u(mean, delta):
        return random.uniform(mean - delta, mean + delta)

    projection_angles = [
        f"RAO {u(configs.view1_rao_mean, configs.view1_rao_delta):.2f}, CRA {u(configs.view1_cra_mean, configs.view1_cra_delta):.2f}",
        f"RAO {u(configs.view2_rao_mean, configs.view2_rao_delta):.2f}, CAU {u(configs.view2_cau_mean, configs.view2_cau_delta):.2f}",
    ]

    theta_list, phi_list = convert_projection_to_standard_angles(projection_angles)

    theta_array = np.array(
        [
            theta_list[0],
            theta_list[1],
        ]
    )[:num_projections]

    phi_array = -(
        np.array(
            [
                phi_list[0],
                phi_list[1],
            ]
        )
        + 90
    )[:num_projections]

    V_sensor, V_source, localX, localY = get_local_params(
        theta_array, phi_array, numImg, OID, SOD, coord_system_change=True
    )

    points_on_plane = []
    for i in range(numImg):
        current_plane_points = ray_image_intersection(
            rotated_spatial_coords_3D, V_source[i], localX[i], localY[i], V_sensor[i]
        )
        points_on_plane.append(current_plane_points)

    suffixes = ["a", "b", "c", "d"]
    for plane_index in range(numImg):
        plane_points_in_px = np.round(
            convert3D_to_pixels(
                points_on_plane[plane_index],
                plane_index,
                img_dim,
                V_sensor,
                sensorWidth,
                localX,
                localY,
            )
        )
        ind_lower_cutoff = np.all(plane_points_in_px > 0, axis=1)
        ind_upper_cutoff = np.all(plane_points_in_px < img_dim, axis=1)
        cutoff_array = np.stack((ind_lower_cutoff, ind_upper_cutoff), axis=1)
        valid_point_inds = np.all(cutoff_array, axis=1)

        if not valid_point_inds.any():
            raise ValueError(
                f"No detector pixels for case {spline_index:05d}, view {plane_index}"
            )
        valid_points_in_px = plane_points_in_px[valid_point_inds, :].astype("int")
        # img_dim - y to perform 2D image to 3D map transform
        # coords in X,Y but indexing in row,col; origin in top left instead of bottom left

        try:
            valid_points_unraveled = np.ravel_multi_index(
                (img_dim - valid_points_in_px[:, 1], valid_points_in_px[:, 0]),
                (img_dim, img_dim),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid detector indices for case {spline_index:05d}, view {plane_index}"
            ) from error
        binary_image_unraveled = np.zeros((img_dim * img_dim))
        binary_image_unraveled[valid_points_unraveled] = 1
        binary_image = binary_image_unraveled.reshape((img_dim, img_dim)) > 0
        closed_binary_image = morph.binary_closing(
            binary_image, morph.disk(configs.projection_closing_radius_px)
        )
        blurred_binary_image = (
            filters.gaussian(
                closed_binary_image, sigma=configs.projection_blur_sigma_px
            )
            > configs.projection_blur_threshold
        )

        if save_path is not None:
            image_dir = save_path if flat_layout else os.path.join(save_path, "images")
            os.makedirs(image_dir, exist_ok=True)
            width = 5 if flat_layout else 4
            filename = (
                f"image{spline_index:0{width}d}{suffixes[plane_index]}{prefix}.png"
            )
            plt.imsave(
                os.path.join(image_dir, filename), blurred_binary_image, cmap="gray"
            )
        image_list.append(blurred_binary_image)

    return image_list, theta_array, phi_array
