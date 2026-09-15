"""RCA configuration; lengths are metres unless a field specifies mm or pixels."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
save_path = str(PROJECT_DIR.parent / "outputs" / "generated_rca")
num_trees = 1
save_visualization = False
output_layout, split = "dataset", "train"

vessel_type, num_branches = "RCA", 0
control_point_path = str(PROJECT_DIR / "RCA_branch_control_points" / "moderate")
num_centerline_points, centerline_supersampling = 140, 10
projection_circle_points = 120
L_RCA_min, L_RCA_max = 0.130, 0.150
D_RCA_min, D_RCA_max = 0.004, 0.005
shear, warp = True, True
shear_strength, warp_strength = 0.12, 0.1
# Control points: uniform sampling from mean-1.5*std to mean+2.5*std.
control_std_lower, control_std_upper = 1.5, 2.5
main_taper_min, main_taper_max = 0.3, 0.4
radius_noise_std_m = 0.00001
constant_radius = False

max_num_stenosis, num_stenoses = 1, None
stenosis_position = None  # one-element list: index on supersampled curve (1,400)
stenosis_severity = None  # one-element list: local fractional diameter reduction
stenosis_length = None  # one-element list: even supersampled point count
ds_min, ds_max = 0.05, 0.80  # local reduction, NOT 1-r_min/r_ROI_inlet
stenosis_len_min_mm, stenosis_len_max_mm = 10.7, 20.7
# mm-to-points conversion uses mean configured RCA length, not each realised length.
start_id, end_id = 0.20, 0.80
stenosis_profile_sigma = 0.5  # lesion shape, not radius postprocessing

generate_projections, num_projections = True, 2
img_dim, crop_size = 512, 128
ImagerPixelSpacing = 0.258390625  # detector mm/pixel
SID, SOD = 1.1, 0.765  # m
view1_rao_mean, view1_rao_delta = 40.0, 10.0
view1_cra_mean, view1_cra_delta = 0.0, 5.0
view2_rao_mean, view2_rao_delta = 0.0, 5.0
view2_cau_mean, view2_cau_delta = 25.0, 10.0
projection_closing_radius_px = 2
projection_blur_sigma_px, projection_blur_threshold = 0.5, 0.25
roi_band_radius_px = 15
distance_transform_mask_size = 5

PARAMETER_NAMES = tuple(
    name
    for name in list(globals())
    if not name.startswith("_")
    and name
    not in {
        "PROJECT_DIR",
        "Path",
        "save_path",
        "num_trees",
        "save_visualization",
        "output_layout",
        "split",
    }
)


def parameters():
    return {name: globals()[name] for name in PARAMETER_NAMES}


def validate():
    import math

    for name, value in parameters().items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if vessel_type != "RCA" or num_branches != 0:
        raise ValueError("The public dataset supports one RCA and no branches")
    if max_num_stenosis != 1 or (
        num_stenoses is not None
        and (type(num_stenoses) is not int or num_stenoses != 1)
    ):
        raise ValueError("The public dataset requires exactly one stenosis")
    if not generate_projections or num_projections != 2:
        raise ValueError("The dataset requires exactly two projections")
    for name in (
        "num_centerline_points",
        "centerline_supersampling",
        "projection_circle_points",
        "img_dim",
        "crop_size",
    ):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    for lower, upper in (
        (L_RCA_min, L_RCA_max),
        (D_RCA_min, D_RCA_max),
        (stenosis_len_min_mm, stenosis_len_max_mm),
    ):
        if not 0 < lower <= upper:
            raise ValueError("Length and diameter bounds must be positive and ordered")
    if not 0 <= ds_min <= ds_max < 1 or not 0 < start_id < end_id < 1:
        raise ValueError("Invalid local reduction or stenosis position bounds")
    if not 0 < main_taper_min <= main_taper_max <= 1:
        raise ValueError("Invalid taper ratios")
    if not 0 < SOD < SID or ImagerPixelSpacing <= 0:
        raise ValueError("Projection requires 0 < SOD < SID and positive pixel spacing")
    if crop_size > img_dim or crop_size % 2:
        raise ValueError("crop_size must be even and no larger than img_dim")
    if distance_transform_mask_size not in (3, 5):
        raise ValueError("distance_transform_mask_size must be 3 or 5")
    if stenosis_profile_sigma <= 0 or not 0 < projection_blur_threshold < 1:
        raise ValueError("Invalid lesion width or projection threshold")
    for name in (
        "shear_strength",
        "warp_strength",
        "control_std_lower",
        "control_std_upper",
        "radius_noise_std_m",
        "roi_band_radius_px",
        "projection_blur_sigma_px",
        "projection_closing_radius_px",
        "view1_rao_delta",
        "view1_cra_delta",
        "view2_rao_delta",
        "view2_cau_delta",
    ):
        if globals()[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("stenosis_position", "stenosis_severity", "stenosis_length"):
        value = globals()[name]
        if value is not None and (not isinstance(value, list) or len(value) != 1):
            raise ValueError(f"{name} must be null or a one-element list")
    if stenosis_severity is not None and not 0 <= stenosis_severity[0] < 1:
        raise ValueError("stenosis_severity must be in [0, 1)")
    n = num_centerline_points * centerline_supersampling
    if stenosis_position is not None and (
        not isinstance(stenosis_position[0], int) or not 0 < stenosis_position[0] < n
    ):
        raise ValueError("stenosis_position must index the supersampled curve")
    if stenosis_length is not None and (
        not isinstance(stenosis_length[0], int)
        or stenosis_length[0] < 6
        or stenosis_length[0] % 2
    ):
        raise ValueError("stenosis_length must contain an even point count >= 6")
    if not (Path(control_point_path) / "RCA_ctrl_points.npy").is_file():
        raise FileNotFoundError("RCA control points not found")
