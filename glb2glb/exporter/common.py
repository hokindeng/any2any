import io
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from glb2glb.exporter.utils_internal import get_cache_directory
from glb2glb.exporter.utils_internal import mulQuat, mat2quat, quat2mat, quatvec2H, H2quatvec
from numpy.typing import NDArray
from PIL import Image

DEF_TARGET_FPS = 25
DEF_TARGET_FRAME_STEP = 2
DEF_FREE_FALL_LEN = 1000

CACHE_PATH = get_cache_directory() / "glb2glb_cache/smpl"


class ExporterError(Exception):
    pass


class SkeletonError(ExporterError):
    pass


class ExporterWarning(Warning):
    pass


# ====== Quaternions for required rotations =======
# Rotate +90 degrees around the X-axis
R90X = np.array([0.7071, 0.7071, 0, 0])
# Rotate -90 degrees around the X-axis
Rm90X = np.array([0.7071, -0.7071, 0, 0])
# lambda to improve legibility of the quat mult operation
rq = lambda q, R: mulQuat(R, q)  # noqa: E731


def get_logger(name="EXPORTER", level=logging.INFO):
    """Get the logger"""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_debug_level(name="EXPORTER"):  # pragma: no cover
    """Set debug level to the logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)


logger = get_logger()


def calculate_fps(times, max_var):
    """
    Calculate the fps from an array of time values.
    This function will assume fps as constant, and return an approximate value while logging
    the error when they are not.

    Args:
        - times (np.array[horizon,]): array of time values in seconds
        - max_var (float): variation tolerance (above this it's considered not constant)
    """
    try:
        if times.size > 0:
            deltas = np.roll(times, -1) - times
            delta_avg = np.mean(deltas[:-1])
            delta_std = np.std(deltas[:-1])
            rel_variation = delta_std / delta_avg
            fps = int(1 / delta_avg)

            if rel_variation > max_var:
                logger.error(
                    "Variable frame rate found!. Time deltas=[mean: {:.3f}, std: {:.3f}, variation: {:.3f}] Using fps={:d}".format(
                        delta_avg, delta_std, rel_variation, fps
                    )
                )

            return fps
    except Exception:
        logger.error(
            "There was a problem calculating the source fps, it might be caused by a corrupted `time` array in the animation source (normally H5 file)"
        )
        pass
    return 0


def calc_anim_skip(anim_qpos, source_fps, target_fps, target_frame_step, logger=None):
    """
    Calculate the number of frames to skip when extracting animation data from the source.
    This function helps optimize the extraction process by determining the appropriate frame
    extraction rate based on the target (real) frame rate.

    Real frame rate == target_fps / target_frame_step. For example:
    - target_fps = 30
    - target_frame_step = 2
    We only need to extract frames at source at 15 fps

    Args:
        anim_qpos (list): The list of animation keypoints.
        source_fps (float): The frame rate of the source animation.
        target_fps (float): The desired frame rate for the target animation.
        target_frame_step (int): The step size for frames in the target animation.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        int: The number of frames to skip in the source animation.
    """

    if target_fps > 120 or target_fps < 1:
        raise ValueError("Unsupported target_fps value, should be in range [1,120]")

    if target_frame_step > target_fps / 2 or target_frame_step < 1:
        raise ValueError(
            "Unsupported target_frame_step value, should be in range [1, target_fps/2]"
        )

    if source_fps <= target_fps:
        return 1

    target_frame_step = int(target_frame_step)

    keypoints = len(anim_qpos)
    duration = keypoints / source_fps

    skip_source = int((source_fps * target_frame_step) / target_fps)
    new_keypoints = int(keypoints / skip_source)
    new_duration = (new_keypoints * target_frame_step) / target_fps

    if logger:
        logger.info(
            "Source animation data: {} keypoints @ {:.0f}fps = {:.2f}seconds".format(
                keypoints, source_fps, duration
            )
        )
        logger.info(
            "Resampled animation data: {} keypoints @ {:.0f}fps (skip={:d}) = {:.2f}seconds".format(
                new_keypoints, target_fps, target_frame_step, new_duration
            )
        )

    return skip_source


def snap_to_ground(vertices: NDArray[np.float32]):  # pragma: no cover
    """
    Offset vertices to ensure minimum Y value is 0

    Args:
        vertices: vertex positions as float32 array (N x 3)

    Return:
        vertex positions as float32 array (N x 3)
    """
    min_y = np.min(vertices[:, 1])
    vert_c = vertices.copy()  # Make a copy of the output vertices array
    vert_c[:, 1] -= min_y  # Add offset to Y coordinates
    logging.debug("Offset Y coordinates by %f to snap scaled twin to ground", min_y)
    return vert_c


def create_unique_vertices_with_uvs(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int16],
    uvs: NDArray[np.float32],
    weights: Optional[NDArray[np.float32]] = None,
    no_cache: bool = False,
    verify_output: bool = True,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.int16],
    NDArray[np.float32],
    Optional[NDArray[np.float32]],
]:
    """
    Create new arrays where each vertex has exactly one pair of UV coordinates.
    This may duplicate vertices that share position but have different UV coordinates.

    Args:
        vertices (NDArray[np.float32]): Original vertex positions (N x 3)
        faces (NDArray[np.int16]): Original face indices with format [v1_idx, uv1_idx, n1_idx, v2_idx, uv2_idx, n2_idx, v3_idx, uv3_idx, n3_idx]
        uvs (NDArray[np.float32]): Original UV coordinates (P x 2)
        weights (Optional[NDArray[np.float32]]): Optional weights array (N x 4)
        no_cache (bool, optional): If True, disable caching. Defaults to False.
        verify_output (bool, optional): If True, verify output correctness. Defaults to True.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.int16], NDArray[np.float32], Optional[NDArray[np.float32]]]:
            - new_vertices: Array of unique vertex positions (M x 3)
            - new_faces: Updated face indices (K x 9) with format [v1_idx, v1_idx, n1_idx, v2_idx, v2_idx, n2_idx, v3_idx, v3_idx, n3_idx]
            - new_uvs: UV coordinates (M x 2) matching 1:1 with new vertices
            - new_weights: Rearranged weights array (M x 4) if provided
    """
    # Create a mapping of (vertex_idx, uv_idx) -> new_vertex_idx
    vert_uv_to_new_idx = {}
    new_vertices = []
    new_weights = [] if weights is not None else None

    # Create compact shape string for cache filename
    cache_filename = (
        f"cache_unique_v{vertices.shape[0]}_f{faces.shape[0]}_uv{uvs.shape[0]}.npz"
    )
    cache_filename = Path(CACHE_PATH) / cache_filename
    # Check if cache exists for faces and UVs
    is_cached = Path(cache_filename).exists() and not no_cache

    # If using cache, disable verify_output
    verify_output &= ~is_cached

    if is_cached:
        # Load cached data for faces and UVs only
        cached = np.load(cache_filename)
        nf, nu = cached["faces"], cached["uvs"]
    else:
        new_uvs = []
        new_faces = []

    # Process each face
    for face in faces:
        new_face = []
        # Process vertex-UV pairs (v1,uv1), (v2,uv2), (v3,uv3)
        for i in range(0, 9, 3):  # 9 is the total elements per face
            vert_idx = face[i]  # Vertex index
            uv_idx = face[i + 1]  # UV index
            normal_idx = face[i + 2]  # Normal index

            key = (vert_idx, uv_idx)
            if key not in vert_uv_to_new_idx:
                # Create new vertex with its UV
                vert_uv_to_new_idx[key] = len(new_vertices)
                new_vertices.append(vertices[vert_idx])
                if not is_cached:
                    new_uvs.append(uvs[uv_idx])
                if weights is not None:
                    new_weights.append(weights[vert_idx])

            # Add vertex index, UV index and normal index to new face
            if not is_cached:
                new_face.extend(
                    [vert_uv_to_new_idx[key], vert_uv_to_new_idx[key], normal_idx]
                )

        # Verify that the new face points to correct vertex/UV pairs
        if verify_output:
            for i in range(3):  # For each vertex in triangle
                orig_vert_idx = face[i * 3]
                orig_uv_idx = face[i * 3 + 1]
                new_idx = new_face[i * 3]  # Get only vertex index from new face

                # Check vertex position matches
                assert np.allclose(
                    vertices[orig_vert_idx], new_vertices[new_idx]
                ), f"Vertex mismatch at face index {i}"

                # Check UV coordinates match
                assert np.allclose(
                    uvs[orig_uv_idx], new_uvs[new_idx]
                ), f"UV mismatch at face index {i}"

        if not is_cached:
            new_faces.append(new_face)

    nv = np.array(new_vertices, dtype=np.float32)

    if not is_cached:
        nf = np.array(new_faces, dtype=np.int16)
        nu = np.array(new_uvs, dtype=np.float32)
        np.savez(cache_filename, faces=nf, uvs=nu)

    nw = np.array(new_weights, dtype=np.float32) if new_weights is not None else None

    return nv, nf, nu, nw


def save_texture(texture_data: str | Path | bytes, output_path: str | Path):
    """
    Save texture data as a PNG file.

    Args:
        texture_data: Binary image data or a path to an image file
        output_path: Directory to save the image

    Returns:
        Path to the saved image file
    """
    if not texture_data:
        return None

    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Handle texture_data based on its type
    if isinstance(texture_data, (str, Path)):
        # If texture_data is a path, open the image from that path
        img = Image.open(texture_data)
    else:
        # Otherwise treat it as binary data
        img = Image.open(io.BytesIO(texture_data))

    # Save the image as PNG regardless of original format
    img.save(output_path, format="PNG")

    return output_path
