"""
Repose utilities for mesh manipulation.

This module contains utilities for repositioning and reshaping meshes
based on skeleton transformations.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Union, Iterable

import numpy as np
import trimesh
from numpy.typing import NDArray
from PIL import Image

from glb2glb.exporter.common import ExporterError

try:
    import bpy

    from glb2glb.exporter.blender import Mjc2Blend, update_mesh_get_verts
except (ModuleNotFoundError, ImportError):
    Mjc2Blend = None

# This version gets included in the cache key.
# Changing it invalidates all previous cache entries
_VERSION = 1
NFS_MORPH_DIR = Path.home() / ".cache/glb2glb/morphable"
NFS_MORPH_XML = NFS_MORPH_DIR / "morphable_model_v2.xml"
NFS_MORPH_GLTF = NFS_MORPH_DIR.parent / "morphable_model_v2.glb"

CACHE_MORPH = Path.home() / ".cache/glb2glb/morphable/.morphable_cache"

# fmt: off
PARAMETRIC_MALE_QPOS_REST = [
    0, 0, 0.95, 1, 0, 0, 0, 0, 0, 0, -0.015424, 0, 0, 0.00265795, 0, 0,
    0.00265795, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07566, 0, 0, -0.269, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0.14842, 0.67553, 0.12206, 0.0416, 0.2618,
    0.01571, -0.003476, 0.188496, 0.2616, -0.2472, 0, 0, 0, 0.002618,
    0, 0, 0.054985, -0.020944, 0.14139, 0.1571, 0.1571, -0.09163,
    0.17281, 0.14139, 0.070695, -0.204204, 0.054985, 0.17281, -0.269,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.14842, 0.64411, 0.12206, 0.0416,
    0.260935, -0.00026, -0.007854, 0.23562, 0.3192, -0.3286, -0.085888,
    0, -0.233925, 0.012859, 0.30363, 0.195366, 0.007032, -0.057234,
    0.14658, 0.0801905, 0.165872, -0.046252, 0.18846, 0.048779,
    0.0925615, -0.156543, 0.45021, -0.3491, 0, 0, 0, 0, 0, -0.104784,
    -0.006994, 0.034905, 0, 0, 0, 0.03354, 0, 0.108222, -0.122185,
    0.07854, -0.0179, 0.05, 0, -0.104784, -0.006994, 0.034905, 0, 0,
    0, 0.03354, 0, 0.0410285, -0.122185, 0.07854, -0.01083, 0.05242,
    0.01051
]

PARAMETRIC_FEMALE_QPOS_REST = [
    0.0, 0.0, 0.976, 1.0, 0.0, 0.0, 0.0, -0.0985, 0.0, 0.0, -0.0789365, 0.0, 0.0,
    -0.003, 0.0, 0.0, 0.06149, 0.0, 0.0, 0.073, 0.0, 0.0, 0.0398, 0.0, 0.0, 0.036,
    0.0, 0.0, 0.0150594, 0.0, 0.0, -0.0682, 0.0, 0.0, -0.0465, 0.0, -0.0002562,
    -0.1023, 0.0, 0.0, -0.02945, 0.0, 0.0, 0.13795, 0.0, 0.0, 0.0992, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.141225, 0.01771, -0.01856, 0.0, -0.0018, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.106835, 0.84834, 0.57962, 0.1149, 0.47124, 0.26707, 0.173656, 0.062832,
    0.204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, -0.141225, 0.01771, -0.01856, 0.0, -0.0018, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.106835, 0.84834, 0.57962, 0.1149, 0.47124, 0.0, 0.0, 0.062832,
    0.204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, -8e-05, -0.0419015, 0.0, 0.0,
    0.0, 0.0, -0.001678, 0.0, 0.114331, 0.0, 0.0, -0.0179, 0.05, 0.0, -8e-05,
    -0.0419015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.114331, 0.0, 0.0, -0.0179, 0.05, 0.0
]

# fmt: on

# Negate horizontally
DEF_PARAMETRIC_SCALE = [1, 1, -1]
# Remove a horizontal offset that is present in the original file
DEF_PARAMETRIC_OFFSET = [0, 0, 0]


def load_and_offset_vertices(
    input_path: Path,
    ob_name: str,
    init_qpos: Optional[list],
    shape_param: float,
    mult_verts: list[float] = None,
    add_verts: list[float] = None,
) -> Union[NDArray[np.float32], None]:
    """
    Load vertices from a GLB/GLTF file and apply morph shape offsets
    using MuJoCo's parametric shape mechanism.

    This method loads a 3D model from a GLTF file, locates the specified object,
    and applies offsets to the vertices based on morph targets defined in the model.
    The shape coefficient determines the scaling of these offsets.

    Args:
        input_path (Path): Path to the input GLTF/GLB file containing the mesh.
        ob_name (str): Name of the object/mesh within the file to be modified.
        init_qpos (Optional[list]): Initial joint positions for skeletal deformation (currently not used).
        shape_param (float): A coefficient that determines the scaling of the morph target offsets.
                             Positive values use one set of morph targets, negative use another.
                             Range is typically [-1, 1], where 0 means no morphing.
        mult_verts (list[float], optional): Multiplicative scaling factors for vertices [x, y, z].
                                           Applied after morphing. Defaults to None (no scaling).
        add_verts (list[float], optional): Additive offset for vertices [x, y, z].
                                          Applied after scaling. Defaults to None (no offset).

    Returns:
        Union[NDArray[np.float32], None]: The modified vertex array with shape (n_vertices, 3)
                                          containing the transformed positions, or None if the
                                          operation fails.

    Raises:
        ExporterError: If the specified object is not found in the file or if there are
                      issues with loading the model.

    Notes:
        - The function caches results to improve performance for repeated calls with the same parameters
        - Morph targets are expected to be in the format "_B" (for positive) and "_S" (for negative)
        - The function uses trimesh for loading GLTF files and extracting vertex data
    """

    v, of, os, _ = get_morphable_mesh_reposed(
        model_path=NFS_MORPH_XML,
        target_file=NFS_MORPH_GLTF,
        i_qpos=init_qpos,
        cache_dir=CACHE_MORPH,
        mult_verts=mult_verts,
        add_verts=add_verts,
        ob_name=ob_name,
    )

    if shape_param > 0:
        loaded_offset = of * shape_param
    elif shape_param < 0:
        loaded_offset = os * shape_param * -1
    else:
        loaded_offset = [0, 0, 0]

    return v + loaded_offset


def create_mods_mark(
    i_qpos: int | Iterable, mult_verts: list = [1, 1, 1], add_verts: list = [0, 0, 0]
):
    if isinstance(i_qpos, Iterable):
        i_qpos_sum = sum(i_qpos)
    else:
        i_qpos_sum = 0

    mult_verts_str = "".join(f"{v*100:03.0f}" for v in mult_verts)
    add_verts_str = "".join(f"{v*100:03.0f}" for v in add_verts)

    unique_id = (
        f"{mult_verts_str}_{add_verts_str}_{i_qpos_sum*10000:05.0f}_v{_VERSION}"
    )
    return unique_id


def get_morphable_mesh_reposed(
    model_path: Path | str,
    target_file: Path | str,
    i_qpos: int | Iterable,
    cache_dir: Path | str,
    mult_verts: list = [1, 1, 1],
    add_verts: list = [0, 0, 0],
    ob_name: str = None,
    save_debug_file: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str, str]]:
    """
    Repose a skinned mesh from a GLB or BLEND file to match the skeleton in the given model path.
    Extracts the vertices array and computes offsets for skinny and fat morph targets.
    This function caches the results as .npy files so multiple calls don't create unnecessary overhead.

    Args:
        model_path (Path | str): Path to the model file containing the skeleton.
        target_file (Path | str): Path to the GLB or BLEND file containing the skinned mesh.
        i_qpos (int | Iterable): The initial pose of the model.
                It can be an int, referring to one of the keyframes already present in the model,
                or an iterable of size=model.nq.
                Defaults to -1, which skips keeps the initial qpos.
        cache_dir (Path | str | None): Directory where the output files will be saved. If None, caching is disabled.
        mult_verts (list, optional): A list of multipliers for scaling the vertices along each axis.
        add_verts (list, optional): A list of values to add to the vertices along each axis.
        ob_name (str, optional): Name of the object to process in Blender. Defaults to None (auto detect).
        save_debug_file (bool, optional): If True, saves a debug BLEND file. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - v (np.ndarray): Vertices array.
            - of (np.ndarray): Offsets array for the fat morph target.
            - os (np.ndarray): Offsets array for the skinny morph target.
            - paths (tuple): Tuple of file paths where the arrays are saved.
    """
    model_path = Path(model_path)
    target_file = Path(target_file)
    cache_dir = Path(cache_dir) if cache_dir else None

    if cache_dir:
        # Construct base filename for output files
        base_npy_filename = cache_dir / target_file.stem

        # Define paths for saving vertices and offsets
        mark = create_mods_mark(i_qpos, mult_verts, add_verts)
        vertices_path = f"{base_npy_filename}_{ob_name}_{mark}_v_.npy"
        offsets_fat_path = f"{base_npy_filename}_{ob_name}_{mark}_o_fat.npy"
        offsets_skinny_path = f"{base_npy_filename}_{ob_name}_{mark}_o_skinny.npy"

        cache_found = False
        try:
            # Attempt to load precomputed vertices and offsets from files
            v = np.load(vertices_path)
            of = np.load(offsets_fat_path)
            os = np.load(offsets_skinny_path)
            cache_found = True
            print(f"Found morphable mesh cache at: {base_npy_filename}")

        except (FileNotFoundError, ValueError):
            print(
                f"Morphable mesh cache not found at: {base_npy_filename}, processing from {target_file}"
            )
    else:
        vertices_path = offsets_fat_path = offsets_skinny_path = None
        cache_found = False

    if not cache_found:
        if not Mjc2Blend:
            raise ExporterError(
                "Blender Python module not found. Please install Blender as Python module"
            )
        # If files are not found or cannot be loaded, process the mesh and save new data
        m2b = Mjc2Blend(model_path, clean_scene=True)

        # Define the NPOSE keyframe for the skeleton
        N_POSE = [np.array([0, 0, 0.95, 1, 0, 0, 0] + [0] * (161 - 7))]
        assert N_POSE[0].shape == (161,)
        m2b.apply_animation_data(N_POSE)

        # Set the target armature and apply the NPOSE
        m2b.set_target_armature(target_file, ob_name, i_qpos=i_qpos)
        m2b.set_animation_keyframes()

        # Optionally save a debug BLEND file
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if save_debug_file:
                blend_file_path = (
                    cache_dir / f"export_{target_file.stem}.blend"
                    if isinstance(save_debug_file, bool)
                    else save_debug_file
                )
                bpy.ops.wm.save_as_mainfile(filepath=str(blend_file_path))
                print(f">>> Debug blend file saved to: {blend_file_path}")

        if ob_name:
            if ob_name not in bpy.data.objects:
                raise ExporterError(f"Object '{ob_name}' not found in file")

            mesh = bpy.data.objects[ob_name]
        else:
            mesh = next(
                obj
                for obj in bpy.data.objects
                if obj.type == "MESH" and obj.parent and obj.parent.type == "ARMATURE"
            )

        # Process and save vertices and offsets for different morph targets
        v = update_mesh_get_verts(mesh, 0, 0, mult_verts, add_verts)
        of = update_mesh_get_verts(mesh, 0, 1, mult_verts, add_verts, offset_from=v)
        os = update_mesh_get_verts(mesh, 1, 0, mult_verts, add_verts, offset_from=v)

        if cache_dir:
            np.save(vertices_path, v)
            print(f"==== Saved cache: {vertices_path}")
            np.save(offsets_fat_path, of)
            print(f"==== Saved cache: {offsets_fat_path}")
            np.save(offsets_skinny_path, os)
            print(f"==== Saved cache: {offsets_skinny_path}")

    return v, of, os, (vertices_path, offsets_fat_path, offsets_skinny_path)
