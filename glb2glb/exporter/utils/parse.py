"""
Parsing utilities for various file formats.
"""

import base64
import logging
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pygltflib import BYTE, GLTF2, SHORT, UNSIGNED_BYTE, UNSIGNED_INT, UNSIGNED_SHORT

from glb2glb.exporter.common import create_unique_vertices_with_uvs

GLTF_2_dtype = {
    BYTE: np.int8,
    UNSIGNED_BYTE: np.uint8,
    SHORT: np.int16,
    UNSIGNED_SHORT: np.uint16,
    UNSIGNED_INT: np.uint32,
}


def parse_obj(
    obj_file: str | Path,
) -> Tuple[NDArray[np.float32], NDArray[np.int16], NDArray[np.float32]]:
    """Extract vertices, faces and UV coordinates from an OBJ file into numpy arrays.

    Args:
        obj_file: Path to the OBJ file as string or Path object

    Returns:
        A tuple containing:
            - vertices: Vertex positions as float32 array (N x 3)
            - faces: Face indices as int16 array (M x 9), each face contains 3 triplets of (vertex_idx, uv_idx, normal_idx)
            - uvs: UV coordinates as float32 array (P x 2)
    """
    vertices = []
    faces = []
    uvs = []

    # Regex patterns for parsing vertex, UV and face data
    vertex_pattern = re.compile(
        r"v\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)"
    )
    uv_pattern = re.compile(r"vt\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)")
    face_pattern = re.compile(r"f\s+(\d+/\d+/?\d*)\s+(\d+/\d+/?\d*)\s+(\d+/\d+/?\d*)")

    with open(obj_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("v "):
                # Parse vertex (x,y,z) coordinates
                match = vertex_pattern.match(line)
                if match:
                    vertices.append(
                        [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                    )

            elif line.startswith("vt "):
                # Parse UV (u,v) coordinates
                match = uv_pattern.match(line)
                if match:
                    uvs.append([float(match.group(1)), float(match.group(2))])

            elif line.startswith("f "):
                # Parse face vertex/uv/normal indices
                match = face_pattern.match(line)
                if match:
                    face = []
                    for group in match.groups():
                        # Extract vertex/uv indices from v/vt/vn format
                        if "/" in group:
                            indices = group.split("/")
                            # Convert from 1-based to 0-based indexing
                            v_idx = int(indices[0]) - 1
                            # Use UV index if present, otherwise use vertex index
                            uv_idx = (
                                int(indices[1]) - 1
                                if len(indices) > 1 and indices[1]
                                else v_idx
                            )
                            # Use normal index if present, otherwise use vertex index
                            n_idx = (
                                int(indices[2]) - 1
                                if len(indices) > 2 and indices[2]
                                else v_idx
                            )
                        else:
                            # Handle case with only vertex index
                            v_idx = int(group) - 1
                            uv_idx = v_idx  # Use vertex index for UV
                            n_idx = v_idx  # Use vertex index for normal
                        face.extend([v_idx, uv_idx, n_idx])
                    faces.append(face)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int16),
        np.array(uvs, dtype=np.float32),
    )


def save_obj(
    filepath: str | Path,
    vertices: NDArray[np.float32],
    faces: NDArray[np.int16],
    uvs: NDArray[np.float32],
) -> None:
    """Save vertices, faces and UV coordinates to an OBJ file.

    Args:
        filepath: Output OBJ file path as string or Path object
        vertices: Vertex positions as float32 array (N x 3)
        faces: Face indices as int16 array (M x 9), each face contains 3 triplets of (vertex_idx, uv_idx, normal_idx)
        uvs: UV coordinates as float32 array (P x 2)
    """
    with open(filepath, "w") as f:
        # Write vertex positions
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write UV coordinates
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Write faces with 1-based indexing
        for face in faces:
            # Build face string from vertex/uv/normal triplets
            face_str = []
            for i in range(0, len(face), 3):
                # Convert indices back to 1-based for OBJ format
                v_idx = face[i] + 1
                uv_idx = face[i + 1] + 1
                n_idx = face[i + 2] + 1
                face_str.append(f"{v_idx}/{uv_idx}/{n_idx}")
            f.write(f'f {" ".join(face_str)}\n')


def mesh_from_obj(
    obj_path: Path,
    vertices: NDArray[np.float32] = None,
    weights: NDArray[np.float32] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.int16], NDArray[np.float32]]:
    """Get UV mapped mesh data for vertices.

    This function takes arbitrary vertex positions and returns a new mesh with UV coordinates.
    It uses a template UV mapping stored in an .obj file.
    It checks that the input array is compatible with the template (they should have the same shape).

    The template .obj is parsed into np.arrays and cached, so that subsequent calls are faster.

    Args:
        obj_path: Path to the obj file containing UV mapping template
        vertices: Vertex positions as float32 array (N x 3) or None (default template
            vertex positions will be used in that case)
        weights: Weights array as float32 array (N x 4) or None

    Returns:
        A tuple containing:
            - new_vertices: Vertex positions with unique UV mappings (M x 3)
            - new_faces: Face indices referencing vertices (K x 3)
            - new_uvs: UV coordinates matching vertices (M x 2)

    Raises:
        AssertionError: If input vertices shape doesn't match template vertices
    """
    # Path for cached numpy arrays
    npz_path = obj_path.parent / f"{obj_path.stem}_mesh_data.npz"

    # Try to load cached mesh data first
    if npz_path.exists():
        data = np.load(str(npz_path))
        obj_v = data["vertices"]
        obj_f = data["faces"]
        obj_uvs = data["uvs"]
    else:
        # Parse template obj and cache the results
        obj_v, obj_f, obj_uvs = parse_obj(str(obj_path))
        np.savez(str(npz_path), vertices=obj_v, faces=obj_f, uvs=obj_uvs)

    if vertices is not None:
        # Verify input vertices match template shape
        assert (
            vertices.shape == obj_v.shape
        ), f"Vertex shapes do not match: vertices shape {vertices.shape} != obj_v shape {obj_v.shape}"
    else:
        vertices = obj_v

    # Verify that weights array matches the length of vertices
    if weights is not None:
        assert (
            weights.shape[0] == vertices.shape[0]
        ), f"Weights length {weights.shape[0]} does not match vertices length {vertices.shape[0]}"

    # Create mesh with unique vertices per UV coordinate
    new_vertices, new_faces, new_uvs, new_weights = create_unique_vertices_with_uvs(
        vertices, obj_f, obj_uvs, weights
    )

    # Extract vertex indices only (discard UV and normal indices)
    new_faces = new_faces[:, [0, 3, 6]]

    return {
        "vertices": new_vertices,
        "faces": new_faces,
        "uvs": new_uvs,
        "weights": new_weights,
        "normals": None,
        "colors": None,
    }


def analyze_gltf(gltf_path: str | Path):
    """
    Analyze a glTF file and extract mesh, material, and texture information.

    Args:
        gltf_path: Path to the glTF file

    Returns:
        Tuple containing:
        - List of skin names
        - Dictionary of mesh names to material indices
        - List of material information including raw binary data for textures
    """
    gltf = GLTF2.load(str(gltf_path))
    meshes = {}
    materials = []

    # Extract mesh data
    for node in gltf.nodes:
        if getattr(node, "mesh", None) is not None:
            mesh_name = node.name
            material_list = []

            for primitive in gltf.meshes[node.mesh].primitives:
                material_index = primitive.material
                material_list.append(material_index)

            meshes[mesh_name] = material_list

    # Extract material and texture data
    for material in gltf.materials:
        texture_info = None
        texture_data = None
        texture_mime_type = None

        if hasattr(material, "pbrMetallicRoughness") and material.pbrMetallicRoughness:
            if (
                hasattr(material.pbrMetallicRoughness, "baseColorTexture")
                and material.pbrMetallicRoughness.baseColorTexture
            ):
                texture_index = material.pbrMetallicRoughness.baseColorTexture.index
                texture = gltf.textures[texture_index]
                image_index = texture.source
                image = gltf.images[image_index]

                # Case 1: External URI
                if image.uri:
                    texture_info = image.uri
                    file_extension = Path(texture_info).suffix
                    texture_mime_type = f"image/{file_extension.lower()[1:]}"

                    # If it's a data URI
                    if image.uri.startswith("data:"):
                        mime_type, b64data = image.uri.split(",", 1)
                        mime_type = mime_type.split(";")[0].split(":")[1]
                        texture_mime_type = mime_type
                        texture_data = base64.b64decode(b64data)

                # Case 2: Binary blob in buffer
                elif image.bufferView is not None:
                    buffer_view = gltf.bufferViews[image.bufferView]
                    buffer = gltf.buffers[buffer_view.buffer]

                    # Handle binary glTF (.glb)
                    if (
                        not buffer.uri
                        and hasattr(gltf, "binary_blob")
                        and gltf.binary_blob is not None
                    ):
                        # Extract raw data from binary blob
                        start = buffer_view.byteOffset
                        end = start + buffer_view.byteLength
                        texture_data = gltf.binary_blob()[start:end]
                        texture_mime_type = image.mimeType
                    # Handle base64 data
                    elif buffer.uri and buffer.uri.startswith("data:"):
                        mime_type, b64data = buffer.uri.split(",", 1)
                        buffer_data = base64.b64decode(b64data)
                        start = buffer_view.byteOffset
                        end = start + buffer_view.byteLength
                        texture_data = buffer_data[start:end]
                        texture_mime_type = image.mimeType

        materials.append(
            {
                "name": material.name,
                "texture_uri": texture_info,
                "texture_data": texture_data,
                "texture_mime_type": texture_mime_type,
            }
        )

    # Extract skin names
    skin_names = (
        [skin.name for skin in gltf.skins]
        if hasattr(gltf, "skins") and gltf.skins
        else []
    )

    return skin_names, meshes, materials


def mesh_from_gltf(
    gltf_path: str | Path,
    object_name: str,
    morph_targets: list[float] = None,
) -> dict:
    """Extract vertex positions, faces, UV coordinates, weights, joint indices, and apply morph targets from a glTF file for a specific mesh.

    Args:
        gltf_path: Path to the glTF file
        object_name: Name of the mesh object to extract data from
        morph_targets: List of weights for each morph target

    Returns:
        A dictionary containing:
            - vertices: List of vertex positions arrays, each as float32 array (N x 3)
            - faces: List of face indices arrays, each as int16 array (M x 3)
            - uvs: List of UV coordinates arrays, each as float32 array (N x 2)
            - weights: List of vertex weights arrays, each as float32 array (N x 4)
            - joints: List of joint indices arrays, each as int16 array (N x 4)
            - joint_names: Dictionary mapping joint indices to names

    Raises:
        ValueError: If mesh with given name is not found in the glTF file
    """

    # Load the glTF file
    gltf = GLTF2.load(str(gltf_path))

    # Find the mesh with matching name
    mesh_node = None
    for i, node in enumerate(gltf.nodes):
        logging.debug(
            f"Checking node #{i}: {node.name if hasattr(node, 'name') else 'unnamed'}"
        )

        if object_name:
            if node.mesh is not None and node.name == object_name:
                mesh_node = node
                break
        else:
            if node.mesh is not None and node.skin is not None:
                mesh_node = node
                break

    if mesh_node is None:
        if not object_name:
            object_name = "<auto detect>"
        raise ValueError(f"Node:'{object_name}' not found in glTF file")

    logging.info(f"Using node: {mesh_node} to extract GTLF mesh data")
    # Initialize lists to collect data from all primitives
    all_vertices = []
    all_faces = []
    all_uvs = []
    all_weights = []
    all_joints = []

    # Iterate over all primitives in the mesh
    for primitive_index, primitive in enumerate(gltf.meshes[mesh_node.mesh].primitives):
        # Get vertex positions
        pos_accessor = gltf.accessors[primitive.attributes.POSITION]
        pos_view = gltf.bufferViews[pos_accessor.bufferView]
        pos_data = gltf.get_data_from_buffer_uri(gltf.buffers[pos_view.buffer].uri)
        vertices = np.frombuffer(
            pos_data[pos_view.byteOffset : pos_view.byteOffset + pos_view.byteLength],
            dtype=np.float32,
        ).reshape(-1, 3)

        # Apply morph targets if provided
        if morph_targets and any(weight != 0 for weight in morph_targets):
            vertices = vertices.copy()
            for i, weight in enumerate(morph_targets):
                if weight != 0:
                    morph_target = primitive.targets[i]
                    morph_pos_accessor = gltf.accessors[morph_target["POSITION"]]
                    morph_pos_view = gltf.bufferViews[morph_pos_accessor.bufferView]
                    morph_pos_data = gltf.get_data_from_buffer_uri(
                        gltf.buffers[morph_pos_view.buffer].uri
                    )
                    morph_positions = np.frombuffer(
                        morph_pos_data[
                            morph_pos_view.byteOffset : morph_pos_view.byteOffset
                            + morph_pos_view.byteLength
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 3)
                    vertices += weight * morph_positions

        # Get UV coordinates
        uv_accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
        uv_view = gltf.bufferViews[uv_accessor.bufferView]
        uv_data = gltf.get_data_from_buffer_uri(gltf.buffers[uv_view.buffer].uri)
        uvs = np.frombuffer(
            uv_data[uv_view.byteOffset : uv_view.byteOffset + uv_view.byteLength],
            dtype=np.float32,
        ).reshape(-1, 2)

        # Get face indices
        idx_accessor = gltf.accessors[primitive.indices]
        idx_view = gltf.bufferViews[idx_accessor.bufferView]
        idx_data = gltf.get_data_from_buffer_uri(gltf.buffers[idx_view.buffer].uri)
        faces = np.frombuffer(
            idx_data[idx_view.byteOffset : idx_view.byteOffset + idx_view.byteLength],
            dtype=np.uint16,
        ).reshape(-1, 3)

        # Get weights
        weights_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
        weights_view = gltf.bufferViews[weights_accessor.bufferView]
        weights_data = gltf.get_data_from_buffer_uri(
            gltf.buffers[weights_view.buffer].uri
        )
        weights = np.frombuffer(
            weights_data[
                weights_view.byteOffset : weights_view.byteOffset
                + weights_view.byteLength
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

        # Get joint indices
        joints_accessor = gltf.accessors[primitive.attributes.JOINTS_0]
        joints_view = gltf.bufferViews[joints_accessor.bufferView]
        joints_data = gltf.get_data_from_buffer_uri(
            gltf.buffers[joints_view.buffer].uri
        )

        # Make sure we're using the correct data type based on the accessor
        dtype = GLTF_2_dtype[joints_accessor.componentType]

        joints = np.frombuffer(
            joints_data[
                joints_view.byteOffset : joints_view.byteOffset + joints_view.byteLength
            ],
            dtype=dtype,
        ).reshape(-1, 4)

        logging.debug(
            f"{primitive_index} GET MESH DATA | Faces max value: {faces.max()}"
        )

        # Append data from this primitive to the lists
        all_vertices.append(vertices)
        all_faces.append(faces)
        all_uvs.append(uvs)
        all_weights.append(weights)
        all_joints.append(joints)

    # Assert that all arrays have compatible shapes
    num_primitives = len(all_vertices)
    for i in range(num_primitives):
        assert (
            all_vertices[i].shape[0] == all_uvs[i].shape[0]
        ), f"Vertices and UVs count mismatch in primitive {i}"
        assert (
            all_vertices[i].shape[0] == all_weights[i].shape[0]
        ), f"Vertices and Weights count mismatch in primitive {i}"
        assert (
            all_vertices[i].shape[0] == all_joints[i].shape[0]
        ), f"Vertices and Joints count mismatch in primitive {i}"

    joint_names = {i: gltf.nodes[j].name for i, j in enumerate(gltf.skins[0].joints)}

    logging.debug(f"Number of primitives: {num_primitives}")
    for i in range(num_primitives):
        logging.debug(
            f"Primitive {i}: verts: {all_vertices[i].shape[0]}, faces: {all_faces[i].shape[0]}"
        )

    return {
        "vertices": all_vertices,
        "faces": [f.astype(np.uint16) for f in all_faces],
        "uvs": all_uvs,
        "weights": all_weights,
        "joints": [j.astype(np.uint16) for j in all_joints],
        "joint_names": joint_names,
    }
