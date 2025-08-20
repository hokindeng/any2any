"""
Mesh data structures.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..common import transform_point_y_to_z, sanitize_name


@dataclass
class MeshData:
    """
    Container for mesh data.
    """
    name: str
    vertices: np.ndarray  # N x 3 array
    indices: Optional[np.ndarray] = None  # M x 3 array for triangles
    normals: Optional[np.ndarray] = None  # N x 3 array
    uvs: Optional[np.ndarray] = None  # N x 2 array
    colors: Optional[np.ndarray] = None  # N x 3 or N x 4 array
    
    # Skinning data
    joints: Optional[np.ndarray] = None  # N x 4 array
    weights: Optional[np.ndarray] = None  # N x 4 array
    
    # Material reference
    material_index: Optional[int] = None
    
    # Source references
    mesh_index: Optional[int] = None
    primitive_index: Optional[int] = None
    
    def __post_init__(self):
        """Validate and sanitize data."""
        self.name = sanitize_name(self.name)
        
        # Ensure vertices is numpy array
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        # Validate indices
        if self.indices is not None:
            self.indices = np.array(self.indices, dtype=np.uint32)
            
        # Ensure other arrays are numpy if present
        if self.normals is not None:
            self.normals = np.array(self.normals, dtype=np.float32)
        if self.uvs is not None:
            self.uvs = np.array(self.uvs, dtype=np.float32)
        if self.colors is not None:
            self.colors = np.array(self.colors, dtype=np.float32)
        if self.joints is not None:
            self.joints = np.array(self.joints, dtype=np.uint16)
        if self.weights is not None:
            self.weights = np.array(self.weights, dtype=np.float32)
            
    @property
    def vertices_mujoco(self) -> np.ndarray:
        """Get vertices in MuJoCo coordinate system."""
        # Transform each vertex from Y-up to Z-up
        vertices_z_up = np.zeros_like(self.vertices)
        for i in range(len(self.vertices)):
            vertices_z_up[i] = transform_point_y_to_z(self.vertices[i])
        return vertices_z_up
        
    @property
    def normals_mujoco(self) -> Optional[np.ndarray]:
        """Get normals in MuJoCo coordinate system."""
        if self.normals is None:
            return None
            
        # Transform normals (same as points, they're directions)
        normals_z_up = np.zeros_like(self.normals)
        for i in range(len(self.normals)):
            normals_z_up[i] = transform_point_y_to_z(self.normals[i])
        return normals_z_up
        
    @property
    def faces(self) -> Optional[np.ndarray]:
        """Get faces (alias for indices reshaped to Nx3)."""
        if self.indices is None:
            return None
        return self.indices.reshape(-1, 3)
        
    def export_stl(self, file_path: str):
        """
        Export mesh as STL file.
        
        Args:
            file_path: Output file path
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for STL export")
            
        # Create trimesh object with MuJoCo coordinates
        mesh = trimesh.Trimesh(
            vertices=self.vertices_mujoco,
            faces=self.faces,
            vertex_normals=self.normals_mujoco
        )
        
        # Export
        mesh.export(file_path, file_type='stl')
        
    def export_obj(self, file_path: str):
        """
        Export mesh as OBJ file.
        
        Args:
            file_path: Output file path
        """
        file_path = Path(file_path)
        
        with open(file_path, 'w') as f:
            f.write(f"# Mesh: {self.name}\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            if self.faces is not None:
                f.write(f"# Faces: {len(self.faces)}\n")
            f.write("\n")
            
            # Write vertices (in MuJoCo coordinates)
            vertices_mj = self.vertices_mujoco
            for v in vertices_mj:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            # Write normals if available
            if self.normals is not None:
                f.write("\n")
                normals_mj = self.normals_mujoco
                for n in normals_mj:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                    
            # Write UVs if available
            if self.uvs is not None:
                f.write("\n")
                for uv in self.uvs:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                    
            # Write faces
            if self.faces is not None:
                f.write("\n")
                f.write(f"g {self.name}\n")
                
                has_normals = self.normals is not None
                has_uvs = self.uvs is not None
                
                for face in self.faces:
                    # OBJ uses 1-based indexing
                    v1, v2, v3 = face + 1
                    
                    if has_uvs and has_normals:
                        f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                    elif has_uvs:
                        f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                    elif has_normals:
                        f.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
                    else:
                        f.write(f"f {v1} {v2} {v3}\n")
                        
    def get_bounds(self) -> Dict[str, np.ndarray]:
        """Get bounding box of mesh."""
        vertices_mj = self.vertices_mujoco
        return {
            'min': np.min(vertices_mj, axis=0),
            'max': np.max(vertices_mj, axis=0),
            'center': np.mean(vertices_mj, axis=0),
            'size': np.max(vertices_mj, axis=0) - np.min(vertices_mj, axis=0)
        }
