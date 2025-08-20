"""
Mesh extractor for GLB files.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from .mesh import MeshData
from ..glb import GLBParser
from ..exceptions import MeshError

logger = logging.getLogger(__name__)


class MeshExtractor:
    """
    Extracts mesh data from GLB files.
    """
    
    def __init__(self, parser: GLBParser):
        """
        Initialize mesh extractor.
        
        Args:
            parser: GLB parser instance
        """
        self.parser = parser
        self.meshes: List[MeshData] = []
        
    def extract_all_meshes(self) -> List[MeshData]:
        """
        Extract all meshes from the GLB file.
        
        Returns:
            List of mesh data objects
        """
        logger.info("Extracting meshes from GLB")
        
        for mesh_idx in range(len(self.parser.gltf.meshes)):
            mesh = self.parser.gltf.meshes[mesh_idx]
            mesh_name = mesh.name or f"mesh_{mesh_idx}"
            
            try:
                # Get primitives
                primitives = self.parser.get_mesh_primitives(mesh_idx)
                
                for prim_idx, prim_data in enumerate(primitives):
                    mesh_data = self._extract_primitive(
                        mesh_idx, prim_idx, prim_data, mesh_name
                    )
                    if mesh_data:
                        self.meshes.append(mesh_data)
                        logger.info(f"Extracted mesh: {mesh_data.name} ({len(mesh_data.vertices)} vertices)")
                        
            except Exception as e:
                logger.warning(f"Failed to extract mesh {mesh_idx}: {e}")
                
        return self.meshes
        
    def _extract_primitive(
        self, 
        mesh_idx: int, 
        prim_idx: int, 
        prim_data: Dict[str, Any],
        base_name: str
    ) -> Optional[MeshData]:
        """Extract mesh data from a primitive."""
        attributes = prim_data.get('attributes', {})
        
        # Check for required position data
        if 'POSITION' not in attributes or attributes['POSITION'] is None:
            logger.warning(f"Primitive {prim_idx} of mesh {mesh_idx} has no position data")
            return None
            
        # Create unique name for primitive
        if len(self.parser.gltf.meshes[mesh_idx].primitives) > 1:
            name = f"{base_name}_prim{prim_idx}"
        else:
            name = base_name
            
        # Create mesh data
        mesh_data = MeshData(
            name=name,
            vertices=attributes['POSITION'],
            indices=prim_data.get('indices'),
            normals=attributes.get('NORMAL'),
            uvs=attributes.get('TEXCOORD_0'),
            colors=attributes.get('COLOR_0'),
            joints=attributes.get('JOINTS_0'),
            weights=attributes.get('WEIGHTS_0'),
            material_index=prim_data.get('material'),
            mesh_index=mesh_idx,
            primitive_index=prim_idx
        )
        
        return mesh_data
        
    def export_meshes(self, output_dir: str, format: str = 'stl') -> Dict[str, str]:
        """
        Export all meshes to files.
        
        Args:
            output_dir: Output directory
            format: Export format ('stl' or 'obj')
            
        Returns:
            Dictionary mapping mesh names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for mesh_data in self.meshes:
            filename = f"{mesh_data.name}.{format}"
            file_path = output_dir / filename
            
            try:
                if format == 'stl':
                    mesh_data.export_stl(str(file_path))
                elif format == 'obj':
                    mesh_data.export_obj(str(file_path))
                else:
                    raise ValueError(f"Unsupported format: {format}")
                    
                exported_files[mesh_data.name] = str(file_path)
                logger.info(f"Exported {mesh_data.name} to {filename}")
                
            except Exception as e:
                logger.warning(f"Failed to export {mesh_data.name}: {e}")
                
        return exported_files
        
    def get_skinned_meshes(self) -> List[MeshData]:
        """Get only meshes that have skinning data."""
        return [m for m in self.meshes if m.joints is not None and m.weights is not None]
        
    def get_static_meshes(self) -> List[MeshData]:
        """Get only meshes without skinning data."""
        return [m for m in self.meshes if m.joints is None or m.weights is None]
        
    def find_meshes_for_bone(self, bone_index: int) -> List[MeshData]:
        """Find meshes that are influenced by a specific bone."""
        influenced_meshes = []
        
        for mesh_data in self.get_skinned_meshes():
            if mesh_data.joints is None:
                continue
                
            # Check if bone index appears in joint indices
            unique_joints = np.unique(mesh_data.joints.flatten())
            if bone_index in unique_joints:
                influenced_meshes.append(mesh_data)
                
        return influenced_meshes
