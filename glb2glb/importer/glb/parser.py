"""
Main GLB/GLTF parser.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from pygltflib import GLTF2
import numpy as np

from .accessor import AccessorReader
from ..exceptions import GLBParseError

logger = logging.getLogger(__name__)


class GLBParser:
    """
    Parser for GLB/GLTF files.
    
    This class handles loading and parsing GLB files, providing
    access to scenes, nodes, meshes, skins, and animations.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize GLB parser.
        
        Args:
            file_path: Path to GLB/GLTF file
            
        Raises:
            GLBParseError: If file cannot be loaded
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise GLBParseError(f"File not found: {file_path}")
            
        logger.info(f"Loading GLB file: {self.file_path}")
        
        try:
            self.gltf = GLTF2.load(str(self.file_path))
        except Exception as e:
            raise GLBParseError(f"Failed to load GLB file: {e}")
            
        self.accessor_reader = AccessorReader(self.gltf)
        
        # Log basic info
        logger.info(f"GLB loaded successfully:")
        logger.info(f"  Scenes: {len(self.gltf.scenes)}")
        logger.info(f"  Nodes: {len(self.gltf.nodes)}")
        logger.info(f"  Meshes: {len(self.gltf.meshes)}")
        logger.info(f"  Skins: {len(self.gltf.skins)}")
        logger.info(f"  Animations: {len(self.gltf.animations)}")
        
    def get_scene_info(self, scene_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a scene.
        
        Args:
            scene_idx: Scene index (default: active scene)
            
        Returns:
            Dictionary with scene information
        """
        if scene_idx is None:
            scene_idx = self.gltf.scene or 0
            
        if scene_idx >= len(self.gltf.scenes):
            raise GLBParseError(f"Scene index {scene_idx} out of range")
            
        scene = self.gltf.scenes[scene_idx]
        
        return {
            'name': scene.name or f'Scene_{scene_idx}',
            'nodes': scene.nodes or [],
        }
        
    def get_node_info(self, node_idx: int) -> Dict[str, Any]:
        """
        Get information about a node.
        
        Args:
            node_idx: Node index
            
        Returns:
            Dictionary with node information
        """
        if node_idx >= len(self.gltf.nodes):
            raise GLBParseError(f"Node index {node_idx} out of range")
            
        node = self.gltf.nodes[node_idx]
        
        info = {
            'name': node.name or f'Node_{node_idx}',
            'children': node.children or [],
            'mesh': node.mesh,
            'skin': node.skin,
            'camera': node.camera,
        }
        
        # Get transform
        if node.matrix is not None:
            # Matrix transform
            info['matrix'] = np.array(node.matrix).reshape(4, 4)
        else:
            # TRS transform
            info['translation'] = node.translation or [0, 0, 0]
            info['rotation'] = node.rotation or [0, 0, 0, 1]
            info['scale'] = node.scale or [1, 1, 1]
            
        return info
        
    def get_mesh_primitives(self, mesh_idx: int) -> List[Dict[str, Any]]:
        """
        Get mesh primitives.
        
        Args:
            mesh_idx: Mesh index
            
        Returns:
            List of primitive data dictionaries
        """
        if mesh_idx >= len(self.gltf.meshes):
            raise GLBParseError(f"Mesh index {mesh_idx} out of range")
            
        mesh = self.gltf.meshes[mesh_idx]
        primitives = []
        
        for prim_idx, primitive in enumerate(mesh.primitives):
            prim_data = {
                'index': prim_idx,
                'mode': primitive.mode or 4,  # Default to TRIANGLES
                'material': primitive.material,
            }
            
            # Get attributes
            attributes = {}
            
            # Helper to read attribute safely
            def _read_attr(attr_name: str):
                idx = getattr(primitive.attributes, attr_name, None)
                if idx is not None:
                    return self.accessor_reader.read_accessor(idx)
                return None

            # Position (required)
            pos = _read_attr('POSITION')
            if pos is not None:
                attributes['POSITION'] = pos
                
            # Normal
            norm = _read_attr('NORMAL')
            if norm is not None:
                attributes['NORMAL'] = norm
                
            # Texture coordinates
            uv0 = _read_attr('TEXCOORD_0')
            if uv0 is not None:
                attributes['TEXCOORD_0'] = uv0
                
            # Vertex colors
            col0 = _read_attr('COLOR_0')
            if col0 is not None:
                attributes['COLOR_0'] = col0
                
            # Skinning data
            j0 = _read_attr('JOINTS_0')
            if j0 is not None:
                attributes['JOINTS_0'] = j0
                
            w0 = _read_attr('WEIGHTS_0')
            if w0 is not None:
                attributes['WEIGHTS_0'] = w0
                
            prim_data['attributes'] = attributes
            
            # Get indices
            if primitive.indices is not None:
                prim_data['indices'] = self.accessor_reader.read_accessor(
                    primitive.indices
                )
                
            primitives.append(prim_data)
            
        return primitives
        
    def get_skin_data(self, skin_idx: int) -> Dict[str, Any]:
        """
        Get skin data.
        
        Args:
            skin_idx: Skin index
            
        Returns:
            Dictionary with skin data
        """
        if skin_idx >= len(self.gltf.skins):
            raise GLBParseError(f"Skin index {skin_idx} out of range")
            
        skin = self.gltf.skins[skin_idx]
        
        skin_data = {
            'name': skin.name or f'Skin_{skin_idx}',
            'joints': skin.joints,
            'skeleton': skin.skeleton,
        }
        
        # Get inverse bind matrices
        if skin.inverseBindMatrices is not None:
            matrices = self.accessor_reader.read_accessor(skin.inverseBindMatrices)
            # Reshape to 4x4 matrices
            skin_data['inverseBindMatrices'] = matrices.reshape(-1, 4, 4)
            
        return skin_data
        
    def get_animation_data(self, anim_idx: int) -> Dict[str, Any]:
        """
        Get animation data.
        
        Args:
            anim_idx: Animation index
            
        Returns:
            Dictionary with animation data
        """
        if anim_idx >= len(self.gltf.animations):
            raise GLBParseError(f"Animation index {anim_idx} out of range")
            
        animation = self.gltf.animations[anim_idx]
        
        anim_data = {
            'name': animation.name or f'Animation_{anim_idx}',
            'channels': [],
            'duration': 0.0,
        }
        
        # Process channels
        for channel in animation.channels:
            sampler = animation.samplers[channel.sampler]
            
            # Read time and value data
            times = self.accessor_reader.read_accessor(sampler.input)
            values = self.accessor_reader.read_accessor(sampler.output)
            
            channel_data = {
                'target_node': channel.target.node,
                'target_path': channel.target.path,
                'interpolation': sampler.interpolation or 'LINEAR',
                'times': times,
                'values': values,
            }
            
            anim_data['channels'].append(channel_data)
            
            # Update duration
            if len(times) > 0:
                anim_data['duration'] = max(anim_data['duration'], float(times[-1]))
                
        return anim_data
        
    def find_armature_roots(self) -> List[int]:
        """
        Find potential armature root nodes.
        
        Returns:
            List of node indices that could be armature roots
        """
        armature_roots = []
        
        # Look for nodes with skins
        for node_idx, node in enumerate(self.gltf.nodes):
            if node.skin is not None:
                # This node has a skin, might be part of armature
                armature_roots.append(node_idx)
                
        # Also look for nodes that are parents of joint nodes
        for skin in self.gltf.skins:
            if skin.skeleton is not None:
                armature_roots.append(skin.skeleton)
                
        # Remove duplicates
        return list(set(armature_roots))
