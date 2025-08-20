"""
Skeleton extractor for GLB files.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from .bone import Bone, BoneHierarchy
from ..glb import GLBParser
from ..exceptions import SkeletonError

logger = logging.getLogger(__name__)


class SkeletonExtractor:
    """
    Extracts skeleton/armature information from GLB files.
    """
    
    def __init__(self, parser: GLBParser):
        """
        Initialize skeleton extractor.
        
        Args:
            parser: GLB parser instance
        """
        self.parser = parser
        self.hierarchies: List[BoneHierarchy] = []
        
    def extract_all_skeletons(self) -> List[BoneHierarchy]:
        """
        Extract all skeletons from the GLB file.
        
        Returns:
            List of bone hierarchies
        """
        logger.info("Extracting skeletons from GLB")
        
        # Process each skin
        for skin_idx in range(len(self.parser.gltf.skins)):
            try:
                hierarchy = self.extract_skeleton(skin_idx)
                self.hierarchies.append(hierarchy)
                logger.info(f"Extracted skeleton {skin_idx} with {len(hierarchy.bones)} bones")
            except Exception as e:
                logger.warning(f"Failed to extract skeleton {skin_idx}: {e}")
                
        # Only look for unskinned armatures if no skins were found
        # This avoids creating duplicate skeletons from nodes that are already part of skins
        if len(self.hierarchies) == 0:
            logger.info("No skinned skeletons found, looking for unskinned armatures...")
            armature_roots = self._find_unskinned_armatures()
            for root_idx in armature_roots:
                try:
                    hierarchy = self._extract_unskinned_skeleton(root_idx)
                    if hierarchy and len(hierarchy.bones) > 0:
                        self.hierarchies.append(hierarchy)
                        logger.info(f"Extracted unskinned skeleton from node {root_idx} with {len(hierarchy.bones)} bones")
                except Exception as e:
                    logger.warning(f"Failed to extract unskinned skeleton from node {root_idx}: {e}")
                
        return self.hierarchies
        
    def extract_skeleton(self, skin_idx: int) -> BoneHierarchy:
        """
        Extract skeleton from a specific skin.
        
        Args:
            skin_idx: Skin index
            
        Returns:
            Bone hierarchy
            
        Raises:
            SkeletonError: If extraction fails
        """
        logger.debug(f"Extracting skeleton from skin {skin_idx}")
        
        # Get skin data
        skin_data = self.parser.get_skin_data(skin_idx)
        joint_indices = skin_data['joints']
        
        if not joint_indices:
            raise SkeletonError(f"Skin {skin_idx} has no joints")
            
        # Create hierarchy
        hierarchy = BoneHierarchy()
        
        # Get node info for all nodes (not only joints), we will reference by index
        nodes_info = [self.parser.get_node_info(i) for i in range(len(self.parser.gltf.nodes))]
            
        # Build hierarchy from nodes
        hierarchy.build_from_nodes(nodes_info, joint_indices)
        
        # Remove duplicate bones if any
        unique_bones = {}
        for bone in hierarchy.bones:
            if bone.name not in unique_bones:
                unique_bones[bone.name] = bone
                
        # Rebuild hierarchy with unique bones only
        if len(unique_bones) < len(hierarchy.bones):
            logger.warning(f"Found duplicate bones, keeping unique ones only ({len(unique_bones)} unique from {len(hierarchy.bones)} total)")
            hierarchy.bones = list(unique_bones.values())
            hierarchy.bone_map = {b.index: b for b in hierarchy.bones}
        
        # Apply inverse bind matrices if available
        if 'inverseBindMatrices' in skin_data:
            matrices = skin_data['inverseBindMatrices']
            for i, joint_idx in enumerate(joint_indices):
                bone = hierarchy.get_bone(joint_idx)
                if bone and i < len(matrices):
                    bone.inverse_bind_matrix = matrices[i]

        # Validate tree connectivity: ensure a single connected component rooted at skin.skeleton if provided
        if skin_data.get('skeleton') is not None:
            skel_root_idx = skin_data['skeleton']
            # If we didn't set this root as parentless, enforce it
            root_bone = hierarchy.get_bone(skel_root_idx)
            if root_bone and root_bone.parent is not None:
                # Detach and make it a root
                root_bone.parent = None
            # Ensure all unreachable bones are attached under this root preserving original parents order
            visited = set()
            def dfs(b):
                if not b or b.index in visited:
                    return
                visited.add(b.index)
                for c in b.children:
                    dfs(c)
            dfs(root_bone)
            for b in list(hierarchy.bones):
                if b.index not in visited and b is not root_bone:
                    # Attach stray components under the root to preserve a single tree
                    root_bone.add_child(b)
                    dfs(b)
        
        # Ensure roots are properly computed after all modifications            
        hierarchy.roots = [b for b in hierarchy.bones if b.parent is None]
                    
        return hierarchy
        
    def _find_unskinned_armatures(self) -> List[int]:
        """Find potential unskinned armature roots."""
        potential_roots = []
        
        # Look for nodes that:
        # 1. Have children
        # 2. Don't have meshes
        # 3. Have names suggesting armature/skeleton
        for node_idx, node in enumerate(self.parser.gltf.nodes):
            node_info = self.parser.get_node_info(node_idx)
            
            if (node_info.get('children') and 
                node_info.get('mesh') is None and
                self._is_armature_name(node_info.get('name', ''))):
                potential_roots.append(node_idx)
                
        return potential_roots
        
    def _is_armature_name(self, name: str) -> bool:
        """Check if name suggests an armature/skeleton."""
        armature_keywords = ['armature', 'skeleton', 'rig', 'bone', 'joint', 'root']
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in armature_keywords)
        
    def _extract_unskinned_skeleton(self, root_idx: int) -> Optional[BoneHierarchy]:
        """Extract skeleton from unskinned hierarchy."""
        hierarchy = BoneHierarchy()
        
        # Recursively process nodes
        def process_node(node_idx: int, parent_bone: Optional[Bone] = None):
            node_info = self.parser.get_node_info(node_idx)
            
            # Create bone
            bone = Bone(
                index=node_idx,
                name=node_info.get('name', f'node_{node_idx}'),
                translation=np.array(node_info.get('translation', [0, 0, 0])),
                rotation=np.array(node_info.get('rotation', [0, 0, 0, 1])),
                scale=np.array(node_info.get('scale', [1, 1, 1])),
                node_index=node_idx
            )
            
            if parent_bone:
                parent_bone.add_child(bone)
                
            hierarchy.add_bone(bone)
            
            # Process children
            for child_idx in node_info.get('children', []):
                process_node(child_idx, bone)
                
        # Start from root
        process_node(root_idx)
        
        # Compute roots after building hierarchy
        hierarchy.roots = [b for b in hierarchy.bones if b.parent is None]
        
        return hierarchy if len(hierarchy.bones) > 1 else None
        
    def get_primary_skeleton(self) -> Optional[BoneHierarchy]:
        """
        Get the primary skeleton (usually the first or largest).
        
        Returns:
            Primary bone hierarchy or None
        """
        if not self.hierarchies:
            return None
            
        # Return the skeleton with most bones
        return max(self.hierarchies, key=lambda h: len(h.bones))
