"""
Bone and skeleton hierarchy classes.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..common import transform_point_y_to_z, transform_quaternion_y_to_z, sanitize_name


@dataclass
class Bone:
    """
    Represents a single bone in the skeleton hierarchy.
    """
    index: int
    name: str
    parent: Optional['Bone'] = None
    children: List['Bone'] = field(default_factory=list)
    
    # Transform in GLB space (Y-up)
    translation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # XYZW
    scale: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    
    # Node reference
    node_index: Optional[int] = None
    
    # Inverse bind matrix (if skinned)
    inverse_bind_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.translation = np.array(self.translation)
        self.rotation = np.array(self.rotation)
        self.scale = np.array(self.scale)
        self.name = sanitize_name(self.name)
        
    @property
    def translation_mujoco(self) -> np.ndarray:
        """Get translation in MuJoCo coordinate system (Z-up).
        
        Only transforms root bone from Y-up to Z-up.
        Child bones keep their local transforms relative to parent.
        """
        if self.parent is None:
            # Root bone: transform from Y-up to Z-up coordinate system
            return transform_point_y_to_z(self.translation)
        else:
            # Child bones: keep local transform as-is (already relative to parent)
            return self.translation
        
    @property
    def rotation_mujoco(self) -> np.ndarray:
        """Get rotation in MuJoCo coordinate system (WXYZ quaternion).
        
        For root bone: Apply inverse rotation to stand the model upright.
        The GLB has the model rotated -90° around X (from MuJoCo→GLB conversion).
        We need to apply +90° around X to get it back upright.
        
        For child bones: Just convert format from XYZW to WXYZ.
        """
        if self.parent is None:
            # Root bone: Apply inverse of the Y-up to Z-up rotation
            # We need +90° around X to stand the model back up
            # This undoes the -90° rotation that was applied during MuJoCo→GLB
            x, y, z, w = self.rotation
            
            # Apply +90° rotation around X-axis
            # Quaternion for +90° around X: [sin(45°), 0, 0, cos(45°)] = [0.7071, 0, 0, 0.7071]
            rot_x_inv = np.array([0.7071068, 0, 0, 0.7071068])  # +90° around X in XYZW
            rx, ry, rz, rw = rot_x_inv
            
            # Quaternion multiplication: q_new = rot_x_inv * q_orig
            w_new = rw * w - rx * x - ry * y - rz * z
            x_new = rw * x + rx * w + ry * z - rz * y
            y_new = rw * y - rx * z + ry * w + rz * x
            z_new = rw * z + rx * y - ry * x + rz * w
            
            # Return in WXYZ format
            result = np.array([w_new, x_new, y_new, z_new])
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            return result
        else:
            # Child bones: just convert XYZW to WXYZ format
            x, y, z, w = self.rotation
            return np.array([w, x, y, z])
        
    @property
    def local_transform_matrix(self) -> np.ndarray:
        """Get local transformation matrix."""
        # Create transformation matrix from TRS
        matrix = np.eye(4)
        
        # Apply scale
        matrix[0, 0] = self.scale[0]
        matrix[1, 1] = self.scale[1]
        matrix[2, 2] = self.scale[2]
        
        # Apply rotation (quaternion to matrix)
        x, y, z, w = self.rotation
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        
        rot_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])
        
        matrix[:3, :3] = rot_matrix @ matrix[:3, :3]
        
        # Apply translation
        matrix[:3, 3] = self.translation
        
        return matrix
        
    @property
    def global_transform_matrix(self) -> np.ndarray:
        """Get global transformation matrix."""
        if self.parent is None:
            return self.local_transform_matrix
        else:
            return self.parent.global_transform_matrix @ self.local_transform_matrix
            
    def add_child(self, child: 'Bone'):
        """Add a child bone."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            
    def get_joint_name(self) -> str:
        """Get name for MuJoCo joint."""
        return f"{self.name}_joint"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'index': self.index,
            'name': self.name,
            'parent_index': self.parent.index if self.parent else None,
            'children_indices': [c.index for c in self.children],
            'translation': self.translation.tolist(),
            'rotation': self.rotation.tolist(),
            'scale': self.scale.tolist(),
            'node_index': self.node_index,
        }


class BoneHierarchy:
    """
    Represents the complete bone hierarchy.
    """
    
    def __init__(self):
        """Initialize empty hierarchy."""
        self.bones: List[Bone] = []
        self.bone_map: Dict[int, Bone] = {}
        self.roots: List[Bone] = []
        
    def add_bone(self, bone: Bone):
        """Add a bone to the hierarchy."""
        # Deduplicate by node index
        if bone.index in self.bone_map:
            return
        self.bones.append(bone)
        self.bone_map[bone.index] = bone
            
    def get_bone(self, index: int) -> Optional[Bone]:
        """Get bone by index."""
        return self.bone_map.get(index)
        
    def get_bone_by_name(self, name: str) -> Optional[Bone]:
        """Get bone by name."""
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None
        
    def build_from_nodes(self, nodes: List[Dict[str, Any]], joint_indices: List[int]):
        """
        Build hierarchy from GLB nodes and joint indices.
        
        Args:
            nodes: List of node info dictionaries
            joint_indices: List of node indices that are joints
        """
        # Remove duplicates from joint indices
        unique_joint_indices = list(dict.fromkeys(joint_indices))
        
        # Create bones for each unique joint
        for joint_idx in unique_joint_indices:
            if joint_idx >= len(nodes):
                continue
                
            node = nodes[joint_idx]
            
            bone_name = node.get('name', f'joint_{joint_idx}')
            
            bone = Bone(
                index=joint_idx,
                name=bone_name,
                translation=np.array(node.get('translation', [0, 0, 0])),
                rotation=np.array(node.get('rotation', [0, 0, 0, 1])),
                scale=np.array(node.get('scale', [1, 1, 1])),
                node_index=joint_idx
            )
            
            self.add_bone(bone)
            
        # Establish parent-child relationships
        for joint_idx in unique_joint_indices:
            if joint_idx >= len(nodes):
                continue
                
            node = nodes[joint_idx]
            bone = self.get_bone(joint_idx)
            if not bone:
                continue
            
            # Find parent among joints
            for parent_idx, parent_node in enumerate(nodes):
                if parent_idx in unique_joint_indices and parent_idx != joint_idx:
                    children = parent_node.get('children', [])
                    if joint_idx in children:
                        parent_bone = self.get_bone(parent_idx)
                        if parent_bone and bone.parent is None:  # Only set parent if not already set
                            parent_bone.add_child(bone)
                        break

        # Recompute roots after parenting
        self.roots = [b for b in self.bones if b.parent is None]
                        
    def print_hierarchy(self, bone: Optional[Bone] = None, indent: int = 0, visited: Optional[set] = None):
        """Print hierarchy structure."""
        if visited is None:
            visited = set()
            
        if bone is None:
            # Print all roots
            for root in self.roots:
                self.print_hierarchy(root, indent, visited)
        else:
            # Skip if already visited (prevent infinite loops)
            if bone.index in visited:
                return
            visited.add(bone.index)
            
            print('  ' * indent + f'- {bone.name} (index={bone.index})')
            for child in bone.children:
                self.print_hierarchy(child, indent + 1, visited)
                
    def get_ordered_bones(self) -> List[Bone]:
        """Get bones in hierarchical order (parents before children)."""
        ordered = []
        visited = set()
        
        def visit(bone: Bone):
            if bone.index in visited:
                return
            visited.add(bone.index)
            ordered.append(bone)
            for child in bone.children:
                visit(child)
                
        # Visit all roots
        for root in self.roots:
            visit(root)
            
        return ordered
