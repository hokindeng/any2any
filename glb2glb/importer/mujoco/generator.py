"""
MuJoCo XML generator.
"""

import logging
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..skeleton import BoneHierarchy, Bone
from ..mesh import MeshData
from ..common import format_float_array, DEFAULT_JOINT_DAMPING, DEFAULT_MOTOR_GEAR, DEFAULT_TIMESTEP, DEFAULT_GRAVITY

logger = logging.getLogger(__name__)


class MuJoCoGenerator:
    """
    Generates MuJoCo XML from extracted GLB data.
    """
    
    def __init__(
        self,
        model_name: str,
        skeleton: Optional[BoneHierarchy] = None,
        meshes: Optional[List[MeshData]] = None
    ):
        """
        Initialize MuJoCo generator.
        
        Args:
            model_name: Name for the MuJoCo model
            skeleton: Bone hierarchy
            meshes: List of mesh data
        """
        self.model_name = model_name
        self.skeleton = skeleton
        self.meshes = meshes or []
        
        # XML elements
        self.root = None
        self.worldbody = None
        self.asset = None
        self.actuator = None
        
        # Track created elements
        self.mesh_refs = {}
        self.joint_refs = {}
        
    def generate(self) -> ET.Element:
        """
        Generate complete MuJoCo XML.
        
        Returns:
            Root XML element
        """
        logger.info(f"Generating MuJoCo XML for model '{self.model_name}'")
        
        # Create root element
        self.root = ET.Element('mujoco')
        self.root.set('model', self.model_name)
        
        # Add compiler settings
        self._add_compiler()
        
        # Add options
        self._add_options()
        
        # Add defaults
        self._add_defaults()
        
        # Add assets (meshes)
        self._add_assets()
        
        # Add worldbody
        self._add_worldbody()
        
        # Add actuators
        self._add_actuators()
        
        # Add sensors (optional)
        self._add_sensors()
        
        # Add keyframes
        self._add_keyframes()
        
        return self.root
        
    def _add_compiler(self):
        """Add compiler settings."""
        compiler = ET.SubElement(self.root, 'compiler')
        compiler.set('angle', 'radian')
        compiler.set('meshdir', 'meshes')
        compiler.set('autolimits', 'true')
        
    def _add_options(self):
        """Add simulation options."""
        option = ET.SubElement(self.root, 'option')
        option.set('timestep', str(DEFAULT_TIMESTEP))
        option.set('gravity', DEFAULT_GRAVITY)
        
    def _add_defaults(self):
        """Add default settings."""
        default = ET.SubElement(self.root, 'default')
        
        # Joint defaults
        joint_default = ET.SubElement(default, 'joint')
        joint_default.set('damping', str(DEFAULT_JOINT_DAMPING))
        joint_default.set('armature', '0.01')
        
        # Geom defaults
        geom_default = ET.SubElement(default, 'geom')
        geom_default.set('contype', '1')
        geom_default.set('conaffinity', '1')
        geom_default.set('friction', '0.8 0.005 0.0001')  # More realistic friction
        
    def _add_assets(self):
        """Add mesh assets."""
        if not self.meshes:
            return
            
        self.asset = ET.SubElement(self.root, 'asset')
        
        for mesh_data in self.meshes:
            mesh_elem = ET.SubElement(self.asset, 'mesh')
            mesh_elem.set('name', mesh_data.name)
            mesh_elem.set('file', f'{mesh_data.name}.stl')
            
            # Store reference
            self.mesh_refs[mesh_data.name] = mesh_elem
            
        logger.info(f"Added {len(self.meshes)} mesh assets")
        
    def _add_worldbody(self):
        """Add worldbody with skeleton hierarchy."""
        self.worldbody = ET.SubElement(self.root, 'worldbody')
        
        # Add ground plane
        self._add_ground_plane()
        
        # Add skeleton if available
        if self.skeleton:
            # Add each root bone with a floating base
            for root_bone in self.skeleton.roots:
                self._add_body_recursive(self.worldbody, root_bone, is_root=True)
        else:
            # Add static meshes
            self._add_static_meshes()
            
    def _add_ground_plane(self):
        """Add ground plane."""
        ground = ET.SubElement(self.worldbody, 'geom')
        ground.set('name', 'ground')
        ground.set('type', 'plane')
        ground.set('size', '10 10 0.1')
        ground.set('rgba', '0.8 0.8 0.8 1')
        
    def _add_body_recursive(self, parent_elem: ET.Element, bone: Bone, is_root: bool = False):
        """Recursively add body elements for bone hierarchy."""
        # Create body element
        body = ET.SubElement(parent_elem, 'body')
        body.set('name', bone.name)
        
        # Set position (relative to parent)
        if bone.parent:
            # Use local transform
            pos = bone.translation_mujoco
        else:
            # Root bone - use global position with reasonable height
            pos = bone.translation_mujoco
            # If Z is too low, set a reasonable height
            if pos[2] < 0.5:
                pos = np.array([pos[0], pos[1], 0.8])  # Default 0.8m height
            
        body.set('pos', format_float_array(pos))
        
        # Set orientation
        quat = bone.rotation_mujoco  # WXYZ format
        body.set('quat', format_float_array(quat))
        
        # Add freejoint for root body
        if is_root:
            freejoint = ET.SubElement(body, 'freejoint')
            freejoint.set('name', f'{bone.name}_free')
        
        # Add joint (except for root and end effectors)
        if bone.parent is not None and not self._is_end_effector(bone):
            joint = self._add_joint(body, bone)
            # Track by actual joint name so actuators can reference correctly
            jname = joint.get('name')
            if jname:
                self.joint_refs[jname] = joint
            
        # Add inertial properties
        self._add_inertial(body, bone)
        
        # Add geometry
        self._add_geometry(body, bone)
        
        # Add child bodies
        for child in bone.children:
            self._add_body_recursive(body, child, is_root=False)
            
    def _is_end_effector(self, bone: Bone) -> bool:
        """
        Check if a bone is an end effector (foot, hand, etc.) that shouldn't have a joint.
        
        Args:
            bone: Bone to check
            
        Returns:
            True if bone is an end effector
        """
        name_lower = bone.name.lower()
        
        # List of keywords that indicate end effectors
        end_effector_keywords = ['foot', 'hand', 'end', 'toe', 'finger', 'tip']
        
        # Check if it's a leaf node (no children)
        is_leaf = len(bone.children) == 0
        
        # Check if name contains end effector keywords
        has_end_name = any(keyword in name_lower for keyword in end_effector_keywords)
        
        # End effector if it's a leaf node with an end effector name
        # Special case: LINK_FOOT_L/R and LINK_ELBOW_END_L/R should not have joints
        if is_leaf and has_end_name:
            logger.info(f"Skipping joint for end effector: {bone.name}")
            return True
            
        return False
    
    def _add_joint(self, body: ET.Element, bone: Bone) -> ET.Element:
        """Add joint to body."""
        joint = ET.SubElement(body, 'joint')
        joint.set('name', bone.get_joint_name())
        
        # Determine joint type and axis based on bone name
        bone_lower = bone.name.lower()
        
        # Always use hinge joints for better stability
        joint.set('type', 'hinge')
        
        # Determine axis based on joint name
        if 'yaw' in bone_lower:
            joint.set('axis', '0 0 1')  # Z-axis for yaw
        elif 'pitch' in bone_lower:
            joint.set('axis', '0 1 0')  # Y-axis for pitch
        elif 'roll' in bone_lower:
            joint.set('axis', '1 0 0')  # X-axis for roll
        elif 'knee' in bone_lower or 'elbow' in bone_lower:
            joint.set('axis', '0 1 0')  # Y-axis for knee/elbow
        elif 'ankle' in bone_lower and 'roll' not in bone_lower:
            joint.set('axis', '0 1 0')  # Y-axis for ankle pitch
        else:
            # Default to Y-axis
            joint.set('axis', '0 1 0')
            
        # Add reasonable joint limits based on joint type
        if 'hip' in bone_lower:
            if 'yaw' in bone_lower:
                joint.set('range', '-1.57 1.57')  # +/- 90 degrees
            else:
                joint.set('range', '-2.0 2.0')  # Wider range for hip
        elif 'knee' in bone_lower:
            joint.set('range', '-0.1 2.5')  # Knee typically bends one way
        elif 'ankle' in bone_lower:
            joint.set('range', '-0.8 0.8')  # Limited ankle range
        elif 'shoulder' in bone_lower:
            joint.set('range', '-3.14 3.14')  # Full rotation for shoulder
        elif 'elbow' in bone_lower:
            joint.set('range', '0 2.5')  # Elbow typically bends one way
        elif 'head' in bone_lower or 'neck' in bone_lower:
            joint.set('range', '-0.8 0.8')  # Limited head movement
        else:
            joint.set('range', '-1.57 1.57')  # Default +/- 90 degrees
            
        # Add damping and armature
        joint.set('damping', '0.5')
        joint.set('armature', '0.01')
            
        return joint
        
    def _add_inertial(self, body: ET.Element, bone: Bone):
        """Add inertial properties to body."""
        inertial = ET.SubElement(body, 'inertial')
        
        # More realistic mass estimation based on body part
        bone_lower = bone.name.lower()
        
        # Base masses for different body parts (in kg)
        if 'base' in bone_lower or 'pelvis' in bone_lower:
            mass = 5.0  # Heavier base/pelvis
        elif 'torso' in bone_lower or 'chest' in bone_lower:
            mass = 8.0  # Heavy torso
        elif 'hip' in bone_lower:
            mass = 2.0  # Hip joint mass
        elif 'thigh' in bone_lower or ('hip' in bone_lower and 'yaw' in bone_lower):
            mass = 3.5  # Thigh mass
        elif 'knee' in bone_lower or 'shin' in bone_lower:
            mass = 2.5  # Lower leg
        elif 'ankle' in bone_lower:
            mass = 0.5  # Ankle joint
        elif 'foot' in bone_lower:
            mass = 0.8  # Foot mass
        elif 'shoulder' in bone_lower:
            mass = 1.5  # Shoulder mass
        elif 'elbow' in bone_lower or 'forearm' in bone_lower:
            mass = 1.0  # Forearm
        elif 'head' in bone_lower:
            mass = 2.0  # Head mass
        else:
            # Estimate based on bone size
            if bone.children:
                child_distances = [np.linalg.norm(child.translation) for child in bone.children]
                avg_dist = np.mean(child_distances)
                mass = max(0.5, min(5.0, 2.0 + avg_dist * 10))  # Scale with size
            else:
                mass = 0.5  # Default for leaf bones
                
        # Calculate center of mass offset
        if bone.children:
            # Place COM towards children
            child_positions = [child.translation for child in bone.children]
            avg_child_dir = np.mean(child_positions, axis=0)
            com_offset = avg_child_dir * 0.3  # 30% towards children
        else:
            com_offset = np.array([0, 0, 0])
            
        inertial.set('pos', format_float_array(com_offset))
        inertial.set('mass', f'{mass:.3f}')
        
        # More realistic inertia tensor (cylinder/box approximation)
        if bone.children:
            # Get characteristic length
            lengths = [np.linalg.norm(child.translation) for child in bone.children]
            length = np.mean(lengths)
        else:
            length = 0.05
            
        # Approximate as cylinder with radius = length/10
        radius = length * 0.1
        
        # Cylinder inertia: Ixx = Iyy = m*(3r² + h²)/12, Izz = mr²/2
        ixx = mass * (3 * radius**2 + length**2) / 12
        iyy = ixx
        izz = mass * radius**2 / 2

        # Ensure minimum inertia values (MuJoCo mjMINVAL ~1e-10)
        min_inertia = 1e-6
        ixx = max(ixx, min_inertia)
        iyy = max(iyy, min_inertia)
        izz = max(izz, min_inertia)
        
        inertial.set('diaginertia', f'{ixx:.6f} {iyy:.6f} {izz:.6f}')
        
    def _add_geometry(self, body: ET.Element, bone: Bone):
        """Add geometry to body."""
        bone_lower = bone.name.lower()
        
        # Check if there's a mesh for this bone
        mesh_name = None
        for mesh_data in self.meshes:
            if bone.node_index == mesh_data.mesh_index:
                mesh_name = mesh_data.name
                break
                
        # Add collision geometry based on body part
        if 'base' in bone_lower or 'pelvis' in bone_lower:
            # Box for base/pelvis
            geom = ET.SubElement(body, 'geom')
            geom.set('name', f'{bone.name}_collision')
            geom.set('type', 'box')
            geom.set('size', '0.08 0.06 0.08')
            geom.set('rgba', '0.8 0.3 0.3 0.5')
        elif 'foot' in bone_lower:
            # Box for foot
            geom = ET.SubElement(body, 'geom')
            geom.set('name', f'{bone.name}_collision')
            geom.set('type', 'box')
            geom.set('size', '0.10 0.05 0.02')
            geom.set('pos', '0.05 0 -0.02')
            geom.set('rgba', '0.8 0.3 0.3 0.5')
        elif 'head' in bone_lower:
            # Sphere for head
            geom = ET.SubElement(body, 'geom')
            geom.set('name', f'{bone.name}_collision')
            geom.set('type', 'sphere')
            geom.set('size', '0.08')
            geom.set('rgba', '0.8 0.3 0.3 0.5')
        else:
            # Capsule for limbs
            if bone.children:
                # Calculate capsule to first child
                child = bone.children[0]
                end_pos = child.translation_mujoco
                
                # Only add capsule if there's significant distance
                length = np.linalg.norm(end_pos)
                if length > 0.01:
                    geom = ET.SubElement(body, 'geom')
                    geom.set('name', f'{bone.name}_collision')
                    geom.set('type', 'capsule')
                    
                    # Size based on body part
                    if 'thigh' in bone_lower or 'hip' in bone_lower:
                        radius = '0.06'
                    elif 'knee' in bone_lower or 'shin' in bone_lower:
                        radius = '0.05'
                    elif 'shoulder' in bone_lower or 'arm' in bone_lower:
                        radius = '0.04'
                    elif 'elbow' in bone_lower or 'forearm' in bone_lower:
                        radius = '0.03'
                    else:
                        radius = '0.04'
                        
                    geom.set('size', radius)
                    
                    # Calculate fromto
                    from_pos = np.array([0, 0, 0])
                    to_pos = end_pos * 0.9  # Slightly shorter to avoid overlap
                    fromto = np.concatenate([from_pos, to_pos])
                    geom.set('fromto', format_float_array(fromto))
                    geom.set('rgba', '0.8 0.3 0.3 0.5')
            else:
                # Small sphere for leaf nodes
                geom = ET.SubElement(body, 'geom')
                geom.set('name', f'{bone.name}_collision')
                geom.set('type', 'sphere')
                geom.set('size', '0.03')
                geom.set('rgba', '0.8 0.3 0.3 0.5')
                
        # Add visual mesh if available
        if mesh_name and self.meshes:
            geom_visual = ET.SubElement(body, 'geom')
            geom_visual.set('name', f'{bone.name}_visual')
            geom_visual.set('type', 'mesh')
            geom_visual.set('mesh', mesh_name)
            geom_visual.set('contype', '0')  # No collision
            geom_visual.set('conaffinity', '0')  # No collision
            geom_visual.set('rgba', '0.75 0.75 0.75 1')
                    
    def _add_static_meshes(self):
        """Add static meshes (when no skeleton)."""
        for mesh_data in self.meshes:
            if mesh_data.joints is None:  # Static mesh
                body = ET.SubElement(self.worldbody, 'body')
                body.set('name', f'{mesh_data.name}_body')
                
                geom = ET.SubElement(body, 'geom')
                geom.set('name', f'{mesh_data.name}_geom')
                geom.set('type', 'mesh')
                geom.set('mesh', mesh_data.name)
                geom.set('rgba', '0.8 0.8 0.8 1')
                
    def _add_actuators(self):
        """Add actuators for joints."""
        if not self.joint_refs:
            return
            
        self.actuator = ET.SubElement(self.root, 'actuator')
        
        for joint_name in self.joint_refs:
            motor = ET.SubElement(self.actuator, 'motor')
            motor.set('name', f'{joint_name}_motor')
            motor.set('joint', joint_name)
            
            # Set gear and control limits based on joint type
            joint_lower = joint_name.lower()
            
            if 'hip' in joint_lower or 'knee' in joint_lower:
                # Stronger joints
                motor.set('gear', '150')
                motor.set('ctrllimited', 'true')
                motor.set('ctrlrange', '-150 150')
            elif 'ankle' in joint_lower:
                # Medium strength
                motor.set('gear', '50')
                motor.set('ctrllimited', 'true')
                motor.set('ctrlrange', '-50 50')
            elif 'shoulder' in joint_lower or 'elbow' in joint_lower:
                # Arm joints
                motor.set('gear', '50')
                motor.set('ctrllimited', 'true')
                motor.set('ctrlrange', '-50 50')
            else:
                # Default
                motor.set('gear', '30')
                motor.set('ctrllimited', 'true')
                motor.set('ctrlrange', '-30 30')
            
        logger.info(f"Added {len(self.joint_refs)} actuators")
        
    def _add_sensors(self):
        """Add optional sensors."""
        # Could add joint position/velocity sensors, IMU, etc.
        pass
        
    def _add_keyframes(self):
        """Add default keyframes."""
        if not self.skeleton or not self.joint_refs:
            return
            
        keyframe = ET.SubElement(self.root, 'keyframe')
        
        # Add standing pose
        key = ET.SubElement(keyframe, 'key')
        key.set('name', 'standing')
        
        # Build qpos string
        # Start with freejoint pos/quat (7 values)
        qpos_values = ['0', '0', '0.8', '1', '0', '0', '0']
        
        # Add zeros for all other joints
        n_joints = len(self.joint_refs)
        qpos_values.extend(['0'] * n_joints)
        
        key.set('qpos', ' '.join(qpos_values))
        
    def save(self, output_path: str):
        """
        Save MuJoCo XML to file.
        
        Args:
            output_path: Output file path
        """
        if self.root is None:
            raise ValueError("XML not generated yet. Call generate() first.")
            
        # Convert to pretty-printed string
        xml_str = ET.tostring(self.root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines[1:])  # Skip XML declaration
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(pretty_xml)
        
        logger.info(f"Saved MuJoCo XML to: {output_file}")
