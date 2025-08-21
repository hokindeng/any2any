"""
Joint-Centric GLB to MuJoCo Importer

This module implements the joint-centric import where:
- GLB nodes are interpreted as MuJoCo joints
- Bodies are created between joints based on translation offsets
- Animation channels map directly to joint DOFs
"""

import numpy as np
import mujoco as mj  # Avoid conflict with local mujoco module
from typing import List, Dict, Optional, Tuple
import pygltflib
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger(__name__)


class JointFromNode:
    """Represents a MuJoCo joint extracted from a GLB node."""
    
    def __init__(self, node_idx: int, node: pygltflib.Node, gltf: pygltflib.GLTF2):
        """Initialize from GLB node."""
        self.node_idx = node_idx
        self.node = node
        self.name = node.name or f"joint_{node_idx}"
        
        # Translation defines body offset
        self.body_offset = node.translation if node.translation else [0, 0, 0]
        
        # Check for metadata in extras field
        if node.extras:
            self.joint_type = node.extras.get('joint_type')
            self.joint_axis = node.extras.get('joint_axis', [0, 1, 0])  # Default Y axis
            self.body_id = node.extras.get('body_id')
            self.dof_count = node.extras.get('dof_count', 1)
        else:
            # Fallback: Determine joint type from animation channels
            self.has_translation_channel = False
            self.has_rotation_channel = False
            
            if gltf.animations:
                for anim in gltf.animations:
                    for channel in anim.channels:
                        if channel.target.node == node_idx:
                            if channel.target.path == "translation":
                                self.has_translation_channel = True
                            elif channel.target.path == "rotation":
                                self.has_rotation_channel = True
            
            # Provisional inference (final decision after we know hierarchy):
            # - Only the top-most root may become 'free'
            # - If both channels present on non-root, prefer 'hinge' (treat translation as FK)
            if self.has_translation_channel and self.has_rotation_channel:
                self.joint_type = "provisional_free"
            elif self.has_translation_channel:
                self.joint_type = "slide"  # 1 DOF
            elif self.has_rotation_channel:
                self.joint_type = "hinge"  # 1 DOF
            else:
                self.joint_type = None  # No joint, just a fixed attachment
                
            self.joint_axis = [0, 1, 0]  # Default Y axis
            self.dof_count = 7 if self.joint_type == "free" else 1
            
        # Parent and children
        self.parent = None
        self.children = []
        
        # Animation data
        self.translation_data = None
        self.rotation_data = None
        
    def is_root(self) -> bool:
        """Check if this is a root joint."""
        return self.parent is None
        
    def has_body_offset(self) -> bool:
        """Check if this joint creates a new body (non-zero translation)."""
        return any(abs(x) > 1e-6 for x in self.body_offset)


class JointCentricImporter:
    """Import GLB to MuJoCo using joint-centric architecture."""
    
    def __init__(self, glb_path: str):
        """Initialize importer with GLB file."""
        self.gltf = pygltflib.GLTF2.load(str(glb_path))
        self.joints = []
        self.joint_map = {}
        
        # Track which joints are actually added to MuJoCo XML
        self.joints_in_xml = set()
        
        # Detect coordinate system
        self.coordinate_system = self._detect_coordinate_system()
        logger.info(f"Detected coordinate system: {self.coordinate_system}")
        
        # Build joint hierarchy from nodes
        self._build_joint_hierarchy()
        
        # Extract animation data
        self._extract_animation()
    
    def _detect_coordinate_system(self):
        """Detect if GLB uses Y-up or Z-up coordinate system."""
        # Check for Blender in generator (strong hint of Z-up)
        if self.gltf.asset and self.gltf.asset.generator:
            if 'Blender' in self.gltf.asset.generator:
                return 'Z-up'
        
        # Analyze node translations for vertical patterns
        y_offsets = []
        z_offsets = []
        
        for node in self.gltf.nodes:
            if node.translation:
                y_offsets.append(abs(node.translation[1]))
                z_offsets.append(abs(node.translation[2]))
        
        # If significant offsets in Z but not Y, likely Z-up
        max_y = max(y_offsets) if y_offsets else 0
        max_z = max(z_offsets) if z_offsets else 0
        
        if max_z > 0.1 and max_y < 0.01:
            return 'Z-up'
        elif max_y > 0.1 and max_z < 0.01:
            return 'Y-up'
        
        # Default to Y-up (GLTF standard)
        return 'Y-up'
        
    def _build_joint_hierarchy(self):
        """Build joint hierarchy from GLB nodes."""
        # Create JointFromNode for each node
        for i, node in enumerate(self.gltf.nodes):
            joint = JointFromNode(i, node, self.gltf)
            self.joints.append(joint)
            self.joint_map[i] = joint
            
        # Establish parent-child relationships
        for i, node in enumerate(self.gltf.nodes):
            if node.children:
                for child_idx in node.children:
                    self.joint_map[child_idx].parent = self.joint_map[i]
                    self.joint_map[i].children.append(self.joint_map[child_idx])
                    
        # Find root joints
        self.root_joints = [j for j in self.joints if j.is_root()]
        
        logger.info(f"Built joint hierarchy with {len(self.joints)} joints, {len(self.root_joints)} roots")
        
        # Resolve joint types properly:
        # - Root joint should be free (even if no animation)
        # - Joints with rotation should be ball (not hinge)
        # - Only use hinge if explicitly specified or single-axis rotation
        root_free_assigned = False
        for joint in self.joints:
            # Force root to be free joint
            if joint.is_root() and not root_free_assigned:
                joint.joint_type = "free"
                root_free_assigned = True
                logger.info(f"Set root joint {joint.name} as free")
            elif joint.joint_type == "provisional_free":
                # Non-root with both channels - make it ball
                joint.joint_type = "ball"
                joint.has_translation_channel = False
            elif joint.joint_type == "hinge" and joint.has_rotation_channel:
                # Convert hinge to ball for proper 3D rotation
                # (keep as hinge only if explicitly marked or constrained)
                joint.joint_type = "ball"
                logger.debug(f"Converted {joint.name} from hinge to ball")
        
    def _extract_animation(self):
        """Extract animation data from GLB."""
        if not self.gltf.animations:
            logger.info("No animations found")
            return

        # Helper: read accessor into numpy array
        def read_accessor(acc_idx: int) -> np.ndarray:
            acc = self.gltf.accessors[acc_idx]
            bv = self.gltf.bufferViews[acc.bufferView]
            buf = self.gltf.binary_blob()
            start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            ncomp = 4 if acc.type == pygltflib.VEC4 else 3 if acc.type == pygltflib.VEC3 else 2 if acc.type == pygltflib.VEC2 else 1
            nbytes = acc.count * ncomp * 4
            raw = buf[start:start + nbytes]
            arr = np.frombuffer(raw, dtype=np.float32).reshape(acc.count, ncomp)
            return arr

        # For simplicity, use first animation
        anim = self.gltf.animations[0]
        
        # Initialize storage
        for j in self.joints:
            j.translation_data = None
            j.rotation_data = None
            j.time_input = None

        # Extract per-node channels
        for channel in anim.channels:
            node_idx = channel.target.node
            joint = self.joint_map.get(node_idx)
            if not joint:
                continue
            sampler = anim.samplers[channel.sampler]
            # Read times
            if sampler.input is not None:
                joint.time_input = read_accessor(sampler.input)
            # Read values
            if channel.target.path == "translation" and sampler.output is not None:
                joint.translation_data = read_accessor(sampler.output)
            elif channel.target.path == "rotation" and sampler.output is not None:
                joint.rotation_data = read_accessor(sampler.output)
                    
    def export_to_mujoco(self, output_path: str):
        """Export to MuJoCo XML file."""
        logger.info(f"Exporting to MuJoCo XML: {output_path}")
        
        # Clear the set of joints that will be added to XML
        self.joints_in_xml.clear()
        
        # Create root element
        root = ET.Element("mujoco")
        
        # Add compiler settings for better visualization
        compiler = ET.SubElement(root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("autolimits", "true")
        
        # Add options for better defaults
        option = ET.SubElement(root, "option")
        option.set("gravity", "0 0 -9.81")  # Z-up gravity
        option.set("timestep", "0.01")
        
        # Add visual settings
        visual = ET.SubElement(root, "visual")
        headlight = ET.SubElement(visual, "headlight")
        headlight.set("diffuse", "0.6 0.6 0.6")
        headlight.set("ambient", "0.3 0.3 0.3")
        headlight.set("specular", "0 0 0")
        
        # Add assets for better visualization
        asset = ET.SubElement(root, "asset")
        texture = ET.SubElement(asset, "texture")
        texture.set("name", "grid")
        texture.set("type", "2d")
        texture.set("builtin", "checker")
        texture.set("width", "512")
        texture.set("height", "512")
        texture.set("rgb1", "0.1 0.2 0.3")
        texture.set("rgb2", "0.2 0.3 0.4")
        
        material = ET.SubElement(asset, "material")
        material.set("name", "grid")
        material.set("texture", "grid")
        material.set("texrepeat", "1 1")
        material.set("texuniform", "true")
        material.set("reflectance", "0.2")
        
        worldbody = ET.SubElement(root, "worldbody")
        
        # Add ground plane
        ground = ET.SubElement(worldbody, "geom")
        ground.set("name", "floor")
        ground.set("type", "plane")
        ground.set("size", "10 10 0.1")
        ground.set("material", "grid")
        ground.set("pos", "0 0 0")
        
        # Add light
        light = ET.SubElement(worldbody, "light")
        light.set("name", "spotlight")
        light.set("directional", "true")
        light.set("diffuse", "0.8 0.8 0.8")
        light.set("specular", "0.2 0.2 0.2")
        light.set("pos", "0 -2 4")
        light.set("dir", "0 1 -1")
        
        # Process each root joint
        for root_joint in self.root_joints:
            self._add_joint_tree(worldbody, root_joint, is_first=True)
            
        # Add actuators for all joints in the model
        if self.joints_in_xml:
            actuator = ET.SubElement(root, "actuator")
            for joint_name in sorted(self.joints_in_xml):
                # Find the joint to determine its type
                joint = next((j for j in self.joints if j.name == joint_name), None)
                if joint:
                    # Create motor actuator with appropriate gear ratio
                    motor = ET.SubElement(actuator, "motor")
                    motor.set("name", f"{joint_name}_motor")
                    motor.set("joint", joint_name)
                    
                    # Set gear ratio based on joint type
                    if joint.joint_type == "free":
                        # Free joints need 6 actuators (3 position + 3 rotation)
                        # MuJoCo will handle this automatically
                        motor.set("gear", "250")
                    elif joint.joint_type == "ball":
                        # Ball joints need higher torque
                        motor.set("gear", "100")
                    else:  # hinge or slide
                        motor.set("gear", "50")
                    
                    # Add control range
                    motor.set("ctrllimited", "true")
                    motor.set("ctrlrange", "-1 1")
                    
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(xml_str)
            
        logger.info(f"Exported MuJoCo XML with {len(self.joints)} joints")

        # Cache model nq to ensure motion export matches exactly
        try:
            model = mj.MjModel.from_xml_path(output_path)
            self.exported_model_nq = int(model.nq)
            logger.info(f"Cached model DOFs (nq): {self.exported_model_nq}")
        except Exception as e:
            logger.warning(f"Could not load model to cache nq: {e}")
        
    def _add_joint_tree(self, parent_element: ET.Element, joint: JointFromNode, is_first: bool = False):
        """Recursively add joint and its children to XML."""
        
        # Special handling for Hip-Pelvis situation
        # If this is Pelvis and it has no offset from Hip, add a small artificial offset
        # to allow both joints to exist in the physics simulation
        if joint.name == "Pelvis" and not joint.has_body_offset():
            # Add 1mm offset in Y direction to separate them physically
            # This preserves both animations while being physically valid
            logger.info(f"Adding small offset to {joint.name} to preserve independent animation")
            joint.body_offset = [0.0, 0.001, 0.0]  # 1mm offset
        
        # Decide whether to create a new body based on:
        # 1. If this is the first/root joint (always needs a body)
        # 2. If there's a translation offset from parent (bone length)
        # 3. If this is a free joint (needs its own body for proper DOFs)
        # 4. Special case: Pelvis joints always need their own body for animation
        # Note: Ball joints only need a new body if they have an offset
        is_pelvis = joint.name.lower() == 'pelvis'
        needs_new_body = is_first or joint.has_body_offset() or joint.joint_type == "free" or is_pelvis
        
        # If pelvis has no offset, give it a small one to ensure it gets its own body
        if is_pelvis and not joint.has_body_offset():
            joint.body_offset = [0.0, 0.001, 0.0]  # 1mm offset
            logger.info(f"Added small offset to {joint.name} to ensure separate body")
        
        if needs_new_body:
            # Create body element
            body_name = f"{joint.name}_body"
            body = ET.SubElement(parent_element, "body")
            body.set("name", body_name)
            
            # Set body position from joint's translation
            if joint.has_body_offset():
                x, y, z = joint.body_offset
                
                # Apply coordinate transformation based on source system
                if self.coordinate_system == 'Y-up':
                    # Convert from GLTF Y-up to MuJoCo Z-up:
                    # GLTF: X=right, Y=up, Z=forward
                    # MuJoCo: X=forward, Y=left, Z=up
                    # Transform: GLTF(X,Y,Z) -> MuJoCo(Z, -X, Y)
                    mujoco_pos = [z, -x, y]
                else:
                    # Source is already Z-up (like MuJoCo), no transformation needed!
                    # Both use: X=forward, Y=left, Z=up
                    mujoco_pos = [x, y, z]
                
                body.set("pos", f"{mujoco_pos[0]} {mujoco_pos[1]} {mujoco_pos[2]}")
            # Note: If no offset, body stays at parent's position (no pos attribute needed)
                
            # Add geom for visualization - use capsule for better skeleton viz
            if joint.children:  # Has children, so it's not an end effector
                # Calculate capsule length based on children offsets
                max_child_dist = 0
                for child in joint.children:
                    if child.has_body_offset():
                        dist = np.linalg.norm(child.body_offset)
                        max_child_dist = max(max_child_dist, dist)
                
                if max_child_dist > 0.01:  # Only create capsule if meaningful length
                    geom = ET.SubElement(body, "geom")
                    geom.set("type", "capsule")
                    geom.set("size", f"0.02")  # radius
                    # Point toward first child with offset
                    for child in joint.children:
                        if child.has_body_offset():
                            cx, cy, cz = child.body_offset
                            # Transform child offset to MuJoCo coords
                            if self.coordinate_system == 'Y-up':
                                mx, my, mz = cz, -cx, cy
                            else:
                                mx, my, mz = cx, cy, cz
                            length = np.sqrt(mx*mx + my*my + mz*mz) * 0.9
                            geom.set("fromto", f"0 0 0 {mx*0.9} {my*0.9} {mz*0.9}")
                            break
                    else:
                        geom.set("fromto", f"0 0 0 0 0 {max_child_dist * 0.9}")
                    geom.set("rgba", "0.9 0.9 0.9 0.8")
                else:
                    # Small sphere for joints with no offset children
                    geom = ET.SubElement(body, "geom")
                    geom.set("type", "sphere")
                    geom.set("size", "0.02")
                    geom.set("rgba", "0.8 0.8 0.8 0.8")
            else:
                # End effector - small sphere
                geom = ET.SubElement(body, "geom")
                geom.set("type", "sphere")
                geom.set("size", "0.015")
                geom.set("rgba", "1 0.5 0.5 0.8")
            
            current_element = body
        else:
            # No translation offset - add joint to parent body
            current_element = parent_element
            
        # Add joint if it has DOFs
        if joint.joint_type:
            joint_elem = ET.SubElement(current_element, "joint", name=joint.name)
            
            # Track that this joint was actually added to the XML
            self.joints_in_xml.add(joint.name)
            
            if joint.joint_type == "free":
                joint_elem.set("type", "free")
            elif joint.joint_type == "hinge":
                joint_elem.set("type", "hinge")
                # Transform axis from GLTF to MuJoCo coordinates
                gx, gy, gz = joint.joint_axis
                if self.coordinate_system == 'Y-up':
                    mujoco_axis = [gz, -gx, gy]  # GLTF(X,Y,Z) -> MuJoCo(Z,-X,Y)
                else:
                    mujoco_axis = [gx, gy, gz]  # Already in MuJoCo coords
                axis_str = f"{mujoco_axis[0]} {mujoco_axis[1]} {mujoco_axis[2]}"
                joint_elem.set("axis", axis_str)
            elif joint.joint_type == "slide":
                joint_elem.set("type", "slide")
                # Transform axis from GLTF to MuJoCo coordinates
                gx, gy, gz = joint.joint_axis
                if self.coordinate_system == 'Y-up':
                    mujoco_axis = [gz, -gx, gy]  # GLTF(X,Y,Z) -> MuJoCo(Z,-X,Y)
                else:
                    mujoco_axis = [gx, gy, gz]  # Already in MuJoCo coords
                axis_str = f"{mujoco_axis[0]} {mujoco_axis[1]} {mujoco_axis[2]}"
                joint_elem.set("axis", axis_str)
            elif joint.joint_type == "ball":
                joint_elem.set("type", "ball")
                
        # Process children
        for child in joint.children:
            self._add_joint_tree(current_element, child)
            
    def export_motion_data(self, output_path: str):
        """Export animation data to NPY file."""
        # Determine frame count from animation samplers - use the maximum frame count
        n_frames = 1
        fps = 30.0
        if self.gltf.animations:
            try:
                anim = self.gltf.animations[0]
                counts = []
                for s in anim.samplers:
                    if s.input is not None and s.input < len(self.gltf.accessors):
                        acc = self.gltf.accessors[s.input]
                        if hasattr(acc, 'count') and acc.count:
                            counts.append(acc.count)
                if counts:
                    n_frames = max(counts)  # Use the maximum frame count (should be 32)
                    logger.info(f"Found {n_frames} frames in animation (max from {len(counts)} samplers)")
            except Exception as e:
                logger.warning(f"Could not determine frame count: {e}")
                n_frames = 1

        # Build zero-trajectory matching model DOFs so that viewer loads cleanly
        # Prefer exact nq from exported model if available
        if hasattr(self, 'exported_model_nq'):
            nq = self.exported_model_nq
        else:
            try:
                # Infer from joint types as a fallback
                nq = 0
                for joint in self.joints:
                    if joint.joint_type == 'free':
                        nq += 7
                    elif joint.joint_type == 'ball':
                        nq += 4
                    elif joint.joint_type in ('hinge', 'slide'):
                        nq += 1
                if nq <= 0:
                    nq = 1
            except Exception:
                nq = 1

        # Build qpos per frame in joint order
        qpos = np.zeros((n_frames, nq), dtype=np.float32)
        
        # Get the actual time values from the animation
        # Find the time array that matches n_frames (the full animation timeline)
        actual_times = None
        for j in self.joints:
            if getattr(j, 'time_input', None) is not None:
                if len(j.time_input) == n_frames:
                    # Found the full timeline!
                    actual_times = j.time_input
                    logger.info(f"Using time array from joint {j.name} with {len(actual_times)} frames")
                    break
        
        # If we didn't find one matching n_frames, use the longest one
        if actual_times is None:
            max_len = 0
            for j in self.joints:
                if getattr(j, 'time_input', None) is not None and len(j.time_input) > max_len:
                    max_len = len(j.time_input)
                    actual_times = j.time_input
                    logger.info(f"Using longest time array from joint {j.name} with {len(actual_times)} frames")
        
        if actual_times is not None and len(actual_times) > 1:
            # Use the exact times from the GLB
            if len(actual_times) == n_frames:
                time = actual_times.astype(np.float32).flatten()  # Ensure 1D array
            else:
                # Need to interpolate or resample to match n_frames
                logger.warning(f"Time array has {len(actual_times)} samples but need {n_frames}")
                # Simple linear interpolation to n_frames
                start_time = float(actual_times.flat[0]) if actual_times.ndim > 1 else float(actual_times[0])
                end_time = float(actual_times.flat[-1]) if actual_times.ndim > 1 else float(actual_times[-1])
                time = np.linspace(start_time, end_time, n_frames, dtype=np.float32)
            
            # Calculate FPS from the actual time values
            duration = float(time[-1] - time[0])
            fps = (len(time) - 1) / duration if duration > 0 else 30.0
        else:
            # Fallback if no time data
            fps = 30.0
            time = np.arange(n_frames, dtype=np.float32) / fps

        # Fill qpos
        col = 0
        for j in self.joints:
            jt = j.joint_type
            # Helper to get rotation quaternion WXYZ from GLTF XYZW per frame idx
            def get_quat_wxyz(frame_idx: int) -> np.ndarray:
                if getattr(j, 'rotation_data', None) is None:
                    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
                # Handle interpolation if we have fewer keyframes than n_frames
                if len(j.rotation_data) < n_frames and len(j.rotation_data) >= 2:
                    # SLERP between keyframes for smooth interpolation
                    key_idx = frame_idx * (len(j.rotation_data) - 1) / (n_frames - 1)
                    idx0 = int(np.floor(key_idx))
                    idx1 = min(idx0 + 1, len(j.rotation_data) - 1)
                    alpha = key_idx - idx0
                    
                    q0 = j.rotation_data[idx0]  # XYZW format
                    q1 = j.rotation_data[idx1]  # XYZW format
                    
                    # Check dot product to ensure shortest path
                    dot = np.dot(q0, q1)
                    if dot < 0:
                        q1 = -q1
                        dot = -dot
                    
                    # SLERP
                    if dot > 0.9995:
                        # Linear for very close quaternions
                        xyzw = (1 - alpha) * q0 + alpha * q1
                    else:
                        theta = np.arccos(np.clip(dot, -1, 1))
                        sin_theta = np.sin(theta)
                        w0 = np.sin((1 - alpha) * theta) / sin_theta
                        w1 = np.sin(alpha * theta) / sin_theta
                        xyzw = w0 * q0 + w1 * q1
                    
                    # Normalize
                    norm = np.linalg.norm(xyzw)
                    if norm > 0:
                        xyzw = xyzw / norm
                else:
                    # Direct indexing if we have enough frames
                    xyzw = j.rotation_data[min(frame_idx, len(j.rotation_data)-1)]
                
                # Convert from GLTF XYZW to WXYZ format
                # Note: We're NOT doing coordinate transformation here for hinge joints
                # as they need the original rotation to extract the angle
                return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)

            # Helper to get translation vector
            def get_trans(frame_idx: int) -> np.ndarray:
                if getattr(j, 'translation_data', None) is None:
                    return np.zeros(3, dtype=np.float32)
                
                # Handle interpolation if we have fewer keyframes than n_frames
                if len(j.translation_data) < n_frames and len(j.translation_data) >= 2:
                    # Linear interpolation between keyframes
                    key_idx = frame_idx * (len(j.translation_data) - 1) / (n_frames - 1)
                    idx0 = int(np.floor(key_idx))
                    idx1 = min(idx0 + 1, len(j.translation_data) - 1)
                    t = key_idx - idx0
                    
                    v = (1 - t) * j.translation_data[idx0] + t * j.translation_data[idx1]
                else:
                    # Direct indexing if we have enough frames
                    v = j.translation_data[min(frame_idx, len(j.translation_data)-1)]
                
                # Transform from GLTF to MuJoCo coordinates
                gltf_x, gltf_y, gltf_z = v
                mujoco_trans = np.array([gltf_z, -gltf_x, gltf_y], dtype=np.float32)
                return mujoco_trans

            if jt == 'free':
                for f in range(n_frames):
                    t = get_trans(f)
                    q = get_quat_wxyz(f)
                    # TODO: Also need to transform quaternion from GLTF to MuJoCo frame
                    qpos[f, col:col+3] = t
                    qpos[f, col+3:col+7] = q
                col += 7
            elif jt == 'ball':
                for f in range(n_frames):
                    q = get_quat_wxyz(f)
                    qpos[f, col:col+4] = q
                col += 4
            elif jt == 'hinge':
                # Import the proper transformation
                from .motion_transform import extract_hinge_angle_from_quaternion, transform_coordinate_system
                
                # Get joint axis in GLTF coords and transform to MuJoCo
                axis_gltf = np.array(getattr(j, 'joint_axis', [0,1,0]), dtype=np.float32)
                axis_mujoco = transform_coordinate_system(axis_gltf, from_gltf=True)
                axis_mujoco = axis_mujoco / (np.linalg.norm(axis_mujoco) + 1e-8)
                
                for f in range(n_frames):
                    q_wxyz = get_quat_wxyz(f)
                    # Already in MuJoCo frame from get_quat_wxyz, extract angle
                    angle = extract_hinge_angle_from_quaternion(q_wxyz, axis_mujoco)
                    qpos[f, col] = float(angle)
                col += 1
            elif jt == 'slide':
                axis = np.array(getattr(j, 'joint_axis', [1,0,0]), dtype=np.float32)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                for f in range(n_frames):
                    t = get_trans(f)
                    qpos[f, col] = float(np.dot(t, axis))
                col += 1

        # If any discrepancy (due to conservative downgrades), trim to nq
        if qpos.shape[1] != nq:
            if qpos.shape[1] > nq:
                qpos = qpos[:, :nq]
            else:
                pad = np.zeros((n_frames, nq - qpos.shape[1]), dtype=np.float32)
                qpos = np.concatenate([qpos, pad], axis=1)

        motion_data = {
            'qpos': qpos,
            'fps': float(fps),
            'time': time.astype(np.float32),  # Exact time values from GLB
            'source': 'glb_import',
            'original_frames': n_frames,
        }
        
        np.save(output_path, motion_data)
        logger.info(f"Exported motion data to {output_path}")


def test_joint_centric_import():
    """Test the joint-centric importer."""
    
    # First, make sure we have a GLB file to import
    glb_path = "temporary/test_joint_centric.glb"
    
    if not Path(glb_path).exists():
        print(f"❌ Test GLB not found: {glb_path}")
        print("Run the exporter test first to create it")
        return
        
    # Import the GLB
    importer = JointCentricImporter(glb_path)
    
    # Export to MuJoCo
    output_xml = "temporary/test_imported.xml"
    importer.export_to_mujoco(output_xml)
    
    print(f"✅ Import successful: {output_xml}")
    
    # Display the generated XML
    with open(output_xml, 'r') as f:
        lines = f.readlines()
        print("\nGenerated MuJoCo XML (first 30 lines):")
        print("=" * 50)
        for line in lines[:30]:
            print(line.rstrip())
            
    # Check if it's valid MuJoCo
    try:
        model = mj.MjModel.from_xml_path(output_xml)
        print(f"\n✅ Valid MuJoCo model with {model.njnt} joints, {model.nbody} bodies")
    except Exception as e:
        print(f"\n❌ Invalid MuJoCo model: {e}")


if __name__ == "__main__":
    test_joint_centric_import()
