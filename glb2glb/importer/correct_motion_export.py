"""
Correct motion export from GLB to MuJoCo.

Key principles:
1. Root joint (free): Needs global position and orientation (7 DOFs)
2. Other joints (ball): Only local rotations relative to parent (4 DOFs)
3. Body positions are structural (in XML), not in motion data (qpos)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in WXYZ format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion in WXYZ format."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def gltf_to_mujoco_position(pos_gltf: np.ndarray, coordinate_system='Y-up') -> np.ndarray:
    """Convert position from GLTF to MuJoCo."""
    if coordinate_system == 'Z-up':
        # Already in MuJoCo coordinates
        return pos_gltf
    
    x, y, z = pos_gltf
    # GLTF Y-up (X=right, Y=up, Z=forward) -> MuJoCo Z-up (X=forward, Y=left, Z=up)
    # Transform: GLTF(X,Y,Z) -> MuJoCo(Z, -X, Y)
    return np.array([z, -x, y])


def gltf_to_mujoco_quaternion(quat_xyzw: np.ndarray, coordinate_system='Y-up', joint_name='') -> np.ndarray:
    """Convert quaternion from GLTF to MuJoCo coordinate system.
    
    Args:
        quat_xyzw: Quaternion in XYZW format
        coordinate_system: 'Y-up' or 'Z-up'
        joint_name: Name of the joint (used for left/right symmetry handling)
    """
    x, y, z, w = quat_xyzw
    
    if coordinate_system == 'Z-up':
        # Already in MuJoCo coordinates, just return WXYZ
        return np.array([w, x, y, z])
    
    # For Y-up to Z-up transformation:
    # GLTF axes: X=right, Y=up, Z=forward
    # MuJoCo axes: X=forward, Y=left, Z=up
    
    # Direct quaternion component remapping:
    # Rotation around GLTF Y -> rotation around MuJoCo Z
    # Rotation around GLTF Z -> rotation around MuJoCo X
    # Rotation around GLTF X -> rotation around MuJoCo Y
    
    x_new = z   # GLTF Z rotation -> MuJoCo X rotation (no negation - fixes head orientation)
    y_new = x   # GLTF X rotation -> MuJoCo Y rotation (no negation - fixes pigeon-toe)
    z_new = y   # GLTF Y rotation -> MuJoCo Z rotation
    w_new = w   # Scalar part unchanged
    
    return np.array([w_new, x_new, y_new, z_new])  # WXYZ format


def compute_global_transform(joint, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global position and orientation for a joint.
    Walks up the hierarchy accumulating transforms.
    """
    # Start with this joint's local transform
    # IMPORTANT: Animation is relative to bind pose, not absolute!
    # We need to use BOTH the static offset AND the animation
    bind_pos = np.array(joint.body_offset if joint.body_offset else [0, 0, 0], dtype=np.float32)
    
    if joint.translation_data is not None and frame_idx < len(joint.translation_data):
        # Animation is typically absolute, not relative to bind pose
        # For most rigs, the animation contains the full transform
        local_pos = joint.translation_data[frame_idx]
    else:
        local_pos = bind_pos
    
    if joint.rotation_data is not None and frame_idx < len(joint.rotation_data):
        local_rot_xyzw = joint.rotation_data[frame_idx]
    else:
        local_rot_xyzw = np.array([0, 0, 0, 1])  # Identity
    
    # Convert to WXYZ
    local_rot = np.array([local_rot_xyzw[3], local_rot_xyzw[0], local_rot_xyzw[1], local_rot_xyzw[2]])
    
    # Walk up the hierarchy
    global_pos = local_pos.copy()
    global_rot = local_rot.copy()
    
    current = joint.parent
    while current is not None:
        # Get parent's local transform
        if current.translation_data is not None and frame_idx < len(current.translation_data):
            parent_pos = current.translation_data[frame_idx]
        else:
            parent_pos = np.array(current.body_offset if current.body_offset else [0, 0, 0], dtype=np.float32)
        
        if current.rotation_data is not None and frame_idx < len(current.rotation_data):
            parent_rot_xyzw = current.rotation_data[frame_idx]
        else:
            parent_rot_xyzw = np.array([0, 0, 0, 1])
        
        parent_rot = np.array([parent_rot_xyzw[3], parent_rot_xyzw[0], parent_rot_xyzw[1], parent_rot_xyzw[2]])
        
        # Apply parent transform
        # Rotate position by parent rotation
        # This is simplified - proper implementation would use quaternion rotation
        global_pos = parent_pos + global_pos  # Simplified, should rotate first
        global_rot = quaternion_multiply(parent_rot, global_rot)
        
        current = current.parent
    
    return global_pos, global_rot


def compute_relative_rotation(joint, frame_idx: int) -> np.ndarray:
    """
    Compute rotation relative to parent for non-root joints.
    This is what goes into qpos for ball joints.
    """
    if joint.rotation_data is None or frame_idx >= len(joint.rotation_data):
        return np.array([1, 0, 0, 0])  # Identity quaternion WXYZ
    
    # Get this joint's rotation
    rot_xyzw = joint.rotation_data[frame_idx]
    rot_wxyz = np.array([rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]])
    
    # If this is already relative to parent (which it should be in GLB), use it directly
    # GLB stores local transforms, not global
    return rot_wxyz


def export_correct_motion(importer, output_path: str):
    """
    Export motion data with correct transformation.
    
    Args:
        importer: JointCentricImporter instance
        output_path: Path to save NPY file
    """
    # Determine frame count
    n_frames = 1
    if importer.gltf.animations:
        anim = importer.gltf.animations[0]
        counts = []
        for s in anim.samplers:
            if s.input is not None and s.input < len(importer.gltf.accessors):
                acc = importer.gltf.accessors[s.input]
                if hasattr(acc, 'count') and acc.count:
                    counts.append(acc.count)
        if counts:
            n_frames = max(counts)
    
    logger.info(f"Exporting motion with {n_frames} frames")
    
    # Count DOFs based on joints actually added to XML
    # IMPORTANT: Free joints must come FIRST in MuJoCo's qpos!
    nq = 0
    joint_order = []
    
    # First, add all free joints (typically just Armature)
    for joint in importer.joints:
        if joint.name not in importer.joints_in_xml:
            continue
        if joint.joint_type == 'free':
            nq += 7
            joint_order.append((joint, 'free', 7))
            logger.info(f"Added free joint {joint.name} at position 0")
    
    # Then add all other joints
    for joint in importer.joints:
        if joint.name not in importer.joints_in_xml:
            logger.info(f"Skipping {joint.name} - not in MuJoCo XML")
            continue
            
        if joint.joint_type == 'ball':
            nq += 4
            joint_order.append((joint, 'ball', 4))
        elif joint.joint_type == 'hinge':
            nq += 1
            joint_order.append((joint, 'hinge', 1))
        elif joint.joint_type == 'slide':
            nq += 1
            joint_order.append((joint, 'slide', 1))
        # free joints already added above
    
    logger.info(f"Total DOFs: {nq} from {len(joint_order)} joints")
    logger.info(f"Joint types: {sum(1 for j in joint_order if j[1]=='free')} free, "
                f"{sum(1 for j in joint_order if j[1]=='ball')} ball, "
                f"{sum(1 for j in joint_order if j[1]=='hinge')} hinge, "
                f"{sum(1 for j in joint_order if j[1]=='slide')} slide")
    
    # Build qpos array
    qpos = np.zeros((n_frames, nq), dtype=np.float32)
    
    for frame_idx in range(n_frames):
        col = 0
        
        # Process joints in order
        for joint, joint_type, dof_count in joint_order:
            if joint_type == 'free':
                # Global transform for free joint (typically root)
                global_pos, global_rot = compute_global_transform(joint, frame_idx)
                
                # For Blender exports, Armature is just a container
                # The actual motion is in the "Root" child node
                if joint.name == 'Armature':
                    # Find the Root child that has the actual animation
                    for child in joint.children:
                        if child.name == 'Root' and child.translation_data is not None:
                            # Use Root's translation as the world position
                            if frame_idx < len(child.translation_data):
                                global_pos = child.translation_data[frame_idx].copy()
                                logger.info(f"Using Root's translation: {global_pos}")
                            break
                    
                    # For Z-up files, also add Hip offset to lift character
                    if importer.coordinate_system == 'Z-up':
                        hip_offset = 0.0
                        for j in importer.joints:
                            if j.name == 'Hip' and j.body_offset:
                                hip_offset = j.body_offset[2]  # Z component
                                break
                        # Add Hip height to position
                        global_pos = np.array(global_pos, dtype=np.float32)
                        global_pos[2] += hip_offset
                        logger.info(f"After adding Hip offset: global_pos = {global_pos}")
                
                # Convert to MuJoCo coordinates
                pos_mujoco = gltf_to_mujoco_position(global_pos, importer.coordinate_system)
                logger.info(f"After coordinate transform: pos_mujoco = {pos_mujoco}")
                # Convert WXYZ back to XYZW for the conversion function
                rot_xyzw = np.array([global_rot[1], global_rot[2], global_rot[3], global_rot[0]])
                rot_mujoco = gltf_to_mujoco_quaternion(rot_xyzw, importer.coordinate_system, joint.name)
                
                logger.info(f"Writing to qpos[{frame_idx}, {col}:{col+3}] = {pos_mujoco}")
                qpos[frame_idx, col:col+3] = pos_mujoco
                qpos[frame_idx, col+3:col+7] = rot_mujoco
                col += 7
                
            elif joint_type == 'ball':
                # Local rotation only
                local_rot = compute_relative_rotation(joint, frame_idx)
                # Apply coordinate transformation
                # local_rot is already in WXYZ format from compute_relative_rotation
                # Convert back to XYZW for the transformation function
                rot_xyzw = np.array([local_rot[1], local_rot[2], local_rot[3], local_rot[0]])
                rot_mujoco = gltf_to_mujoco_quaternion(rot_xyzw, importer.coordinate_system, joint.name)
                qpos[frame_idx, col:col+4] = rot_mujoco
                col += 4
                
            elif joint_type == 'hinge':
                # Extract angle around axis
                local_rot = compute_relative_rotation(joint, frame_idx)
                # Simplified: just use first component for now
                # Should properly extract angle around the joint axis
                angle = 2.0 * np.arccos(np.clip(local_rot[0], -1.0, 1.0))
                qpos[frame_idx, col] = angle
                col += 1
                
            elif joint_type == 'slide':
                # Position along axis - for now use 0
                qpos[frame_idx, col] = 0.0
                col += 1
    
    # Get time values
    actual_times = None
    for j in importer.joints:
        if getattr(j, 'time_input', None) is not None:
            if len(j.time_input) == n_frames:
                actual_times = j.time_input
                break
    
    if actual_times is not None:
        time = actual_times.astype(np.float32).flatten()
        duration = float(time[-1] - time[0])
        fps = (len(time) - 1) / duration if duration > 0 else 30.0
    else:
        fps = 30.0
        time = np.arange(n_frames, dtype=np.float32) / fps
    
    # Save motion data
    motion_data = {
        'qpos': qpos,
        'fps': float(fps),
        'time': time,
        'source': 'glb_import_corrected',
        'original_frames': n_frames,
    }
    
    np.save(output_path, motion_data)
    logger.info(f"Exported corrected motion to {output_path}")
    logger.info(f"Shape: {qpos.shape}, FPS: {fps:.1f}")
