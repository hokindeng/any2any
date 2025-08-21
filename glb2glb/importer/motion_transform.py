"""
Proper motion transformation from GLB to MuJoCo.

GLB Motion Representation:
- Hierarchical transforms (parent-relative)
- Each node: translation + rotation (quaternion XYZW)
- Y-up coordinate system

MuJoCo Motion Representation:
- Flat qpos array
- Joint-specific values (angles for hinges, quats for balls)
- Z-up coordinate system
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def transform_coordinate_system(vec: np.ndarray, from_gltf=True, coordinate_system='Y-up') -> np.ndarray:
    """
    Transform between GLTF and MuJoCo coordinate systems.
    
    For Y-up GLTF: X=right, Y=up, Z=forward
    For Z-up GLTF: X=forward, Y=left, Z=up (same as MuJoCo!)
    MuJoCo: X=forward, Y=left, Z=up
    
    Transform: GLTF(X,Y,Z) -> MuJoCo(Z, -X, Y) for Y-up only
    """
    if coordinate_system == 'Z-up':
        # Already in MuJoCo coordinates, no transformation needed
        return vec
    
    if from_gltf:
        # GLTF Y-up to MuJoCo Z-up
        x, y, z = vec
        return np.array([z, -x, y])
    else:
        # MuJoCo to GLTF Y-up (inverse)
        x, y, z = vec
        return np.array([-y, z, x])


def transform_quaternion(quat: np.ndarray, from_gltf=True, coordinate_system='Y-up') -> np.ndarray:
    """
    Transform quaternion between coordinate systems.
    
    This is more complex than just reordering - we need to account for
    the change in rotation axes.
    """
    if coordinate_system == 'Z-up':
        # Already in MuJoCo coordinates
        # Just need to handle XYZW vs WXYZ format
        if from_gltf:
            x, y, z, w = quat
            return np.array([w, x, y, z])  # XYZW to WXYZ
        else:
            w, x, y, z = quat
            return np.array([x, y, z, w])  # WXYZ to XYZW
    
    if from_gltf:
        # GLTF XYZW to MuJoCo WXYZ with coordinate transform
        x, y, z, w = quat
        
        # The rotation needs to be transformed to account for axis changes
        # This is a simplified version - may need more sophisticated handling
        # Transform the quaternion to match the new coordinate frame
        # GLTF Y-up to MuJoCo Z-up requires a 90-degree rotation around X
        
        # Apply coordinate system rotation
        # This rotates the quaternion to account for Y-up to Z-up conversion
        sqrt2 = np.sqrt(2) / 2
        
        # Rotation matrix for Y-up to Z-up (90 degrees around X)
        # q_transform = [sqrt2, 0, 0, sqrt2]  # 90 deg around X
        
        # Compose quaternions: q_new = q_transform * q * q_transform^-1
        # For now, simplified direct mapping
        return np.array([w, z, -x, y])  # WXYZ format with axis swap
    else:
        # MuJoCo to GLTF (inverse)
        w, x, y, z = quat
        return np.array([-y, z, x, w])  # XYZW format


def extract_hinge_angle_from_quaternion(
    quat_wxyz: np.ndarray, 
    axis: np.ndarray,
    parent_quat_wxyz: Optional[np.ndarray] = None
) -> float:
    """
    Extract the rotation angle around a specific axis from a quaternion.
    
    This is crucial for converting ball/free joint rotations to hinge angles.
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # If we have a parent quaternion, compute relative rotation
    if parent_quat_wxyz is not None:
        # Compute relative quaternion: q_rel = q_parent^-1 * q_child
        w_p, x_p, y_p, z_p = parent_quat_wxyz
        # Conjugate of parent (inverse for unit quaternions)
        parent_inv = np.array([w_p, -x_p, -y_p, -z_p])
        
        # Quaternion multiplication
        quat_wxyz = quaternion_multiply(parent_inv, quat_wxyz)
    
    w, x, y, z = quat_wxyz
    
    # Convert quaternion to axis-angle
    # For unit quaternions: q = [cos(θ/2), sin(θ/2) * axis]
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    
    if abs(angle) < 1e-6:
        return 0.0
    
    # Get rotation axis from quaternion
    s = np.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-6:
        rot_axis = np.array([x, y, z])
    else:
        rot_axis = np.array([x, y, z]) / s
    
    # Project onto joint axis to get signed angle
    # The sign indicates rotation direction
    projection = np.dot(rot_axis, axis)
    signed_angle = angle * np.sign(projection)
    
    # Handle angle wrapping
    while signed_angle > np.pi:
        signed_angle -= 2 * np.pi
    while signed_angle < -np.pi:
        signed_angle += 2 * np.pi
    
    return signed_angle


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (WXYZ format).
    q = q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def interpolate_sparse_to_dense(
    sparse_times: np.ndarray,
    sparse_values: np.ndarray,
    dense_times: np.ndarray,
    is_quaternion: bool = False
) -> np.ndarray:
    """
    Interpolate sparse keyframes to dense frames.
    
    GLB often has sparse keyframes (e.g., 2 keyframes)
    MuJoCo needs dense frames (e.g., 32 frames)
    """
    if len(sparse_times) == len(dense_times) and np.allclose(sparse_times, dense_times):
        # Already dense
        return sparse_values
    
    dense_values = np.zeros((len(dense_times), sparse_values.shape[1]))
    
    for i, t in enumerate(dense_times):
        # Find surrounding keyframes
        idx = np.searchsorted(sparse_times, t)
        
        if idx == 0:
            dense_values[i] = sparse_values[0]
        elif idx >= len(sparse_times):
            dense_values[i] = sparse_values[-1]
        else:
            # Linear interpolation
            t0, t1 = sparse_times[idx-1], sparse_times[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            
            if is_quaternion:
                # SLERP for quaternions
                q0, q1 = sparse_values[idx-1], sparse_values[idx]
                
                # Check if quaternions point in same direction
                dot = np.dot(q0, q1)
                if dot < 0:
                    q1 = -q1
                    dot = -dot
                
                # SLERP
                if dot > 0.9995:
                    # Linear interpolation for close quaternions
                    q = (1 - alpha) * q0 + alpha * q1
                else:
                    theta = np.arccos(np.clip(dot, -1, 1))
                    sin_theta = np.sin(theta)
                    w0 = np.sin((1 - alpha) * theta) / sin_theta
                    w1 = np.sin(alpha * theta) / sin_theta
                    q = w0 * q0 + w1 * q1
                
                # Normalize
                dense_values[i] = q / (np.linalg.norm(q) + 1e-8)
            else:
                # Linear interpolation for positions
                dense_values[i] = (1 - alpha) * sparse_values[idx-1] + alpha * sparse_values[idx]
    
    return dense_values


class MotionTransformer:
    """
    Handles the complete transformation from GLB to MuJoCo motion.
    """
    
    def __init__(self, joints, n_frames: int):
        self.joints = joints
        self.n_frames = n_frames
        
    def transform_frame(self, frame_idx: int) -> dict:
        """
        Transform a single frame from GLB to MuJoCo representation.
        
        Returns dict with joint values keyed by joint name.
        """
        joint_values = {}
        
        for joint in self.joints:
            if joint.joint_type is None:
                continue
                
            # Get animation data for this frame
            if joint.joint_type == 'free':
                # Free joint: position + quaternion
                pos = self.get_position(joint, frame_idx)
                quat = self.get_quaternion(joint, frame_idx)
                
                # Transform coordinates
                pos_mujoco = transform_coordinate_system(pos)
                quat_mujoco = transform_quaternion(quat)
                
                joint_values[joint.name] = {
                    'type': 'free',
                    'values': np.concatenate([pos_mujoco, quat_mujoco])
                }
                
            elif joint.joint_type == 'ball':
                # Ball joint: quaternion only
                quat = self.get_quaternion(joint, frame_idx)
                quat_mujoco = transform_quaternion(quat)
                
                joint_values[joint.name] = {
                    'type': 'ball',
                    'values': quat_mujoco
                }
                
            elif joint.joint_type == 'hinge':
                # Hinge joint: extract angle around axis
                quat = self.get_quaternion(joint, frame_idx)
                
                # Transform axis to MuJoCo coordinates
                axis_gltf = np.array(joint.joint_axis)
                axis_mujoco = transform_coordinate_system(axis_gltf)
                
                # Extract angle
                angle = extract_hinge_angle_from_quaternion(quat, axis_mujoco)
                
                joint_values[joint.name] = {
                    'type': 'hinge',
                    'values': np.array([angle])
                }
                
            elif joint.joint_type == 'slide':
                # Slide joint: position along axis
                pos = self.get_position(joint, frame_idx)
                axis_gltf = np.array(joint.joint_axis)
                axis_mujoco = transform_coordinate_system(axis_gltf)
                
                # Project position onto axis
                slide_value = np.dot(pos, axis_mujoco)
                
                joint_values[joint.name] = {
                    'type': 'slide',
                    'values': np.array([slide_value])
                }
        
        return joint_values
    
    def get_position(self, joint, frame_idx: int) -> np.ndarray:
        """Get interpolated position for frame."""
        if joint.translation_data is None:
            return np.zeros(3)
        
        # Handle sparse keyframes
        if len(joint.translation_data) < self.n_frames:
            # Need to interpolate
            if joint.time_input is not None:
                # Use actual times for interpolation
                target_time = frame_idx / (self.n_frames - 1) * joint.time_input[-1]
                # Find surrounding keyframes and interpolate
                # ... (interpolation logic)
            else:
                # Simple linear interpolation by frame index
                key_idx = frame_idx * (len(joint.translation_data) - 1) / (self.n_frames - 1)
                idx0 = int(np.floor(key_idx))
                idx1 = min(idx0 + 1, len(joint.translation_data) - 1)
                alpha = key_idx - idx0
                return (1 - alpha) * joint.translation_data[idx0] + alpha * joint.translation_data[idx1]
        else:
            return joint.translation_data[frame_idx]
    
    def get_quaternion(self, joint, frame_idx: int) -> np.ndarray:
        """Get interpolated quaternion for frame (returns WXYZ)."""
        if joint.rotation_data is None:
            return np.array([1, 0, 0, 0])  # Identity
        
        # Similar interpolation logic as position
        # ... but with SLERP for quaternions
        
        # For now, simple version:
        if frame_idx < len(joint.rotation_data):
            xyzw = joint.rotation_data[frame_idx]
        else:
            xyzw = joint.rotation_data[-1]
        
        # Convert XYZW to WXYZ
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
