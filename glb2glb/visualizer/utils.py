"""
Utility functions for motion visualization.
"""

import numpy as np
import mujoco
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def load_motion_data(motion_path: str) -> Dict[str, Any]:
    """
    Load motion data from NPY file.
    
    Args:
        motion_path: Path to motion NPY file
        
    Returns:
        Dictionary with motion data
    """
    try:
        data = np.load(motion_path, allow_pickle=True)
        
        # Handle different data formats
        if isinstance(data, dict):
            # Already a dictionary
            return data
        elif hasattr(data, 'item'):
            # Numpy object that can be converted to dict
            return data.item()
        elif isinstance(data, np.ndarray):
            # Check if it's a 0-d array containing a dict
            if data.ndim == 0:
                return data.item()
            # Otherwise treat as raw trajectory array
            return {
                'qpos': data,
                'fps': 30.0,  # Default FPS
                'duration': data.shape[0] / 30.0 if len(data.shape) > 0 else 0,
                'source_file': motion_path
            }
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
            
    except Exception as e:
        logger.error(f"Failed to load motion from {motion_path}: {e}")
        raise


def validate_motion_model_compatibility(
    model: mujoco.MjModel,
    motion_data: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Check if motion data is compatible with model.
    
    Args:
        model: MuJoCo model
        motion_data: Motion data dictionary
        
    Returns:
        (is_compatible, message) tuple
    """
    qpos = motion_data.get('qpos')
    if qpos is None:
        return False, "Motion data missing 'qpos' field"
    
    if len(qpos.shape) != 2:
        return False, f"Expected 2D qpos array, got shape {qpos.shape}"
    
    n_frames, n_dofs = qpos.shape
    
    if n_dofs == model.nq:
        return True, f"Perfect match: {n_dofs} DOFs"
    
    # Check for common mismatches
    if model.njnt > 0 and model.joint(0).type[0] == 0:  # Has freejoint
        expected_joint_dofs = model.nq - 7
        if n_dofs == expected_joint_dofs:
            return False, f"Motion missing floating base (has {n_dofs}, needs {model.nq})"
        elif n_dofs == expected_joint_dofs + 2:
            return False, f"Motion has extra joints (has {n_dofs}, model expects {model.nq})"
    
    return False, f"Dimension mismatch: motion has {n_dofs} DOFs, model needs {model.nq}"


def pad_motion_to_model(
    model: mujoco.MjModel,
    motion_data: Dict[str, Any],
    default_pos: Optional[np.ndarray] = None,
    default_quat: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Pad or trim motion data to match model dimensions.
    
    Args:
        model: MuJoCo model
        motion_data: Motion data dictionary
        default_pos: Default position for floating base [x, y, z]
        default_quat: Default quaternion for floating base [w, x, y, z]
        
    Returns:
        Padded motion data dictionary
    """
    qpos = motion_data['qpos']
    n_frames, n_dofs = qpos.shape
    
    if n_dofs == model.nq:
        logger.info("Motion dimensions already match model")
        return motion_data
    
    # Default position and orientation
    if default_pos is None:
        default_pos = [0, 0, 0.8]
    if default_quat is None:
        default_quat = [1, 0, 0, 0]
    
    padded_qpos = np.zeros((n_frames, model.nq))
    
    # Check if model has floating base
    has_freejoint = model.njnt > 0 and model.joint(0).type[0] == 0
    
    if has_freejoint and n_dofs < model.nq:
        # Motion missing floating base - add it
        logger.info(f"Adding floating base: motion {n_dofs} -> {model.nq} DOFs")
        
        # Set floating base
        padded_qpos[:, 0:3] = default_pos
        padded_qpos[:, 3:7] = default_quat
        
        # Copy joint angles
        available_joints = min(n_dofs, model.nq - 7)
        padded_qpos[:, 7:7+available_joints] = qpos[:, :available_joints]
        
    elif n_dofs > model.nq:
        # Motion has more DOFs than model - trim
        logger.warning(f"Trimming motion: {n_dofs} -> {model.nq} DOFs")
        padded_qpos = qpos[:, :model.nq]
        
    else:
        # Other cases - try to copy what we can
        logger.warning(f"Partial copy: motion {n_dofs} -> model {model.nq} DOFs")
        min_dofs = min(n_dofs, model.nq)
        padded_qpos[:, :min_dofs] = qpos[:, :min_dofs]
    
    # Create padded data
    padded_data = motion_data.copy()
    padded_data['qpos'] = padded_qpos
    padded_data['original_shape'] = qpos.shape
    padded_data['padded'] = True
    
    return padded_data


def create_test_motion(
    model_path: str,
    duration: float = 3.0,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Create a simple test motion for debugging.
    
    Args:
        model_path: Path to MuJoCo XML model
        duration: Duration in seconds
        fps: Frames per second
        
    Returns:
        Motion data dictionary
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    n_frames = int(duration * fps)
    
    trajectory = np.zeros((n_frames, model.nq))
    
    for t in range(n_frames):
        # Start from default pose
        trajectory[t] = model.qpos0.copy()
        
        # Add simple oscillation to joints
        phase = 2 * np.pi * t / (fps * 2)  # 2 second period
        
        # Skip floating base if present
        start_idx = 7 if (model.njnt > 0 and model.joint(0).type[0] == 0) else 0
        
        # Animate joints with different phases
        for i in range(start_idx, min(start_idx + 10, model.nq)):
            trajectory[t, i] = 0.2 * np.sin(phase + i * 0.3)
        
        # Add vertical motion if floating base
        if start_idx == 7:
            trajectory[t, 2] = model.qpos0[2] + 0.1 * np.sin(phase)
    
    return {
        'qpos': trajectory,
        'fps': fps,
        'duration': duration,
        'time': np.linspace(0, duration, n_frames),
        'animation_name': 'test_motion',
        'joint_names': [model.joint(i).name for i in range(model.njnt)]
    }


def extract_joint_info(model: mujoco.MjModel) -> Dict[str, Any]:
    """
    Extract detailed joint information from model.
    
    Args:
        model: MuJoCo model
        
    Returns:
        Dictionary with joint details
    """
    joint_info = {
        'total_dofs': model.nq,
        'total_joints': model.njnt,
        'joints': []
    }
    
    joint_types = ['free', 'ball', 'slide', 'hinge']
    joint_dofs = [7, 4, 1, 1]
    
    for i in range(model.njnt):
        joint = model.joint(i)
        jtype = joint.type[0]
        
        joint_info['joints'].append({
            'index': i,
            'name': joint.name,
            'type': joint_types[jtype],
            'dofs': joint_dofs[jtype],
            'range': joint.range.tolist() if joint.limited else None
        })
    
    # Check for floating base
    if model.njnt > 0 and model.joint(0).type[0] == 0:
        joint_info['has_floating_base'] = True
        joint_info['floating_base_name'] = model.joint(0).name
    else:
        joint_info['has_floating_base'] = False
    
    return joint_info
