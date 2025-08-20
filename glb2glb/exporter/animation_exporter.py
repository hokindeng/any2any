"""
Export animation data from MuJoCo to NPY format for GLB animation transfer.
"""

import numpy as np
import mujoco
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def export_animation_to_npy(
    model_path: str,
    qpos_data: List[np.ndarray],
    output_path: str,
    fps: float = 30.0
) -> Dict:
    """
    Export MuJoCo animation data to NPY format that can be applied to GLB.
    
    Args:
        model_path: Path to MuJoCo XML model
        qpos_data: List of qpos arrays (one per frame)
        output_path: Path to save NPY file
        fps: Frames per second
        
    Returns:
        Dictionary with animation data
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Prepare animation data for each joint
    animation_data = {
        'fps': fps,
        'n_frames': len(qpos_data),
        'joints': {}
    }
    
    # Process each joint
    for joint_idx in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
        joint_type = model.jnt_type[joint_idx]
        joint_axis = model.jnt_axis[joint_idx].copy()
        qpos_addr = model.jnt_qposadr[joint_idx]
        
        # Collect animation data for this joint
        rotations = []
        translations = []
        
        for qpos in qpos_data:
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                # Free joint: 3 translations + 4 rotations (quaternion)
                trans = qpos[qpos_addr:qpos_addr + 3]
                rot = qpos[qpos_addr + 3:qpos_addr + 7]
                translations.append(trans)
                rotations.append(rot)
                
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                # Ball joint: 4 rotations (quaternion)
                rot = qpos[qpos_addr:qpos_addr + 4]
                rotations.append(rot)
                
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                # Hinge joint: 1 rotation (convert to quaternion)
                angle = qpos[qpos_addr]
                q = np.zeros(4)
                mujoco.mju_axisAngle2Quat(q, joint_axis, angle)
                rotations.append(q)
                
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                # Slide joint: 1 translation along axis
                value = qpos[qpos_addr]
                trans = joint_axis * value
                translations.append(trans)
        
        # Store joint animation
        joint_data = {
            'name': joint_name,
            'type': joint_type
        }
        
        if rotations:
            joint_data['rotations'] = np.array(rotations, dtype=np.float32)
        if translations:
            joint_data['translations'] = np.array(translations, dtype=np.float32)
            
        animation_data['joints'][joint_name] = joint_data
    
    # Also store the raw qpos data for reference
    animation_data['qpos'] = np.array(qpos_data, dtype=np.float32)
    
    # Save to NPY
    np.save(output_path, animation_data)
    
    logger.info(f"Exported animation to {output_path}")
    logger.info(f"  Frames: {len(qpos_data)}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Joints: {len(animation_data['joints'])}")
    
    return animation_data
