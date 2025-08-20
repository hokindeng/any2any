"""
Common utilities and constants for the importer module.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FPS = 30
DEFAULT_JOINT_DAMPING = 0.5
DEFAULT_MOTOR_GEAR = 100.0
DEFAULT_TIMESTEP = 0.002
DEFAULT_GRAVITY = '0 0 -9.81'

# Coordinate system transformations
# GLB uses Y-up right-handed system
# MuJoCo uses Z-up right-handed system

def transform_point_y_to_z(point: np.ndarray) -> np.ndarray:
    """
    Transform a point from Y-up to Z-up coordinate system.
    
    GLB (Y-up): X-right, Y-up, Z-forward
    MuJoCo (Z-up): X-forward, Y-left, Z-up
    
    This is the inverse of MuJoCo→GLB transformation:
    MuJoCo→GLB: [x, y, z] → [x, z, -y]
    GLB→MuJoCo: [x, y, z] → [x, -z, y]
    
    Args:
        point: 3D point in Y-up system
        
    Returns:
        3D point in Z-up system
    """
    x, y, z = point
    # GLB to MuJoCo: Y-up to Z-up transformation
    # GLB X (right) → MuJoCo -Y (left, negated)
    # GLB Y (up) → MuJoCo Z (up)
    # GLB Z (forward) → MuJoCo X (forward)
    return np.array([z, -x, y])


def transform_quaternion_y_to_z(quat: np.ndarray) -> np.ndarray:
    """
    Transform a quaternion from Y-up to Z-up coordinate system.
    
    Applies a -90 degree rotation around X-axis to convert from Y-up to Z-up.
    
    Args:
        quat: Quaternion in XYZW format (Y-up system)
        
    Returns:
        Quaternion in WXYZ format (Z-up system, MuJoCo convention)
    """
    x, y, z, w = quat
    
    # Rotation of -90 degrees around X-axis
    # Quaternion for -90° around X: [sin(-45°), 0, 0, cos(-45°)] in XYZW
    # = [-0.7071068, 0, 0, 0.7071068]
    rot_x = np.array([-0.7071068, 0, 0, 0.7071068])  # -90 deg around X in XYZW
    rx, ry, rz, rw = rot_x
    
    # Quaternion multiplication: q_new = rot_x * q_orig
    # Formula: (w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #           w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #           w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #           w1*z2 + x1*y2 - y1*x2 + z1*w2)
    w_new = rw * w - rx * x - ry * y - rz * z
    x_new = rw * x + rx * w + ry * z - rz * y
    y_new = rw * y - rx * z + ry * w + rz * x
    z_new = rw * z + rx * y - ry * x + rz * w
    
    # Return in WXYZ format (MuJoCo convention)
    result = np.array([w_new, x_new, y_new, z_new])
    
    # Normalize
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
        
    return result


def quaternion_to_axis_angle(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        quat: Quaternion in WXYZ format
        
    Returns:
        Tuple of (axis, angle) where axis is unit vector and angle in radians
    """
    w, x, y, z = quat
    
    # Handle near-identity quaternion
    if abs(w) > 0.9999:
        return np.array([0, 0, 1]), 0.0
    
    # Calculate angle
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    # Calculate axis
    s = np.sqrt(1 - w * w)
    if s < 0.001:
        # Arbitrary axis
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([x, y, z]) / s
        
    return axis, angle


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.
    
    Args:
        matrix: 3x3 rotation matrix
        
    Returns:
        Quaternion in WXYZ format
    """
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
        
    return np.array([w, x, y, z])


def format_float_array(arr: np.ndarray, precision: int = 6) -> str:
    """
    Format numpy array as space-separated string for XML attributes.
    
    Args:
        arr: Numpy array
        precision: Decimal precision
        
    Returns:
        Space-separated string
    """
    return ' '.join(f'{x:.{precision}f}' for x in arr)


def sanitize_name(name: str) -> str:
    """
    Sanitize name for use in XML/MuJoCo.
    
    Args:
        name: Original name
        
    Returns:
        Sanitized name
    """
    # Replace invalid characters
    sanitized = name.replace(' ', '_')
    sanitized = sanitized.replace('.', '_')
    sanitized = sanitized.replace('-', '_')
    sanitized = sanitized.replace('/', '_')
    sanitized = sanitized.replace('\\', '_')
    
    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
        
    return sanitized or 'unnamed'
