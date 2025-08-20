"""
Internal utilities extracted to make exporter self-contained.
These are minimal implementations of only the functions actually used.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Union

# ============= Quaternion utilities =============

def mulQuat(qa, qb):
    """
    Multiply two quaternions.
    Quaternions are in scalar-first format: [w, x, y, z]
    
    Args:
        qa: First quaternion (4,) array
        qb: Second quaternion (4,) array
    
    Returns:
        Product quaternion (4,) array
    """
    # Quaternion multiplication formula
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def mat2quat(mat):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
        mat: 3x3 rotation matrix
    
    Returns:
        Quaternion [w, x, y, z]
    """
    # Use the most numerically stable method
    m = mat
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


def quat2mat(quat):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    mat = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return mat


def quatvec2H(q, t):
    """
    Convert quaternion and translation to 4x4 homogeneous transformation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        t: Translation [x, y, z]
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    H = np.eye(4)
    H[:3, :3] = quat2mat(q)
    H[:3, 3] = t
    return H


def H2quatvec(H):
    """
    Convert 4x4 homogeneous transformation matrix to quaternion and translation.
    
    Args:
        H: 4x4 homogeneous transformation matrix
    
    Returns:
        Tuple of (quaternion [w, x, y, z], translation [x, y, z])
    """
    R = H[:3, :3]
    t = H[:3, 3]
    q = mat2quat(R)
    return q, t


# ============= MuJoCo model loading =============

def get_model(model_path: Union[str, Path, mujoco.MjModel], **kwargs):
    """
    Load a MuJoCo model from path or return if already loaded.
    
    Args:
        model_path: Path to XML/MJB file or MjModel object
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Tuple of (model, data) for compatibility with original API
    """
    if isinstance(model_path, mujoco.MjModel):
        # Already a model
        return model_path, mujoco.MjData(model_path)
    
    # Load from file
    model_path = Path(model_path)
    if model_path.suffix == '.xml':
        model = mujoco.MjModel.from_xml_path(str(model_path))
    elif model_path.suffix == '.mjb':
        model = mujoco.MjModel.from_binary_path(str(model_path))
    else:
        raise ValueError(f"Unsupported file type: {model_path.suffix}")
    
    data = mujoco.MjData(model)
    return model, data


# ============= Cache utilities =============

def get_cache_directory():
    """
    Get the cache directory for the exporter.
    Simple implementation that uses home directory.
    
    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir