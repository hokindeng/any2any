"""
GLB/GLTF parsing module.
"""

from .parser import GLBParser
from .accessor import AccessorReader

__all__ = ['GLBParser', 'AccessorReader']
