"""
GLB accessor reader for extracting data from buffers.
"""

import numpy as np
from typing import Optional
from pygltflib import GLTF2, Accessor, BufferView, Buffer
import logging

from ..exceptions import GLBParseError

logger = logging.getLogger(__name__)


class AccessorReader:
    """Reads data from GLB accessors."""
    
    # Component type to numpy dtype mapping
    COMPONENT_TYPE_MAP = {
        5120: np.int8,    # BYTE
        5121: np.uint8,   # UNSIGNED_BYTE
        5122: np.int16,   # SHORT
        5123: np.uint16,  # UNSIGNED_SHORT
        5125: np.uint32,  # UNSIGNED_INT
        5126: np.float32, # FLOAT
    }
    
    # Type to component count mapping
    TYPE_SIZE_MAP = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }
    
    def __init__(self, gltf: GLTF2):
        """
        Initialize accessor reader.
        
        Args:
            gltf: GLTF2 object
        """
        self.gltf = gltf
        self._buffer_cache = {}
        
    def read_accessor(self, accessor_idx: int) -> np.ndarray:
        """
        Read data from accessor.
        
        Args:
            accessor_idx: Index of accessor
            
        Returns:
            Numpy array with accessor data
            
        Raises:
            GLBParseError: If accessor cannot be read
        """
        if accessor_idx is None or accessor_idx < 0:
            raise GLBParseError(f"Invalid accessor index: {accessor_idx}")
            
        if accessor_idx >= len(self.gltf.accessors):
            raise GLBParseError(f"Accessor index {accessor_idx} out of range")
            
        accessor = self.gltf.accessors[accessor_idx]
        
        # Get buffer view
        if accessor.bufferView is None:
            # Sparse accessor or zero-initialized
            return self._create_zero_data(accessor)
            
        buffer_view = self.gltf.bufferViews[accessor.bufferView]
        buffer = self.gltf.buffers[buffer_view.buffer]
        
        # Get buffer data
        buffer_data = self._get_buffer_data(buffer)
        
        # Calculate offsets
        offset = buffer_view.byteOffset or 0
        if accessor.byteOffset:
            offset += accessor.byteOffset
            
        # Determine data type and shape
        dtype = self.COMPONENT_TYPE_MAP.get(accessor.componentType)
        if dtype is None:
            raise GLBParseError(f"Unknown component type: {accessor.componentType}")
            
        elements_per_item = self.TYPE_SIZE_MAP.get(accessor.type)
        if elements_per_item is None:
            raise GLBParseError(f"Unknown accessor type: {accessor.type}")
            
        # Calculate byte length
        byte_length = accessor.count * elements_per_item * np.dtype(dtype).itemsize
        
        # Extract data
        try:
            data_bytes = buffer_data[offset:offset + byte_length]
            data = np.frombuffer(data_bytes, dtype=dtype)
            
            # Reshape if needed
            if elements_per_item > 1:
                data = data.reshape((accessor.count, elements_per_item))
                
            logger.debug(f"Read accessor {accessor_idx}: shape={data.shape}, dtype={data.dtype}")
            return data
            
        except Exception as e:
            raise GLBParseError(f"Failed to read accessor {accessor_idx}: {e}")
            
    def _get_buffer_data(self, buffer: Buffer) -> bytes:
        """Get raw buffer data."""
        # Check cache
        buffer_id = id(buffer)
        if buffer_id in self._buffer_cache:
            return self._buffer_cache[buffer_id]
            
        # Load buffer data
        if buffer.uri is None:
            # Binary chunk (GLB format)
            if hasattr(self.gltf, 'binary_blob') and self.gltf.binary_blob is not None:
                data = self.gltf.binary_blob()
            else:
                raise GLBParseError("Binary buffer expected but not found")
        else:
            # External reference or data URI
            data = self.gltf.get_data_from_buffer_uri(buffer.uri)
            
        # Cache for future use
        self._buffer_cache[buffer_id] = data
        return data
        
    def _create_zero_data(self, accessor: Accessor) -> np.ndarray:
        """Create zero-initialized data for accessor."""
        dtype = self.COMPONENT_TYPE_MAP[accessor.componentType]
        elements_per_item = self.TYPE_SIZE_MAP[accessor.type]
        
        if elements_per_item > 1:
            shape = (accessor.count, elements_per_item)
        else:
            shape = (accessor.count,)
            
        return np.zeros(shape, dtype=dtype)
