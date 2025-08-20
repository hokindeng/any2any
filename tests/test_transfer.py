"""
Unit tests for the animation transfer module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock, Mock, call

from glb2glb.transfer.npy_to_glb import (
    load_gltf,
    save_gltf,
    create_accessor,
    finalize_animation_buffer,
    apply_animation_from_npy
)
from glb2glb.transfer.animation_transfer import (
    get_all_binary_data,
    copy_accessor_data_simple,
    consolidate_buffers_simple,
    transfer_animation_from_glb,
    transfer_retargeted_animation
)


class TestNPYToGLB(unittest.TestCase):
    """Test NPY to GLB animation transfer functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('pygltflib.GLTF2')
    def test_load_gltf(self, mock_gltf2):
        """Test loading GLTF file."""
        mock_instance = MagicMock()
        mock_gltf2.return_value.load.return_value = mock_instance
        
        result = load_gltf("test.glb")
        
        mock_gltf2.return_value.load.assert_called_once_with("test.glb")
        self.assertEqual(result, mock_instance)
        
    @patch('pygltflib.GLTF2')
    def test_save_gltf(self, mock_gltf2):
        """Test saving GLTF file."""
        mock_gltf = MagicMock()
        
        save_gltf(mock_gltf, "output.glb")
        
        mock_gltf.save.assert_called_once_with("output.glb")
        
    def test_create_accessor(self):
        """Test creating an accessor for animation data."""
        # Mock GLTF object
        mock_gltf = MagicMock()
        mock_gltf.bufferViews = []
        mock_gltf.accessors = []
        
        # Test data
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        
        # Create accessor
        accessor_idx = create_accessor(mock_gltf, data, "SCALAR")
        
        # Verify accessor was created
        self.assertEqual(accessor_idx, 0)
        self.assertEqual(len(mock_gltf.bufferViews), 1)
        self.assertEqual(len(mock_gltf.accessors), 1)
        
        # Check accessor properties
        accessor = mock_gltf.accessors[0]
        self.assertEqual(accessor.count, 3)
        self.assertEqual(accessor.type, "SCALAR")
        
    def test_create_accessor_vec3(self):
        """Test creating a VEC3 accessor."""
        mock_gltf = MagicMock()
        mock_gltf.bufferViews = []
        mock_gltf.accessors = []
        
        # Test VEC3 data
        data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        
        accessor_idx = create_accessor(mock_gltf, data.flatten(), "VEC3")
        
        self.assertEqual(accessor_idx, 0)
        accessor = mock_gltf.accessors[0]
        self.assertEqual(accessor.count, 3)
        self.assertEqual(accessor.type, "VEC3")
        
    def test_finalize_animation_buffer(self):
        """Test finalizing the animation buffer."""
        mock_gltf = MagicMock()
        mock_gltf.buffers = []
        
        # Add test animation data
        mock_gltf._animation_data = bytearray(b"test_data")
        
        # Finalize buffer
        finalize_animation_buffer(mock_gltf)
        
        # Verify buffer was created
        self.assertEqual(len(mock_gltf.buffers), 1)
        self.assertEqual(mock_gltf.buffers[0].byteLength, 12)  # Aligned to 4 bytes
        
        # Verify binary blob was set
        mock_gltf.set_binary_blob.assert_called_once()
        
        # Verify _animation_data was cleaned up
        self.assertFalse(hasattr(mock_gltf, '_animation_data'))
        
    @patch('glb2glb.transfer.npy_to_glb.load_gltf')
    @patch('glb2glb.transfer.npy_to_glb.save_gltf')
    @patch('numpy.load')
    def test_apply_animation_from_npy_success(
        self,
        mock_np_load,
        mock_save_gltf,
        mock_load_gltf
    ):
        """Test successful NPY animation application."""
        # Mock GLTF
        mock_gltf = MagicMock()
        mock_load_gltf.return_value = mock_gltf
        
        # Mock nodes
        node1 = MagicMock()
        node1.name = "Joint1"
        node2 = MagicMock()
        node2.name = "Joint2"
        mock_gltf.nodes = [node1, node2]
        
        # Mock skin
        mock_skin = MagicMock()
        mock_skin.joints = [0, 1]
        mock_gltf.skins = [mock_skin]
        
        mock_gltf.animations = []
        mock_gltf.bufferViews = []
        mock_gltf.accessors = []
        mock_gltf.buffers = []
        
        # Mock NPY animation data
        anim_data = {
            'fps': 30.0,
            'n_frames': 10,
            'joints': {
                'Joint1': {
                    'rotations': np.random.randn(10, 4).astype(np.float32)
                },
                'Joint2': {
                    'rotations': np.random.randn(10, 4).astype(np.float32),
                    'translations': np.random.randn(10, 3).astype(np.float32)
                }
            }
        }
        mock_np_load.return_value.item.return_value = anim_data
        
        # Apply animation
        result = apply_animation_from_npy(
            target_glb="target.glb",
            animation_npy="animation.npy",
            output_glb="output.glb",
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify animation was added
        self.assertEqual(len(mock_gltf.animations), 1)
        
        # Verify save was called
        mock_save_gltf.assert_called_once_with(mock_gltf, "output.glb")
        
    @patch('glb2glb.transfer.npy_to_glb.load_gltf')
    @patch('numpy.load')
    def test_apply_animation_joint_name_matching(
        self,
        mock_np_load,
        mock_load_gltf
    ):
        """Test joint name matching with fallback."""
        # Mock GLTF with different naming
        mock_gltf = MagicMock()
        mock_load_gltf.return_value = mock_gltf
        
        # Nodes with prefix
        node1 = MagicMock()
        node1.name = "Armature:Joint1"
        node2 = MagicMock()
        node2.name = "Armature:Joint2"
        mock_gltf.nodes = [node1, node2]
        
        mock_gltf.skins = []
        mock_gltf.animations = []
        mock_gltf.bufferViews = []
        mock_gltf.accessors = []
        mock_gltf.buffers = []
        
        # NPY with simpler names
        anim_data = {
            'fps': 30.0,
            'n_frames': 5,
            'joints': {
                'Joint1': {'rotations': np.zeros((5, 4), dtype=np.float32)},
                'Joint2': {'rotations': np.zeros((5, 4), dtype=np.float32)}
            }
        }
        mock_np_load.return_value.item.return_value = anim_data
        
        # Apply animation
        with patch('glb2glb.transfer.npy_to_glb.save_gltf'):
            result = apply_animation_from_npy(
                target_glb="target.glb",
                animation_npy="animation.npy",
                output_glb="output.glb",
                verbose=False
            )
            
            # Should find matches via substring matching
            self.assertTrue(result)


class TestAnimationTransfer(unittest.TestCase):
    """Test GLB to GLB animation transfer functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_get_all_binary_data_with_blob(self):
        """Test getting binary data from GLB with blob."""
        mock_gltf = MagicMock()
        test_data = b"test_binary_data"
        mock_gltf.binary_blob.return_value = test_data
        
        result = get_all_binary_data(mock_gltf)
        
        self.assertEqual(result, test_data)
        mock_gltf.binary_blob.assert_called_once()
        
    def test_get_all_binary_data_no_blob(self):
        """Test getting binary data when no blob available."""
        mock_gltf = MagicMock()
        mock_gltf.binary_blob.return_value = None
        
        # Mock buffer with no URI (embedded)
        mock_buffer = MagicMock()
        mock_buffer.uri = None
        mock_gltf.buffers = [mock_buffer]
        
        # Mock _glb_data attribute
        test_data = b"embedded_data"
        mock_gltf._glb_data = test_data
        
        result = get_all_binary_data(mock_gltf)
        
        self.assertEqual(result, test_data)
        
    def test_get_all_binary_data_external_file(self):
        """Test getting binary data from external file."""
        mock_gltf = MagicMock()
        mock_gltf.binary_blob.return_value = None
        
        # Mock buffer with URI
        mock_buffer = MagicMock()
        mock_buffer.uri = "data.bin"
        mock_gltf.buffers = [mock_buffer]
        
        # Mock get_data_from_buffer_uri
        test_data = b"external_data"
        mock_gltf.get_data_from_buffer_uri.return_value = test_data
        
        result = get_all_binary_data(mock_gltf)
        
        self.assertEqual(result, test_data)
        mock_gltf.get_data_from_buffer_uri.assert_called_once_with("data.bin")
        
    def test_copy_accessor_data_simple(self):
        """Test copying accessor data between GLTFs."""
        # Source GLTF
        src_gltf = MagicMock()
        src_accessor = MagicMock()
        src_accessor.bufferView = 0
        src_accessor.byteOffset = 0
        src_accessor.componentType = 5126  # FLOAT
        src_accessor.count = 10
        src_accessor.type = "SCALAR"
        src_accessor.max = [1.0]
        src_accessor.min = [0.0]
        src_accessor.normalized = False
        src_accessor.sparse = None
        src_accessor.name = "test_accessor"
        src_gltf.accessors = [src_accessor]
        
        src_bv = MagicMock()
        src_bv.byteOffset = 100
        src_bv.byteLength = 40  # 10 floats
        src_bv.byteStride = None
        src_bv.target = None
        src_gltf.bufferViews = [src_bv]
        
        # Target GLTF
        tgt_gltf = MagicMock()
        tgt_gltf.bufferViews = []
        tgt_gltf.accessors = []
        
        # Binary data
        src_binary = b'\x00' * 200  # Enough to cover the buffer view
        
        # Pending data list
        pending_data = []
        
        # Copy accessor
        new_idx = copy_accessor_data_simple(
            0, src_gltf, tgt_gltf, src_binary, pending_data
        )
        
        # Verify
        self.assertEqual(new_idx, 0)
        self.assertEqual(len(tgt_gltf.bufferViews), 1)
        self.assertEqual(len(tgt_gltf.accessors), 1)
        self.assertEqual(len(pending_data), 1)
        
        # Check pending data
        self.assertEqual(len(pending_data[0]['data']), 40)
        
    def test_consolidate_buffers_simple(self):
        """Test consolidating buffer data."""
        mock_gltf = MagicMock()
        mock_gltf.buffers = []
        
        # Create pending data
        mock_gltf._pending_data = [
            {
                'data': b'data1',
                'bufferView': 0,
            },
            {
                'data': b'data2',
                'bufferView': 1,
            }
        ]
        
        # Mock buffer views
        bv1 = MagicMock()
        bv2 = MagicMock()
        mock_gltf.bufferViews = [bv1, bv2]
        
        # Consolidate
        consolidate_buffers_simple(mock_gltf)
        
        # Verify buffer was created
        self.assertEqual(len(mock_gltf.buffers), 1)
        
        # Verify binary blob was set
        mock_gltf.set_binary_blob.assert_called_once()
        
        # Verify offsets were set
        self.assertEqual(bv1.byteOffset, 0)
        self.assertGreater(bv2.byteOffset, 0)
        
        # Verify _pending_data was cleaned up
        self.assertFalse(hasattr(mock_gltf, '_pending_data'))
        
    @patch('glb2glb.transfer.animation_transfer.load_gltf')
    @patch('glb2glb.transfer.animation_transfer.save_gltf')
    @patch('glb2glb.transfer.animation_transfer.get_all_binary_data')
    def test_transfer_animation_from_glb(
        self,
        mock_get_binary,
        mock_save,
        mock_load
    ):
        """Test transferring animation between GLB files."""
        # Mock source GLB with animation
        src_gltf = MagicMock()
        src_anim = MagicMock()
        src_anim.name = "TestAnimation"
        src_anim.channels = []
        src_anim.samplers = []
        src_gltf.animations = [src_anim]
        src_gltf.nodes = [MagicMock(name="Joint1")]
        src_gltf.skins = []
        src_gltf.accessors = []
        src_gltf.bufferViews = []
        
        # Mock target GLB
        tgt_gltf = MagicMock()
        tgt_gltf.animations = []
        tgt_gltf.nodes = [MagicMock(name="Joint1")]
        tgt_gltf.skins = []
        tgt_gltf.accessors = []
        tgt_gltf.bufferViews = []
        tgt_gltf.buffers = []
        
        mock_load.side_effect = [src_gltf, tgt_gltf]
        mock_get_binary.return_value = b'\x00' * 100
        
        # Transfer
        result = transfer_animation_from_glb(
            "source.glb",
            "target.glb",
            "output.glb",
            verbose=False
        )
        
        # Should succeed even with no channels
        self.assertFalse(result)  # No channels transferred
        
    def test_transfer_retargeted_animation(self):
        """Test the convenience wrapper function."""
        with patch('glb2glb.transfer.animation_transfer.transfer_animation_from_glb') as mock_transfer:
            mock_transfer.return_value = True
            
            result = transfer_retargeted_animation(
                "retargeted.glb",
                "original.glb",
                "output.glb",
                verbose=False
            )
            
            self.assertTrue(result)
            mock_transfer.assert_called_once_with(
                "retargeted.glb",
                "original.glb",
                "output.glb",
                False
            )


if __name__ == '__main__':
    unittest.main()
