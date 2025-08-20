"""
Unit tests for the GLB importer module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import xml.etree.ElementTree as ET

from glb2glb.importer.joint_centric_importer import JointCentricImporter


class TestJointCentricImporter(unittest.TestCase):
    """Test the JointCentricImporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    def test_importer_initialization(self, mock_gltf2):
        """Test importer initialization."""
        # Mock GLTF2 load
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Mock basic structure
        mock_gltf.nodes = [MagicMock(name=f"node_{i}") for i in range(3)]
        mock_gltf.skins = []
        mock_gltf.animations = []
        
        # Create importer
        importer = JointCentricImporter("test.glb")
        
        # Verify GLTF was loaded
        mock_gltf2.load.assert_called_once_with("test.glb")
        self.assertIsNotNone(importer.gltf)
        
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    def test_detect_coordinate_system_y_up(self, mock_gltf2):
        """Test Y-up coordinate system detection."""
        # Mock GLTF with Y-up indicators
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Create nodes with Y-up naming
        mock_node = MagicMock()
        mock_node.name = "mixamo:Hips"
        mock_node.translation = [0, 1, 0]  # Y-up position
        mock_gltf.nodes = [mock_node]
        mock_gltf.skins = []
        mock_gltf.animations = []
        
        # Create importer
        importer = JointCentricImporter("test.glb")
        
        # Should detect Y-up
        self.assertEqual(importer.coordinate_system, 'Y-up')
        
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    def test_detect_coordinate_system_z_up(self, mock_gltf2):
        """Test Z-up coordinate system detection."""
        # Mock GLTF with Z-up indicators
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Create nodes with Z-up naming
        mock_node = MagicMock()
        mock_node.name = "pelvis"
        mock_node.translation = [0, 0, 1]  # Z-up position
        mock_gltf.nodes = [mock_node]
        mock_gltf.skins = []
        mock_gltf.animations = []
        
        # Create importer
        importer = JointCentricImporter("test.glb")
        
        # Should detect Z-up
        self.assertEqual(importer.coordinate_system, 'Z-up')
        
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    def test_export_to_mujoco(self, mock_gltf2):
        """Test export to MuJoCo XML."""
        # Mock GLTF
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Create simple node hierarchy
        root_node = MagicMock()
        root_node.name = "Root"
        root_node.children = [1]
        root_node.translation = [0, 0, 0]
        root_node.rotation = [1, 0, 0, 0]
        root_node.scale = None
        
        child_node = MagicMock()
        child_node.name = "Child"
        child_node.children = []
        child_node.translation = [0, 1, 0]
        child_node.rotation = [1, 0, 0, 0]
        child_node.scale = None
        
        mock_gltf.nodes = [root_node, child_node]
        mock_gltf.skins = []
        mock_gltf.animations = []
        
        # Create importer
        importer = JointCentricImporter("test.glb")
        
        # Export to MuJoCo
        output_path = Path(self.temp_dir) / "test.xml"
        importer.export_to_mujoco(str(output_path))
        
        # Verify XML was created
        self.assertTrue(output_path.exists())
        
        # Parse and verify XML structure
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        self.assertEqual(root.tag, 'mujoco')
        worldbody = root.find('worldbody')
        self.assertIsNotNone(worldbody)
        
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    @patch('numpy.save')
    def test_export_motion_data(self, mock_np_save, mock_gltf2):
        """Test motion data export."""
        # Mock GLTF with animation
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Create nodes
        mock_gltf.nodes = [MagicMock(name=f"node_{i}") for i in range(2)]
        mock_gltf.skins = []
        
        # Mock animation with channels
        mock_animation = MagicMock()
        mock_animation.name = "TestAnimation"
        mock_animation.channels = []
        mock_animation.samplers = []
        
        # Create mock channel for translation
        mock_channel = MagicMock()
        mock_channel.target.node = 0
        mock_channel.target.path = "translation"
        mock_channel.sampler = 0
        
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler.input = 0  # Time accessor
        mock_sampler.output = 1  # Data accessor
        
        mock_animation.channels = [mock_channel]
        mock_animation.samplers = [mock_sampler]
        mock_gltf.animations = [mock_animation]
        
        # Mock accessors with proper attributes
        time_accessor = MagicMock(count=10, bufferView=0, byteOffset=0, type='SCALAR')
        data_accessor = MagicMock(count=10, bufferView=1, byteOffset=0, type='VEC3')
        mock_gltf.accessors = [time_accessor, data_accessor]
        
        # Mock buffer views
        mock_gltf.bufferViews = [
            MagicMock(byteOffset=0),
            MagicMock(byteOffset=40)
        ]
        
        # Mock binary blob to return actual bytes
        mock_gltf.binary_blob.return_value = b'\x00' * 1000  # Dummy binary data
        
        # Create importer with mocked _extract_accessor_data
        with patch.object(JointCentricImporter, '_extract_accessor_data') as mock_extract:
            # Return time values and translation values for animation extraction
            mock_extract.side_effect = [
                np.linspace(0, 1, 10).reshape(-1, 1),  # Time values
                np.random.randn(10, 3)   # Translation values  
            ]
            importer = JointCentricImporter("test.glb")
            
            # Reset mock for export_motion_data call
            mock_extract.side_effect = [
                np.linspace(0, 1, 10).reshape(-1, 1),  # Time values
                np.random.randn(10, 3)   # Translation values
            ]
            
            # Export motion data
            output_path = Path(self.temp_dir) / "motion.npy"
            importer.export_motion_data(str(output_path))
            
            # Verify numpy save was called
            mock_np_save.assert_called_once()
            
            # Verify the structure of saved data
            saved_data = mock_np_save.call_args[0][1]
            self.assertIn('qpos', saved_data)
            self.assertIn('fps', saved_data)
            self.assertIn('n_frames', saved_data)
            
    @patch('glb2glb.importer.joint_centric_importer.GLTF2')
    def test_build_joint_hierarchy(self, mock_gltf2):
        """Test joint hierarchy building."""
        # Mock GLTF
        mock_gltf = MagicMock()
        mock_gltf2.load.return_value = mock_gltf
        
        # Create hierarchical nodes
        root_node = MagicMock()
        root_node.name = "Root"
        root_node.children = [1, 2]
        root_node.translation = None
        root_node.rotation = None
        root_node.scale = None
        
        child1 = MagicMock()
        child1.name = "Child1"
        child1.children = []
        child1.translation = None
        child1.rotation = None
        child1.scale = None
        
        child2 = MagicMock()
        child2.name = "Child2"
        child2.children = []
        child2.translation = None
        child2.rotation = None
        child2.scale = None
        
        mock_gltf.nodes = [root_node, child1, child2]
        
        # Create skin with joints
        mock_skin = MagicMock()
        mock_skin.joints = [0, 1, 2]
        mock_gltf.skins = [mock_skin]
        
        mock_gltf.animations = []
        
        # Create importer
        importer = JointCentricImporter("test.glb")
        
        # Build hierarchy
        importer._build_joint_hierarchy()
        
        # Verify hierarchy was built
        self.assertEqual(len(importer.joints), 3)
        self.assertIn("Root", importer.joint_map)
        self.assertIn("Child1", importer.joint_map)
        self.assertIn("Child2", importer.joint_map)
        
        # Verify parent-child relationships
        root_joint = importer.joint_map["Root"]
        self.assertEqual(len(root_joint['children']), 2)
        
        child1_joint = importer.joint_map["Child1"]
        self.assertEqual(child1_joint['parent'], 0)
        
        child2_joint = importer.joint_map["Child2"]
        self.assertEqual(child2_joint['parent'], 0)


class TestCoordinateTransform(unittest.TestCase):
    """Test coordinate system transformations."""
    
    def test_y_up_to_z_up_position(self):
        """Test Y-up to Z-up position transformation."""
        from glb2glb.importer.joint_centric_importer import JointCentricImporter
        
        # Create mock importer
        with patch('glb2glb.importer.joint_centric_importer.GLTF2'):
            importer = JointCentricImporter.__new__(JointCentricImporter)
            importer.coordinate_system = 'Y-up'
            
            # Y-up position (right, up, forward)
            pos_y_up = np.array([1.0, 2.0, 3.0])
            
            # Transform to Z-up (forward, left, up)
            pos_z_up = importer._transform_position(pos_y_up)
            
            # Expected: (Z, -X, Y) = (3, -1, 2)
            np.testing.assert_array_almost_equal(pos_z_up, [3.0, -1.0, 2.0])
            
    def test_z_up_position_unchanged(self):
        """Test Z-up position remains unchanged."""
        from glb2glb.importer.joint_centric_importer import JointCentricImporter
        
        # Create mock importer
        with patch('glb2glb.importer.joint_centric_importer.GLTF2'):
            importer = JointCentricImporter.__new__(JointCentricImporter)
            importer.coordinate_system = 'Z-up'
            
            # Z-up position
            pos = np.array([1.0, 2.0, 3.0])
            
            # Should remain unchanged
            transformed = importer._transform_position(pos)
            np.testing.assert_array_equal(transformed, pos)
            
    def test_y_up_to_z_up_quaternion(self):
        """Test Y-up to Z-up quaternion transformation."""
        from glb2glb.importer.joint_centric_importer import JointCentricImporter
        
        # Create mock importer
        with patch('glb2glb.importer.joint_centric_importer.GLTF2'):
            importer = JointCentricImporter.__new__(JointCentricImporter)
            importer.coordinate_system = 'Y-up'
            
            # Identity quaternion (XYZW format)
            quat_y_up = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Transform to Z-up
            quat_z_up = importer._transform_quaternion(quat_y_up)
            
            # Should apply coordinate rotation
            # The exact value depends on the implementation
            self.assertEqual(len(quat_z_up), 4)
            # Verify it's still normalized
            norm = np.linalg.norm(quat_z_up)
            self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
