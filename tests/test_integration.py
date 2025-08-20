"""
Integration tests for the complete GLB2GLB pipeline.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

import glb2glb


class TestFullPipeline(unittest.TestCase):
    """Test the complete GLB to GLB pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_mock_glb_data(self):
        """Create mock GLB data structure."""
        mock_gltf = MagicMock()
        
        # Create nodes
        root_node = MagicMock()
        root_node.name = "Root"
        root_node.children = [1]
        root_node.translation = [0, 0, 0]
        root_node.rotation = [0, 0, 0, 1]
        root_node.scale = None
        
        child_node = MagicMock()
        child_node.name = "Child"
        child_node.children = []
        child_node.translation = [0, 1, 0]
        child_node.rotation = [0, 0, 0, 1]
        child_node.scale = None
        
        mock_gltf.nodes = [root_node, child_node]
        
        # Create skin
        mock_skin = MagicMock()
        mock_skin.joints = [0, 1]
        mock_gltf.skins = [mock_skin]
        
        # Create mesh
        mock_mesh = MagicMock()
        mock_mesh.name = "TestMesh"
        mock_mesh.primitives = [MagicMock()]
        mock_gltf.meshes = [mock_mesh]
        
        # Create animation
        mock_animation = MagicMock()
        mock_animation.name = "TestAnimation"
        mock_animation.channels = []
        mock_animation.samplers = []
        mock_gltf.animations = [mock_animation]
        
        return mock_gltf
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('glb2glb.pipeline.motion_transfer.export_animation_to_npy')
    @patch('glb2glb.pipeline.motion_transfer.apply_animation_from_npy')
    @patch('glb2glb.pipeline.motion_transfer.ik_retarget_motion')
    @patch('numpy.load')
    def test_transfer_animation_api(
        self,
        mock_np_load,
        mock_ik_retarget,
        mock_apply_npy,
        mock_export_npy,
        mock_importer
    ):
        """Test the main API function transfer_animation."""
        # Mock motion data
        mock_motion_data = {
            'qpos': np.random.randn(10, 7),
            'fps': 30.0,
            'n_frames': 10
        }
        mock_np_load.return_value.item.return_value = mock_motion_data
        
        # Mock importer with no file IO
        mock_importer_instance = MagicMock()
        def fake_export_to_mujoco(path):
            Path(path).write_text('<mujoco><worldbody/></mujoco>')
        def fake_export_motion_data(path):
            np.save(path, {'qpos': np.zeros((10,7)), 'fps': 30.0})
        mock_importer_instance.export_to_mujoco.side_effect = fake_export_to_mujoco
        mock_importer_instance.export_motion_data.side_effect = fake_export_motion_data
        mock_importer.return_value = mock_importer_instance
        
        # Mock successful pipeline
        mock_ik_retarget.return_value = True
        mock_apply_npy.return_value = True
        
        # Use test assets
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Call main API
        result = glb2glb.transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify pipeline was executed
        self.assertEqual(mock_importer.call_count, 2)
        mock_export_npy.assert_called_once()
        mock_apply_npy.assert_called_once()
        
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    def test_glb_to_mujoco_api(self, mock_mjdata, mock_mjmodel):
        """Test the glb_to_mujoco API function."""
        # Mock model
        mock_model = MagicMock()
        mock_model.nq = 7
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Use test asset
        assets_dir = Path(__file__).parent / "assets"
        test_glb = assets_dir / "source.glb"
        
        # Call API
        with patch('glb2glb.joint_centric.JointCentricImporter') as mock_importer:
            mock_importer_instance = MagicMock()
            mock_importer.return_value = mock_importer_instance
            
            # Mock motion data
            mock_motion = {
                'qpos': np.zeros((1, 7)),
                'fps': 30.0
            }
            with patch('numpy.load') as mock_load:
                mock_load.return_value.item.return_value = mock_motion
                
                glb2glb.glb_to_mujoco(
                    glb_path=str(test_glb),
                    output_dir=self.temp_dir
                )
                
                # Verify importer was used
                mock_importer.assert_called_once_with(str(test_glb))
                mock_importer_instance.export_to_mujoco.assert_called_once()
                
    @patch('mujoco.MjModel')
    @patch('pygltflib.GLTF2')
    def test_mujoco_to_glb_api(self, mock_gltf2, mock_mjmodel):
        """Test the mujoco_to_glb API function."""
        # Mock model
        mock_model = MagicMock()
        mock_model.njnt = 2
        mock_model.nq = 8
        mock_model.nbody = 3
        
        # Mock joint and body data
        mock_joint = MagicMock()
        mock_joint.name = "joint1"
        mock_model.joint.return_value = mock_joint
        mock_model.jnt_type = [1, 1]
        mock_model.jnt_axis = np.array([[0, 1, 0], [1, 0, 0]])
        mock_model.jnt_bodyid = [1, 2]
        mock_model.jnt_qposadr = [0, 1]
        mock_model.jnt_dofadr = [0, 1]
        
        mock_body = MagicMock()
        mock_body.name = "body1"
        mock_body.pos = np.array([0, 0, 0])
        mock_body.parentid = 0
        mock_model.body.return_value = mock_body
        
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Create test XML
        test_xml = Path(self.temp_dir) / "test.xml"
        with open(test_xml, 'w') as f:
            f.write('<mujoco><worldbody></worldbody></mujoco>')
            
        # Call API
        with patch('glb2glb.joint_centric._get_exporter') as mock_get_exporter:
            mock_exporter_class = MagicMock()
            mock_exporter_instance = MagicMock()
            mock_exporter_class.return_value = mock_exporter_instance
            mock_get_exporter.return_value = mock_exporter_class
            
            output_glb = Path(self.temp_dir) / "output.glb"
            
            glb2glb.mujoco_to_glb(
                xml_path=str(test_xml),
                output_path=str(output_glb)
            )
            
            # Verify exporter was used
            mock_exporter_class.assert_called_once_with(str(test_xml))
            mock_exporter_instance.export_to_glb.assert_called_once_with(str(output_glb))


class TestRoundTrip(unittest.TestCase):
    """Test round-trip conversion GLB -> MuJoCo -> GLB."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('glb2glb.joint_centric.JointCentricImporter')
    @patch('glb2glb.joint_centric._get_exporter')
    @patch('mujoco.MjModel')
    @patch('numpy.load')
    @patch('numpy.save')
    def test_round_trip_test_api(
        self,
        mock_np_save,
        mock_np_load,
        mock_mjmodel,
        mock_get_exporter,
        mock_importer
    ):
        """Test the round_trip_test API function."""
        # Mock importer
        mock_importer_instance = MagicMock()
        mock_importer.return_value = mock_importer_instance
        
        # Mock exporter
        mock_exporter_class = MagicMock()
        mock_exporter_instance = MagicMock()
        mock_exporter_class.return_value = mock_exporter_instance
        mock_get_exporter.return_value = mock_exporter_class
        
        # Mock model
        mock_model = MagicMock()
        mock_model.nq = 7
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Mock motion data
        mock_motion = {
            'qpos': np.zeros((10, 7)),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion
        
        # Use test asset
        assets_dir = Path(__file__).parent / "assets"
        test_glb = assets_dir / "source.glb"
        
        # Run round trip test
        glb2glb.round_trip_test(
            input_glb=str(test_glb),
            temp_dir=self.temp_dir
        )
        
        # Verify both import and export were called
        mock_importer.assert_called()
        mock_exporter_class.assert_called()
        
        # Verify files were created
        mock_importer_instance.export_to_mujoco.assert_called()
        mock_exporter_instance.export_to_glb.assert_called()


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    def test_transfer_animation_invalid_source(self, mock_importer):
        """Test handling of invalid source file."""
        # Make importer raise exception
        mock_importer.side_effect = Exception("Invalid GLB file")
        
        # Use test assets and invalid source
        assets_dir = Path(__file__).parent / "assets"
        invalid_file = "invalid.glb"  # This doesn't exist
        target_file = assets_dir / "target.glb"
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Should handle gracefully
        result = glb2glb.transfer_animation(
            source_glb=invalid_file,
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertFalse(result)
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('numpy.load')
    def test_transfer_animation_no_animation(
        self,
        mock_np_load,
        mock_importer
    ):
        """Test handling when source has no animation."""
        # Mock importer
        mock_importer_instance = MagicMock()
        mock_importer.return_value = mock_importer_instance
        
        # Mock motion with only 1 frame (no animation)
        mock_motion = {
            'qpos': np.zeros((1, 7)),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion
        
        # Use test assets
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Mock importer to write temp files
        def fake_export_to_mujoco(path):
            Path(path).write_text('<mujoco><worldbody/></mujoco>')
        def fake_export_motion_data(path):
            np.save(path, {'qpos': np.zeros((1,7)), 'fps': 30.0})  # Only 1 frame
        mock_importer_instance.export_to_mujoco.side_effect = fake_export_to_mujoco
        mock_importer_instance.export_motion_data.side_effect = fake_export_motion_data
        
        # Should detect no animation
        result = glb2glb.transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertFalse(result)
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('glb2glb.pipeline.motion_transfer.export_animation_to_npy')
    @patch('glb2glb.pipeline.motion_transfer.apply_animation_from_npy')
    @patch('glb2glb.pipeline.motion_transfer.ik_retarget_motion')
    @patch('numpy.load')
    def test_transfer_animation_apply_failure(
        self,
        mock_np_load,
        mock_ik_retarget,
        mock_apply_npy,
        mock_export_npy,
        mock_importer
    ):
        """Test handling when NPY application fails."""
        # Mock successful early stages
        mock_motion = {
            'qpos': np.random.randn(10, 7),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion
        
        mock_importer_instance = MagicMock()
        mock_importer.return_value = mock_importer_instance
        
        mock_ik_retarget.return_value = True
        
        # Make NPY application fail
        mock_apply_npy.return_value = False
        
        # Use test assets
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Mock importer to write temp files
        def fake_export_to_mujoco(path):
            Path(path).write_text('<mujoco><worldbody/></mujoco>')
        def fake_export_motion_data(path):
            np.save(path, {'qpos': np.zeros((10,7)), 'fps': 30.0})
        mock_importer_instance.export_to_mujoco.side_effect = fake_export_to_mujoco
        mock_importer_instance.export_motion_data.side_effect = fake_export_motion_data
        
        # Should handle failure
        result = glb2glb.transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
