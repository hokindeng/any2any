"""
Unit tests for the motion transfer pipeline.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock

from glb2glb.pipeline.motion_transfer import (
    prepare_model_with_sites,
    transfer_animation
)


class TestPrepareModelWithSites(unittest.TestCase):
    """Test the prepare_model_with_sites function."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
        
    def test_add_sites_to_simple_model(self):
        """Test adding sites to a simple MuJoCo model."""
        # Create a simple test XML
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="joint1" type="hinge"/>
                    <geom size="0.1"/>
                    <body name="body2">
                        <joint name="joint2" type="hinge"/>
                        <geom size="0.1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
        
        input_path = Path(self.temp_dir) / "input.xml"
        output_path = Path(self.temp_dir) / "output.xml"
        
        with open(input_path, 'w') as f:
            f.write(xml_content)
            
        # Add sites
        sites_added = prepare_model_with_sites(
            str(input_path), 
            str(output_path),
            verbose=False
        )
        
        # Check that sites were added
        self.assertEqual(sites_added, 2)
        self.assertTrue(output_path.exists())
        
        # Verify the output XML contains sites
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn('site', content)
            self.assertIn('body1_site', content)
            self.assertIn('body2_site', content)
            
    def test_skip_existing_sites(self):
        """Test that existing sites are not duplicated."""
        # Create XML with existing site
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="body1">
                    <site name="body1_site" pos="0 0 0"/>
                    <joint name="joint1" type="hinge"/>
                </body>
            </worldbody>
        </mujoco>"""
        
        input_path = Path(self.temp_dir) / "input.xml"
        output_path = Path(self.temp_dir) / "output.xml"
        
        with open(input_path, 'w') as f:
            f.write(xml_content)
            
        # Add sites
        sites_added = prepare_model_with_sites(
            str(input_path), 
            str(output_path),
            verbose=False
        )
        
        # Should not add duplicate site
        self.assertEqual(sites_added, 0)


class TestTransferAnimation(unittest.TestCase):
    """Test the transfer_animation function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('glb2glb.pipeline.motion_transfer.export_animation_to_npy')
    @patch('glb2glb.pipeline.motion_transfer.apply_animation_from_npy')
    @patch('glb2glb.pipeline.motion_transfer.ik_retarget_motion')
    @patch('numpy.load')
    def test_transfer_animation_success(
        self, 
        mock_np_load,
        mock_ik_retarget,
        mock_apply_npy,
        mock_export_npy,
        mock_importer
    ):
        """Test successful animation transfer."""
        # Mock the motion data
        mock_motion_data = {
            'qpos': np.zeros((10, 7)),  # 10 frames, 7 DOFs
            'fps': 30.0
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
        
        # Mock retargeting success
        mock_ik_retarget.return_value = True
        
        # Mock NPY application success
        mock_apply_npy.return_value = True
        
        # Use test assets for source/target GLB
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"  
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Run transfer
        result = transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify the pipeline was called correctly
        self.assertEqual(mock_importer.call_count, 2)  # Once for source, once for target
        mock_export_npy.assert_called_once()
        mock_apply_npy.assert_called_once()
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('numpy.load')
    def test_transfer_animation_no_source_animation(
        self,
        mock_np_load,
        mock_importer
    ):
        """Test handling of source without animation."""
        # Mock motion data with only 1 frame (no animation)
        mock_motion_data = {
            'qpos': np.zeros((1, 7)),  # Only 1 frame
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion_data
        
        # Mock importer
        mock_importer_instance = MagicMock()
        mock_importer.return_value = mock_importer_instance
        
        # Use test assets for source/target GLB
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"  
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Run transfer
        result = transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        # Should fail due to no animation
        self.assertFalse(result)
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('glb2glb.pipeline.motion_transfer.export_animation_to_npy')
    @patch('glb2glb.pipeline.motion_transfer.apply_animation_from_npy')
    @patch('glb2glb.pipeline.motion_transfer.ik_retarget_motion')
    @patch('numpy.load')
    @patch('shutil.copy')
    def test_transfer_animation_fallback(
        self,
        mock_shutil_copy,
        mock_np_load,
        mock_ik_retarget,
        mock_apply_npy,
        mock_export_npy,
        mock_importer
    ):
        """Test fallback when IK retargeting fails."""
        # Mock the motion data
        mock_motion_data = {
            'qpos': np.zeros((10, 7)),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion_data
        
        # Mock importer writes required temp outputs
        mock_importer_instance = MagicMock()
        def fake_export_to_mujoco(path):
            Path(path).write_text('<mujoco><worldbody/></mujoco>')
        def fake_export_motion_data(path):
            np.save(path, {'qpos': np.zeros((10,7)), 'fps': 30.0})
        mock_importer_instance.export_to_mujoco.side_effect = fake_export_to_mujoco
        mock_importer_instance.export_motion_data.side_effect = fake_export_motion_data
        mock_importer.return_value = mock_importer_instance
        
        # Mock IK retargeting failure
        mock_ik_retarget.side_effect = RuntimeError("IK failed")
        
        # Mock NPY application success
        mock_apply_npy.return_value = True
        
        # Use test assets for source/target GLB
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"  
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Run transfer
        result = transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify fallback (shutil.copy) was called
        mock_shutil_copy.assert_called_once()
        
    @patch('glb2glb.pipeline.motion_transfer.JointCentricImporter')
    @patch('glb2glb.pipeline.motion_transfer.export_animation_to_npy')
    @patch('glb2glb.pipeline.motion_transfer.apply_animation_from_npy')
    @patch('glb2glb.pipeline.motion_transfer.ik_retarget_motion')
    @patch('numpy.load')
    @patch('shutil.copy')
    def test_transfer_animation_keep_npy(
        self,
        mock_shutil_copy,
        mock_np_load,
        mock_ik_retarget,
        mock_apply_npy,
        mock_export_npy,
        mock_importer
    ):
        """Test keeping NPY file option."""
        # Mock the motion data
        mock_motion_data = {
            'qpos': np.zeros((10, 7)),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = mock_motion_data
        
        # Mock importer writes required temp outputs
        mock_importer_instance = MagicMock()
        def fake_export_to_mujoco(path):
            Path(path).write_text('<mujoco><worldbody/></mujoco>')
        def fake_export_motion_data(path):
            np.save(path, {'qpos': np.zeros((10,7)), 'fps': 30.0})
        mock_importer_instance.export_to_mujoco.side_effect = fake_export_to_mujoco
        mock_importer_instance.export_motion_data.side_effect = fake_export_motion_data
        mock_importer.return_value = mock_importer_instance
        
        # Mock retargeting and NPY application success
        mock_ik_retarget.return_value = True
        mock_apply_npy.return_value = True
        
        # Use test assets for source/target GLB
        assets_dir = Path(__file__).parent / "assets"
        source_file = assets_dir / "source.glb"
        target_file = assets_dir / "target.glb"  
        output_file = Path(self.temp_dir) / "output.glb"
        
        # Run transfer with keep_npy=True
        result = transfer_animation(
            source_glb=str(source_file),
            target_glb=str(target_file),
            output_glb=str(output_file),
            keep_npy=True,
            temp_dir=self.temp_dir,
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify NPY file was copied for keep_npy
        self.assertGreaterEqual(mock_shutil_copy.call_count, 1)


if __name__ == '__main__':
    unittest.main()
