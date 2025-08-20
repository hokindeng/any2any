"""
Unit tests for the retargeter module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock, Mock

# Only test if mink is available
try:
    import mink
    from glb2glb.retargeter.ik_based_retargeting import (
        IKBasedRetargeter,
        ik_retarget_motion
    )
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False


@unittest.skipUnless(MINK_AVAILABLE, "mink not installed")
class TestIKBasedRetargeter(unittest.TestCase):
    """Test the IKBasedRetargeter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_xml_with_sites(self):
        """Create a test MuJoCo XML with sites."""
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="joint1" type="free"/>
                    <geom size="0.1"/>
                    <site name="body1_site" pos="0 0 0"/>
                    <body name="body2" pos="1 0 0">
                        <joint name="joint2" type="hinge"/>
                        <geom size="0.1"/>
                        <site name="body2_site" pos="0 0 0"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
        
        xml_path = Path(self.temp_dir) / "test_with_sites.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        return str(xml_path)
        
    def create_target_xml(self):
        """Create a target MuJoCo XML."""
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="target_body1">
                    <joint name="target_joint1" type="free"/>
                    <geom size="0.1"/>
                    <body name="target_body2" pos="0.5 0 0">
                        <joint name="target_joint2" type="ball"/>
                        <geom size="0.1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
        
        xml_path = Path(self.temp_dir) / "target.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        return str(xml_path)
        
    @patch('mujoco.mj_forward')
    @patch('mujoco.mj_id2name')
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    @patch('mink.Configuration')
    def test_retargeter_initialization(
        self,
        mock_config,
        mock_mjdata,
        mock_mjmodel,
        mock_mj_id2name,
        mock_mj_forward
    ):
        """Test IKBasedRetargeter initialization."""
        # Mock source model
        source_model = MagicMock()
        source_model.njnt = 2
        source_model.nq = 8  # free + hinge
        source_model.nv = 7
        source_model.nsite = 2
        source_model.site_name = ["body1_site", "body2_site"]
        mock_mjmodel.from_xml_path.side_effect = [source_model, None]
        
        # Mock target model
        target_model = MagicMock()
        target_model.njnt = 2
        target_model.nq = 11  # free + ball
        target_model.nv = 9
        target_model.nbody = 3
        mock_mjmodel.from_xml_path.side_effect = [source_model, target_model]
        
        # Mock mink configuration
        mock_config_instance = MagicMock()
        mock_config_instance.model = target_model
        mock_config_instance.data = mock_mjdata()
        mock_config_instance.q = np.zeros(11)  # Match target_model.nq
        mock_config.return_value = mock_config_instance
        
        # Mock mj_id2name to return appropriate names
        mock_mj_id2name.side_effect = [
            "body1_site", "body2_site",  # Source sites
            "target_body0", "target_body1", "target_body2"  # Target bodies
        ] * 10  # Repeat for multiple calls
        
        # Mock mj_forward to do nothing
        mock_mj_forward.return_value = None
        
        # Create retargeter
        source_xml = self.create_test_xml_with_sites()
        target_xml = self.create_target_xml()
        
        retargeter = IKBasedRetargeter(
            source_xml=source_xml,
            target_xml=target_xml,
            verbose=False
        )
        
        # Verify initialization
        self.assertIsNotNone(retargeter.source_model)
        self.assertIsNotNone(retargeter.target_model)
        self.assertIsNotNone(retargeter.configuration)
        
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    @patch('mujoco.mj_id2name')
    def test_build_site_mapping(
        self,
        mock_id2name,
        mock_mjdata,
        mock_mjmodel
    ):
        """Test building site to body mapping."""
        # Mock models
        source_model = MagicMock()
        source_model.nsite = 2
        source_model.site_bodyid = [1, 2]
        
        target_model = MagicMock()
        target_model.nbody = 3
        
        mock_mjmodel.from_xml_path.side_effect = [source_model, target_model]
        
        # Mock site and body names
        mock_id2name.side_effect = [
            "body1_site",  # Site 0 name
            "body2_site",  # Site 1 name
            "target_body1",  # Body 0 name
            "target_body2",  # Body 1 name
            "target_body3",  # Body 2 name
        ]
        
        # Create retargeter
        from glb2glb.retargeter.ik_based_retargeting import IKBasedRetargeter
        retargeter = IKBasedRetargeter.__new__(IKBasedRetargeter)
        retargeter.source_model = source_model
        retargeter.target_model = target_model
        retargeter.source_data = mock_mjdata()
        retargeter.target_data = mock_mjdata()
        
        # Build mapping and store it
        retargeter.site_map = retargeter._create_site_mapping()
        
        # Verify mapping
        self.assertIn("body1_site", retargeter.site_map)
        self.assertIn("body2_site", retargeter.site_map)
        
    @patch('mink.solve_ik')
    @patch('mink.Configuration')
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    @patch('mujoco.mj_kinematics')
    @patch('mujoco.mj_forward')
    def test_retarget_motion(
        self,
        mock_forward,
        mock_kinematics,
        mock_mjdata,
        mock_mjmodel,
        mock_config,
        mock_solve_ik
    ):
        """Test motion retargeting."""
        # Mock models
        source_model = MagicMock()
        source_model.njnt = 1
        source_model.nq = 7  # free joint
        source_model.nv = 6
        source_model.nsite = 1
        
        target_model = MagicMock()
        target_model.njnt = 1
        target_model.nq = 7
        target_model.nv = 6
        target_model.nbody = 2
        
        mock_mjmodel.from_xml_path.side_effect = [source_model, target_model]
        
        # Mock data
        source_data = MagicMock()
        source_data.qpos = np.zeros(7)
        source_data.site_xpos = np.zeros((1, 3))
        source_data.site_xmat = np.eye(3).flatten()
        
        target_data = MagicMock()
        target_data.qpos = np.zeros(7)
        
        mock_mjdata.side_effect = [source_data, target_data]
        
        # Mock configuration
        mock_config_instance = MagicMock()
        mock_config_instance.q = np.zeros(7)
        mock_config.return_value = mock_config_instance
        
        # Mock IK solver
        mock_solve_ik.return_value = np.zeros(6)  # Velocity
        
        # Create retargeter
        from glb2glb.retargeter.ik_based_retargeting import IKBasedRetargeter
        retargeter = IKBasedRetargeter.__new__(IKBasedRetargeter)
        retargeter.source_model = source_model
        retargeter.target_model = target_model
        retargeter.source_data = source_data
        retargeter.target_data = target_data
        retargeter.configuration = mock_config_instance
        retargeter.site_to_body_map = {}
        retargeter.tasks = []
        retargeter.frame_tasks = []
        retargeter.dt = 0.01
        retargeter.max_iters = 10
        retargeter.solver = "quadprog"
        retargeter.verbose = False
        
        # Test motion
        source_motion = np.random.randn(5, 7)  # 5 frames
        result = retargeter.retarget_motion(source_motion, fps=30.0)
        
        # Verify result
        self.assertIn('qpos', result)
        self.assertIn('fps', result)
        self.assertIn('n_frames', result)
        self.assertEqual(result['qpos'].shape, (5, 7))
        self.assertEqual(result['fps'], 30.0)
        self.assertEqual(result['n_frames'], 5)


@unittest.skipUnless(MINK_AVAILABLE, "mink not installed")
class TestIKRetargetMotionFunction(unittest.TestCase):
    """Test the ik_retarget_motion convenience function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('glb2glb.retargeter.ik_based_retargeting.IKBasedRetargeter')
    @patch('numpy.load')
    @patch('numpy.save')
    def test_ik_retarget_motion_success(
        self,
        mock_np_save,
        mock_np_load,
        mock_retargeter_class
    ):
        """Test successful motion retargeting."""
        # Mock motion data
        motion_data = {
            'qpos': np.random.randn(10, 7),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = motion_data
        
        # Mock retargeter
        mock_retargeter = MagicMock()
        mock_retargeter_class.return_value = mock_retargeter
        
        # Mock retarget result
        retargeted = {
            'qpos': np.random.randn(10, 7),
            'fps': 30.0,
            'n_frames': 10
        }
        mock_retargeter.retarget_motion.return_value = retargeted
        
        # Call function
        result = ik_retarget_motion(
            source_xml="source.xml",
            target_xml="target.xml",
            source_motion_path="motion.npy",
            output_motion_path="output.npy",
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify calls
        mock_retargeter_class.assert_called_once_with(
            source_xml="source.xml",
            target_xml="target.xml",
            verbose=False
        )
        mock_retargeter.retarget_motion.assert_called_once()
        mock_np_save.assert_called_once()
        
    @patch('glb2glb.retargeter.ik_based_retargeting.IKBasedRetargeter')
    @patch('numpy.load')
    def test_ik_retarget_motion_failure(
        self,
        mock_np_load,
        mock_retargeter_class
    ):
        """Test handling of retargeting failure."""
        # Mock motion data
        motion_data = {
            'qpos': np.random.randn(10, 7),
            'fps': 30.0
        }
        mock_np_load.return_value.item.return_value = motion_data
        
        # Mock retargeter failure
        mock_retargeter_class.side_effect = Exception("Retargeting failed")
        
        # Call function
        result = ik_retarget_motion(
            source_xml="source.xml",
            target_xml="target.xml",
            source_motion_path="motion.npy",
            output_motion_path="output.npy",
            verbose=False
        )
        
        self.assertFalse(result)


class TestRetargeterInit(unittest.TestCase):
    """Test the retargeter module initialization."""
    
    def test_check_retargeter_dependencies(self):
        """Test dependency checking function."""
        from glb2glb.retargeter import check_retargeter_dependencies
        
        # Call the function
        result = check_retargeter_dependencies()
        
        # Verify result structure
        self.assertIn('ik', result)
        self.assertIn('visualization', result)
        
        # Check that it returns booleans
        self.assertIsInstance(result['ik'], bool)
        self.assertIsInstance(result['visualization'], bool)
        
    @patch('glb2glb.retargeter.warnings.warn')
    def test_import_warnings(self, mock_warn):
        """Test that import warnings are generated for missing dependencies."""
        # Re-import to trigger warnings
        import importlib
        import glb2glb.retargeter
        
        # Force reload to trigger import logic
        importlib.reload(glb2glb.retargeter)
        
        # If mink is not installed, should have warning
        if not MINK_AVAILABLE:
            # Check that warning was called
            self.assertTrue(any(
                'mink' in str(call) 
                for call in mock_warn.call_args_list
            ))


if __name__ == '__main__':
    unittest.main()
