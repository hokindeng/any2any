"""
Unit tests for the GLB exporter module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import xml.etree.ElementTree as ET

from glb2glb.exporter.animation_exporter import export_animation_to_npy
from glb2glb.exporter.joint_centric_exporter import JointCentricExporter, JointNode


class TestAnimationExporter(unittest.TestCase):
    """Test the animation exporter functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_model(self):
        """Create a simple test MuJoCo model XML."""
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="joint1" type="hinge" axis="0 1 0"/>
                    <geom size="0.1"/>
                    <body name="body2" pos="1 0 0">
                        <joint name="joint2" type="ball"/>
                        <geom size="0.1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
        
        model_path = Path(self.temp_dir) / "test_model.xml"
        with open(model_path, 'w') as f:
            f.write(xml_content)
        return str(model_path)
        
    @patch('mujoco.MjModel')
    @patch('mujoco.mj_id2name')
    @patch('mujoco.mju_axisAngle2Quat')
    @patch('numpy.save')
    def test_export_animation_to_npy(
        self,
        mock_np_save,
        mock_axisAngle2Quat,
        mock_mj_id2name,
        mock_mjmodel
    ):
        """Test exporting animation to NPY format."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Setup model properties
        mock_model.njnt = 2
        mock_model.jnt_type = [1, 2]  # HINGE, BALL
        mock_model.jnt_axis = np.array([[0, 1, 0], [0, 0, 1]])
        mock_model.jnt_qposadr = [0, 1]
        
        # Mock joint names
        mock_mj_id2name.side_effect = ["joint1", "joint2"]
        
        # Mock quaternion conversion
        mock_axisAngle2Quat.side_effect = lambda q, axis, angle: q.__setitem__(slice(None), [0, 0, 0, 1])
        
        # Create test qpos data (3 frames)
        qpos_data = [
            np.array([0.1, 1.0, 0.0, 0.0, 0.0]),  # Frame 1
            np.array([0.2, 0.707, 0.0, 0.0, 0.707]),  # Frame 2
            np.array([0.3, 0.0, 0.0, 0.0, 1.0]),  # Frame 3
        ]
        
        # Export animation
        output_path = Path(self.temp_dir) / "animation.npy"
        result = export_animation_to_npy(
            model_path="test.xml",
            qpos_data=qpos_data,
            output_path=str(output_path),
            fps=24.0
        )
        
        # Verify the result structure
        self.assertIn('fps', result)
        self.assertEqual(result['fps'], 24.0)
        self.assertIn('n_frames', result)
        self.assertEqual(result['n_frames'], 3)
        self.assertIn('joints', result)
        self.assertEqual(len(result['joints']), 2)
        
        # Verify numpy save was called
        mock_np_save.assert_called_once()
        
    @patch('mujoco.MjModel')
    @patch('mujoco.mj_id2name')
    @patch('numpy.save')
    def test_export_animation_free_joint(
        self,
        mock_np_save,
        mock_mj_id2name,
        mock_mjmodel
    ):
        """Test exporting animation with free joint."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Setup model with free joint
        mock_model.njnt = 1
        mock_model.jnt_type = [0]  # FREE joint
        mock_model.jnt_axis = np.array([[0, 0, 0]])
        mock_model.jnt_qposadr = [0]
        
        # Mock joint name
        mock_mj_id2name.return_value = "root"
        
        # Create test qpos data with free joint (7 DOFs)
        qpos_data = [
            np.array([0, 0, 1, 1, 0, 0, 0]),  # pos + quat
            np.array([0, 0, 2, 0.707, 0.707, 0, 0]),
        ]
        
        # Export animation
        output_path = Path(self.temp_dir) / "animation.npy"
        result = export_animation_to_npy(
            model_path="test.xml",
            qpos_data=qpos_data,
            output_path=str(output_path),
            fps=30.0
        )
        
        # Verify free joint has both translations and rotations
        self.assertIn('root', result['joints'])
        root_joint = result['joints']['root']
        self.assertIn('translations', root_joint)
        self.assertIn('rotations', root_joint)
        self.assertEqual(root_joint['translations'].shape, (2, 3))
        self.assertEqual(root_joint['rotations'].shape, (2, 4))


class TestJointCentricExporter(unittest.TestCase):
    """Test the JointCentricExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_xml(self):
        """Create a test MuJoCo XML file."""
        xml_content = """<?xml version="1.0"?>
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="joint1" type="hinge" axis="0 1 0"/>
                    <geom size="0.1"/>
                </body>
            </worldbody>
        </mujoco>"""
        
        xml_path = Path(self.temp_dir) / "test.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        return str(xml_path)
        
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    @patch('pygltflib.GLTF2')
    def test_exporter_initialization(self, mock_gltf2, mock_mjdata, mock_mjmodel):
        """Test JointCentricExporter initialization."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Setup model properties
        mock_model.njnt = 1
        mock_model.joint.return_value.name = "joint1"
        mock_model.jnt_type = [3]  # HINGE
        mock_model.jnt_axis = np.array([[0, 1, 0]])
        mock_model.jnt_bodyid = [0]
        mock_model.jnt_qposadr = [0]
        mock_model.jnt_dofadr = [0]
        
        mock_body = MagicMock()
        mock_body.name = "body1"
        mock_body.pos = np.array([0, 0, 0])
        mock_body.parentid = 0
        mock_model.body.return_value = mock_body
        mock_model.nbody = 1
        
        # Create exporter
        xml_path = self.create_test_xml()
        exporter = JointCentricExporter(xml_path)
        
        # Verify initialization
        self.assertIsNotNone(exporter.model)
        self.assertIsNotNone(exporter.data)
        self.assertEqual(len(exporter.joints), 1)
        
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    def test_joint_node_creation(self, mock_mjdata, mock_mjmodel):
        """Test JointNode creation."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        
        # Setup joint properties
        mock_joint = MagicMock()
        mock_joint.name = "test_joint"
        mock_model.joint.return_value = mock_joint
        mock_model.jnt_type = [1]  # BALL
        mock_model.jnt_axis = np.array([[0, 1, 0]])
        mock_model.jnt_bodyid = [0]
        mock_model.jnt_qposadr = [0]
        mock_model.jnt_dofadr = [0]
        
        # Setup body
        mock_body = MagicMock()
        mock_body.name = "test_body"
        mock_body.pos = np.array([1, 0, 0])
        mock_model.body.return_value = mock_body
        
        # Create JointNode
        joint_node = JointNode(0, mock_model)
        
        # Verify properties
        self.assertEqual(joint_node.name, "test_joint")
        self.assertEqual(joint_node.type, 1)
        np.testing.assert_array_equal(joint_node.axis, [0, 1, 0])
        self.assertEqual(joint_node.body_name, "test_body")
        np.testing.assert_array_equal(joint_node.local_pos, [1, 0, 0])
        
    def test_joint_node_animation_frame(self):
        """Test adding animation frame to JointNode."""
        # Mock model
        mock_model = MagicMock()
        mock_joint = MagicMock()
        mock_joint.name = "joint1"
        mock_model.joint.return_value = mock_joint
        mock_model.jnt_type = [1]  # BALL
        mock_model.jnt_axis = np.array([[0, 1, 0]])
        mock_model.jnt_bodyid = [0]
        mock_model.jnt_qposadr = [0]
        mock_model.jnt_dofadr = [0]
        
        mock_body = MagicMock()
        mock_body.name = "body1"
        mock_body.pos = np.array([0, 0, 0])
        mock_model.body.return_value = mock_body
        
        # Create JointNode
        joint_node = JointNode(0, mock_model)
        
        # Add animation frames for ball joint (quaternions)
        qpos1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        qpos2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90 deg rotation
        
        joint_node.add_animation_frame(qpos1)
        joint_node.add_animation_frame(qpos2)
        
        # Verify rotation keys were added
        self.assertEqual(len(joint_node.rotation_keys), 2)
            
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    @patch('pygltflib.GLTF2')
    def test_add_animation(self, mock_gltf2, mock_mjdata, mock_mjmodel):
        """Test adding animation to exporter."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Setup simple model
        mock_model.njnt = 1
        mock_model.nbody = 1
        mock_model.nq = 1
        
        mock_joint = MagicMock()
        mock_joint.name = "joint1"
        mock_model.joint.return_value = mock_joint
        mock_model.jnt_type = [1]  # BALL
        mock_model.jnt_axis = np.array([[0, 1, 0]])
        mock_model.jnt_bodyid = [0]
        mock_model.jnt_qposadr = [0]
        mock_model.jnt_dofadr = [0]
        
        mock_body = MagicMock()
        mock_body.name = "body1"
        mock_body.pos = np.array([0, 0, 0])
        mock_body.parentid = 0
        mock_model.body.return_value = mock_body
        
        # Create exporter
        xml_path = self.create_test_xml()
        exporter = JointCentricExporter(xml_path)
        
        # Add animation data for ball joint (4 DOF quaternion)
        qpos_data = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            np.array([0.707, 0.707, 0.0, 0.0]),  # 90 deg rotation
            np.array([0.0, 1.0, 0.0, 0.0]),  # 180 deg rotation
        ]
        
        # Ball joints don't need mju_axisAngle2Quat
        exporter.add_animation(qpos_data, fps=30.0)
        
        # Verify animation was added to joints
        for joint in exporter.joints:
            self.assertEqual(len(joint.rotation_keys), 3)


class TestJointHierarchy(unittest.TestCase):
    """Test joint hierarchy building in exporter."""
    
    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    def test_build_joint_hierarchy_simple(self, mock_mjdata, mock_mjmodel):
        """Test building simple joint hierarchy."""
        # Mock MuJoCo model
        mock_model = MagicMock()
        mock_mjmodel.from_xml_path.return_value = mock_model
        
        # Setup 2 joints in parent-child relationship
        mock_model.njnt = 2
        mock_model.nbody = 3  # world + 2 bodies
        
        # Joint 0 (parent)
        mock_joint0 = MagicMock()
        mock_joint0.name = "joint0"
        
        # Joint 1 (child)
        mock_joint1 = MagicMock()
        mock_joint1.name = "joint1"
        
        mock_model.joint.side_effect = [mock_joint0, mock_joint1]
        mock_model.jnt_type = [1, 1]  # Both BALL
        mock_model.jnt_axis = np.array([[0, 1, 0], [0, 1, 0]])
        mock_model.jnt_bodyid = [1, 2]  # Different bodies
        mock_model.jnt_qposadr = [0, 1]
        mock_model.jnt_dofadr = [0, 1]
        
        # Body 0 (world)
        mock_body0 = MagicMock()
        mock_body0.name = "world"
        mock_body0.parentid = 0
        mock_body0.pos = np.array([0, 0, 0])
        
        # Body 1 (parent joint's body)
        mock_body1 = MagicMock()
        mock_body1.name = "body1"
        mock_body1.parentid = 0  # Child of world
        mock_body1.pos = np.array([0, 0, 0])
        
        # Body 2 (child joint's body)
        mock_body2 = MagicMock()
        mock_body2.name = "body2"
        mock_body2.parentid = 1  # Child of body1
        mock_body2.pos = np.array([1, 0, 0])
        
        # Create a proper body getter
        body_map = {0: mock_body0, 1: mock_body1, 2: mock_body2}
        mock_model.body.side_effect = lambda idx: body_map[idx]
        
        # Create exporter
        from glb2glb.exporter.joint_centric_exporter import JointCentricExporter
        exporter = JointCentricExporter.__new__(JointCentricExporter)
        exporter.model = mock_model
        exporter.data = mock_mjdata
        exporter.joints = []
        exporter.joint_map = {}
        
        # Build hierarchy
        exporter._build_joint_hierarchy()
        
        # Verify hierarchy
        self.assertEqual(len(exporter.joints), 2)
        self.assertEqual(len(exporter.root_joints), 1)
        
        # Check parent-child relationship
        parent_joint = exporter.joints[0]
        child_joint = exporter.joints[1]
        
        self.assertIsNone(parent_joint.parent)
        self.assertEqual(child_joint.parent, parent_joint)
        self.assertIn(child_joint, parent_joint.children)


if __name__ == '__main__':
    unittest.main()
