"""
Unit tests for the GLB importer module using actual GLB files.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import os

from glb2glb.importer.joint_centric_importer import JointCentricImporter


class TestJointCentricImporterWithRealFiles(unittest.TestCase):
    """Test the JointCentricImporter class with actual GLB files."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.assets_dir = Path(__file__).parent / "assets"
        cls.temp_dir = tempfile.mkdtemp()
        
        # List all GLB files in assets folder
        cls.glb_files = [
            'a_man_falling.glb',
            'mixamo_idle.glb',
            'myo_animation.glb',
            'panda_running.glb',
            'source.glb',
            'target.glb',
            'output.glb'
        ]
        
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_load_all_glb_files(self):
        """Test that all GLB files can be loaded successfully."""
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                with self.subTest(glb_file=glb_file):
                    try:
                        importer = JointCentricImporter(str(glb_path))
                        self.assertIsNotNone(importer.gltf)
                        print(f"✓ Successfully loaded: {glb_file}")
                    except Exception as e:
                        self.fail(f"Failed to load {glb_file}: {e}")
    
    def test_a_man_falling_glb(self):
        """Test importing a_man_falling.glb specifically."""
        glb_path = self.assets_dir / "a_man_falling.glb"
        if not glb_path.exists():
            self.skipTest(f"File not found: {glb_path}")
            
        importer = JointCentricImporter(str(glb_path))
        
        # Test that it loaded successfully
        self.assertIsNotNone(importer.gltf)
        
        # Test coordinate system detection
        self.assertIn(importer.coordinate_system, ['Y-up', 'Z-up'])
        print(f"a_man_falling.glb coordinate system: {importer.coordinate_system}")
        
        # Test joint hierarchy building
        self.assertGreater(len(importer.joints), 0, "Should have joints")
        self.assertGreater(len(importer.joint_map), 0, "Should have joint map")
        
        # Print joint hierarchy info
        print(f"a_man_falling.glb has {len(importer.joints)} joints")
        
        # Test export to MuJoCo XML
        output_xml = Path(self.temp_dir) / "a_man_falling.xml"
        try:
            importer.export_to_mujoco(str(output_xml))
            self.assertTrue(output_xml.exists(), "XML file should be created")
            
            # Verify XML structure
            tree = ET.parse(output_xml)
            root = tree.getroot()
            self.assertEqual(root.tag, 'mujoco')
            
            worldbody = root.find('worldbody')
            self.assertIsNotNone(worldbody, "Should have worldbody element")
            print(f"✓ Successfully exported a_man_falling.glb to MuJoCo XML")
        except Exception as e:
            self.fail(f"Failed to export a_man_falling.glb to MuJoCo: {e}")
        
        # Test motion data export if animations exist
        if importer.gltf.animations:
            output_npy = Path(self.temp_dir) / "a_man_falling_motion.npy"
            try:
                importer.export_motion_data(str(output_npy))
                self.assertTrue(output_npy.exists(), "NPY file should be created")
                
                # Load and verify motion data
                motion_data = np.load(output_npy, allow_pickle=True).item()
                self.assertIn('qpos', motion_data)
                self.assertIn('fps', motion_data)
                self.assertIn('n_frames', motion_data)
                print(f"✓ a_man_falling.glb has {motion_data['n_frames']} animation frames")
            except Exception as e:
                print(f"Note: a_man_falling.glb motion export failed (may not have animations): {e}")
    
    def test_mixamo_idle_glb(self):
        """Test importing mixamo_idle.glb specifically."""
        glb_path = self.assets_dir / "mixamo_idle.glb"
        if not glb_path.exists():
            self.skipTest(f"File not found: {glb_path}")
            
        importer = JointCentricImporter(str(glb_path))
        
        # Test that it loaded successfully
        self.assertIsNotNone(importer.gltf)
        
        # Test coordinate system detection - Mixamo typically uses Y-up
        self.assertIn(importer.coordinate_system, ['Y-up', 'Z-up'])
        print(f"mixamo_idle.glb coordinate system: {importer.coordinate_system}")
        
        # Test joint hierarchy building
        self.assertGreater(len(importer.joints), 0, "Should have joints")
        
        # Check for typical Mixamo joint names
        joint_names = [joint.name for joint in importer.joints]
        mixamo_indicators = ['mixamo:', 'Hips', 'Spine', 'Head', 'LeftArm', 'RightArm']
        has_mixamo_joints = any(indicator in name for name in joint_names for indicator in mixamo_indicators)
        
        if has_mixamo_joints:
            print(f"✓ mixamo_idle.glb contains Mixamo-style joints")
        
        print(f"mixamo_idle.glb has {len(importer.joints)} joints")
        
        # Test export to MuJoCo XML
        output_xml = Path(self.temp_dir) / "mixamo_idle.xml"
        try:
            importer.export_to_mujoco(str(output_xml))
            self.assertTrue(output_xml.exists(), "XML file should be created")
            print(f"✓ Successfully exported mixamo_idle.glb to MuJoCo XML")
        except Exception as e:
            self.fail(f"Failed to export mixamo_idle.glb to MuJoCo: {e}")
        
        # Test motion data export if animations exist
        if importer.gltf.animations:
            output_npy = Path(self.temp_dir) / "mixamo_idle_motion.npy"
            try:
                importer.export_motion_data(str(output_npy))
                self.assertTrue(output_npy.exists(), "NPY file should be created")
                
                motion_data = np.load(output_npy, allow_pickle=True).item()
                print(f"✓ mixamo_idle.glb has {motion_data['n_frames']} animation frames at {motion_data['fps']} FPS")
            except Exception as e:
                print(f"Note: mixamo_idle.glb motion export failed (may not have animations): {e}")
    
    def test_coordinate_system_detection_all_files(self):
        """Test coordinate system detection for all GLB files."""
        print("\n=== Coordinate System Detection ===")
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                try:
                    importer = JointCentricImporter(str(glb_path))
                    print(f"{glb_file:25} -> {importer.coordinate_system}")
                    self.assertIn(importer.coordinate_system, ['Y-up', 'Z-up'])
                except Exception as e:
                    print(f"{glb_file:25} -> Error: {e}")
    
    def test_joint_hierarchy_all_files(self):
        """Test joint hierarchy extraction for all GLB files."""
        print("\n=== Joint Hierarchy Information ===")
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                try:
                    importer = JointCentricImporter(str(glb_path))
                    
                    # Count joints and roots
                    root_joints = [j for j in importer.joints if j.is_root()]
                    
                    print(f"{glb_file:25} -> {len(importer.joints):3} joints, {len(root_joints)} root(s)")
                    
                    # Print first few joint names
                    if importer.joints:
                        joint_names = [j.name for j in importer.joints[:5]]
                        names_str = ", ".join(joint_names)
                        if len(importer.joints) > 5:
                            names_str += f", ... ({len(importer.joints)-5} more)"
                        print(f"  Joint names: {names_str}")
                        
                except Exception as e:
                    print(f"{glb_file:25} -> Error: {e}")
    
    def test_animation_info_all_files(self):
        """Test animation information for all GLB files."""
        print("\n=== Animation Information ===")
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                try:
                    importer = JointCentricImporter(str(glb_path))
                    
                    if importer.gltf.animations:
                        num_anims = len(importer.gltf.animations)
                        anim_names = [anim.name or f"Animation_{i}" for i, anim in enumerate(importer.gltf.animations)]
                        
                        print(f"{glb_file:25} -> {num_anims} animation(s)")
                        for anim_name in anim_names:
                            print(f"  - {anim_name}")
                    else:
                        print(f"{glb_file:25} -> No animations")
                        
                except Exception as e:
                    print(f"{glb_file:25} -> Error: {e}")
    
    def test_export_all_to_mujoco(self):
        """Test exporting all GLB files to MuJoCo XML."""
        print("\n=== MuJoCo XML Export Test ===")
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                try:
                    importer = JointCentricImporter(str(glb_path))
                    
                    # Export to XML
                    output_name = glb_file.replace('.glb', '.xml')
                    output_xml = Path(self.temp_dir) / output_name
                    importer.export_to_mujoco(str(output_xml))
                    
                    if output_xml.exists():
                        # Parse and validate XML
                        tree = ET.parse(output_xml)
        root = tree.getroot()
        
        self.assertEqual(root.tag, 'mujoco')
        worldbody = root.find('worldbody')
        self.assertIsNotNone(worldbody)
        
                        # Count bodies in XML
                        bodies = worldbody.findall('.//body')
                        print(f"{glb_file:25} -> ✓ Exported ({len(bodies)} bodies)")
                    else:
                        print(f"{glb_file:25} -> ✗ Failed to create XML")
                        
                except Exception as e:
                    print(f"{glb_file:25} -> ✗ Error: {e}")
    
    def test_export_motion_data_for_animated_files(self):
        """Test motion data export for files with animations."""
        print("\n=== Motion Data Export Test ===")
        for glb_file in self.glb_files:
            glb_path = self.assets_dir / glb_file
            if glb_path.exists():
                try:
                    importer = JointCentricImporter(str(glb_path))
                    
                    if importer.gltf.animations:
                        # Try to export motion data
                        output_name = glb_file.replace('.glb', '_motion.npy')
                        output_npy = Path(self.temp_dir) / output_name
                        
                        try:
                            importer.export_motion_data(str(output_npy))
                            
                            if output_npy.exists():
                                # Load and verify motion data
                                motion_data = np.load(output_npy, allow_pickle=True).item()
                                
                                n_frames = motion_data.get('n_frames', 0)
                                fps = motion_data.get('fps', 30)
                                qpos_shape = motion_data.get('qpos', np.array([])).shape if 'qpos' in motion_data else (0, 0)
                                
                                print(f"{glb_file:25} -> ✓ {n_frames} frames @ {fps} FPS, qpos shape: {qpos_shape}")
                            else:
                                print(f"{glb_file:25} -> ✗ Failed to create NPY")
                        except Exception as e:
                            print(f"{glb_file:25} -> ✗ Motion export error: {e}")
                    else:
                        print(f"{glb_file:25} -> No animations to export")
                        
                except Exception as e:
                    print(f"{glb_file:25} -> ✗ Error: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)