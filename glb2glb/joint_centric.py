"""
Joint-Centric GLB↔MuJoCo Conversion Pipeline

This module provides the main interface for the joint-centric conversion approach
where GLB nodes represent joints (not bodies) for lossless bidirectional conversion.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import mujoco
from typing import Optional

from glb2glb.importer.joint_centric_importer import JointCentricImporter

# Lazy import for exporter to avoid pygltflib dependency when not needed
JointCentricExporter = None

def _get_exporter():
    global JointCentricExporter
    if JointCentricExporter is None:
        from glb2glb.exporter.joint_centric_exporter import JointCentricExporter as _JCE
        JointCentricExporter = _JCE
    return JointCentricExporter

logger = logging.getLogger(__name__)


def mujoco_to_glb(
    xml_path: str,
    output_path: str,
    motion_data: Optional[str] = None,
    fps: float = 30.0
) -> None:
    """
    Convert MuJoCo model to GLB using joint-centric approach.
    
    Args:
        xml_path: Path to MuJoCo XML file
        output_path: Path for output GLB file
        motion_data: Optional path to NPY motion data
        fps: Frames per second for animation
    """
    logger.info(f"Converting MuJoCo to GLB (joint-centric)")
    logger.info(f"  Input: {xml_path}")
    logger.info(f"  Output: {output_path}")
    
    # Create exporter
    Exporter = _get_exporter()
    exporter = Exporter(xml_path)
    
    # Add motion data if provided
    if motion_data:
        logger.info(f"  Loading motion from: {motion_data}")
        data = np.load(motion_data, allow_pickle=True)
        
        # Handle different NPY formats
        if hasattr(data, 'item') and callable(data.item):
            data = data.item()  # Convert numpy scalar to python object
            
        if isinstance(data, dict):
            qpos_data = data.get('qpos', data.get('motion', []))
            if 'fps' in data:
                fps = data['fps']
        else:
            qpos_data = data
            
        if hasattr(qpos_data, '__len__') and len(qpos_data) > 0:
            # Convert to list of arrays if needed
            if isinstance(qpos_data, np.ndarray):
                qpos_list = [qpos_data[i] for i in range(len(qpos_data))]
            else:
                qpos_list = qpos_data
                
            exporter.add_animation(qpos_list, fps)
            logger.info(f"  Added {len(qpos_list)} animation frames at {fps} FPS")
    
    # Export
    exporter.export_to_glb(output_path)
    
    logger.info(f"✅ Export complete:")
    logger.info(f"   Joints: {len(exporter.joints)}")
    logger.info(f"   Root joints: {len(exporter.root_joints)}")
    

def glb_to_mujoco(
    glb_path: str,
    xml_output: str,
    motion_output: Optional[str] = None
) -> None:
    """
    Convert GLB to MuJoCo model using joint-centric approach.
    
    Args:
        glb_path: Path to GLB file
        xml_output: Path for output MuJoCo XML
        motion_output: Optional path for output motion NPY
    """
    logger.info(f"Converting GLB to MuJoCo (joint-centric)")
    logger.info(f"  Input: {glb_path}")
    logger.info(f"  Output XML: {xml_output}")
    
    # Create importer
    importer = JointCentricImporter(glb_path)
    
    # Export MuJoCo model
    importer.export_to_mujoco(xml_output)
    
    # Export motion if requested
    if motion_output:
        importer.export_motion_data(motion_output)
        logger.info(f"  Output motion: {motion_output}")
    
    logger.info(f"✅ Import complete:")
    logger.info(f"   Joints found: {len(importer.joints)}")
    logger.info(f"   Root joints: {len(importer.root_joints)}")
    
    # Validate the output
    try:
        model = mujoco.MjModel.from_xml_path(xml_output)
        logger.info(f"   Valid MuJoCo model: {model.njnt} joints, {model.nq} DOFs")
    except Exception as e:
        logger.warning(f"   Model validation failed: {e}")


def round_trip_test(xml_path: str) -> bool:
    """
    Test round-trip conversion: MuJoCo → GLB → MuJoCo.
    
    Args:
        xml_path: Path to MuJoCo XML file
        
    Returns:
        True if round-trip preserves joint count and DOFs
    """
    import tempfile
    
    logger.info(f"Testing round-trip conversion for: {xml_path}")
    
    # Load original
    original = mujoco.MjModel.from_xml_path(xml_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to GLB
        glb_path = Path(tmpdir) / "temp.glb"
        mujoco_to_glb(xml_path, str(glb_path))
        
        # Import back to MuJoCo
        xml_reimported = Path(tmpdir) / "reimported.xml"
        glb_to_mujoco(str(glb_path), str(xml_reimported))
        
        # Compare
        reimported = mujoco.MjModel.from_xml_path(str(xml_reimported))
        
        joints_match = original.njnt == reimported.njnt
        dofs_match = original.nq == reimported.nq
        
        logger.info(f"Round-trip results:")
        logger.info(f"  Joints: {original.njnt} → {reimported.njnt} {'✅' if joints_match else '❌'}")
        logger.info(f"  DOFs: {original.nq} → {reimported.nq} {'✅' if dofs_match else '❌'}")
        
        return joints_match and dofs_match


def main():
    """Command-line interface for joint-centric conversion."""
    parser = argparse.ArgumentParser(
        description="Joint-Centric GLB↔MuJoCo Conversion"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Conversion direction')
    
    # MuJoCo to GLB
    mj2glb = subparsers.add_parser('mj2glb', help='Convert MuJoCo to GLB')
    mj2glb.add_argument('xml', help='Input MuJoCo XML file')
    mj2glb.add_argument('output', help='Output GLB file')
    mj2glb.add_argument('--motion', help='Motion data NPY file')
    mj2glb.add_argument('--fps', type=float, default=30.0, help='Animation FPS')
    
    # GLB to MuJoCo
    glb2mj = subparsers.add_parser('glb2mj', help='Convert GLB to MuJoCo')
    glb2mj.add_argument('glb', help='Input GLB file')
    glb2mj.add_argument('xml', help='Output MuJoCo XML file')
    glb2mj.add_argument('--motion', help='Output motion NPY file')
    
    # Round-trip test
    test = subparsers.add_parser('test', help='Test round-trip conversion')
    test.add_argument('xml', help='MuJoCo XML file to test')
    
    # Logging
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )
    
    # Execute command
    if args.command == 'mj2glb':
        mujoco_to_glb(args.xml, args.output, args.motion, args.fps)
    elif args.command == 'glb2mj':
        glb_to_mujoco(args.glb, args.xml, args.motion)
    elif args.command == 'test':
        success = round_trip_test(args.xml)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())
