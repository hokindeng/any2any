#!/usr/bin/env python
"""
Script to import GLB files and export to MuJoCo XML and NPY motion data.
"""

from pathlib import Path
import sys
sys.path.append('/Users/myo/any2any')

from glb2glb.importer.joint_centric_importer import JointCentricImporter

def process_glb_file(glb_path, output_dir):
    """Process a single GLB file and export to XML and NPY."""
    glb_name = glb_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {glb_path.name}")
    print(f"{'='*60}")
    
    try:
        # Load the GLB file
        importer = JointCentricImporter(str(glb_path))
        print(f"✓ Loaded GLB file")
        print(f"  Coordinate system: {importer.coordinate_system}")
        print(f"  Number of joints: {len(importer.joints)}")
        
        # Export to MuJoCo XML
        xml_output = output_dir / f"{glb_name}.xml"
        importer.export_to_mujoco(str(xml_output))
        print(f"✓ Exported to XML: {xml_output.name}")
        
        # Export motion data to NPY (if animations exist)
        if importer.gltf.animations:
            npy_output = output_dir / f"{glb_name}_motion.npy"
            try:
                importer.export_motion_data(str(npy_output))
                print(f"✓ Exported motion to NPY: {npy_output.name}")
                
                # Load and display motion info
                import numpy as np
                motion_data = np.load(npy_output, allow_pickle=True).item()
                print(f"  Animation info:")
                print(f"    - Frames: {motion_data.get('n_frames', 0)}")
                print(f"    - FPS: {motion_data.get('fps', 30)}")
                if 'qpos' in motion_data:
                    print(f"    - qpos shape: {motion_data['qpos'].shape}")
            except Exception as e:
                print(f"⚠ Motion export failed: {e}")
        else:
            print("  No animations found in GLB file")
            
    except Exception as e:
        print(f"✗ Failed to process {glb_path.name}: {e}")
        return False
    
    return True

def main():
    # Define paths
    assets_dir = Path("/Users/myo/any2any/tests/assets")
    output_dir = Path("/Users/myo/any2any/temporary")
    
    # List of GLB files to process (excluding the problematic output.glb)
    glb_files = [
        'a_man_falling.glb',
        'mixamo_idle.glb',
        'myo_animation.glb',
        'panda_running.glb',
        'source.glb',
        'target.glb'
    ]
    
    print("GLB to MuJoCo Converter")
    print(f"Input directory: {assets_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(glb_files)}")
    
    # Process each GLB file
    successful = 0
    failed = 0
    
    for glb_file in glb_files:
        glb_path = assets_dir / glb_file
        if glb_path.exists():
            if process_glb_file(glb_path, output_dir):
                successful += 1
            else:
                failed += 1
        else:
            print(f"⚠ File not found: {glb_file}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(glb_files)} files")
    if failed > 0:
        print(f"Failed: {failed} files")
    
    # List generated files
    print(f"\nGenerated files in {output_dir}:")
    xml_files = list(output_dir.glob("*.xml"))
    npy_files = list(output_dir.glob("*_motion.npy"))
    
    if xml_files:
        print(f"\nXML files ({len(xml_files)}):")
        for f in sorted(xml_files):
            print(f"  - {f.name}")
    
    if npy_files:
        print(f"\nNPY motion files ({len(npy_files)}):")
        for f in sorted(npy_files):
            print(f"  - {f.name}")

if __name__ == "__main__":
    main()
