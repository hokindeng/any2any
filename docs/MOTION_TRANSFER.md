# GLB Motion Transfer Pipeline

## Overview

This pipeline transfers animations from one GLB model to another while **preserving the target model's mesh**. It uses MuJoCo as an intermediate representation and IK-based retargeting for motion transfer.

## Key Innovation

The main challenge was that MuJoCo only handles skeletons (armatures), not meshes. When converting:
- **GLB → MuJoCo**: Mesh data is lost, only skeleton is preserved
- **MuJoCo → GLB**: Only skeleton can be exported, no mesh

**Solution**: Export retargeted animation to NPY format, then apply it to the original target GLB that still has its mesh.

## Pipeline Architecture

```
Source GLB (with animation)
    ↓
1. Import to MuJoCo
    ↓
Source MuJoCo + Motion NPY
    ↓
2. Add tracker sites
    ↓
3. IK-based Retargeting ← Target MuJoCo (from Target GLB)
    ↓
Retargeted Motion NPY
    ↓
4. Export animation to structured NPY
    ↓
5. Apply NPY animation to Target GLB
    ↓
Output GLB (mesh + retargeted animation)
```

## Usage

### Basic Usage

```bash
# Transfer animation from MyoSkeleton to Tripo panda (preserves mesh)
python final_motion_transfer.py \
    -s animations/myo_animation.glb \
    -t animations/tripo/panda_running.glb \
    -o output/panda_animated.glb
```

### Options

- `-s, --source`: Source GLB file with animation
- `-t, --target`: Target GLB file to receive animation  
- `-o, --output`: Output GLB file
- `--keep-npy`: Keep intermediate NPY animation file for debugging
- `--temp-dir`: Directory for temporary files (default: temporary)
- `-q, --quiet`: Suppress output messages

## Key Components

### 1. NPY Animation Exporter (`glb2glb/exporter/animation_exporter.py`)

Exports MuJoCo animation data to NPY format with joint-level animation data:

```python
from glb2glb.exporter.animation_exporter import export_animation_to_npy

animation_data = export_animation_to_npy(
    model_path="model.xml",
    qpos_data=qpos_frames,
    output_path="animation.npy",
    fps=30.0
)
```

The NPY file contains:
- `fps`: Frame rate
- `n_frames`: Number of frames
- `joints`: Dictionary of joint animations
  - Each joint has `rotations` and/or `translations` arrays
- `qpos`: Raw MuJoCo qpos data for reference

### 2. NPY to GLB Transfer (`glb2glb/transfer/npy_to_glb.py`)

Applies NPY animation to GLB with matching armature:

```python
from glb2glb.transfer import apply_animation_from_npy

success = apply_animation_from_npy(
    target_glb="model.glb",
    animation_npy="animation.npy",
    output_glb="animated.glb",
    verbose=True
)
```

### 3. IK-based Retargeting (`glb2glb/retargeter/ik_based_retargeting.py`)

Uses mink library for proper IK-based motion retargeting:

```python
from glb2glb.retargeter.ik_based_retargeting import ik_retarget_motion

success = ik_retarget_motion(
    source_xml="source.xml",
    target_xml="target.xml",
    source_motion_path="source_motion.npy",
    output_motion_path="retargeted_motion.npy",
    verbose=True
)
```

## Results

The final output GLB contains:
- ✅ **Original mesh preserved** from target GLB
- ✅ **Retargeted animation** from source GLB
- ✅ **Proper GLB structure** with all components

Example output structure:
```
Final GLB:
  Nodes: 41 (full skeleton hierarchy)
  Meshes: 1 (preserved from target)
  Animations: 1 (transferred from source)
  Channels: 39 (one per animated joint)
```

## Dependencies

```bash
# Core dependencies
pip install mujoco pygltflib numpy

# For IK retargeting
pip install pink  # or conda install -c conda-forge pink
```

## Advanced Usage

### Keep NPY for Custom Processing

```bash
python final_motion_transfer.py -s source.glb -t target.glb -o output.glb --keep-npy
```

This saves `output.animation.npy` alongside the GLB for custom processing.

### Direct NPY Application

If you have a pre-computed NPY animation:

```python
from glb2glb.transfer import apply_animation_from_npy

apply_animation_from_npy(
    target_glb="model.glb",
    animation_npy="custom_animation.npy", 
    output_glb="result.glb"
)
```

## Troubleshooting

1. **No animation in output**: Check that source GLB has animation and IK retargeting found matching joints
2. **Missing mesh**: Ensure you're using the NPY transfer method, not direct MuJoCo export
3. **IK retargeting fails**: Install mink/pink library: `pip install pink`
4. **Mismatched armatures**: The NPY transfer works best when target GLB armature matches the MuJoCo model structure

## Architecture Benefits

1. **Mesh Preservation**: Original geometry, textures, and materials are preserved
2. **Modular Pipeline**: Each step can be customized or replaced
3. **NPY Format**: Animation data can be inspected, modified, or generated programmatically
4. **Reusable Components**: NPY animations can be applied to multiple models with same armature

## Future Improvements

- [ ] Automatic armature matching and bone name mapping
- [ ] Support for multiple animations in single GLB
- [ ] Blend shape/morph target support
- [ ] Animation compression and optimization
- [ ] GUI for visual retargeting setup
