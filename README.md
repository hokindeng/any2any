# glb2glb - Complete GLB↔MuJoCo Conversion & Motion Transfer

A powerful toolkit for converting between GLB/GLTF and MuJoCo formats, with advanced motion retargeting capabilities.

## 🎯 Key Features

- ✅ **Bidirectional Conversion** - Seamless GLB ↔ MuJoCo conversion
- ✅ **Motion Transfer** - Transfer animations between different GLB models
- ✅ **Coordinate System Handling** - Automatic Y-up/Z-up conversion
- ✅ **IK-Based Retargeting** - Advanced retargeting with inverse kinematics (optional)
- ✅ **Physics Preservation** - Maintains dynamics for simulation
- ✅ **Joint-Centric Architecture** - Preserves DOF structure
- ✅ **NPY Motion Export** - Export and reuse animations as NumPy arrays

## 🚀 Quick Start

### Installation

```bash
# Using conda (recommended)
conda create -n glb2glb python=3.11
conda activate glb2glb
pip install -e .

# Install with all dependencies
pip install -e .

# Optional: Install IK retargeting support
pip install mink  # For advanced IK-based motion retargeting
```

### Basic Usage

#### Command Line Interface

```bash
# GLB to MuJoCo conversion
glbjc glb2mj model.glb --output model.xml --motion motion.npy

# MuJoCo to GLB conversion
glbjc mj2glb model.xml --output model.glb --motion motion.npy --fps 30

# Round-trip test
glbjc test model.glb --temp-dir ./temp
```

#### GLB to GLB Motion Transfer

Transfer animation from one model to another while preserving the target mesh:

```python
import glb2glb

# Transfer animation from source to target model
glb2glb.transfer_animation(
    source_glb='character_walking.glb',  # Model with animation
    target_glb='robot_model.glb',        # Model to receive animation
    output_glb='robot_walking.glb',      # Output with transferred motion
    keep_npy=True                        # Optionally save animation as NPY
)
```

#### GLB → MuJoCo Conversion

```python
import glb2glb

# Convert GLB to MuJoCo
glb2glb.glb_to_mujoco(
    glb_path='model.glb',
    xml_output='model.xml',
    motion_output='motion.npy'  # Optional: extract animation
)
```

#### MuJoCo → GLB Conversion

```python
import glb2glb

# Convert MuJoCo to GLB
glb2glb.mujoco_to_glb(
    xml_path='model.xml',
    output_path='model.glb',
    motion_data='motion.npy',  # Optional: add animation
    fps=30.0                   # Animation frame rate
)
```

## 📦 Module Structure

```
glb2glb/
├── importer/               # GLB → MuJoCo conversion
│   ├── joint_centric_importer.py   # Core importer
│   ├── glb/                        # GLB parsing utilities
│   ├── mesh/                       # Mesh extraction
│   ├── skeleton/                   # Skeleton extraction
│   └── mujoco/                     # MuJoCo model generation
├── exporter/               # MuJoCo → GLB conversion
│   ├── joint_centric_exporter.py   # Core exporter
│   ├── animation_exporter.py       # Animation export to NPY
│   ├── common.py                   # Shared utilities
│   └── utils/                      # Export utilities
├── pipeline/               # High-level pipelines
│   └── motion_transfer.py          # Complete motion transfer
├── transfer/               # Animation transfer utilities
│   ├── animation_transfer.py       # GLB animation transfer
│   └── npy_to_glb.py               # NPY to GLB conversion
├── retargeter/             # Motion retargeting
│   └── ik_based_retargeting.py     # IK-based retargeting (requires mink)
├── visualizer/             # Visualization tools
│   ├── mjviewer.py                 # MuJoCo motion viewer
│   ├── visualize_retargeting.py    # Retargeting visualization
│   └── utils.py                    # Visualization utilities
└── utils/
    └── inspect_glb.py              # GLB inspection utility
```

## 🎬 Visualization

### MuJoCo Viewer

```bash
# View animated MuJoCo model
conda activate glb2glb
python -m glb2glb.visualizer.mjviewer model.xml motion.npy --loop

# Controls:
#   Space: Pause/Resume
#   →/←: Next/Previous frame
#   R: Reset to start
#   ESC: Exit
```

### GLB Inspector

```bash
# Inspect GLB file structure and animations
python -m glb2glb.utils.inspect_glb model.glb

# Inspect multiple files
python -m glb2glb.utils.inspect_glb *.glb
```

### Blender

1. Open Blender
2. File → Import → glTF 2.0
3. Select your .glb file
4. Press Space to play animation

## 🔧 Advanced Features

### IK-Based Motion Retargeting

```python
from glb2glb.retargeter import IKBasedRetargeter
import numpy as np

# Setup retargeter (requires mink)
retargeter = IKBasedRetargeter(
    source_model,  # MuJoCo model
    target_model,  # MuJoCo model
    joint_mapping  # Optional joint mapping dict
)

# Retarget motion frame by frame
for frame in source_motion:
    target_frame = retargeter.retarget_frame(frame)
    # Process retargeted frame
```

### Batch Processing

```python
import glb2glb
from pathlib import Path

# Process multiple animations
animations = ['walk.glb', 'run.glb', 'jump.glb']
target = 'character.glb'

for anim in animations:
    output = f"character_{Path(anim).stem}.glb"
    glb2glb.transfer_animation(anim, target, output, verbose=False)
```

## ⚙️ Technical Details

### Coordinate System Handling

The toolkit automatically handles coordinate system conversions:
- **GLTF/GLB**: Y-up, Z-forward (OpenGL convention)
- **MuJoCo**: Z-up, X-forward (robotics convention)

### Quaternion Transformation

Proper quaternion mapping for rotations:
```
GLTF (XYZW) → MuJoCo (WXYZ)
Axis remapping: X→Y, Y→Z, Z→X
```

### Joint Handling

- **Free joints**: 7 DOF (position + quaternion)
- **Ball joints**: 4 DOF (quaternion only)
- **Hinge joints**: 1 DOF (angle)
- **Slide joints**: 1 DOF (position)

### Special Cases

- **Hip/Pelvis**: Handled with small offset for proper body creation
- **Twist joints**: Preserved as auxiliary joints
- **Root motion**: Transferred with free joint representation

## 📊 Example Workflows

### 1. Character Animation Transfer

```python
# Transfer walking animation to different character
transfer_animation(
    'animations/human_walking.glb',
    'models/robot.glb',
    'output/robot_walking.glb'
)
```

### 2. Motion Library Building

```python
# Convert motion capture to multiple characters
source_mocap = 'mocap/dance.glb'
characters = ['panda.glb', 'robot.glb', 'avatar.glb']

for char in characters:
    name = Path(char).stem
    transfer_animation(source_mocap, char, f'library/{name}_dance.glb')
```

### 3. Physics Simulation

```python
import glb2glb
import numpy as np

# Convert GLB to MuJoCo for simulation
glb2glb.glb_to_mujoco('model.glb', 'model.xml')

# Run MuJoCo simulation
import mujoco
model = mujoco.MjModel.from_xml_path('model.xml')
# ... simulation code generates qpos trajectory ...

# Save simulation results
np.save('simulation.npy', {'qpos': qpos_trajectory, 'fps': 60})

# Convert back to GLB with simulated motion
glb2glb.mujoco_to_glb(
    xml_path='model.xml',
    output_path='simulated.glb',
    motion_data='simulation.npy',
    fps=60
)
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: mink**
   ```bash
   pip install mink  # Optional, only needed for IK-based retargeting
   ```

2. **Coordinate system issues**
   - The toolkit auto-detects Y-up/Z-up systems
   - Check model orientation in Blender first

3. **Motion transfer quality**
   - Ensure similar skeleton structures
   - Check joint mapping in verbose output
   - Consider using IK-based retargeting for complex transfers

## 📝 Motion Data Format

### NPY Format

```python
# Simple array format
motion = np.random.randn(100, model.nq)  # 100 frames
np.save('motion.npy', motion)

# Dictionary format with metadata
motion = {
    'qpos': np.random.randn(100, model.nq),
    'fps': 30.0,
    'n_frames': 100
}
np.save('motion.npy', motion)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/hokindeng/glb2glb.git
cd glb2glb
conda create -n glb2glb-dev python=3.11
conda activate glb2glb-dev
pip install -e .

# Install development tools
pip install pytest black isort flake8
```

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- MuJoCo physics engine by DeepMind
- pygltflib for GLTF handling
- trimesh for 3D processing
- mink for inverse kinematics

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/hokindeng/glb2glb/issues)
- Repository: [https://github.com/hokindeng/glb2glb](https://github.com/hokindeng/glb2glb)

---

**Note**: This project is under active development. APIs may change between versions.