"""
Joint-Centric MuJoCo to GLB Exporter

This module implements the new joint-centric architecture where:
- GLB nodes represent MuJoCo joints (not bodies)
- Body lengths are encoded as static translations between nodes
- Animation channels map directly to joint DOFs
"""

import numpy as np
import mujoco
from typing import List, Dict, Optional, Tuple
import pygltflib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JointNode:
    """Represents a joint that will become a GLB node."""
    
    def __init__(self, joint_id: int, model: mujoco.MjModel):
        """Initialize a joint node from MuJoCo joint ID."""
        self.joint_id = joint_id
        self.model = model
        
        # Get joint data through model
        jnt_name = model.joint(joint_id).name
        jnt_type = model.jnt_type[joint_id]
        jnt_axis = model.jnt_axis[joint_id].copy()
        jnt_bodyid = model.jnt_bodyid[joint_id]
        jnt_qposadr = model.jnt_qposadr[joint_id]
        jnt_dofadr = model.jnt_dofadr[joint_id]
        
        self.name = jnt_name
        self.type = jnt_type
        self.axis = jnt_axis
        self.body_id = jnt_bodyid
        
        # Get the body this joint moves
        self.body = model.body(self.body_id)
        self.body_name = self.body.name
        
        # Position relative to parent body
        self.local_pos = self.body.pos.copy()
        
        # Joint limits and parameters
        self.qpos_addr = jnt_qposadr
        self.dof_addr = jnt_dofadr
        
        # Get DOF count based on joint type
        # Accept both enum and integer (tests often use int literals)
        jnt_type_int = int(jnt_type)
        if jnt_type_int == int(getattr(mujoco.mjtJoint, 'mjJNT_FREE', 0)) or jnt_type_int == 0:
            self.dof_count = 7
        elif jnt_type_int == int(getattr(mujoco.mjtJoint, 'mjJNT_BALL', 1)) or jnt_type_int == 1:
            self.dof_count = 4
        else:
            self.dof_count = 1
        
        # Parent and children (to be filled later)
        self.parent = None
        self.children = []
        
        # Animation data
        self.translation_keys = []  # For free/slide joints
        self.rotation_keys = []     # For hinge/free joints
        
        # GLB node index (assigned during export)
        self.glb_idx = -1
        
    def get_static_translation(self) -> List[float]:
        """
        Get the static translation to this joint from its parent.
        This represents the body length/offset.
        """
        # If this is the root or has same body as parent, no translation
        if self.parent is None:
            return [0.0, 0.0, 0.0]
            
        # If parent joint is on the same body, zero translation
        if self.parent.body_id == self.body_id:
            return [0.0, 0.0, 0.0]
            
        # Otherwise, use the body's local position
        return self.local_pos.tolist()
        
    def add_animation_frame(self, qpos: np.ndarray):
        """Add animation data for this joint from a qpos frame."""
        jt = int(self.type)
        if jt == int(getattr(mujoco.mjtJoint, 'mjJNT_FREE', 0)) or jt == 0:
            # Free joint: extract translation and rotation
            trans = qpos[self.qpos_addr:self.qpos_addr + 3].copy()
            rot = qpos[self.qpos_addr + 3:self.qpos_addr + 7].copy()
            self.translation_keys.append(trans)
            self.rotation_keys.append(rot)
        
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_BALL', 1)) or jt == 1:
            # Ball joint: extract quaternion directly (4 DOF)
            quat = qpos[self.qpos_addr:self.qpos_addr + 4].copy()
            self.rotation_keys.append(quat)
        
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_HINGE', 3)) or jt == 3:
            # Hinge joint: convert angle to quaternion
            angle = qpos[self.qpos_addr]
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, self.axis, angle)
            self.rotation_keys.append(q)
        
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', 2)) or jt == 2:
            # Slide joint: translation along axis
            value = qpos[self.qpos_addr]
            trans = self.axis * value
            self.translation_keys.append(trans)


class JointCentricExporter:
    """Export MuJoCo model to GLB using joint-centric architecture."""
    
    def __init__(self, model_path: str):
        """Initialize exporter with MuJoCo model."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Build joint hierarchy
        self.joints = []
        self.joint_map = {}
        self._build_joint_hierarchy()
        
        # GLB components
        self.gltf = pygltflib.GLTF2()
        self.scene = pygltflib.Scene()
        self.gltf.scenes.append(self.scene)
        self.gltf.scene = 0
        
    def _build_joint_hierarchy(self):
        """Build hierarchy of joints from MuJoCo model."""
        # Create JointNode for each joint
        for i in range(self.model.njnt):
            node = JointNode(i, self.model)
            self.joints.append(node)
            self.joint_map[node.name] = node
            
        # Group joints by body they belong to
        body_joints = {}
        for node in self.joints:
            if node.body_id not in body_joints:
                body_joints[node.body_id] = []
            body_joints[node.body_id].append(node)
            
        # Chain multiple joints on same body
        for body_id, joints in body_joints.items():
            if len(joints) > 1:
                # Chain joints together (first → second → third...)
                for i in range(len(joints) - 1):
                    joints[i+1].parent = joints[i]
                    joints[i].children.append(joints[i+1])
                    
        # Establish parent-child relationships between bodies
        for node in self.joints:
            if node.parent is None:  # Only process unchained joints
                body = self.model.body(node.body_id)
                parent_body_id = int(body.parentid[0]) if hasattr(body.parentid, '__len__') else int(body.parentid)
                
                if parent_body_id > 0:  # Not world body
                    # Find joints that move the parent body
                    parent_joints = body_joints.get(parent_body_id, [])
                    
                    if parent_joints:
                        # Connect to the last joint of parent body
                        # (the end of the chain if multiple joints)
                        node.parent = parent_joints[-1]
                        parent_joints[-1].children.append(node)
                    
        # Find root joints (no parent)
        self.root_joints = [j for j in self.joints if j.parent is None]
        
        logger.info(f"Built joint hierarchy with {len(self.joints)} joints, {len(self.root_joints)} roots")
        
    def add_animation(self, qpos_data: List[np.ndarray], fps: float = 30.0):
        """Add animation data from qpos sequence."""
        logger.info(f"Adding animation with {len(qpos_data)} frames at {fps} FPS")
        
        # Add each frame to joints
        for frame_idx, qpos in enumerate(qpos_data):
            for joint in self.joints:
                joint.add_animation_frame(qpos)
                
        # Debug: Check if keyframes were added
        for joint in self.joints:
            if joint.rotation_keys or joint.translation_keys:
                logger.debug(f"Joint {joint.name}: {len(joint.rotation_keys)} rot keys, {len(joint.translation_keys)} trans keys")
                
    def _get_joint_type_name(self, joint_type: int) -> str:
        """Convert MuJoCo joint type to string name."""
        jt = int(joint_type)
        if jt == int(getattr(mujoco.mjtJoint, 'mjJNT_FREE', 0)) or jt == 0:
            return "free"
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_BALL', 1)) or jt == 1:
            return "ball"
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', 2)) or jt == 2:
            return "slide"
        elif jt == int(getattr(mujoco.mjtJoint, 'mjJNT_HINGE', 3)) or jt == 3:
            return "hinge"
        else:
            return "unknown"
            
    def export_to_glb(self, output_path: str):
        """Export to GLB file."""
        logger.info(f"Exporting to {output_path}")
        
        # Assign GLB indices to joints
        idx = 0
        for joint in self.joints:
            joint.glb_idx = idx
            idx += 1
            
        # Create GLB nodes from joints
        for joint in self.joints:
            # Get static translation (body offset)
            translation = joint.get_static_translation()
            
            # Initial rotation (identity for now)
            rotation = [0.0, 0.0, 0.0, 1.0]  # XYZW format
            
            # Child indices
            children = [c.glb_idx for c in joint.children]
            
            # Add metadata about joint type and axis
            extras = {
                'joint_type': self._get_joint_type_name(joint.type),
                'joint_axis': joint.axis.tolist(),
                'body_id': int(joint.body_id),
                'qpos_addr': int(joint.qpos_addr),
                'dof_count': int(joint.dof_count)
            }
            
            node = pygltflib.Node(
                name=joint.name,
                translation=translation,
                rotation=rotation,
                children=children if children else None,
                extras=extras  # Store metadata
            )
            self.gltf.nodes.append(node)
            
        # Add root nodes to scene
        for root in self.root_joints:
            self.scene.nodes.append(root.glb_idx)
            
        # Create animation if we have keyframes
        if self.joints and (self.joints[0].rotation_keys or self.joints[0].translation_keys):
            self._create_animation()
            
        # Set binary blob if we have buffer data
        if hasattr(self, 'binary_blob') and self.binary_blob:
            self.gltf.set_binary_blob(bytes(self.binary_blob))
            
        # Save GLB file
        self.gltf.save(output_path)
        logger.info(f"Exported GLB with {len(self.gltf.nodes)} nodes")
        
    def _create_animation(self):
        """Create animation from joint keyframes."""
        animation = pygltflib.Animation(name="JointAnimation")
        samplers = []
        channels = []
        
        # Find a joint with keyframes to determine frame count
        n_frames = 0
        for joint in self.joints:
            if joint.rotation_keys:
                n_frames = len(joint.rotation_keys)
                break
            elif joint.translation_keys:
                n_frames = len(joint.translation_keys)
                break
                
        if n_frames == 0:
            logger.warning("No animation keyframes found")
            return
            
        time_values = np.arange(n_frames, dtype=np.float32) / 30.0  # Assuming 30 FPS
        
        # Create samplers and channels for each joint
        for joint in self.joints:
            # Rotation channel
            if joint.rotation_keys:
                # Convert rotation keys to flat array
                rot_data = np.array(joint.rotation_keys, dtype=np.float32)
                
                # Convert WXYZ to XYZW for GLTF
                rot_data_gltf = np.roll(rot_data, -1, axis=1)
                
                # Create sampler
                sampler_idx = len(samplers)
                sampler = pygltflib.AnimationSampler(
                    input=self._add_accessor(time_values, pygltflib.SCALAR),
                    output=self._add_accessor(rot_data_gltf.flatten(), pygltflib.VEC4),
                    interpolation=pygltflib.LINEAR
                )
                samplers.append(sampler)
                
                # Create channel
                channel = pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=joint.glb_idx,
                        path=pygltflib.ROTATION
                    )
                )
                channels.append(channel)
                
            # Translation channel (for free/slide joints)
            if joint.translation_keys:
                trans_data = np.array(joint.translation_keys, dtype=np.float32)
                
                sampler_idx = len(samplers)
                sampler = pygltflib.AnimationSampler(
                    input=self._add_accessor(time_values, pygltflib.SCALAR),
                    output=self._add_accessor(trans_data.flatten(), pygltflib.VEC3),
                    interpolation=pygltflib.LINEAR
                )
                samplers.append(sampler)
                
                channel = pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=joint.glb_idx,
                        path=pygltflib.TRANSLATION
                    )
                )
                channels.append(channel)
                
        animation.samplers = samplers
        animation.channels = channels
        self.gltf.animations.append(animation)
        
        logger.info(f"Created animation with {len(channels)} channels")
        
    def _add_accessor(self, data: np.ndarray, accessor_type: str) -> int:
        """Add data accessor to GLB with proper binary buffer handling."""
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Calculate component count
        if accessor_type == pygltflib.VEC4:
            component_count = 4
        elif accessor_type == pygltflib.VEC3:
            component_count = 3
        elif accessor_type == pygltflib.VEC2:
            component_count = 2
        else:  # SCALAR
            component_count = 1
            
        count = len(data) // component_count
        
        # Create or get buffer
        if not self.gltf.buffers:
            # Create initial buffer
            self.gltf.buffers.append(pygltflib.Buffer(byteLength=0))
            self.binary_blob = bytearray()
        else:
            self.binary_blob = getattr(self, 'binary_blob', bytearray())
            
        # Add data to binary blob
        byte_offset = len(self.binary_blob)
        byte_data = data.tobytes()
        self.binary_blob.extend(byte_data)
        byte_length = len(byte_data)
        
        # Update buffer byte length
        self.gltf.buffers[0].byteLength = len(self.binary_blob)
        
        # Create buffer view
        buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=byte_offset,
            byteLength=byte_length,
            target=None  # For animation data
        )
        buffer_view_idx = len(self.gltf.bufferViews)
        self.gltf.bufferViews.append(buffer_view)
        
        # Create accessor
        accessor = pygltflib.Accessor(
            bufferView=buffer_view_idx,
            componentType=pygltflib.FLOAT,
            count=count,
            type=accessor_type,
            min=data.reshape(-1, component_count).min(axis=0).tolist() if count > 0 else None,
            max=data.reshape(-1, component_count).max(axis=0).tolist() if count > 0 else None
        )
        accessor_idx = len(self.gltf.accessors)
        self.gltf.accessors.append(accessor)
        
        return accessor_idx


def test_joint_centric_export():
    """Test the joint-centric exporter with a simple model."""
    import tempfile
    
    # Create a simple test model
    xml = """
    <mujoco>
      <worldbody>
        <body name="upper_arm">
          <joint name="shoulder" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.05"/>
          <body name="forearm" pos="0.3 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.04"/>
            <body name="hand" pos="0.25 0 0">
              <joint name="wrist" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.03"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    
    # Save test model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml)
        model_path = f.name
        
    try:
        # Create exporter
        exporter = JointCentricExporter(model_path)
        
        # Create some test animation data
        n_frames = 30
        qpos_data = []
        for i in range(n_frames):
            t = i / float(n_frames - 1)
            # Animate joints with sine waves
            qpos = np.array([
                np.sin(2 * np.pi * t) * 0.5,  # shoulder
                np.sin(2 * np.pi * t + np.pi/2) * 0.7,  # elbow
                np.sin(2 * np.pi * t + np.pi) * 0.3,  # wrist
            ])
            qpos_data.append(qpos)
            
        # Add animation
        exporter.add_animation(qpos_data)
        
        # Export
        output_path = "temporary/test_joint_centric.glb"
        exporter.export_to_glb(output_path)
        
        print(f"✅ Test export successful: {output_path}")
        
    finally:
        # Clean up
        Path(model_path).unlink()
        

if __name__ == "__main__":
    test_joint_centric_export()
