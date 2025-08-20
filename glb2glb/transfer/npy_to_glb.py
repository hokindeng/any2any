"""
Transfer animation from NPY file to GLB with matching armature.
Adapted from rig2rig for glb2glb pipeline.
"""

import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import logging

import pygltflib
from pygltflib import (
    GLTF2, Accessor, BufferView, Buffer,
    Animation, AnimationChannel, AnimationChannelTarget,
    AnimationSampler
)

logger = logging.getLogger(__name__)


def load_gltf(file_path: str) -> GLTF2:
    """Load a GLTF2 file."""
    # Use pygltflib.GLTF2 from module so tests can patch
    return pygltflib.GLTF2().load(file_path)


def save_gltf(gltf: GLTF2, file_path: str):
    """Save a GLTF2 file."""
    gltf.save(file_path)


def create_accessor(gltf: GLTF2, data: np.ndarray, accessor_type: str, 
                   component_type: int = 5126) -> int:
    """
    Create an accessor for animation data.
    
    Args:
        gltf: GLTF2 object
        data: Numpy array of data
        accessor_type: Type like "SCALAR", "VEC3", "VEC4"
        component_type: Component type (5126 = FLOAT)
        
    Returns:
        Accessor index
    """
    # Get or create binary blob
    if not hasattr(gltf, '_animation_data'):
        gltf._animation_data = bytearray()
    
    # Align to 4-byte boundary
    while len(gltf._animation_data) % 4 != 0:
        gltf._animation_data.append(0)
    
    byte_offset = len(gltf._animation_data)
    
    # Add data to blob
    data_bytes = data.astype(np.float32).tobytes()
    gltf._animation_data.extend(data_bytes)
    
    # Create buffer view
    buffer_view = BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=len(data_bytes)
    )
    bv_index = len(gltf.bufferViews)
    gltf.bufferViews.append(buffer_view)
    
    # Determine count based on type
    type_sizes = {
        'SCALAR': 1,
        'VEC2': 2, 
        'VEC3': 3,
        'VEC4': 4
    }
    
    element_size = type_sizes.get(accessor_type, 1)
    count = len(data) // element_size if len(data.shape) == 1 else data.shape[0]
    
    # Calculate min/max
    if len(data.shape) == 1:
        data_reshaped = data.reshape(-1, element_size)
    else:
        data_reshaped = data
        
    min_vals = data_reshaped.min(axis=0).tolist()
    max_vals = data_reshaped.max(axis=0).tolist()
    
    # Create accessor
    accessor = Accessor(
        bufferView=bv_index,
        byteOffset=0,
        componentType=component_type,
        count=count,
        type=accessor_type,
        min=min_vals,
        max=max_vals
    )
    
    accessor_index = len(gltf.accessors)
    gltf.accessors.append(accessor)
    
    return accessor_index


def finalize_animation_buffer(gltf: GLTF2):
    """Finalize the animation buffer after all accessors are added."""
    if not hasattr(gltf, '_animation_data'):
        return
        
    # Align final size to 4 bytes
    while len(gltf._animation_data) % 4 != 0:
        gltf._animation_data.append(0)
    
    # Update or create buffer
    buffer_data = bytes(gltf._animation_data)
    
    if not gltf.buffers:
        gltf.buffers = [Buffer(byteLength=len(buffer_data))]
    else:
        # Update existing buffer size
        gltf.buffers[0].byteLength = len(buffer_data)
    
    # Set binary blob for GLB
    gltf.set_binary_blob(buffer_data)
    
    # Clean up
    delattr(gltf, '_animation_data')


def apply_animation_from_npy(
    target_glb: str,
    animation_npy: str,
    output_glb: str,
    verbose: bool = True
) -> bool:
    """
    Apply animation from NPY file to GLB with matching armature.
    
    Args:
        target_glb: Path to target GLB file (with mesh and armature)
        animation_npy: Path to NPY file with animation data
        output_glb: Path for output GLB with animation
        verbose: Print progress
        
    Returns:
        True if successful
    """
    if verbose:
        print("="*60)
        print("APPLYING NPY ANIMATION TO GLB")
        print("="*60)
        print(f"Target GLB: {target_glb}")
        print(f"Animation: {animation_npy}")
        
    # Load target GLB
    gltf = load_gltf(target_glb)
    
    # Load animation data
    anim_data = np.load(animation_npy, allow_pickle=True).item()
    
    fps = anim_data.get('fps', 30.0)
    n_frames = anim_data.get('n_frames', 0)
    joints = anim_data.get('joints', {})
    
    if verbose:
        print(f"\nAnimation info:")
        print(f"  Frames: {n_frames}")
        print(f"  FPS: {fps}")
        print(f"  Animated joints: {len(joints)}")
    
    # Build node name to index mapping
    node_map = {}
    if gltf.skins and gltf.skins[0].joints:
        for joint_idx in gltf.skins[0].joints:
            node = gltf.nodes[joint_idx]
            if node.name:
                node_map[node.name] = joint_idx
    else:
        # Fall back to all nodes
        for i, node in enumerate(gltf.nodes):
            if node.name:
                node_map[node.name] = i
    
    if verbose:
        print(f"\nTarget armature:")
        print(f"  Nodes: {len(node_map)}")
    
    # Remove existing animations
    if gltf.animations:
        if verbose:
            print(f"  Removing {len(gltf.animations)} existing animations")
        gltf.animations = []
    
    # Create new animation
    animation = Animation(name="TransferredAnimation")
    animation.channels = []
    animation.samplers = []
    
    # Create time accessor (shared by all channels)
    time_values = np.linspace(0, (n_frames - 1) / fps, n_frames, dtype=np.float32)
    time_accessor = create_accessor(gltf, time_values, "SCALAR")
    
    # Process each joint
    channels_created = 0
    joints_mapped = 0
    
    for joint_name, joint_data in joints.items():
        # Find corresponding node in target
        if joint_name not in node_map:
            # Try alternative names (e.g., with/without prefixes)
            found = False
            for node_name in node_map:
                if joint_name in node_name or node_name in joint_name:
                    joint_name = node_name
                    found = True
                    break
            if not found:
                continue
        
        node_idx = node_map[joint_name]
        joints_mapped += 1
        
        # Add rotation channel if present
        if 'rotations' in joint_data:
            rotations = joint_data['rotations']
            
            # Create sampler
            sampler_idx = len(animation.samplers)
            output_accessor = create_accessor(gltf, rotations.flatten(), "VEC4")
            
            sampler = AnimationSampler(
                input=time_accessor,
                output=output_accessor,
                interpolation="LINEAR"
            )
            animation.samplers.append(sampler)
            
            # Create channel
            channel = AnimationChannel(
                sampler=sampler_idx,
                target=AnimationChannelTarget(
                    node=node_idx,
                    path="rotation"
                )
            )
            animation.channels.append(channel)
            channels_created += 1
        
        # Add translation channel if present
        if 'translations' in joint_data:
            translations = joint_data['translations']
            
            # Create sampler
            sampler_idx = len(animation.samplers)
            output_accessor = create_accessor(gltf, translations.flatten(), "VEC3")
            
            sampler = AnimationSampler(
                input=time_accessor,
                output=output_accessor,
                interpolation="LINEAR"
            )
            animation.samplers.append(sampler)
            
            # Create channel
            channel = AnimationChannel(
                sampler=sampler_idx,
                target=AnimationChannelTarget(
                    node=node_idx,
                    path="translation"
                )
            )
            animation.channels.append(channel)
            channels_created += 1
    
    # Add animation to GLB
    gltf.animations = [animation]
    
    # Finalize buffer
    finalize_animation_buffer(gltf)
    
    # Save output
    save_gltf(gltf, output_glb)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Joints mapped: {joints_mapped}/{len(joints)}")
        print(f"  Channels created: {channels_created}")
        print(f"  Output saved to: {output_glb}")
        print("="*60)
    
    return channels_created > 0
