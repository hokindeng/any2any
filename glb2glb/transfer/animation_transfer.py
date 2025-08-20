#!/usr/bin/env python3
"""
Core functionality for transferring retargeted animations to GLB files while preserving meshes.
Adapted from rig2rig for glb2glb retargeting pipeline.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from pygltflib import (
    GLTF2, Accessor, BufferView, Buffer,
    Animation, AnimationChannel, AnimationChannelTarget,
    AnimationSampler
)


def load_gltf(file_path: str) -> GLTF2:
    """Load a GLTF2 file."""
    return GLTF2().load(file_path)


def save_gltf(gltf: GLTF2, file_path: str):
    """Save a GLTF2 file."""
    gltf.save(file_path)


def get_all_binary_data(gltf: GLTF2) -> bytes:
    """Get all binary data from a GLTF file."""
    # For GLB files
    if hasattr(gltf, 'binary_blob'):
        blob = gltf.binary_blob()
        if isinstance(blob, (bytes, bytearray)) and len(blob) > 0:
            return blob
    
    # Try to get data from the first buffer
    if gltf.buffers:
        buffer = gltf.buffers[0]
        if buffer.uri is None:
            # GLB embedded buffer
            if hasattr(gltf, '_glb_data'):
                return gltf._glb_data
        elif hasattr(gltf, 'get_data_from_buffer_uri'):
            # External file buffer
            return gltf.get_data_from_buffer_uri(buffer.uri)
    
    raise RuntimeError("Unable to access buffer data")


def copy_accessor_data_simple(acc_index: int, src: GLTF2, tgt: GLTF2, 
                             src_binary_data: bytes, 
                             pending_data: list) -> int:
    """
    Copy accessor and its buffer data from source glTF to target glTF.
    Uses a simpler approach that directly copies the buffer view data.
    
    Returns:
        New accessor index in target glTF
    """
    src_acc = src.accessors[acc_index]
    src_bv = src.bufferViews[src_acc.bufferView]
    
    # Calculate the data range to copy
    bv_start = src_bv.byteOffset or 0
    bv_end = bv_start + src_bv.byteLength
    
    # Extract the buffer view data
    bv_data = src_binary_data[bv_start:bv_end]
    
    if len(bv_data) != src_bv.byteLength:
        print(f"Warning: Expected {src_bv.byteLength} bytes, got {len(bv_data)} bytes")
    
    # Store for later processing
    new_bv_index = len(tgt.bufferViews)
    pending_data.append({
        'data': bv_data,
        'bufferView': new_bv_index,
        'accessor': len(tgt.accessors),
        'src_accessor': src_acc,
        'src_bufferView': src_bv
    })
    
    # Create new buffer view (offset will be set later)
    new_bv = BufferView(
        buffer=0,
        byteOffset=0,  # Will be set during consolidation
        byteLength=len(bv_data),
        byteStride=src_bv.byteStride,
        target=src_bv.target
    )
    tgt.bufferViews.append(new_bv)
    
    # Create new accessor
    new_acc = Accessor(
        bufferView=new_bv_index,
        byteOffset=src_acc.byteOffset or 0,
        componentType=src_acc.componentType,
        count=src_acc.count,
        type=src_acc.type,
        max=src_acc.max,
        min=src_acc.min,
        normalized=src_acc.normalized,
        sparse=src_acc.sparse,
        name=src_acc.name
    )
    tgt.accessors.append(new_acc)
    
    return len(tgt.accessors) - 1


def consolidate_buffers_simple(gltf: GLTF2):
    """Consolidate all pending buffer data into a single buffer for GLB export."""
    if not hasattr(gltf, '_pending_data') or not gltf._pending_data:
        return
    
    print(f"Consolidating {len(gltf._pending_data)} buffer segments...")
    
    # Get the existing binary data if any
    existing_data = None
    existing_size = 0
    try:
        existing_data = get_all_binary_data(gltf)
        if isinstance(existing_data, (bytes, bytearray)):
            existing_size = len(existing_data)
            print(f"Existing buffer size: {existing_size} bytes")
        else:
            existing_data = None
            existing_size = 0
            print("Existing buffer data is not bytes; ignoring")
    except:
        print("No existing buffer data found")
    
    # First pass: calculate total size with proper alignment
    offset = existing_size
    # Align existing data to 4-byte boundary if needed
    if offset % 4 != 0:
        offset += 4 - (offset % 4)
    
    offsets = []
    for item in gltf._pending_data:
        # Align to 4-byte boundary
        padding = (4 - (offset % 4)) % 4
        offset += padding
        offsets.append(offset)
        offset += len(item['data'])
    
    # Total buffer size must be aligned to 4 bytes
    total_size = offset
    if total_size % 4 != 0:
        total_size += 4 - (total_size % 4)
    
    print(f"Total buffer size: {total_size} bytes")
    
    # Create the consolidated buffer
    buffer_data = bytearray(total_size)
    
    # Copy existing data if any
    if isinstance(existing_data, (bytes, bytearray)) and existing_size > 0:
        buffer_data[:existing_size] = existing_data
    
    # Second pass: copy new data at calculated offsets
    for i, item in enumerate(gltf._pending_data):
        data = item['data']
        start_offset = offsets[i]
        
        # Update the buffer view offset
        bv_idx = item['bufferView']
        gltf.bufferViews[bv_idx].byteOffset = start_offset
        
        # Copy the data
        buffer_data[start_offset:start_offset + len(data)] = data
    
    # Create or update the buffer
    if not gltf.buffers:
        gltf.buffers = []
    
    # For GLB, we need a single buffer with no URI
    if len(gltf.buffers) == 0:
        gltf.buffers.append(Buffer(byteLength=total_size))
    else:
        gltf.buffers[0].byteLength = total_size
    
    # Convert to bytes and set as binary blob
    final_data = bytes(buffer_data)
    gltf.set_binary_blob(final_data)
    
    # Clean up
    delattr(gltf, '_pending_data')
    
    print(f"Buffer consolidation complete. Final size: {len(final_data)} bytes")


def transfer_animation_from_glb(src_glb_path: str, tgt_glb_path: str, output_path: str,
                               verbose: bool = True) -> bool:
    """
    Transfer animation from source GLB to target GLB while preserving target's mesh.
    This is useful for applying retargeted animations to original models.
    
    Args:
        src_glb_path: Path to GLB with the animation (e.g., exported from MuJoCo)
        tgt_glb_path: Path to original GLB with mesh to preserve
        output_path: Path for output GLB with mesh and new animation
        verbose: Print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print("="*60)
        print("ANIMATION TRANSFER (Preserving Mesh)")
        print("="*60)
        print(f"\n📁 Loading models...")
    
    src_gltf = load_gltf(src_glb_path)
    tgt_gltf = load_gltf(tgt_glb_path)
    
    if verbose:
        print(f"✓ Source (animation): {src_glb_path}")
        print(f"✓ Target (mesh): {tgt_glb_path}")
    
    # Check animations
    if not src_gltf.animations:
        if verbose:
            print("❌ Error: Source model has no animations!")
        return False
    
    if verbose:
        print(f"\n🎬 Found {len(src_gltf.animations)} animation(s)")
    
    # Build bone mapping (assuming identical rigs)
    src_bones = {}
    tgt_bones = {}
    
    # Get source bones
    if src_gltf.skins and src_gltf.skins[0].joints:
        for joint_idx in src_gltf.skins[0].joints:
            node = src_gltf.nodes[joint_idx]
            if node.name:
                src_bones[node.name] = joint_idx
    else:
        # Fall back to all nodes
        for i, node in enumerate(src_gltf.nodes):
            if node.name:
                src_bones[node.name] = i
    
    # Get target bones
    if tgt_gltf.skins and tgt_gltf.skins[0].joints:
        for joint_idx in tgt_gltf.skins[0].joints:
            node = tgt_gltf.nodes[joint_idx]
            if node.name:
                tgt_bones[node.name] = joint_idx
    else:
        # Fall back to all nodes
        for i, node in enumerate(tgt_gltf.nodes):
            if node.name:
                tgt_bones[node.name] = i
    
    # Create identity mapping for bones with matching names
    bone_map = {}
    for src_name in src_bones:
        if src_name in tgt_bones:
            bone_map[src_name] = src_name
    
    if verbose:
        print(f"\n🦴 Mapped {len(bone_map)} bones by name")
        if len(bone_map) < len(src_bones):
            unmapped = set(src_bones.keys()) - set(bone_map.keys())
            print(f"⚠️  Warning: {len(unmapped)} source bones not mapped: {list(unmapped)[:5]}...")
    
    # Remove existing animations from target
    if tgt_gltf.animations:
        if verbose:
            print(f"Removing {len(tgt_gltf.animations)} existing animations from target")
        tgt_gltf.animations = []
    
    # Get all binary data from source
    try:
        src_binary_data = get_all_binary_data(src_gltf)
        if verbose:
            print(f"Source binary data size: {len(src_binary_data)} bytes")
    except Exception as e:
        if verbose:
            print(f"Error getting source binary data: {e}")
        return False
    
    # Get source animation
    src_anim = src_gltf.animations[0]
    if verbose:
        print(f"\n🔄 Transferring animation: {src_anim.name or 'Animation_0'}")
    
    # Create new animation for target
    new_anim = Animation(name=src_anim.name or "RetargetedAnimation")
    new_anim.channels = []
    new_anim.samplers = []
    
    # List to store pending buffer data
    if not hasattr(tgt_gltf, '_pending_data'):
        tgt_gltf._pending_data = []
    
    # Transfer each channel
    channels_transferred = 0
    for channel in src_anim.channels:
        src_node_index = channel.target.node
        src_bone_name = src_gltf.nodes[src_node_index].name
        transform_path = channel.target.path  # "rotation", "translation", or "scale"
        
        # Skip if no mapping exists
        if src_bone_name not in bone_map:
            continue
            
        tgt_bone_name = bone_map[src_bone_name]
        if tgt_bone_name not in tgt_bones:
            continue
            
        tgt_node_index = tgt_bones[tgt_bone_name]
        
        # Copy sampler data
        src_sampler = src_anim.samplers[channel.sampler]
        
        if verbose and channels_transferred < 5:  # Only print first few
            print(f"  Transferring {transform_path} for {src_bone_name}")
        
        # Copy input (time) data
        new_input_index = copy_accessor_data_simple(src_sampler.input, src_gltf, tgt_gltf, 
                                                   src_binary_data, tgt_gltf._pending_data)
        
        # Copy output (transform) data
        new_output_index = copy_accessor_data_simple(src_sampler.output, src_gltf, tgt_gltf,
                                                    src_binary_data, tgt_gltf._pending_data)
        
        # Create new sampler
        new_sampler_index = len(new_anim.samplers)
        new_sampler = AnimationSampler(
            input=new_input_index,
            output=new_output_index,
            interpolation=src_sampler.interpolation or "LINEAR"
        )
        new_anim.samplers.append(new_sampler)
        
        # Create new channel
        new_channel = AnimationChannel(
            sampler=new_sampler_index,
            target=AnimationChannelTarget(
                node=tgt_node_index,
                path=transform_path
            )
        )
        new_anim.channels.append(new_channel)
        channels_transferred += 1
    
    # Add animation to target
    if tgt_gltf.animations is None:
        tgt_gltf.animations = []
    tgt_gltf.animations.append(new_anim)
    
    if verbose:
        print(f"Transferred {channels_transferred} animation channels")
    
    # Consolidate buffers
    consolidate_buffers_simple(tgt_gltf)
    
    # Save the result
    save_gltf(tgt_gltf, output_path)
    
    if verbose:
        print(f"\n✅ SUCCESS! Animation transferred with mesh preserved")
        print(f"   Output saved to: {output_path}")
    
    return channels_transferred > 0


def transfer_retargeted_animation(retargeted_glb: str, original_glb: str, output_path: str,
                                 verbose: bool = True) -> bool:
    """
    Transfer retargeted animation to original GLB while preserving mesh.
    
    This is a convenience wrapper specifically for the retargeting pipeline.
    
    Args:
        retargeted_glb: Path to GLB with retargeted animation (from MuJoCo export)
        original_glb: Path to original GLB with mesh
        output_path: Path for final output with mesh and retargeted animation
        verbose: Print progress
        
    Returns:
        True if successful
    """
    return transfer_animation_from_glb(retargeted_glb, original_glb, output_path, verbose)
