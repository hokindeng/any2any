#!/usr/bin/env python3
"""
GLB File Inspector - Understand the actual structure of GLB files
"""

import sys
import json
from pathlib import Path
from pygltflib import GLTF2
import numpy as np


def inspect_glb(file_path):
    """Inspect a GLB file to understand its structure."""
    
    print(f"\n{'='*70}")
    print(f"GLB FILE INSPECTION: {Path(file_path).name}")
    print(f"{'='*70}")
    
    # Load the GLB
    gltf = GLTF2.load(str(file_path))
    
    # 1. Basic statistics
    print("\n1. BASIC STRUCTURE:")
    print(f"   Scenes: {len(gltf.scenes)}")
    print(f"   Nodes: {len(gltf.nodes)}")
    print(f"   Meshes: {len(gltf.meshes)}")
    print(f"   Skins: {len(gltf.skins)}")
    print(f"   Animations: {len(gltf.animations)}")
    print(f"   Materials: {len(gltf.materials)}")
    
    # 2. Node hierarchy - build parent-child relationships
    print("\n2. NODE HIERARCHY:")
    
    # Find which nodes are children of which
    node_parents = {}
    node_children = {}
    for i, node in enumerate(gltf.nodes):
        node_children[i] = node.children if node.children else []
        for child_idx in node_children[i]:
            node_parents[child_idx] = i
    
    # Find root nodes (no parents)
    root_nodes = []
    for i in range(len(gltf.nodes)):
        if i not in node_parents:
            root_nodes.append(i)
    
    print(f"   Root nodes (no parent): {len(root_nodes)}")
    
    # Print hierarchy for each root
    def print_node_tree(idx, indent=0):
        node = gltf.nodes[idx]
        prefix = "  " * indent + "├─" if indent > 0 else ""
        
        # Determine node type
        node_type = []
        if node.mesh is not None:
            node_type.append("MESH")
        if node.skin is not None:
            node_type.append(f"SKIN[{node.skin}]")
        if node.camera is not None:
            node_type.append("CAMERA")
        if not node_type and node.children:
            node_type.append("GROUP")
        if not node_type:
            node_type.append("TRANSFORM")
            
        type_str = f" ({', '.join(node_type)})" if node_type else ""
        
        print(f"   {prefix}[{idx:2d}] {node.name or 'unnamed'}{type_str}")
        
        # Print children (limit depth to avoid huge output)
        if indent < 3 and node_children.get(idx):
            for child_idx in node_children[idx]:
                print_node_tree(child_idx, indent + 1)
    
    for root_idx in root_nodes[:5]:  # Limit to first 5 roots
        print_node_tree(root_idx)
    
    if len(root_nodes) > 5:
        print(f"   ... and {len(root_nodes) - 5} more root nodes")
    
    # 3. Skin information
    if gltf.skins:
        print("\n3. SKIN INFORMATION:")
        for i, skin in enumerate(gltf.skins):
            print(f"   Skin[{i}]: {skin.name or 'unnamed'}")
            print(f"      Joints: {len(skin.joints)} nodes")
            if skin.skeleton is not None:
                skeleton_name = gltf.nodes[skin.skeleton].name if skin.skeleton < len(gltf.nodes) else "invalid"
                print(f"      Skeleton root: [{skin.skeleton}] {skeleton_name}")
            print(f"      Joint nodes: {skin.joints[:5]}{'...' if len(skin.joints) > 5 else ''}")
    
    # 4. Scene information
    print("\n4. SCENE ROOTS:")
    for i, scene in enumerate(gltf.scenes):
        print(f"   Scene[{i}]: {scene.name or 'unnamed'}")
        print(f"      Root nodes: {scene.nodes}")
        for node_idx in scene.nodes[:3]:
            if node_idx < len(gltf.nodes):
                print(f"        └─[{node_idx}] {gltf.nodes[node_idx].name}")
    
    # 5. Animation information
    if gltf.animations:
        print("\n5. ANIMATION DATA:")
        for i, anim in enumerate(gltf.animations):
            print(f"   Animation[{i}]: {anim.name or 'unnamed'}")
            print(f"      Channels: {len(anim.channels)}")
            
            # Count channel types
            channel_types = {}
            animated_nodes = set()
            for channel in anim.channels:
                path = channel.target.path
                node = channel.target.node
                animated_nodes.add(node)
                channel_types[path] = channel_types.get(path, 0) + 1
            
            print(f"      Channel types:")
            for path, count in channel_types.items():
                print(f"        - {path}: {count}")
            print(f"      Animated nodes: {len(animated_nodes)}")
            
            # Show which root nodes are animated
            animated_roots = [n for n in animated_nodes if n in root_nodes]
            if animated_roots:
                print(f"      Animated root nodes: {animated_roots}")
    
    # 6. Potential issues
    print("\n6. POTENTIAL ISSUES:")
    issues = []
    
    # Check for duplicate node names
    node_names = [n.name for n in gltf.nodes if n.name]
    duplicate_names = [name for name in node_names if node_names.count(name) > 1]
    if duplicate_names:
        issues.append(f"Duplicate node names: {set(duplicate_names)}")
    
    # Check for orphaned nodes (not in scene and not referenced)
    referenced_nodes = set()
    for scene in gltf.scenes:
        referenced_nodes.update(scene.nodes)
    for node in gltf.nodes:
        if node.children:
            referenced_nodes.update(node.children)
    for skin in gltf.skins:
        referenced_nodes.update(skin.joints)
        if skin.skeleton is not None:
            referenced_nodes.add(skin.skeleton)
    
    orphaned = set(range(len(gltf.nodes))) - referenced_nodes
    if orphaned and len(orphaned) < 10:
        issues.append(f"Orphaned nodes: {orphaned}")
    
    # Check for multiple skins (can cause confusion)
    if len(gltf.skins) > 1:
        issues.append(f"Multiple skins found ({len(gltf.skins)})")
    
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    else:
        print("   ✅ No obvious issues found")
    
    return {
        'file': file_path,
        'nodes': len(gltf.nodes),
        'root_nodes': root_nodes,
        'skins': len(gltf.skins),
        'animations': len(gltf.animations)
    }


if __name__ == "__main__":
    # Test files
    test_files = [
        "animations/myo_animation.glb",
        "animations/tripo/panda_running.glb",
        "animations/tripo/a_man_falling.glb"
    ]
    
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    
    for file_path in test_files:
        if Path(file_path).exists():
            inspect_glb(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE")
    print("="*70)
