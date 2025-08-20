#!/usr/bin/env python3
"""
Example: Transfer animation from one GLB to another.
"""

import glb2glb

# Transfer animation from source to target GLB
success = glb2glb.transfer_animation(
    source_glb="animations/myo_animation.glb",
    target_glb="animations/tripo/panda_running.glb", 
    output_glb="output/panda_with_myo_animation.glb",
    verbose=True
)

if success:
    print("\n✅ Animation transferred successfully!")
else:
    print("\n❌ Animation transfer failed!")
