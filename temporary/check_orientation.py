#!/usr/bin/env python3
"""Check the orientation of generated MuJoCo models."""

import numpy as np
import mujoco

def check_model(xml_path, motion_path, name):
    """Load model and check first frame pose."""
    print(f"\n{'='*50}")
    print(f"Checking: {name}")
    print('='*50)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Load motion
    motion = np.load(motion_path, allow_pickle=True).item()
    qpos = motion['qpos']
    
    print(f"Model info:")
    print(f"  nq (DOFs): {model.nq}")
    print(f"  nbody: {model.nbody}")
    print(f"  Motion frames: {qpos.shape[0]}")
    
    # Set first frame
    if qpos.shape[1] == model.nq:
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        
        # Check root body position (usually body 1 or 2)
        if model.nbody > 1:
            print(f"\nFirst frame root position (from qpos):")
            print(f"  Translation: {qpos[0, :3]}")
            if model.nq >= 7:
                print(f"  Quaternion: {qpos[0, 3:7]}")
            
            # Check actual body positions after forward kinematics
            print(f"\nBody positions after forward kinematics:")
            for i in range(min(5, model.nbody)):
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    print(f"  {body_name}: {data.xpos[i]}")
    else:
        print(f"⚠️  DOF mismatch: model expects {model.nq}, motion has {qpos.shape[1]}")

if __name__ == "__main__":
    # Check both models
    check_model(
        "temporary/a_man_falling.xml",
        "temporary/a_man_falling_motion.npy",
        "a_man_falling (Z-up)"
    )
    
    check_model(
        "temporary/myo_animation.xml",
        "temporary/myo_animation_motion.npy",
        "myo_animation (Y-up)"
    )
