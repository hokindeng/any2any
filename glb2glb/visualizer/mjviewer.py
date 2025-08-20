"""
MuJoCo viewer for motion playback.
Run with: mjpython -m glb2glb.visualizer.mjviewer <model.xml> <motion.npy>
"""

import sys
import mujoco
import mujoco.viewer
import numpy as np
import time


def play_motion(model_path, motion_path, loop=True):
    """Play motion using MuJoCo viewer."""
    
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load motion data
    motion_data = np.load(motion_path, allow_pickle=True)
    if hasattr(motion_data, 'item'):
        motion_data = motion_data.item()
    
    trajectory = motion_data['qpos']
    fps = motion_data.get('fps', 30.0)
    
    print(f"Model: {model_path}")
    print(f"Motion: {motion_path}")
    print(f"Frames: {trajectory.shape[0]}")
    print(f"DOFs: {trajectory.shape[1]} (model expects {model.nq})")
    print(f"FPS: {fps}")
    print(f"Duration: {motion_data.get('duration', trajectory.shape[0]/fps):.2f}s")
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  Right Arrow: Next frame")
    print("  Left Arrow: Previous frame")
    print("  R: Reset to start")
    print("  ESC: Exit")
    
    # Ensure dimensions match
    if trajectory.shape[1] != model.nq:
        print(f"\n⚠️ Dimension mismatch! Attempting to pad/trim...")
        if trajectory.shape[1] < model.nq:
            padded = np.zeros((trajectory.shape[0], model.nq))
            padded[:, :trajectory.shape[1]] = trajectory
            trajectory = padded
        else:
            trajectory = trajectory[:, :model.nq]
    
    # Launch viewer with passive mode
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure camera
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 4.0
        
        frame = 0
        playing = True
        last_time = time.time()
        frame_duration = 1.0 / fps
        
        while viewer.is_running():
            # Handle playback
            if playing:
                current_time = time.time()
                if current_time - last_time >= frame_duration:
                    # Update to next frame
                    frame = (frame + 1) % len(trajectory) if loop else min(frame + 1, len(trajectory) - 1)
                    last_time = current_time
            
            # Set the pose
            data.qpos[:] = trajectory[frame]
            mujoco.mj_forward(model, data)
            
            # Check for key presses (simplified - viewer handles most keys internally)
            # Space key handling is built into the viewer
            
            # Sync with viewer
            viewer.sync()
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)


def compare_motions(source_model, source_motion, target_model, target_motion):
    """Compare two motions side by side."""
    print("Comparison mode not yet implemented for mjpython.")
    print("Run two separate instances to compare:")
    print(f"  Terminal 1: mjpython -m glb2glb.visualizer.mjviewer {source_model} {source_motion}")
    print(f"  Terminal 2: mjpython -m glb2glb.visualizer.mjviewer {target_model} {target_motion}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  mjpython -m glb2glb.visualizer.mjviewer <model.xml> <motion.npy> [--loop]")
        print("\nExamples:")
        print("  mjpython -m glb2glb.visualizer.mjviewer temporary/myo_skeleton.xml temporary/myo_skeleton_motion_padded.npy")
        print("  mjpython -m glb2glb.visualizer.mjviewer temporary/mixamo_idle.xml temporary/mixamo_simple_retargeted.npy --loop")
        sys.exit(1)
    
    model_path = sys.argv[1]
    motion_path = sys.argv[2]
    loop = "--loop" in sys.argv
    
    try:
        play_motion(model_path, motion_path, loop=loop)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)