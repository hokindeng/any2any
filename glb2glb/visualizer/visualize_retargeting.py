"""
Visualize robot-to-robot motion retargeting results.
Shows source and target robots side by side.
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from typing import Optional


class DualRobotViewer:
    """Visualize source and target robots side by side during retargeting."""
    
    def __init__(
        self,
        source_xml: str,
        target_xml: str,
        sync_cameras: bool = True,
        fps: float = 30.0,
    ):
        """
        Initialize dual robot viewer.
        
        Args:
            source_xml: Path to source robot XML
            target_xml: Path to target robot XML
            sync_cameras: Whether to synchronize camera views
            fps: Playback frames per second
        """
        # Load models
        self.source_model = mujoco.MjModel.from_xml_path(source_xml)
        self.source_data = mujoco.MjData(self.source_model)
        
        self.target_model = mujoco.MjModel.from_xml_path(target_xml)
        self.target_data = mujoco.MjData(self.target_model)
        
        self.sync_cameras = sync_cameras
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.source_viewer = None
        self.target_viewer = None
        
        self.playing = False
        self.current_frame = 0
        self.source_trajectory = None
        self.target_trajectory = None
        
    def load_trajectories(
        self,
        source_trajectory: np.ndarray,
        target_trajectory: np.ndarray,
    ):
        """Load trajectories for playback."""
        self.source_trajectory = source_trajectory
        self.target_trajectory = target_trajectory
        
        # Ensure trajectories have same length
        min_len = min(len(source_trajectory), len(target_trajectory))
        self.source_trajectory = source_trajectory[:min_len]
        self.target_trajectory = target_trajectory[:min_len]
        
        self.current_frame = 0
        
    def update_frame(self, frame_idx: int):
        """Update both robots to specified frame."""
        if (self.source_trajectory is None or 
            self.target_trajectory is None):
            return
            
        # Clamp frame index
        frame_idx = max(0, min(frame_idx, len(self.source_trajectory) - 1))
        self.current_frame = frame_idx
        
        # Update source robot
        self.source_data.qpos[:] = self.source_trajectory[frame_idx]
        mujoco.mj_forward(self.source_model, self.source_data)
        
        # Update target robot  
        self.target_data.qpos[:] = self.target_trajectory[frame_idx]
        mujoco.mj_forward(self.target_model, self.target_data)
        
    def play_trajectories(self):
        """Play trajectories in loop."""
        self.playing = True
        
        def playback_loop():
            while self.playing:
                # Update frame
                self.update_frame(self.current_frame)
                
                # Advance frame
                self.current_frame = (self.current_frame + 1) % len(self.source_trajectory)
                
                # Sleep to maintain FPS
                time.sleep(self.dt)
        
        # Start playback in separate thread
        self.playback_thread = threading.Thread(target=playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
    def stop_playback(self):
        """Stop trajectory playback."""
        self.playing = False
        if hasattr(self, 'playback_thread'):
            self.playback_thread.join(timeout=1.0)
    
    def launch_viewers(self):
        """Launch side-by-side viewers."""
        print("Launching viewers...")
        print("Controls:")
        print("  Space: Play/pause")
        print("  R: Reset to frame 0")
        print("  Left/Right arrows: Previous/next frame")
        print("  Q: Quit")
        
        # Create viewer for source robot
        with mujoco.viewer.launch_passive(
            self.source_model,
            self.source_data,
        ) as source_viewer:
            self.source_viewer = source_viewer
            
            # Create viewer for target robot
            with mujoco.viewer.launch_passive(
                self.target_model,
                self.target_data,
            ) as target_viewer:
                self.target_viewer = target_viewer
                
                # Main interaction loop
                self.playing = False
                
                while (self.source_viewer.is_running() and 
                       self.target_viewer.is_running()):
                    
                    # Sync cameras if enabled
                    if self.sync_cameras:
                        target_viewer.cam.azimuth = source_viewer.cam.azimuth
                        target_viewer.cam.elevation = source_viewer.cam.elevation
                        target_viewer.cam.distance = source_viewer.cam.distance
                        target_viewer.cam.lookat[:] = source_viewer.cam.lookat
                    
                    # Handle keyboard input (simplified)
                    # Note: Real implementation would need proper key handling
                    
                    # Update viewers
                    if not self.playing:
                        self.update_frame(self.current_frame)
                    
                    source_viewer.sync()
                    target_viewer.sync()
                    
                    time.sleep(0.01)
        
        self.stop_playback()


def compare_retargeting_results(
    source_xml: str,
    target_xml: str,
    source_motion_file: str,
    target_motion_file: str,
):
    """
    Compare retargeting results by visualizing both robots.
    
    Args:
        source_xml: Path to source robot XML
        target_xml: Path to target robot XML
        source_motion_file: Path to source motion .npy file
        target_motion_file: Path to target motion .npy file
    """
    # Load trajectories
    source_traj = np.load(source_motion_file)
    target_traj = np.load(target_motion_file)
    
    print(f"Loaded trajectories:")
    print(f"  Source: {source_traj.shape}")
    print(f"  Target: {target_traj.shape}")
    
    # Create viewer
    viewer = DualRobotViewer(source_xml, target_xml)
    viewer.load_trajectories(source_traj, target_traj)
    
    # Start playback
    viewer.play_trajectories()
    
    # Launch viewers
    viewer.launch_viewers()


def create_test_motion(model_path: str, output_path: str):
    """Create a simple test motion for a robot."""
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Create a simple motion (e.g., arm wave + squat)
    T = 200  # Number of frames
    trajectory = np.zeros((T, model.nq))
    
    for t in range(T):
        # Start from default pose
        trajectory[t] = model.qpos0.copy()
        
        # Add some motion
        phase = 2 * np.pi * t / 100
        
        # Animate arms (if joints exist)
        for i in range(7, min(13, model.nq)):  # Typical arm joint indices
            trajectory[t, i] = 0.3 * np.sin(phase + i * 0.5)
        
        # Animate legs (squat motion)
        if model.nq > 13:
            # Knee joints (typical indices)
            for i in [13, 16]:  # Adjust based on your model
                if i < model.nq:
                    trajectory[t, i] = 0.5 * (1 - np.cos(phase))
        
        # Add some root motion
        trajectory[t, 2] = model.qpos0[2] - 0.1 * (1 - np.cos(phase))  # Vertical
    
    # Save trajectory
    np.save(output_path, trajectory)
    print(f"Saved test motion to {output_path}")
    
    return trajectory


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Create test motions
        print("Creating test motions...")
        
        source_motion = create_test_motion(
            "temp_models/booster_t1/booster_t1_with_sites.xml",
            "test_source_motion.npy"
        )
        
        # For testing, use same motion on target (normally would be retargeted)
        target_model = mujoco.MjModel.from_xml_path(
            "temp_models/unitree_h1/unitree_h1_retarget_ready.xml"
        )
        target_motion = np.zeros((len(source_motion), target_model.nq))
        for t in range(len(source_motion)):
            target_motion[t] = target_model.qpos0.copy()
            # Copy some joint angles (simplified)
            if source_motion.shape[1] >= 7 and target_motion.shape[1] >= 7:
                target_motion[t, :7] = source_motion[t, :7]  # Copy free joint
        
        np.save("test_target_motion.npy", target_motion)
        
        # Visualize
        compare_retargeting_results(
            "temp_models/booster_t1/booster_t1_with_sites.xml",
            "temp_models/unitree_h1/unitree_h1_retarget_ready.xml",
            "test_source_motion.npy",
            "test_target_motion.npy"
        )
    
    elif len(sys.argv) == 5:
        # Use provided files
        compare_retargeting_results(
            sys.argv[1],  # source_xml
            sys.argv[2],  # target_xml
            sys.argv[3],  # source_motion
            sys.argv[4],  # target_motion
        )
    
    else:
        print("Usage:")
        print("  Test mode: python visualize_retargeting.py test")
        print("  Custom: python visualize_retargeting.py <source_xml> <target_xml> <source_motion.npy> <target_motion.npy>")
