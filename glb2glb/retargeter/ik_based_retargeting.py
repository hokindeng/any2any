"""
IK-based motion retargeting using mink library.
Based on FastIK approach for proper motion transfer.
"""

import numpy as np
import mujoco as mj
from typing import Dict, List, Optional, Tuple
import logging

# Check if mink is available
try:
    import mink
    from mink import SE3
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: mink not installed. Install with: pip install pink")

logger = logging.getLogger(__name__)


class IKBasedRetargeter:
    """
    IK-based motion retargeting using mink for proper joint solving.
    """
    
    def __init__(
        self,
        source_xml: str,
        target_xml: str,
        dt: float = 0.01,
        max_iters: int = 10,
        position_cost: float = 200.0,
        orientation_cost: float = 50.0,
        posture_cost: float = 1.0,
        lm_damping: float = 0.1,
        solver: str = "daqp",  # Use daqp solver
        verbose: bool = True
    ):
        """
        Initialize IK-based retargeter.
        
        Args:
            source_xml: Path to source MuJoCo model with tracker sites
            target_xml: Path to target MuJoCo model
            dt: Integration timestep for IK
            max_iters: Maximum IK iterations per frame
            position_cost: Cost for position matching
            orientation_cost: Cost for orientation matching
            posture_cost: Cost for posture regularization
            lm_damping: Levenberg-Marquardt damping
            solver: IK solver to use
            verbose: Print progress information
        """
        if not MINK_AVAILABLE:
            raise ImportError("mink library required for IK-based retargeting. Install with: pip install pink")
            
        self.verbose = verbose
        
        # Load models
        self.source_model = mj.MjModel.from_xml_path(source_xml)
        self.source_data = mj.MjData(self.source_model)
        self.target_model = mj.MjModel.from_xml_path(target_xml)
        self.target_data = mj.MjData(self.target_model)
        
        # IK parameters
        self.dt = dt
        self.max_iters = max_iters
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.posture_cost = posture_cost
        self.lm_damping = lm_damping
        self.solver = solver
        
        # Setup IK for target model
        self._setup_ik()
        
        if self.verbose:
            print(f"IK-based retargeter initialized:")
            print(f"  Source: {self.source_model.njnt} joints, {self.source_model.nq} DOFs, {self.source_model.nsite} sites")
            print(f"  Target: {self.target_model.njnt} joints, {self.target_model.nq} DOFs, {self.target_model.nsite} sites")
            print(f"  Mapped sites: {len(self.site_map)}")
            
    def _setup_ik(self):
        """Setup IK solver for target model."""
        # Initialize configuration
        self.configuration = mink.Configuration(self.target_model)
        
        # Create posture task (regularization)
        self.posture_task = mink.PostureTask(self.target_model, cost=self.posture_cost)
        self.tasks = [self.posture_task]
        
        # Map source sites to target sites/bodies
        self.site_map = self._create_site_mapping()
        
        # Create frame tasks for each mapped site
        self.frame_tasks = []
        for source_site, target_name in self.site_map.items():
            # Target is always a body (since target model doesn't have sites)
            task = mink.FrameTask(
                frame_name=target_name,
                frame_type="body",
                position_cost=self.position_cost,
                orientation_cost=self.orientation_cost,
                lm_damping=self.lm_damping,
            )
            self.frame_tasks.append(task)
            
        self.tasks.extend(self.frame_tasks)
        
        # Set initial posture target
        mj.mj_forward(self.target_model, self.target_data)
        self.configuration.update(self.target_data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        
    def _create_site_mapping(self) -> Dict[str, str]:
        """Create mapping from source sites to target bodies."""
        mapping = {}
        
        # Get all source sites
        source_sites = []
        try:
            for i in range(int(self.source_model.nsite)):
                # mj_id2name expects ints under mocks
                site_name = mj.mj_id2name(self.source_model, int(getattr(mj.mjtObj, 'mjOBJ_SITE', 6)), int(i))
                if site_name:
                    source_sites.append(site_name)
        except Exception:
            # Fallback: use names from model if available
            try:
                for i in range(int(self.source_model.nsite)):
                    site_name = getattr(self.source_model, 'site_name', [None]*int(self.source_model.nsite))[i]
                    if site_name:
                        source_sites.append(site_name)
            except Exception:
                pass
                
        # Map source sites to target bodies
        for src_site in source_sites:
            # Remove _site suffix if present
            base_name = src_site.replace('_site', '').replace('_body', '')
            
            # Try to match to target bodies
            mapped = self._find_matching_body(base_name)
            if mapped:
                mapping[src_site] = mapped
            else:
                # Try direct body name match
                for i in range(int(self.target_model.nbody)):
                    body_name = mj.mj_id2name(self.target_model, int(getattr(mj.mjtObj, 'mjOBJ_BODY', 1)), int(i))
                    if body_name and base_name.lower() in body_name.lower():
                        mapping[src_site] = body_name
                        break
                
        return mapping
        
    def _find_matching_body(self, name: str) -> Optional[str]:
        """Find matching body in target model."""
        name_lower = name.lower()
        
        # Mapping patterns
        patterns = [
            ('pelvis', 'Hip'),
            ('lumbar_spine', 'Spine'),
            ('thoracic_spine', 'Chest'),
            ('cervical_spine', 'Neck'),
            ('skull', 'Head'),
            ('clavicle', 'Shoulder'),
            ('shoulder', 'Shoulder'),
            ('elbow', 'Elbow'),
            ('wrist', 'Wrist'),
            ('femur', 'Thigh'),
            ('tibia', 'Shin'),
            ('ankle', 'Ankle'),
            ('foot', 'Foot'),
        ]
        
        # Check for left/right
        is_left = '_l' in name_lower or 'left' in name_lower
        is_right = '_r' in name_lower or 'right' in name_lower
        
        # Try pattern matching
        for src_pattern, tgt_pattern in patterns:
            if src_pattern in name_lower:
                # Build target name with side
                if is_left:
                    candidates = [f"L_{tgt_pattern}", f"Left_{tgt_pattern}", f"{tgt_pattern}_L"]
                elif is_right:
                    candidates = [f"R_{tgt_pattern}", f"Right_{tgt_pattern}", f"{tgt_pattern}_R"]
                else:
                    candidates = [tgt_pattern]
                    
                # Try each candidate
                for candidate in candidates:
                    # Check if body exists
                    for i in range(self.target_model.nbody):
                        body_name = mj.mj_id2name(self.target_model, mj.mjtObj.mjOBJ_BODY, i)
                        if body_name and candidate in body_name:
                            return body_name
                            
        return None
        
    def retarget_motion(
        self,
        source_motion: np.ndarray,
        fps: float = 30.0
    ) -> Dict:
        """
        Retarget motion from source to target using IK.
        
        Args:
            source_motion: Source motion data (n_frames, source_nq)
            fps: Frames per second
            
        Returns:
            Dict with retargeted motion data
        """
        n_frames = source_motion.shape[0]
        target_motion = np.zeros((n_frames, self.target_model.nq))
        
        if self.verbose:
            print(f"\nRetargeting {n_frames} frames...")
            
        # Process each frame
        for frame_idx in range(n_frames):
            if self.verbose and frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{n_frames}")
                
            # Set source pose
            self.source_data.qpos[:] = source_motion[frame_idx]
            mj.mj_forward(self.source_model, self.source_data)
            
            # Get source site positions
            target_positions = []
            for src_site, tgt_name in getattr(self, 'site_map', {}).items():
                try:
                    site_id = self.source_model.site(src_site).id
                except Exception:
                    site_id = 0
                if site_id != -1:
                    # Get global position of source site
                    site_pos = self.source_data.site_xpos[site_id].copy()
                    site_mat = self.source_data.site_xmat[site_id].reshape(3, 3).copy()
                    
                    # Convert rotation matrix to quaternion (WXYZ format)
                    # Using MuJoCo's mat2quat function
                    quat_wxyz = np.zeros(4)
                    try:
                        mj.mju_mat2Quat(quat_wxyz, site_mat.flatten())
                    except Exception:
                        quat_wxyz[:] = [1, 0, 0, 0]
                    
                    # Create SE3 from quaternion and position
                    # SE3 expects [qw, qx, qy, qz, x, y, z]
                    wxyz_xyz = np.concatenate([quat_wxyz, site_pos])
                    se3_target = mink.SE3(wxyz_xyz)
                    target_positions.append(se3_target)
                else:
                    target_positions.append(None)
                    
            # Set targets for frame tasks
            for task, target_se3 in zip(self.frame_tasks, target_positions):
                if target_se3 is not None:
                    task.set_target(target_se3)
                    task.set_position_cost(self.position_cost)
                    task.set_orientation_cost(self.orientation_cost)
                else:
                    # Disable task if no target
                    task.set_position_cost(0)
                    task.set_orientation_cost(0)
                    
            # Solve IK
            for iter_idx in range(self.max_iters):
                vel = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=self.tasks,
                    dt=self.dt,
                    solver=self.solver,
                )
                self.configuration.integrate_inplace(vel, self.dt)
                
                # Check convergence
                errors = []
                for task in self.frame_tasks:
                    if task.position_cost > 0:
                        err = task.compute_error(self.configuration)
                        errors.append(np.linalg.norm(err[:3]))  # Position error
                        
                if errors and np.mean(errors) < 0.001:
                    break
                    
            # Store result
            target_motion[frame_idx] = self.configuration.q.copy()
            
            # Update data for next frame
            self.target_data.qpos[:] = self.configuration.q
            mj.mj_forward(self.target_model, self.target_data)
            
        if self.verbose:
            print(f"  Retargeting complete!")
            
            # Check animation quality
            animated_joints = 0
            for j in range(self.target_model.njnt):
                addr = self.target_model.jnt_qposadr[j]
                jtype = self.target_model.jnt_type[j]
                
                if jtype == 0:  # free
                    variance = np.var(target_motion[:, addr:addr+7], axis=0)
                    if np.any(variance > 0.0001):
                        animated_joints += 1
                elif jtype == 1:  # ball
                    variance = np.var(target_motion[:, addr:addr+4], axis=0)
                    if np.any(variance > 0.0001):
                        animated_joints += 1
                elif jtype == 3:  # hinge
                    variance = np.var(target_motion[:, addr])
                    if variance > 0.0001:
                        animated_joints += 1
                        
            print(f"  Animated joints: {animated_joints}/{self.target_model.njnt}")
            
        return {
            'qpos': target_motion,
            'fps': fps,
            'n_frames': n_frames
        }


def ik_retarget_motion(
    source_xml: str,
    target_xml: str,
    source_motion_path: str,
    output_motion_path: str,
    verbose: bool = True
) -> bool:
    """
    Convenience function for IK-based motion retargeting.
    
    Args:
        source_xml: Path to source MuJoCo model with tracker sites
        target_xml: Path to target MuJoCo model
        source_motion_path: Path to source motion NPY file
        output_motion_path: Path to save retargeted motion
        verbose: Print progress
        
    Returns:
        True if successful
    """
    try:
        # Load source motion
        motion_data = np.load(source_motion_path, allow_pickle=True).item()
        source_motion = motion_data['qpos']
        fps = motion_data.get('fps', 30.0)
        
        # Create retargeter
        retargeter = IKBasedRetargeter(
            source_xml=source_xml,
            target_xml=target_xml,
            verbose=verbose
        )
        
        # Retarget motion
        retargeted = retargeter.retarget_motion(source_motion, fps)
        
        # Save retargeted motion
        np.save(output_motion_path, retargeted)
        
        return True
        
    except Exception as e:
        logger.error(f"IK retargeting failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing IK-based retargeting...")
    
    success = ik_retarget_motion(
        source_xml="temporary/myo_animation_prepared.xml",
        target_xml="temporary/panda_running.xml",
        source_motion_path="temporary/myo_animation_motion.npy",
        output_motion_path="temporary/ik_retargeted.npy",
        verbose=True
    )
    
    if success:
        print("\n✅ IK-based retargeting complete!")
        print("View with: conda activate glb2glb && mjpython -m glb2glb.visualizer.mjviewer temporary/panda_running.xml temporary/ik_retargeted.npy --loop")
    else:
        print("\n❌ Retargeting failed!")
