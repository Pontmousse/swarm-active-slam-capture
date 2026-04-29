"""
Landmark Registry for SwarmDFGO+

Maintains a registry of landmarks with stable IDs for re-identification across frames.
Uses feature descriptors and spatial gating for matching.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Import descriptor dimension from config (default to 33 for FPFH)
try:
    import config
    DESCRIPTOR_DIM = getattr(config, 'DESCRIPTOR_DIM', 33)
except ImportError:
    DESCRIPTOR_DIM = 33


@dataclass
class LandmarkTrack:
    """Stores information about a tracked landmark."""
    id: int
    position_world: np.ndarray        # current estimate (3,)
    descriptor_mean: np.ndarray       # running mean descriptor (DESCRIPTOR_DIM,)
    times_seen: int = 1
    last_seen_frame: int = 0
    gt_index: Optional[int] = None    # Ground truth point cloud index (for error calculation)


class LandmarkRegistry:
    """
    Maintains a registry of landmarks with stable IDs for re-identification across frames.
    Uses feature descriptors and spatial gating for matching.
    """
    
    def __init__(self,
                 desc_match_ratio: float = 0.9,
                 desc_l2_thresh: Optional[float] = None,
                 spatial_gate_m: float = 3.0):
        """
        Initialize the landmark registry.
        
        Args:
            desc_match_ratio: Ratio threshold for descriptor matching (default 0.85)
            desc_l2_thresh: Optional L2 distance threshold for descriptors
            spatial_gate_m: Spatial gating distance in meters (default 0.5)
        """
        self.next_id = 0
        self.tracks: Dict[int, LandmarkTrack] = {}
        self.desc_match_ratio = desc_match_ratio
        self.desc_l2_thresh = desc_l2_thresh
        self.spatial_gate_m = spatial_gate_m

    def _match_descriptor(self, d_curr: np.ndarray, D_prev: np.ndarray) -> Tuple[Optional[int], float, bool]:
        """
        Match a descriptor against a set of previous descriptors using ratio test.
        
        Args:
            d_curr: Current descriptor (DESCRIPTOR_DIM,)
            D_prev: Previous descriptors matrix (N, DESCRIPTOR_DIM)
            
        Returns:
            (best_index, l2_dist, ratio_ok) in D_prev, or (None, inf, False) if empty or no match
        """
        if D_prev.size == 0 or len(D_prev) == 0:
            return None, float('inf'), False
        
        # Compute L2 distances
        dists = np.linalg.norm(D_prev - d_curr, axis=1)
        order = np.argsort(dists)
        best = int(order[0])
        best_dist = float(dists[best])
        
        # Ratio test: best match must be significantly better than second best
        ratio_ok = True
        if len(dists) >= 2:
            second = int(order[1])
            ratio_ok = best_dist < self.desc_match_ratio * float(dists[second])
        
        # Optional absolute threshold
        if self.desc_l2_thresh is not None and best_dist > self.desc_l2_thresh:
            ratio_ok = False
            
        return best, best_dist, ratio_ok

    def propagate_positions(self, com: np.ndarray, vel: np.ndarray, ang_vel: np.ndarray, 
                           step_size: int, num_steps: int = 1):
        """
        Propagate all landmark positions forward using target kinematics.
        Call this BEFORE matching to ensure spatial gating works on a moving target.
        
        Args:
            com: Center of mass of target (3,)
            vel: Linear velocity of target (3,)
            ang_vel: Angular velocity of target (3,)
            step_size: Step size multiplier (from config)
            num_steps: Number of time steps to propagate (default 1)
        """
        dt = 1/240
        DeltaT = dt * step_size * num_steps
        
        com = np.array(com, dtype=np.float64)
        vel = np.array(vel, dtype=np.float64)
        ang_vel = np.array(ang_vel, dtype=np.float64)
        
        for lm_id, track in self.tracks.items():
            pos = np.array(track.position_world, dtype=np.float64).copy()
            # Translate to CoM frame
            pos = pos - com
            # Apply rotation and translation
            pos = pos + (np.cross(ang_vel, pos) + vel) * DeltaT
            # Translate back to world frame
            pos = pos + com
            track.position_world = pos

    def try_match(self,
                  feature_point_sensor: np.ndarray,   # (3,) in sensor/agent frame
                  agent_pose_world: Tuple[np.ndarray, np.ndarray],  # (t, R) where t is (3,), R is (3,3)
                  feature_desc: np.ndarray,           # descriptor
                  candidate_ids: List[int]) -> Optional[int]:
        """
        Try to match an observation to an existing landmark track (no creation).
        Returns matched landmark_id if descriptor + spatial gate passes, otherwise None.
        """
        if len(candidate_ids) == 0:
            return None

        t_w, R_w = agent_pose_world
        pw = R_w @ feature_point_sensor + t_w

        D_prev = np.stack([self.tracks[k].descriptor_mean for k in candidate_ids], axis=0)
        best_idx, _, ratio_ok = self._match_descriptor(feature_desc, D_prev)
        if best_idx is None or not ratio_ok:
            return None

        lm_id = candidate_ids[best_idx]
        lm = self.tracks[lm_id]

        if np.linalg.norm(pw - lm.position_world) > self.spatial_gate_m:
            return None

        return lm_id

    def get_or_create(self,
                      frame_idx: int,
                      feature_point_sensor: np.ndarray,   # (3,) in sensor/agent frame
                      agent_pose_world: Tuple[np.ndarray, np.ndarray],  # (t, R) where t is (3,), R is (3,3)
                      feature_desc: np.ndarray,           # descriptor
                      candidate_ids: List[int],
                      gt_index: Optional[int] = None) -> int:
        """
        Try to re-identify landmark using descriptor + spatial gate against candidate tracks.
        If none passes, create a new landmark ID.
        
        Args:
            frame_idx: Current frame index
            feature_point_sensor: Feature position in sensor frame (3,)
            agent_pose_world: Agent pose as (translation, rotation_matrix)
            feature_desc: Feature descriptor
            candidate_ids: List of candidate landmark IDs to match against
            gt_index: Optional ground truth point cloud index (for error calculation)
            
        Returns:
            landmark_id: Stable landmark ID (either matched or newly created)
        """
        t_w, R_w = agent_pose_world
        
        # Predict world position from current pose
        pw = R_w @ feature_point_sensor + t_w

        # Build descriptor matrix of candidates
        if len(candidate_ids) == 0:
            return self._create_new(pw, feature_desc, frame_idx, gt_index)

        D_prev = np.stack([self.tracks[k].descriptor_mean for k in candidate_ids], axis=0)
        best_idx, _, ratio_ok = self._match_descriptor(feature_desc, D_prev)

        if best_idx is None or not ratio_ok:
            return self._create_new(pw, feature_desc, frame_idx, gt_index)

        lm_id = candidate_ids[best_idx]
        lm = self.tracks[lm_id]
        
        # Spatial gate: ensure predicted position close to current track position
        if np.linalg.norm(pw - lm.position_world) > self.spatial_gate_m:
            return self._create_new(pw, feature_desc, frame_idx, gt_index)

        # Update track
        lm.times_seen += 1
        lm.last_seen_frame = frame_idx
        
        # Simple running mean for descriptor
        alpha = 1.0 / lm.times_seen
        lm.descriptor_mean = (1 - alpha) * lm.descriptor_mean + alpha * feature_desc
        
        # Update position by averaging pre-optimization
        lm.position_world = (lm.position_world * (lm.times_seen - 1) + pw) / lm.times_seen
        
        # Update gt_index if provided and not already set
        if gt_index is not None and lm.gt_index is None:
            lm.gt_index = gt_index
        
        self.tracks[lm_id] = lm
        return lm_id

    def _create_new(self, pw: np.ndarray, desc: np.ndarray, frame_idx: int, 
                    gt_index: Optional[int] = None) -> int:
        """
        Create a new landmark track.
        
        Args:
            pw: World position (3,)
            desc: Feature descriptor
            frame_idx: Current frame index
            gt_index: Optional ground truth point cloud index
            
        Returns:
            New landmark ID
        """
        lm_id = self.next_id
        self.next_id += 1
        self.tracks[lm_id] = LandmarkTrack(
            id=lm_id,
            position_world=pw.copy(),
            descriptor_mean=desc.copy(),
            times_seen=1,
            last_seen_frame=frame_idx,
            gt_index=gt_index
        )
        return lm_id

    def active_ids(self, min_last_seen: int) -> List[int]:
        """
        Get list of active landmark IDs (seen after min_last_seen frame).
        
        Args:
            min_last_seen: Minimum frame index to consider active
            
        Returns:
            List of active landmark IDs
        """
        return [lm_id for lm_id, lm in self.tracks.items() if lm.last_seen_frame >= min_last_seen]
    
    def update_position(self, lm_id: int, new_position: np.ndarray):
        """
        Update the world position of a landmark track.
        Typically called after optimization.
        
        Args:
            lm_id: Landmark ID
            new_position: Updated world position (3,)
        """
        if lm_id in self.tracks:
            self.tracks[lm_id].position_world = np.array(new_position, dtype=np.float64).copy()

    def remove_landmark(self, lm_id: int):
        """
        Remove a landmark from the registry.
        Used for outlier rejection.
        
        Args:
            lm_id: Landmark ID to remove
        """
        if lm_id in self.tracks:
            del self.tracks[lm_id]

    def get_gt_index(self, lm_id: int) -> Optional[int]:
        """
        Get the ground truth index for a landmark.
        
        Args:
            lm_id: Landmark ID
            
        Returns:
            Ground truth point cloud index, or None if not available
        """
        if lm_id in self.tracks:
            return self.tracks[lm_id].gt_index
        return None

    def select_features(self,
                        frame_idx: int,
                        landset_world: np.ndarray,
                        desc_all: np.ndarray,
                        t_w: np.ndarray,
                        R_w: np.ndarray,
                        max_land: int,
                        sw: int,
                        use_legacy_random_sampling: bool = False,
                        use_random_feature_fill: bool = True,
                        gt_indices: Optional[np.ndarray] = None,
                        return_selection_idx: bool = False):
        """
        Select a subset of features for graph construction, prioritizing stable re-observations.

        This function returns aligned arrays: FeatureSet, FeatureDescSet, FeatureIdSet.

        Args:
            frame_idx: Current frame index
            landset_world: Nx3 points in world frame (noisy)
            desc_all: NxD descriptors aligned 1:1 with landset_world
            t_w: (3,) agent translation in world frame
            R_w: (3,3) agent rotation matrix in world frame
            max_land: Max number of landmarks to keep
            sw: sliding window size (used to decide active candidates)
            use_legacy_random_sampling: if True, purely random selection (old behavior)
            use_random_feature_fill: when matched-first is used, fill remaining slots randomly (else deterministic)
            gt_indices: Optional Nx1 array of ground truth indices for each point

        Returns:
            FeatureSet: Mx3 selected points (world frame)
            FeatureDescSet: MxD selected descriptors
            FeatureIdSet: (M,) stable landmark IDs aligned with FeatureSet/FeatureDescSet
        """
        landset_world = np.asarray(landset_world).reshape(-1, 3)
        desc_dim = desc_all.shape[1] if len(desc_all.shape) > 1 and desc_all.shape[0] > 0 else DESCRIPTOR_DIM
        desc_all = np.asarray(desc_all).reshape(-1, desc_dim)
        t_w = np.asarray(t_w).reshape(3,)
        R_w = np.asarray(R_w).reshape(3, 3)

        if landset_world.size == 0 or len(landset_world) == 0:
            if return_selection_idx:
                return (np.array([]).reshape(0, 3),
                        np.array([]).reshape(0, desc_dim),
                        np.array([], dtype=int),
                        np.array([], dtype=int))
            return (np.array([]).reshape(0, 3),
                    np.array([]).reshape(0, desc_dim),
                    np.array([], dtype=int))

        max_land_min = min(int(max_land), len(landset_world))
        candidate_ids = self.active_ids(min_last_seen=max(0, int(frame_idx) - int(sw)))
        agent_pose_world = (t_w, R_w)

        # ----------------------------
        # Choose indices to consider
        # ----------------------------
        if use_legacy_random_sampling:
            selection_idx = random.sample(range(0, len(landset_world)), max_land_min)
        else:
            matched = []
            unmatched = []
            for idx in range(len(landset_world)):
                feature = landset_world[idx]
                desc = desc_all[idx]
                feature_sensor = (R_w.T @ (feature - t_w).T).T
                lm_id = self.try_match(feature_sensor, agent_pose_world, desc, candidate_ids)
                if lm_id is not None:
                    matched.append((idx, lm_id))
                else:
                    unmatched.append(idx)

            selection_idx = []
            used_ids = set()

            # Take unique matched IDs first (stable)
            for idx, lm_id in matched:
                if lm_id in used_ids:
                    continue
                used_ids.add(lm_id)
                selection_idx.append(idx)
                if len(selection_idx) >= max_land_min:
                    break

            # Fill remaining with unmatched
            remaining = max_land_min - len(selection_idx)
            if remaining > 0 and len(unmatched) > 0:
                if use_random_feature_fill:
                    fill_idx = random.sample(unmatched, min(remaining, len(unmatched)))
                else:
                    fill_idx = unmatched[:remaining]
                selection_idx.extend(fill_idx)

        selection_idx = list(selection_idx)[:max_land_min]

        # ----------------------------
        # Assign stable IDs (keep arrays aligned + unique IDs per frame)
        # ----------------------------
        FeatureSet_unique = []
        FeatureDesc_unique = []
        FeatureId_unique = []

        inserted_ids = set()
        candidate_ids_aug = list(candidate_ids)  # allow matching against tracks created earlier in this same frame

        for idx_sel in selection_idx:
            feature = landset_world[idx_sel]
            desc = desc_all[idx_sel]
            feature_sensor = (R_w.T @ (feature - t_w).T).T
            
            # Get ground truth index if available
            gt_idx = int(gt_indices[idx_sel]) if gt_indices is not None else None

            lm_id = self.get_or_create(int(frame_idx), feature_sensor, agent_pose_world, desc, 
                                       candidate_ids_aug, gt_index=gt_idx)

            if lm_id in inserted_ids:
                continue

            inserted_ids.add(lm_id)
            if lm_id not in candidate_ids_aug:
                candidate_ids_aug.append(lm_id)

            FeatureSet_unique.append(feature)
            FeatureDesc_unique.append(desc)
            FeatureId_unique.append(lm_id)

        FeatureSet = np.array(FeatureSet_unique).reshape(-1, 3)
        FeatureDescSet = np.array(FeatureDesc_unique).reshape(-1, desc_dim) if len(FeatureDesc_unique) > 0 else np.array([]).reshape(0, desc_dim)
        FeatureIdSet = np.array(FeatureId_unique, dtype=int)

        if return_selection_idx:
            return FeatureSet, FeatureDescSet, FeatureIdSet, np.array(selection_idx, dtype=int)
        return FeatureSet, FeatureDescSet, FeatureIdSet
