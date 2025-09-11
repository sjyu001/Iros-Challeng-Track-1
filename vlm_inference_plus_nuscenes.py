#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute nuScenes context for current + history frames
- ego status: speed, yaw-rate, qualitative maneuver
- motion cues: displacement since earliest history frame
- anchors: all visible objects (with projected 2D pixel coords)
"""

import argparse, json, os, math
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np


class NuScenesContext:
    CAMERA_ORDER = [
        "CAM_FRONT","CAM_FRONT_RIGHT","CAM_FRONT_LEFT",
        "CAM_BACK","CAM_BACK_RIGHT","CAM_BACK_LEFT"
    ]

    def __init__(self, nusc: NuScenes, top_k_per_cam: int = 9999):
        self.nusc = nusc
        self.top_k_per_cam = top_k_per_cam

    # ---------- Ego status helpers ----------
    @staticmethod
    def _yaw_from_quat(q: Quaternion) -> float:
        R = q.rotation_matrix
        return math.atan2(R[1, 0], R[0, 0])

    def _ego_pose(self, sample_data_token: str) -> Tuple[List[float], Quaternion]:
        sd = self.nusc.get("sample_data", sample_data_token)
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return ep["translation"], Quaternion(ep["rotation"])

    def _safe_has_sample(self, token: str) -> bool:
        try:
            self.nusc.get("sample", token)
            return True
        except Exception:
            return False

    def _speed_yawrate(self, curr_token: str, hist_tokens: List[str]) -> Tuple[Optional[float], Optional[float]]:
        try:
            sample_curr = self.nusc.get("sample", curr_token)
            cam_token_curr = sample_curr["data"].get("CAM_FRONT")
            if cam_token_curr is None:
                return None, None
            sd_curr = self.nusc.get("sample_data", cam_token_curr)
            t_curr = sd_curr["timestamp"] / 1e6
            p_curr, r_curr = self._ego_pose(cam_token_curr)

            hist_valid = [h for h in hist_tokens if self._safe_has_sample(h)]
            if not hist_valid:
                return None, None
            h_last = hist_valid[0]
            sample_prev = self.nusc.get("sample", h_last)
            cam_token_prev = sample_prev["data"].get("CAM_FRONT")
            sd_prev = self.nusc.get("sample_data", cam_token_prev)
            t_prev = sd_prev["timestamp"] / 1e6
            p_prev, r_prev = self._ego_pose(cam_token_prev)

            dt = max(1e-3, t_curr - t_prev)
            dx, dy = p_curr[0]-p_prev[0], p_curr[1]-p_prev[1]
            speed = math.hypot(dx, dy) / dt

            yaw_curr = self._yaw_from_quat(r_curr)
            yaw_prev = self._yaw_from_quat(r_prev)
            dyaw = math.atan2(math.sin(yaw_curr-yaw_prev), math.cos(yaw_curr-yaw_prev))
            yaw_rate = dyaw / dt
            return speed, yaw_rate
        except Exception:
            return None, None

    # ---------- Object helpers ----------
    def _project_box_center(self, box: Box, K) -> Optional[Tuple[float, float]]:
        try:
            pts = view_points(np.array(box.center).reshape(3,1), np.array(K), normalize=True)
            return float(pts[0,0]), float(pts[1,0])
        except Exception:
            return None

    def _collect_objects_for_cam(self, sample: Dict[str, Any], cam: str) -> List[Dict[str, Any]]:
        try:
            sd_token = sample["data"][cam]
            _, boxes, K = self.nusc.get_sample_data(sd_token, box_vis_level=1)
        except Exception:
            return []

        objs = []
        for b in boxes:
            px = self._project_box_center(b, K)
            if px is None:
                continue
            dist = float(np.linalg.norm(b.center))
            objs.append({
                "name": (b.name or "").lower(),
                "distance": dist,
                "px": px
            })
        objs.sort(key=lambda o: o["distance"])
        return objs[: self.top_k_per_cam]

    # ---------- Context builder ----------
    def build_context(self, scene_token: str, sample_token: str, history_tokens: List[str]) -> Dict[str, Any]:
        sample = self.nusc.get("sample", sample_token)

        # Ego status
        speed, yaw_rate = self._speed_yawrate(sample_token, history_tokens)
        ego_desc = []
        if speed is not None:
            ego_desc.append(f"Speed: {speed:.2f} m/s")
        if yaw_rate is not None:
            ego_desc.append(f"Yaw rate: {math.degrees(yaw_rate):.2f} deg/s")

        # Motion cues: displacement since earliest hist
        motion_desc = []
        try:
            if history_tokens:
                h0 = history_tokens[-1]
                if self._safe_has_sample(h0):
                    s0 = self.nusc.get("sample", h0)
                    sd0 = s0["data"].get("CAM_FRONT")
                    p0, _ = self._ego_pose(sd0)

                    sc = self.nusc.get("sample", sample_token)
                    sdc = sc["data"].get("CAM_FRONT")
                    pc, _ = self._ego_pose(sdc)
                    moved = math.hypot(pc[0]-p0[0], pc[1]-p0[1])
                    motion_desc.append(f"Displacement: {moved:.1f} m")
        except Exception:
            pass

        # Anchors
        anchors = []
        cid = 1
        for cam in self.CAMERA_ORDER:
            if cam not in sample["data"]:
                continue
            for o in self._collect_objects_for_cam(sample, cam):
                x, y = o["px"]
                anchors.append({
                    "id": f"c{cid}",
                    "cam": cam,
                    "x": float(x),
                    "y": float(y),
                    "name": o["name"],
                    "distance": o["distance"]
                })
                cid += 1

        return {
            "ego_status": "; ".join(ego_desc) if ego_desc else None,
            "motion_cues": "; ".join(motion_desc) if motion_desc else None,
            "anchors": anchors
        }


# ---------- Runner ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nusc_root", type=str, required=True)
    parser.add_argument("--nusc_version", type=str, default="v1.0-trainval")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=False)
    ctx_builder = NuScenesContext(nusc, top_k_per_cam=9999)

    out_records = []
    for rec in tqdm(records, desc="Precomputing nuScenes context"):
        scene_token = rec.get("scene_token")
        frame_token = rec.get("frame_token")
        hist_tokens = list(rec.get("history_frames", {}).keys())

        ctx = ctx_builder.build_context(scene_token, frame_token, hist_tokens)
        rec["ego_status"] = ctx.get("ego_status")
        rec["motion_cues"] = ctx.get("motion_cues")
        rec["anchors"] = ctx.get("anchors", [])
        out_records.append(rec)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(out_records)} records with nuScenes context → {args.output}")


if __name__ == "__main__":
    main()
