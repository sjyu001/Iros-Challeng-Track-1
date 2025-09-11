#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute nuScenes context for current + history frames
- Fix: history frames also get ego/motion/anchors
- Collect ALL nuScenes objects as anchors (no filtering, no top-k limit)
- Add: anchor status from nuScenes attribute tokens
- Remove: velocity
- Add: anchor bbox (3D center/size/orientation)
- Add: camera intrinsic (for 3D bbox projection)
"""

import argparse, json, os
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from vlm_inference_plus_nuscenes import NuScenesContext


class NuScenesContextFixed(NuScenesContext):
    def _collect_objects_for_cam(self, sample, cam: str):
        try:
            sd_token = sample['data'][cam]
            _, boxes, K = self.nusc.get_sample_data(sd_token, box_vis_level=1)
        except Exception:
            return []

        # calibration → intrinsic
        sd_rec = self.nusc.get("sample_data", sd_token)
        cs_rec = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        intrinsic = cs_rec["camera_intrinsic"]

        objs = []
        for b in boxes:
            name = (b.name or "").lower()

            # ✅ 모든 객체 포함 (필터 제거)

            # status
            status = "unknown"
            try:
                ann = self.nusc.get("sample_annotation", b.token)
                attr_tokens = ann.get("attribute_tokens", [])
                if attr_tokens:
                    status = self.nusc.get("attribute", attr_tokens[0])["name"]
            except Exception:
                pass

            # bbox (3D info)
            bbox = None
            try:
                bbox = {
                    "center": [float(c) for c in b.center],
                    "size": [float(s) for s in b.wlh],
                    "orientation": [float(q) for q in b.orientation.q],  # (w,x,y,z)
                }
            except Exception:
                bbox = None

            # distance / projection
            dist = float(b.center[2]) if b.center is not None else 999.0
            try:
                pts = self._project_box_center(b, K)
            except Exception:
                pts = None
            if pts is None:
                continue

            objs.append(
                {
                    "name": name,
                    "status": status,
                    "bbox": bbox,
                    "intrinsic": intrinsic,  # 3D bbox 투영용 intrinsic 저장
                    "distance": dist,
                    "px": pts,
                    "box": b,  # 디버깅용 (JSON dump 전에는 제거 가능)
                }
            )

        # 거리순 정렬 (가까운 객체 먼저)
        objs.sort(key=lambda o: o["distance"])
        return objs   # ✅ top-k 제한 제거 → 모든 객체 반환

    def build_context(self, scene_token: str, frame_token: str, history_tokens=None):
        """Override to keep extended anchor info (status, bbox, intrinsic)."""
        sample = self.nusc.get("sample", frame_token)

        anchors = []
        cnt = 0
        for cam, sd_token in sample["data"].items():
            if "CAM" not in cam:
                continue
            objs = self._collect_objects_for_cam(sample, cam)
            for obj in objs:
                cnt += 1
                anchor_entry = {
                    "id": f"c{cnt}",
                    "cam": cam,
                    "x": obj["px"][0],
                    "y": obj["px"][1],
                    "name": obj["name"],
                    "distance": obj["distance"],
                    "status": obj.get("status"),
                    "bbox": obj.get("bbox"),
                    "intrinsic": obj.get("intrinsic"),
                }
                anchors.append(anchor_entry)
                
        base_ctx = super().build_context(scene_token, frame_token, history_tokens)

        return {
            "ego_status": base_ctx.get("ego_status"),
            "motion_cues": base_ctx.get("motion_cues"),
            "anchors": anchors,
        }


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
    ctx_builder = NuScenesContextFixed(nusc, top_k_per_cam=999999)  # 사실상 무제한

    out_records = []
    for rec in tqdm(records, desc="Precomputing context (current+history)"):
        scene_token = rec.get("scene_token")
        frame_token = rec.get("frame_token")
        hist_tokens = list(rec.get("history_frames", {}).keys())

        # --- 현재 frame context ---
        try:
            main_ctx = ctx_builder.build_context(scene_token, frame_token, hist_tokens)
        except Exception as e:
            print(f"[WARN] main context 실패: {frame_token} {e}")
            main_ctx = {}

        rec["ego_status"] = main_ctx.get("ego_status")
        rec["motion_cues"] = main_ctx.get("motion_cues")
        rec["anchors"] = main_ctx.get("anchors", [])

        # --- history frame contexts ---
        history_contexts = {}
        for i, htoken in enumerate(hist_tokens):
            try:
                prev_hist = hist_tokens[i + 1 :]
                h_ctx = ctx_builder.build_context(scene_token, htoken, prev_hist)
                history_contexts[htoken] = {
                    "ego_status": h_ctx.get("ego_status"),
                    "motion_cues": h_ctx.get("motion_cues"),
                    "anchors": h_ctx.get("anchors", []),
                }
            except Exception as e:
                print(f"[WARN] history context 실패: {htoken} {e}")
                history_contexts[htoken] = {
                    "ego_status": None,
                    "motion_cues": None,
                    "anchors": [],
                }
        rec["history_contexts"] = history_contexts

        out_records.append(rec)

    with open(args.output, "w", encoding="utf-8") as f:
        # box 객체는 JSON dump 불가 → 제거
        for r in out_records:
            for a in r.get("anchors", []):
                if "box" in a:
                    a.pop("box")
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Saved {len(out_records)} records with current+history context → {args.output}"
    )


if __name__ == "__main__":
    main()
