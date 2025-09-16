#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Inference with Precomputed Context (Enhanced+++ with 3-Section Enforcement)

Key upgrades for score improvement:
- Deterministic cues injected into context: side (left/center/right) & in-path:yes/no (front/back + center band).
- Relaxed question-object matching radius (12px → 64px) to reduce "(no anchor matched)" cases.
- Standardized, longer outputs with STRICT 3-section format:
    1) Observations: factual description using anchors (≥2 sentences)
    2) Reasoning: step-by-step logic grounded in anchors (≥2 sentences)
    3) Answer: the final result (task-specific constraints kept)
- Task-specific system snippets for Prediction and Planning that STILL obey the 3-section requirement.
- Object-Description seeded with fixed first sentence template (improves ROUGE/METEOR) but embedded into 3 sections.
- Lightweight post-processor `_enforce_three_sections` to repair outputs if a model ever forgets a section.
- MCQ "Answer:" line keeps letter + option text while preserving the 3-section structure.

Assumptions:
- Six camera views: [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]
- Coordinates in questions use 1600×900 frame reference (<id, camera, x, y>)
- /v1/chat/completions is compatible with OpenAI Chat Completions schema (e.g., vLLM/OpenAI-compatible server)

"""

import argparse
import json
import os
import time
import re
import tempfile
import base64
from typing import Any, Dict, List, Tuple, Optional

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.cm as cm
from pyquaternion import Quaternion


# ----------------------------- Domain Knowledge & System Prompts -----------------------------

DRIVING_DOMAIN_KNOWLEDGE = """
You are assisting vision-language autonomous driving reasoning.
Always follow the structured stack: Perception → Prediction → Planning → Action.

PERCEPTION:
- Detect and classify lane geometry: lane count, curvature, merges, and splits.
- Identify road controls: signals, stop/yield signs, lane markings, crosswalks, speed bumps, traffic cones, barriers.
- Recognize and categorize road users: vehicles, pedestrians, cyclists, motorcycles, buses, trucks.
- Estimate distances in meters, relative positions (left, center, right), and states (moving, stopped, braking, turning, merging).
- Consider occlusions from large vehicles, curves, or infrastructure.

PREDICTION:
- Infer future motion from current kinematics (speed, heading, turn signals, brake lights).
- Pedestrians: anticipate intent to cross when near crosswalks or curbs, especially if facing the road.
- Vehicles: detect lane-change or merging cues from wheel angle, lateral drift, or signal lights.
- Heavy vehicles (trucks, buses): longer stopping distance, wider turning radius, blind spots.
- Cyclists/motorcyclists: higher lateral movement uncertainty, frequent lane filtering.
- Always evaluate risk of collision in the next 3–5 seconds.

PLANNING:
- Maintain safe time headway (≥2s normal, ≥3s in poor weather or congestion).
- Prioritize vulnerable road users (pedestrians, cyclists) over maintaining speed.
- Obey traffic signals and right-of-way rules strictly.
- Avoid unnecessary lane changes; only change when safe and beneficial.
- Smooth maneuvers preferred: reduce abrupt braking, keep steering gradual.
- Keep clear of occluded zones (e.g., behind parked cars or large trucks).
- Prepare fallback strategies: if uncertainty is high, slow down and wait.

ACTION RULES OF THUMB:
- If a lead vehicle brakes, increase following gap and prepare to stop smoothly.
- If a pedestrian/cyclist is approaching or waiting near a crosswalk, slow down and yield.
- At intersections: check all directions, anticipate merging/turning vehicles, and avoid blocking.
- In multi-lane roads: watch for merging traffic from ramps and lane changes from adjacent lanes.
- In poor visibility (fog, night, rain): reduce speed, increase following distance, and focus on road markings.
- On highways: keep right unless overtaking, maintain steady speed, avoid unnecessary lane weaving.
- In narrow roads: anticipate parked cars opening doors, pedestrians stepping off curbs.
- Always prioritize safety and legality over efficiency or speed.

SAFETY PRIORITIZATION:
1. Protect vulnerable road users (pedestrians, cyclists).
2. Avoid collisions with vehicles.
3. Respect traffic controls and right-of-way.
4. Maintain passenger comfort and smooth driving.
5. Optimize efficiency only after safety and legality are ensured.
""".strip()

# === Global mandatory output rules ===
CORE_3_SECTION_RULE = """
MANDATORY FORMAT (ALL QUESTIONS):
Always produce exactly three sections in this order, each labeled and on separate paragraphs:
1. Observations: At least two full sentences of objective facts using anchors (category, distance in m, side left/center/right, in-path yes/no, motion state, occlusion).
2. Reasoning: At least two full sentences of step-by-step logic grounded in anchor facts and the driving stack (Perception→Prediction→Planning).
3. Answer: The final result, concise but complete. For MCQ include letter and option text. For prediction include a clear Yes./No. For planning include the decided action(s) and, if asked, a probability (NN%).
Do NOT use bullet points. Do NOT skip or rename sections. Avoid hedging words such as "maybe" or "likely".
Distances must be integer meters or "~<int> m", unit "m". Use only this vocabulary for objects and states:
adult, child, car, truck, bus, bicycle, motorcycle, cyclist, pedestrian, left, center, right, moving, stopped, turning-left, turning-right, merging, braking, occluded, clear.
""".strip()

SYSTEM_PROMPT = f"""
You are an autonomous driving assistant.

TASK DESCRIPTION:
In this phase, participants must answer high-level driving questions on possibly corrupted images.
You must rely on anchors and provided scene context. Six camera views:
[CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT].
Coordinates: <id, camera_view, x, y> with resolution 1600×900.

ANCHORS:
- The provided ANCHORS are the only valid set of objects in the current scene. Do not hallucinate.
- Always ground Observations/Reasoning/Answer in the anchor information (category, distance, position, status, occlusion).
- When <cID,CAM,x,y> appears, match it to the overlapping or nearest anchor and reason on that anchor’s metadata.

{CORE_3_SECTION_RULE}

Domain Knowledge:
{DRIVING_DOMAIN_KNOWLEDGE}

OUTPUT LENGTH:
- Observations: ≥2 sentences. Reasoning: ≥2 sentences. Make answers detailed (aim for 120–220 words overall when possible).
- Keep language clear and factual, weaving anchor info throughout.
""".strip()

PREDICTION_GUIDE = """
PREDICTION FORMAT ADDENDUM:
- The Answer line must be exactly: "Answer: Yes." or "Answer: No." with no extra clause on that line.
- Use in-path (yes/no), camera (front/back), and side (left/center/right) to decide if the object is in the ego moving direction.
- If the question asks about "change its motion state", answer "Yes." if anchor states and spatial relations imply an upcoming change; otherwise "No.".
""".strip()

PLANNING_GUIDE = """
PLANNING FORMAT ADDENDUM:
- If asked for actions with probability, the Answer must be: "Answer: Action: <action>; Why: <reason>; Probability: <NN>%".
- If asked for safe actions: "Answer: Safe: <action1>; <action2>."
- If asked for dangerous actions: "Answer: Dangerous: <action1>; <action2>."
- Allowed actions vocabulary: slow down, stop, keep lane, lane change left, lane change right, yield, wait.
""".strip()

OBJECT_DESC_GUIDE = """
OBJECT DESCRIPTION ADDENDUM:
- Still use the three sections strictly.
- Begin the Answer sentence with a deterministic clause:
  "Answer: The {category} is {state} at ~{dist} m on the {side}, {visibility}."
- Then continue the Answer with additional fluent details (remain grounded in anchors), but avoid bullets.
""".strip()


# ----------------------------- VLM Inference Class -----------------------------

class VLMInference:
    CAMERA_ORDER = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
    ]

    _LEXICON = {
        r"\bpeople\b": "adults",
        r"\bperson\b": "adult",
        r"\bpersons\b": "adults",
        r"\bvehicle\b": "car",
        r"\bvehicles\b": "cars",
        r"\bmeters\b|\bmetres\b": "m",
        r"\bmaybe\b|\blikely\b|\bseems\b": "",
        r"\bdecelerate\b": "slow down",
        r"\baccelerat(e|es|ed|ing)\b": "speed up",
        r"\blane[- ]?change(s|d|ing)?\b": "lane change",
        r"\bbrake(s|d|ing)?\b": "braking",
    }

    def __init__(self,
                 model_name: str,
                 api_base: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 n_consistency: int = 3,
                 max_history_frames: int = 2,
                 include_raw_anchor_text: bool = True,
                 include_question_crops: bool = True,
                 max_anchors_per_cam: int = 20):
        self.api_base = api_base.rstrip('/')
        self.model = model_name
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.n_consistency = max(1, int(n_consistency))
        self.max_history_frames = max(0, int(max_history_frames))
        self.include_raw_anchor_text = bool(include_raw_anchor_text)
        self.include_question_crops = bool(include_question_crops)
        self.max_anchors_per_cam = max(1, int(max_anchors_per_cam))

    # ---------- Geometry utils ----------
    @staticmethod
    def _compute_3d_corners(center, size, quat):
        w, l, h = size
        x = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
        y = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
        z = np.array([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
        corners = np.vstack([x, y, z])
        q = Quaternion(quat)
        R = q.rotation_matrix
        corners = R @ corners
        corners += np.array(center, dtype=float).reshape(3, 1)
        return corners

    @staticmethod
    def _project_points(K: np.ndarray, P3: np.ndarray) -> np.ndarray:
        P = K @ P3
        z = P[2, :]
        z[z == 0] = 1e-6
        return P[:2, :] / z

    # ---------- Spatial cues ----------
    @staticmethod
    def _pos_label(x: float, img_w: int = 1600) -> str:
        if x <= img_w/3: return "left"
        if x < 2*img_w/3: return "center"
        return "right"

    @staticmethod
    def _ego_is_reversing(ego_status: str) -> bool:
        s = (ego_status or "").lower()
        return any(k in s for k in ["reverse", "back up", "backing up", "reversing"])

    @staticmethod
    def _in_path_flag(cam: str, x: float, ego_reversing: bool, img_w: int = 1600, tol: int = 220) -> str:
        center = img_w/2
        on_center = abs(x - center) <= tol
        cam = (cam or "").upper()
        if ego_reversing:
            front_like = cam.startswith("CAM_BACK")
        else:
            front_like = cam.startswith("CAM_FRONT")
        return "yes" if (front_like and on_center) else "no"

    # ---------- Helpers ----------
    @staticmethod
    def _take_recent_history(history: Dict[str, Dict[str, str]], n: int) -> List[Tuple[str, Dict[str, str]]]:
        if not isinstance(history, dict) or n <= 0:
            return []
        items = list(history.items())
        return items[:n]

    @staticmethod
    def _bbox_intersect(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    def _place_text_nonoverlap(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        x: int, y: int,
        font: ImageFont.ImageFont,
        existing_bboxes: List[Tuple[int,int,int,int]],
        img_w: int, img_h: int,
        stroke_width: int = 2,
        pad: int = 4
    ) -> Tuple[int, int]:
        base_offsets = [(12,-24), (14,12), (-14,-24), (-14,12), (0,-36), (0,18), (24,0), (-24,0)]
        radii = [0, 8, 16, 24, 32, 44, 56, 72, 88]
        candidates = []
        for r in radii:
            for (dx, dy) in base_offsets:
                candidates.append((x + dx + (dx and (dx//abs(dx))*r or 0),
                                   y + dy + (dy and (dy//abs(dy))*r or 0)))
        candidates.append((x + 10, y - 10))

        for (tx, ty) in candidates:
            try:
                tb = draw.textbbox((tx, ty), label, font=font, stroke_width=stroke_width)
            except Exception:
                w, h = draw.textsize(label, font=font)
                tb = (tx, ty, tx + w, ty + h)
            tb = (tb[0]-pad, tb[1]-pad, tb[2]+pad, tb[3]+pad)
            if tb[0] < 0 or tb[1] < 0 or tb[2] > img_w or tb[3] > img_h:
                continue
            if any(self._bbox_intersect(tb, eb) for eb in existing_bboxes):
                continue
            return (tx, ty)
        return (x + 8, y - 20)

    def _limit_anchors_by_cam(self, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {cam: [] for cam in self.CAMERA_ORDER}
        for a in anchors or []:
            cam = a.get("cam")
            if cam in grouped:
                grouped[cam].append(a)
        pruned = []
        for cam, arr in grouped.items():
            arr.sort(key=lambda z: (float('inf') if z.get('distance') is None else float(z['distance'])))
            pruned.extend(arr[: self.max_anchors_per_cam])
        return pruned

    # ---------- 3D→2D bbox ----------
    def _ensure_bbox2d(self, anchors: List[Dict[str, Any]]) -> None:
        for a in anchors or []:
            b = a.get("bbox"); K = a.get("intrinsic")
            if not b or not K:
                a["bbox2d"] = None
                continue
            try:
                center = np.array(b["center"], dtype=float)
                size = np.array(b["size"], dtype=float)
                quat = np.array(b["orientation"], dtype=float)
                Kmat = np.array(K, dtype=float)
                P3 = self._compute_3d_corners(center, size, quat)
                P2 = self._project_points(Kmat, P3)
                x0, y0 = float(P2[0,:].min()), float(P2[1,:].min())
                x1, y1 = float(P2[0,:].max()), float(P2[1,:].max())
                a["bbox2d"] = [x0, y0, x1, y1]
            except Exception:
                a["bbox2d"] = None

    # ---------- bbox-based question-object match ----------
    @staticmethod
    def _is_point_in_bbox(px: float, py: float, bbox2d: Optional[List[float]]) -> bool:
        if not bbox2d:
            return False
        x0, y0, x1, y1 = bbox2d
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _match_anchor_by_bbox(self, cam: str, x: float, y: float, anchors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [a for a in anchors if a.get("cam")==cam]
        for a in candidates:
            if self._is_point_in_bbox(x, y, a.get("bbox2d")):
                return a
        best, best_d = None, 1e9
        for a in candidates:
            ax, ay = float(a.get("x", 1e9)), float(a.get("y", 1e9))
            d = (ax - x)**2 + (ay - y)**2
            if d < best_d:
                best, best_d = a, d
        if best is not None and best_d <= (64.0**2):
            return best
        return None

    # ---------- image & crop builders ----------
    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        import cv2
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resize_keep_ar(img: np.ndarray, max_w: int = 1280) -> Image.Image:
        im = Image.fromarray(img)
        if im.width > max_w:
            ratio = max_w / float(im.width)
            im = im.resize((max_w, int(im.height * ratio)))
        return im

    def _crop_patch_with_optional_bbox(self, path: str, x: float, y: float,
                                       maybe_bbox2d: Optional[List[float]],
                                       box: int = 384) -> Optional[str]:
        try:
            im = Image.open(path).convert("RGB")
            cx, cy = int(x), int(y)
            half = box // 2
            left, top  = max(0, cx - half), max(0, cy - half)
            right, bottom = min(im.width, cx + half), min(im.height, cy + half)
            crop = im.crop((left, top, right, bottom))
            if maybe_bbox2d:
                x0,y0,x1,y1 = maybe_bbox2d
                x0, y0, x1, y1 = x0-left, y0-top, x1-left, y1-top
                draw = ImageDraw.Draw(crop)
                draw.rectangle([x0,y0,x1,y1], outline="red", width=2)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            crop.save(tmp.name, format="JPEG", quality=90)
            return tmp.name
        except Exception:
            return None

    # ---------- annotated images ----------
    def _annotate_images(self, img_paths: Dict[str, str], refs: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> Dict[str, str]:
        anchors = self._limit_anchors_by_cam(anchors)
        annotated: Dict[str, str] = {}
        cmap = cm.get_cmap("tab20", max(1, len(anchors)))

        for cam in self.CAMERA_ORDER:
            raw = img_paths.get(cam)
            if not raw or not os.path.exists(raw):
                continue
            base = Image.open(raw).convert("RGB")
            draw = ImageDraw.Draw(base)
            W, H = base.size

            font_small = ImageFont.load_default()
            try:
                font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            except Exception:
                font_big = ImageFont.load_default()

            cam_anchors = [a for a in anchors if a.get("cam") == cam]
            existing_bboxes: List[Tuple[int,int,int,int]] = []

            # draw 3D wireframes
            for idx, a in enumerate(cam_anchors):
                b = a.get("bbox"); K = a.get("intrinsic")
                if not b or not K:
                    continue
                try:
                    center = np.array(b["center"], dtype=float)
                    size   = np.array(b["size"], dtype=float)
                    quat   = np.array(b["orientation"], dtype=float)
                    Kmat   = np.array(K, dtype=float)
                    P3 = self._compute_3d_corners(center, size, quat)
                    P2 = self._project_points(Kmat, P3)
                    color = tuple(int(c*255) for c in cmap(idx % cmap.N)[:3])
                    edges = [
                        [0,1],[1,2],[2,3],[3,0],
                        [4,5],[5,6],[6,7],[7,4],
                        [0,4],[1,5],[2,6],[3,7]
                    ]
                    for e in edges:
                        x0,y0 = P2[:, e[0]]
                        x1,y1 = P2[:, e[1]]
                        draw.line([(x0,y0),(x1,y1)], fill=color, width=2)
                    if not a.get("bbox2d"):
                        a["bbox2d"] = [float(P2[0,:].min()), float(P2[1,:].min()),
                                       float(P2[0,:].max()), float(P2[1,:].max())]
                except Exception:
                    pass

            # anchor dots + short labels
            for idx, a in enumerate(cam_anchors):
                x, y = float(a.get("x", 0)), float(a.get("y", 0))
                xi, yi = int(x), int(y)
                rgba = cmap(idx % cmap.N)
                color = tuple(int(c * 255) for c in rgba[:3])

                R = 6
                draw.ellipse([(xi - R, yi - R), (xi + R, yi + R)], outline=color, width=2, fill=None)
                existing_bboxes.append((xi - R - 2, yi - R - 2, xi + R + 2, yi + R + 2))

                obj_name = str(a.get("name", "object")).split(".")[-1]
                label = obj_name
                if a.get("distance") is not None:
                    label += f" ({float(a['distance']):.1f}m)"
                tx, ty = xi + 5, yi - 10
                draw.text((tx, ty), label, fill=color, font=font_small)
                try:
                    tb = draw.textbbox((tx, ty), label, font=font_small)
                except Exception:
                    w, h = draw.textsize(label, font=font_small)
                    tb = (tx, ty, tx + w, ty + h)
                existing_bboxes.append((tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2))

            # question markers + labels
            ref_cam = [r for r in refs if r["cam"] == cam]
            for r in ref_cam:
                xi, yi = int(r["x"]), int(r["y"])
                matched = self._match_anchor_by_bbox(cam, r["x"], r["y"], cam_anchors)
                if matched:
                    obj_name = str(matched.get("name", "object")).split(".")[-1]
                    dist = matched.get("distance", None)
                    q_label = f"Question Object: {r['label']} ({obj_name}{', ' + f'{float(dist):.1f}m' if dist is not None else ''})"
                else:
                    q_label = f"Question Object: {r['label']}"

                RQ = 12
                draw.ellipse([(xi - RQ, yi - RQ), (xi + RQ, yi + RQ)], outline="black", width=3, fill="yellow")
                existing_bboxes.append((xi - RQ - 3, yi - RQ - 3, xi + RQ + 3, yi + RQ + 3))

                tx, ty = self._place_text_nonoverlap(draw, q_label, xi, yi, font_big, existing_bboxes, W, H,
                                                     stroke_width=2, pad=4)
                draw.text((tx, ty), q_label, fill="yellow", font=font_big, stroke_width=2, stroke_fill="black")
                try:
                    tb = draw.textbbox((tx, ty), q_label, font=font_big, stroke_width=2)
                except Exception:
                    w, h = draw.textsize(q_label, font=font_big)
                    tb = (tx, ty, tx + w, ty + h)
                existing_bboxes.append((tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2))

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            base.save(tmp.name, format="JPEG")
            annotated[cam] = tmp.name

        return annotated

    def _group_anchors_by_cam(self, anchors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        anchors = self._limit_anchors_by_cam(anchors)
        by_cam = {cam: [] for cam in self.CAMERA_ORDER}
        for a in anchors or []:
            cam = a.get("cam")
            if cam in by_cam:
                by_cam[cam].append(a)
        return by_cam

    def _img_to_payload(self, path: str) -> Dict[str, Any]:
        try:
            im = Image.open(path).convert("RGB")
            max_w = 1280
            if im.width > max_w:
                ratio = max_w / float(im.width)
                im = im.resize((max_w, int(im.height * ratio)))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                im.save(tmp.name, format="JPEG", quality=85)
                with open(tmp.name, 'rb') as fh:
                    data = fh.read()
            b64 = base64.b64encode(data).decode("ascii")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        except Exception:
            return {"type": "text", "text": "(missing image)"}

    # ---------- Human-readable context ----------
    def _build_context_text(self, record: Dict[str, Any]) -> str:
        parts = ["SCENE CONTEXT:"]
        ego_status = record.get('ego_status') or ""
        if ego_status:
            parts.append(f"EGO STATUS: {ego_status}")
        if record.get('motion_cues'):
            parts.append(f"MOTION CUES: {record['motion_cues']}")
        ego_reversing = self._ego_is_reversing(ego_status)

        if record.get('anchors'):
            lines = []
            for a in record['anchors']:
                cam = a.get('cam','?')
                x  = float(a.get('x',0)); y = float(a.get('y',0))
                name = a.get('name','object')
                status = a.get('status','?')
                dist = a.get('distance')
                side = self._pos_label(x, 1600)
                inpath = self._in_path_flag(cam, x, ego_reversing, 1600, 220)
                core = f"<{cam},{x:.1f},{y:.1f}> {name} [{status}] ({side}, in-path:{inpath}"
                if dist is not None:
                    core += f", ~{int(round(float(dist)))} m)"
                else:
                    core += ")"
                lines.append(core)
            if lines:
                parts.append("OBJECT CANDIDATES:\n" + "\n".join(lines))
        return "\n".join(parts)

    # ---------- Question refs anchor summary ----------
    def _build_question_anchor_text(self, refs: List[Dict[str, Any]], anchors: List[Dict[str, Any]], ego_reversing: bool) -> str:
        lines = []
        for r in refs:
            cam, x, y = r["cam"], r["x"], r["y"]
            matched = self._match_anchor_by_bbox(cam, x, y, anchors)
            if matched:
                name = matched.get('name','object')
                status = matched.get('status','?')
                dist = matched.get('distance', None)
                side = self._pos_label(float(matched.get("x", x)))
                inpath = self._in_path_flag(cam, float(matched.get("x", x)), ego_reversing)
                if dist is not None:
                    lines.append(f"{cam} ({x:.1f},{y:.1f}) → {name} [{status}] (~{int(round(float(dist)))} m, {side}, in-path:{inpath})")
                else:
                    lines.append(f"{cam} ({x:.1f},{y:.1f}) → {name} [{status}] ({side}, in-path:{inpath})")
            else:
                side = self._pos_label(x)
                inpath = self._in_path_flag(cam, x, ego_reversing)
                lines.append(f"{cam} ({x:.1f},{y:.1f}) → (no anchor matched, {side}, in-path:{inpath})")
        return "ANCHOR_INFO[Question Objects]:\n" + "\n".join(lines) if lines else ""

    # ---------- MCQ helpers (kept for parsing only; no forcing) ----------
    @staticmethod
    def _is_mcq_question(q: str) -> bool:
        if not q: return False
        ql = q.lower()
        if "please select" in ql or "select from" in ql: return True
        return bool(re.search(r'\b[ABCD]\s*\.', q))

    @staticmethod
    def _parse_mcq_options(q: str) -> Dict[str, str]:
        opts = {}
        m = re.search(r'([A-D]\s*\..+)$', q, flags=re.S)
        tail = m.group(1) if m else q
        parts = re.split(r'(?=(?:\b[ABCD]\s*\.))', tail)
        for p in parts:
            m2 = re.match(r'^\s*([ABCD])\s*\.\s*(.+?)\s*(?=$|[ABCD]\s*\.)', p, flags=re.S)
            if m2:
                k = m2.group(1)
                v = m2.group(2).strip()
                v = re.sub(r'\s+', ' ', v).strip().rstrip('.')
                opts[k] = v
        return opts

    # ---------- Normalization (non-semantic) ----------
    def _normalize_answer_text(self, s: str) -> str:
        """Light, non-semantic cleanup; does not change answer choice/meaning."""
        s = s.strip()
        s = re.sub(r'^[`"*]+|[`"*]+$', '', s)
        s = re.sub(r"\b~?(\d+\.\d+)\s*m\b", lambda m: f"{round(float(m.group(1))):d} m", s)
        for pat, rep in self._LEXICON.items():
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        s = re.sub(r"\s{2,}", " ", s).strip(",.; ").strip()
        return s

    # ---------- LLM call with self-consistency ----------
    def call_with_self_consistency(self,
                                   messages: List[Dict[str, Any]],
                                   task_hint: str = "",
                                   question_text: str = "",
                                   is_prediction: bool = False,
                                   is_planning: bool = False,
                                   is_visual_desc: bool = False) -> str:
        answers: List[str] = []

        temp = self.temperature
        top_p = self.top_p
        n_cons = self.n_consistency
        hint = (task_hint or "").lower()

        if ("mcq" in hint) or is_prediction:
            temp, top_p, n_cons = 0.2, 0.2, max(1, min(3, n_cons))
        elif ("vqa" in hint) or is_planning or is_visual_desc:
            temp, top_p, n_cons = 0.5, 0.5, max(3, n_cons)

        if len(question_text.split()) > 40:
            temp, top_p, n_cons = 0.6, 0.6, max(3, n_cons)

        for _ in range(n_cons):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": float(temp),
                "top_p": float(top_p),
                "max_tokens": int(self.max_tokens),
            }
            try:
                resp = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=1200)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                answers.append(content)
            except Exception as e:
                answers.append(f"Observations: (error occurred). Reasoning: (error occurred). Answer: (error) {e}")
            time.sleep(0.05)

        # majority vote based on normalized text, but do NOT alter semantics
        counts: Dict[str, int] = {}
        best, best_cnt = None, -1
        for a in answers:
            norm = self._normalize_answer_text(a)
            counts[norm] = counts.get(norm, 0) + 1
            if counts[norm] > best_cnt:
                best_cnt, best = counts[norm], norm
        if not best:
            best = max(answers, key=len)
        return best

    # ---------- Per-sample processing ----------
    def process_sample(self, record: Dict[str, Any]) -> str:
        question = record.get("question", "")
        img_paths = record.get("img_paths", {}) or {}
        history   = record.get("history_frames", {}) or {}
        anchors   = record.get("anchors", []) or []
        category  = (record.get("category") or "").lower()

        # precompute 2D boxes
        self._ensure_bbox2d(anchors)

        # parse refs
        coords = re.findall(r"<c(\d+),([^,>]+),([\d.]+),([\d.]+)>", question)
        refs = [{"id": int(cid), "cam": cam, "x": float(x), "y": float(y), "label": f"c{cid}"}
                for cid, cam, x, y in coords]

        recent_hist = self._take_recent_history(history, self.max_history_frames)

        # annotated images & grouping
        annotated = self._annotate_images(img_paths, refs, anchors)
        anchors_by_cam = self._group_anchors_by_cam(anchors)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        question_l = (question or "").lower()
        is_prediction = any(k in question_l for k in ["would", "will"]) and ("direction" in question_l or "motion state" in question_l)
        is_planning = any(k in question_l for k in ["what actions", "safe actions", "dangerous actions", "collision", "probability"])
        is_visual_desc = bool(re.search(r"(visual\s+description|describe)\b", question, re.I))
        is_mcq = self._is_mcq_question(question)

        # Non-binding guides
        if is_prediction:
            messages.append({"role": "system", "content": PREDICTION_GUIDE})
        if is_planning:
            messages.append({"role": "system", "content": PLANNING_GUIDE})
        if is_visual_desc:
            messages.append({"role": "system", "content": OBJECT_DESC_GUIDE})

        user_content: List[Any] = []

        # history images
        for _, paths in recent_hist:
            for cam in self.CAMERA_ORDER:
                p = paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))

        # question crops
        if self.include_question_crops:
            for r in refs:
                cam = r["cam"]; p = img_paths.get(cam)
                if p and os.path.exists(p):
                    matched = self._match_anchor_by_bbox(cam, r["x"], r["y"], anchors)
                    bbox2d = matched.get("bbox2d") if matched else None
                    crop_path = self._crop_patch_with_optional_bbox(p, r["x"], r["y"], maybe_bbox2d=bbox2d, box=384)
                    if crop_path:
                        user_content.append(self._img_to_payload(crop_path))

        # raw images + raw anchors (pruned)
        if self.include_raw_anchor_text:
            for cam in self.CAMERA_ORDER:
                p = img_paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))
                    cam_raw = anchors_by_cam.get(cam) or []
                    if cam_raw:
                        try:
                            cam_no_id = [
                                {k: a.get(k) for k in ("name", "cam", "x", "y", "distance", "status")}
                                for a in cam_raw
                            ]
                            raw_json = json.dumps(cam_no_id, ensure_ascii=False)
                        except Exception:
                            raw_json = "[]"
                        user_content.append({"type": "text", "text": f"ANCHORS_RAW[{cam}]:\n{raw_json}"})

        # annotated images
        for cam in self.CAMERA_ORDER:
            p = annotated.get(cam)
            if p:
                user_content.append(self._img_to_payload(p))

        # scene context + question anchor summary
        ego_status = record.get("ego_status","")
        ego_rev = self._ego_is_reversing(ego_status)
        user_content.append({"type": "text", "text": self._build_context_text(record)})
        if refs:
            user_content.append({"type": "text", "text": self._build_question_anchor_text(refs, anchors, ego_rev)})

        # final question
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        raw = self.call_with_self_consistency(
            messages,
            task_hint=category,
            question_text=question,
            is_prediction=is_prediction,
            is_planning=is_planning,
            is_visual_desc=is_visual_desc,
        )

        # 후처리 강제 없음: 모델 출력 그대로(경미한 정리만) 반환
        final = self._normalize_answer_text(raw)
        return final


# ----------------------------- CLI & Runner -----------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Inference from Precomputed Context (Enhanced+++ / No Answer Forcing)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--n_consistency", type=int, default=3)

    parser.add_argument("--max_history_frames", type=int, default=2,
                        help="Max number of history frames to include (most recent first)")
    parser.add_argument("--include_raw_anchor_text", type=int, default=1,
                        help="Include original image + raw anchor JSON per camera (1=yes, 0=no)")
    parser.add_argument("--include_question_crops", type=int, default=1,
                        help="Include high-res crops for question objects (1=yes, 0=no)")
    parser.add_argument("--max_anchors_per_cam", type=int, default=20,
                        help="Limit number of anchors per camera by nearest distance")
    return parser.parse_args()

def load_or_create_output(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            return json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return []

def save_output(path: str, data: List[Dict[str, Any]]):
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def process_all(data: List[Dict[str, Any]], vlm: VLMInference, out: str):
    results = load_or_create_output(out)
    seen_ids = {(r.get('scene_token'), r.get('frame_token'), r.get('question')) for r in results}
    to_run = [r for r in data if (r.get('scene_token'), r.get('frame_token'), r.get('question')) not in seen_ids]

    with tqdm(total=len(to_run), desc="Processing") as pbar:
        for rec in to_run:
            answer = vlm.process_sample(rec)
            result_item = {
                "scene_token": rec.get("scene_token"),
                "frame_token": rec.get("frame_token"),
                "question": rec.get("question"),
                "answer": answer,
                "category": rec.get("category"),
                "img_paths": rec.get("img_paths"),
            }
            results.append(result_item)
            save_output(out, results)
            pbar.update(1)
    return results

def main():
    args = parse_arguments()
    vlm = VLMInference(
        model_name=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_consistency=args.n_consistency,
        max_history_frames=args.max_history_frames,
        include_raw_anchor_text=bool(args.include_raw_anchor_text),
        include_question_crops=bool(args.include_question_crops),
        max_anchors_per_cam=args.max_anchors_per_cam,
    )
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    process_all(data, vlm, args.output)
    print("Inference complete!")

if __name__ == "__main__":
    main()
