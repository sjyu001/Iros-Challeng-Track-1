#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Inference with Precomputed Context (Enhanced++ Long-Form)

- 자연어 품질 강화:
  * 태그형 상태: pedestrian.moving → pedestrian is moving
  * 하이픈 상태: turning-left → turning left
  * 앵커 이름/상태 정규화(평가 어휘만 노출)

- 컨텍스트 강화:
  * left/center/right 자동 산출(x/1600)
  * OBJECT CANDIDATES 라인: "pedestrian (left, ~7 m, moving)" 포맷 통일
  * ANCHOR SUMMARY 추가(카운트/좌중우 분포/상위 위험요소)

- PROSE 모드 보강 + 길게 쓰기:
  * 단일 객체 설명형에서 자동 prose 모드(90~140단어 유도)
  * Observations(≥120 words), Reasoning(≥100 words) 유도

- Self-Consistency 집계 개선:
  * 거리/위치/상태/가시성 포함 여부 + 길이/문장수 가점

- MCQ 최종 형식 강제:
  * 항상 "3. Answer: C. <option>" 정규화
  * MCQ Elimination 보조 프롬프트 추가

- 디코딩 파라미터:
  * description/planning: temp=0.7, top_p=0.9, n_cons ≥ 5
  * 기본 max_tokens 1280, n_consistency 4
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


# ----------------------------- Domain Knowledge & Prompt -----------------------------
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
- Pedestrians: anticipate intent to cross when near crosswalks or curbs.
- Vehicles: detect lane-change or merging cues from wheel angle, lateral drift, or signal lights.
- Heavy vehicles (trucks, buses): longer stopping distance, wider turning radius, blind spots.
- Cyclists/motorcyclists: higher lateral movement uncertainty, frequent lane filtering.
- Always evaluate risk of collision in the next 3–5 seconds.

PLANNING:
- Maintain safe time headway (≥2s normal, ≥3s in poor weather or congestion).
- Prioritize vulnerable road users (pedestrians, cyclists).
- Obey traffic signals and right-of-way rules strictly.
- Avoid unnecessary lane changes; only change when safe and beneficial.
- Prefer smooth, low-jerk maneuvers.
- Keep clear of occluded zones (e.g., behind parked cars or large trucks).

ACTION RULES OF THUMB:
- If a lead vehicle brakes, increase following gap and prepare to stop smoothly.
- If a pedestrian/cyclist is approaching or waiting near a crosswalk, slow down and yield.
- At intersections: check all directions and avoid blocking.
- In poor visibility: reduce speed and increase following distance.
- Always prioritize safety and legality over efficiency.

SAFETY PRIORITIZATION:
1) Protect vulnerable road users.
2) Avoid collisions with vehicles.
3) Respect traffic controls and right-of-way.
4) Maintain passenger comfort.
5) Optimize efficiency only after safety and legality are ensured.
""".strip()


SYSTEM_PROMPT = f"""
You are an autonomous driving assistant.

TASK DESCRIPTION:
In this phase, participants must answer high-level driving questions on possibly corrupted images.
Rely strictly on provided anchors and scene context; do not hallucinate new objects.

Six camera views: [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT].
Coordinates: <id, camera_view, x, y> with resolution 1600×900.

ANCHORS:
- The provided ANCHORS are the only valid set of objects.
- Always ground observations and reasoning in anchor metadata (category, ~distance m, left/center/right, motion state, visibility/occlusion).
- When <cID,CAM,x,y> tokens appear, match to the overlapping/nearest anchor and reason with that anchor.

STRUCTURE (for every question):
1. Observations: 6–10 sentences (≥120 words). Use anchors to report category, ~distance (m), left/center/right, motion state, visibility/occlusion.
   - If ≥10 anchors exist, include an aggregated count by category (e.g., "3 pedestrians, 2 cars") and mention the nearest 3 potential hazards.
   - Prefer natural prose (no bullet lists).
2. Reasoning: 5–9 sentences (≥100 words). Explain the causal chain: how anchor states and positions constrain near-term motion and safety; refer to occlusions and right-of-way where relevant.
3. Answer: single sentence.
   - MCQ MUST be "3. Answer: C. Going ahead" (letter + option).
   - For descriptions: concise natural English starting with "Answer:".

STYLE:
- Use only this vocabulary for categories/states: adult, child, car, truck, bus, bicycle, motorcycle, cyclist, pedestrian, left, center, right, moving, stopped, turning left, turning right, merging, braking, occluded, clear.
- Distances as integers with "m" (e.g., ~12 m).
- Avoid hedging (no "maybe/likely/seems").
- Be fluent and descriptive but grounded in anchors. Prefer long, information-dense prose over terse notes.

RISK & PRIORITIZATION HEURISTICS (verbal, no numbers required):
- Potential hazard increases when: distance is small, motion is moving/merging/turning toward ego, path conflict exists, or visibility is occluded.
- Rank at least 3 hazards in Observations when possible and reflect that ranking in Reasoning.

RULE:
For every question (MCQ, perception, planning, prediction), always use exactly three sections:
1. Observations: factual description using anchors.
2. Reasoning: step-by-step logic with anchors.
3. Answer: final result.

- For MCQs: The Answer MUST repeat both the letter and the option text, e.g.:
  "3. Answer: C. Going ahead"
- For others: short factual sentence or action.
- Each Observations and Reasoning must have ≥2 sentences.
- Always weave anchor info (~distance m, category) into Observations/Reasoning.
- Never skip sections.
- Use only controlled vocabulary (adult, car, truck, stopped, moving, etc.).

Domain Knowledge:
{DRIVING_DOMAIN_KNOWLEDGE}

OUTPUT RULES (CRITICAL):
- Use this vocabulary only: adult, child, car, truck, bus, bicycle, motorcycle, cyclist, pedestrian, left, center, right, moving, stopped, turning-left, turning-right, merging, braking, occluded, clear.
- Distances: integer meters or “~<int> m”. Units: “m”.
- No hedging (avoid: maybe, likely, seems).
- Answers must be long, detailed, and descriptive, weaving anchor information into complete sentences.
- Both Step 1 and Step 2 should be multi-sentence outputs, not short phrases.
- Do not use quotes or markdown styling in the final answer.

"""

PROSE_MODE_SNIPPET = """
PROSE MODE (single-object visual description):
- Write 90–140 words of natural English prose.
- Start with “Answer:” and then a fluent description.
- Include: category, left/center/right, distance in meters (“~12 m”), motion state (moving/stopped/turning left/turning right/merging/braking), and visibility (clear/occluded).
- Use only the allowed vocabulary for categories/states.
- No bullet points. No quotes or markdown.
"""

PERCEPTION_POWER_UP = """
PERCEPTION POWER-UP:
- Enumerate the visible road users and controls from anchors without hallucination.
- For each of the nearest 3–5 hazards, say: category, left/center/right, ~distance (m), motion state, visibility.
- Mention occlusions explicitly and their impact on uncertainty (e.g., “occluded by bus, reduce confidence”).
- If crosswalks/cones/signals appear in anchors, relate them to nearby pedestrians/vehicles.
"""

PLANNING_POWER_UP = """
PLANNING POWER-UP:
- Start from constraints: vulnerable users first, active controls (signals, stop/yield), right-of-way, distance headway.
- Discuss at least two alternatives (e.g., keep lane vs. slow-and-yield; lane change vs. maintain gap) and why one is safer/less jerky.
- If objects are occluded near the path, recommend lower speed and increased headway.
- End with a single clear action consistent with rules: slow down, keep lane, prepare to stop, or yield.
"""

MCQ_ELIMINATION_POWER_UP = """
MCQ ELIMINATION:
- Compare each option against anchor-grounded facts. Eliminate options contradicted by motion or geometry.
- State the key anchor features that differentiate the correct option (e.g., moving vs. stopped; turning right vs. going ahead).
- Conclude with the exact required format: "3. Answer: <letter>. <option>".
"""


# ----------------------------- Utility mappers -----------------------------
ALLOWED_CATEGORIES = {"adult","child","car","truck","bus","bicycle","motorcycle","cyclist","pedestrian"}
ALLOWED_STATES = {"moving","stopped","braking","turning left","turning right","merging","occluded","clear"}

CANON_CLASS_MAP = {
    # vehicles
    "vehicle": "car",
    "vehicle.car": "car",
    "vehicle.van": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus": "bus",
    "vehicle.truck": "truck",
    "vehicle.trailer": "truck",
    "vehicle.construction": "truck",
    # humans
    "human.pedestrian": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.cyclist": "cyclist",
    # fallbacks
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "car": "car",
    "cyclist": "cyclist",
}

STATE_NORMALIZER = {
    "turning-left": "turning left",
    "turning-right": "turning right",
    "braking": "braking",
    "moving": "moving",
    "stopped": "stopped",
    "merging": "merging",
    "occluded": "occluded",
    "clear": "clear",
}

def canon_name(name: str) -> str:
    if not name: return "object"
    n = CANON_CLASS_MAP.get(name, None)
    if n: return n
    tail = name.split(".")[-1]
    return tail if tail in ALLOWED_CATEGORIES else (CANON_CLASS_MAP.get(tail, tail))

def canon_state(status: Optional[str]) -> Optional[str]:
    if not status: return None
    s = status.strip().lower()
    m = re.search(r"\b(moving|stopped|braking|turning-?left|turning-?right|merging|occluded|clear)\b", s)
    s = m.group(1) if m else s
    s = s.replace("turning-left","turning left").replace("turning-right","turning right")
    return s if s in ALLOWED_STATES else None

def pos_from_x(x: float, width: int = 1600) -> str:
    if x < width/3: return "left"
    if x > 2*width/3: return "right"
    return "center"

def humanize_dotted_phrases(text: str) -> str:
    # vehicle.car → car ; pedestrian.moving → pedestrian is moving
    def repl(m):
        obj = m.group(1).lower()
        st  = m.group(2).lower()
        obj = CANON_CLASS_MAP.get(obj, obj.split(".")[-1])
        st  = canon_state(st) or st
        st  = st.replace("-", " ")
        return f"{obj} is {st}"
    text = re.sub(r"\b([A-Za-z\.]+)\.(moving|stopped|braking|turning-?left|turning-?right|merging|occluded|clear)\b", repl, text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ----------------------------- Main Inference -----------------------------
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
        r"\bmaybe\b|\blikely\b|\bseems\b": ""
    }

    def __init__(self,
                 model_name: str,
                 api_base: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 n_consistency: int = 4,
                 max_history_frames: int = 5,
                 include_raw_anchor_text: bool = True,
                 include_question_crops: bool = True,
                 max_anchors_per_cam: int = 20,
                 richness_boost: bool = True):
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
        self.richness_boost = bool(richness_boost)

    # ---------- math / geometry utils ----------
    @staticmethod
    def _compute_3d_corners(center, size, quat):
        w, l, h = size
        x = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
        y = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
        z = np.array([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
        corners = np.vstack([x, y, z])
        q = Quaternion(quat)
        R = q.rotation_matrix
        return (R @ corners) + np.array(center, dtype=float).reshape(3,1)

    @staticmethod
    def _project_points(K: np.ndarray, P3: np.ndarray) -> np.ndarray:
        P = K @ P3
        z = P[2, :]
        z[z == 0] = 1e-6
        return P[:2, :] / z

    # ---------- helpers ----------
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

    # ---------- 3D → 2D bbox ----------
    def _ensure_bbox2d(self, anchors: List[Dict[str, Any]]) -> None:
        for a in anchors or []:
            b = a.get("bbox"); K = a.get("intrinsic")
            if not b or not K:
                a["bbox2d"] = None
                continue
            try:
                P3 = self._compute_3d_corners(np.array(b["center"], float),
                                              np.array(b["size"], float),
                                              np.array(b["orientation"], float))
                P2 = self._project_points(np.array(K, float), P3)
                a["bbox2d"] = [float(P2[0,:].min()), float(P2[1,:].min()),
                               float(P2[0,:].max()), float(P2[1,:].max())]
            except Exception:
                a["bbox2d"] = None

    # ---------- bbox-based matching ----------
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
        if best is not None and best_d <= (12.0**2):
            return best
        return None

    # ---------- image/crop ----------
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
    def _annotate_images(self,
                         img_paths: Dict[str, str],
                         refs: List[Dict[str, Any]],
                         anchors: List[Dict[str, Any]]) -> Dict[str, str]:
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

            # 3D bbox wireframe
            for idx, a in enumerate(cam_anchors):
                b = a.get("bbox"); K = a.get("intrinsic")
                if not b or not K:
                    continue
                try:
                    P3 = self._compute_3d_corners(np.array(b["center"], float),
                                                  np.array(b["size"], float),
                                                  np.array(b["orientation"], float))
                    P2 = self._project_points(np.array(K, float), P3)
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

            # anchor points + short labels
            for idx, a in enumerate(cam_anchors):
                x, y = float(a.get("x", 0)), float(a.get("y", 0))
                xi, yi = int(x), int(y)
                rgba = cmap(idx % cmap.N)
                color = tuple(int(c * 255) for c in rgba[:3])

                R = 6
                draw.ellipse([(xi - R, yi - R), (xi + R, yi + R)],
                             outline=color, width=2, fill=None)
                existing_bboxes.append((xi - R - 2, yi - R - 2, xi + R + 2, yi + R + 2))

                obj_name = canon_name(str(a.get("name","object")))
                d = a.get("distance")
                label = obj_name + (f" ({float(d):.1f}m)" if d is not None else "")
                tx, ty = xi + 5, yi - 10
                draw.text((tx, ty), label, fill=color, font=font_small)
                try:
                    tb = draw.textbbox((tx, ty), label, font=font_small)
                except Exception:
                    w, h = draw.textsize(label, font=font_small)
                    tb = (tx, ty, tx + w, ty + h)
                existing_bboxes.append((tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2))

            # question points + labels
            ref_cam = [r for r in refs if r["cam"] == cam]
            for r in ref_cam:
                xi, yi = int(r["x"]), int(r["y"])
                matched = self._match_anchor_by_bbox(cam, r["x"], r["y"], cam_anchors)
                if matched:
                    obj_name = canon_name(str(matched.get("name","object")))
                    dist = matched.get("distance", None)
                    q_label = f"Question Object: {r['label']} ({obj_name}{', ' + f'{float(dist):.1f}m' if dist is not None else ''})"
                else:
                    q_label = f"Question Object: {r['label']}"

                RQ = 12
                draw.ellipse([(xi - RQ, yi - RQ), (xi + RQ, yi + RQ)],
                             outline="black", width=3, fill="yellow")
                existing_bboxes.append((xi - RQ - 3, yi - RQ - 3, xi + RQ + 3, yi + RQ + 3))

                tx, ty = self._place_text_nonoverlap(draw, q_label, xi, yi, font_big, existing_bboxes, W, H,
                                                     stroke_width=2, pad=4)
                draw.text((tx, ty), q_label, fill="yellow", font=font_big,
                          stroke_width=2, stroke_fill="black")
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

    # ---------- Human-readable SCENE CONTEXT ----------
    def _build_context_text(self, record: Dict[str, Any]) -> str:
        parts = ["SCENE CONTEXT:"]
        if record.get('ego_status'):
            parts.append(f"EGO STATUS: {record['ego_status']}")
        if record.get('motion_cues'):
            parts.append(f"MOTION CUES: {record['motion_cues']}")

        if record.get('anchors'):
            lines = []
            for a in record['anchors']:
                cam = a.get('cam','?')
                x = float(a.get('x',0)); y = float(a.get('y',0))
                name_raw = a.get('name','object')
                name = canon_name(name_raw)
                status = canon_state(a.get('status',''))  # normalized
                dist = a.get('distance')
                pos = pos_from_x(x, 1600)
                item = f"<{cam},{x:.1f},{y:.1f}> {name} ({pos}"
                if dist is not None:
                    item += f", ~{round(float(dist))} m"
                if status:
                    item += f", {status}"
                item += ")"
                lines.append(item)
            if lines:
                parts.append("OBJECT CANDIDATES:\n" + "\n".join(lines))
        return "\n".join(parts)

    # ---------- Question-object anchor summary ----------
    def _build_question_anchor_text(self, refs: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> str:
        lines = []
        for r in refs:
            cam, x, y = r["cam"], r["x"], r["y"]
            matched = self._match_anchor_by_bbox(cam, x, y, anchors)
            if matched:
                name = canon_name(matched.get('name','object'))
                status = canon_state(matched.get('status',''))
                dist = matched.get('distance', None)
                pos = pos_from_x(float(matched.get('x',x)), 1600)
                parts = [name, pos]
                if dist is not None: parts.append(f"~{round(float(dist))} m")
                if status: parts.append(status)
                lines.append(f"{cam} ({x:.1f},{y:.1f}) → {', '.join(parts)}")
            else:
                lines.append(f"{cam} ({x:.1f},{y:.1f}) → (no anchor matched)")
        return "ANCHOR_INFO[Question Objects]:\n" + "\n".join(lines) if lines else ""

    # ---------- Anchor quick summary for perception/planning ----------
    def _anchor_quick_summary(self, anchors: List[Dict[str, Any]]) -> str:
        if not anchors:
            return "ANCHOR SUMMARY: none."
        from collections import Counter
        cats = [canon_name(a.get("name","")) for a in anchors]
        cnt = Counter(cats)
        parts = []
        parts.append("ANCHOR SUMMARY:")
        parts.append("Counts: " + ", ".join(f"{k}={v}" for k,v in cnt.most_common()))
        pos_cnt = Counter(pos_from_x(float(a.get("x",800)), 1600) for a in anchors)
        parts.append("L/C/R: " + ", ".join(f"{k}={pos_cnt.get(k,0)}" for k in ["left","center","right"]))
        # top hazards by (distance + state weight)
        def risk_key(a):
            d = float(a.get("distance", 1e9) or 1e9)
            state = str(canon_state(a.get("status","")) or "")
            state_w = 0
            if state in ("merging","turning left","turning right"):
                state_w = -5
            elif state in ("moving",):
                state_w = -3
            elif state in ("braking",):
                state_w = -2
            return (d + state_w)
        hazards = sorted([a for a in anchors if a.get("distance") is not None], key=risk_key)[:3]
        if hazards:
            hs = []
            for h in hazards:
                hs.append(f"{canon_name(h.get('name','object'))} ({pos_from_x(float(h.get('x',800)))}, ~{round(float(h.get('distance')))} m, {canon_state(h.get('status','')) or 'clear'})")
            parts.append("Top hazards: " + "; ".join(hs))
        return "\n".join(parts)

    # ---------- MCQ helpers ----------
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

    def _force_mcq_letter(self, llm_answer_text: str, question_text: str) -> str:
        opts = self._parse_mcq_options(question_text)
        m = re.search(r'\b([ABCD])\b', llm_answer_text.strip())
        letter = m.group(1) if m else "A"
        if opts and letter in opts:
            return f"{letter}. {opts[letter]}"
        return f"{letter}."

    def _normalize_answer(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r'^[`"*]+|[`"*]+$', '', s)
        s = re.sub(r"\b~?(\d+\.\d+)\s*m\b",
                   lambda m: f"{round(float(m.group(1))):d}m", s)
        for pat, rep in self._LEXICON.items():
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        s = humanize_dotted_phrases(s)
        s = re.sub(r"\s{2,}", " ", s).strip(",.; ").strip()
        return s

    def _post_format_description(self, s: str) -> str:
        s = self._normalize_answer(s)
        s = re.sub(r",\s*", ", ", s)
        s = re.sub(r",\s*,", ", ", s)
        return s

    # ---------- scoring & enforcement ----------
    @staticmethod
    def _contains_any(s: str, patterns: List[str]) -> bool:
        return any(re.search(p, s, re.I) for p in patterns)

    def _score_answer_quality(self, text: str) -> int:
        # 핵심 정보 가점
        score = 0
        if re.search(r"\b\d+\s*m\b", text): score += 2
        if self._contains_any(text, [r"\bleft\b", r"\bcenter\b", r"\bright\b"]): score += 2
        if self._contains_any(text, [r"\bmoving\b", r"\bstopped\b", r"\bturning left\b", r"\bturning right\b", r"\bmerging\b", r"\bbraking\b"]): score += 2
        if self._contains_any(text, [r"\boccluded\b", r"\bclear\b"]): score += 1
        if self._contains_any(text, [r"1\. Observations", r"2\. Reasoning", r"3\. Answer"]): score += 2
        # 길이/문장수 가점 (긴 답 선호)
        words = len(re.findall(r"\w+", text))
        sents = max(1, len(re.findall(r"[.!?](?:\s|$)", text)))
        if words >= 120: score += 3
        if words >= 180: score += 2
        if words >= 240: score += 1
        if sents >= 8: score += 2
        if sents >= 12: score += 1
        return score

    def _ensure_structured_output(self,
                                  base_answer: str,
                                  is_mcq: bool,
                                  question_text: str,
                                  refs: List[Dict[str, Any]],
                                  anchors: List[Dict[str, Any]]) -> str:
        s = self._normalize_answer(base_answer)

        # MCQ: force final line
        if is_mcq:
            normalized = self._force_mcq_letter(s, question_text)
            if not re.search(r"^3\.\s*Answer:", s):
                s = (
                    "1. Observations: The decision is derived from the anchor states visible in the images. "
                    "We consider positions, motion cues, and visibility of the referenced objects.\n"
                    "2. Reasoning: We compare options against the matched anchors and their motion/heading, "
                    "eliminating those that contradict observed states or geometry.\n"
                    f"3. Answer: {normalized}"
                )
            else:
                s = re.sub(r"(3\.\s*Answer:\s*)(.+)$", lambda m: f"{m.group(1)}{normalized}", s, flags=re.S)
            return s

        # Non-MCQ: ensure "Answer:" line present
        if "answer:" not in s.lower():
            obs, rea, ans = self._synthesize_from_anchor(refs, anchors)
            return f"1. Observations: {obs}\n2. Reasoning: {rea}\n3. Answer: {ans}"
        return s

    def _synthesize_from_anchor(self, refs: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> Tuple[str,str,str]:
        if refs:
            r = refs[0]
            matched = self._match_anchor_by_bbox(r["cam"], r["x"], r["y"], anchors)
        else:
            matched = None

        if matched:
            name = canon_name(matched.get("name","object"))
            dist = matched.get("distance", None)
            status = canon_state(matched.get("status",""))
            pos = pos_from_x(float(matched.get("x", r["x"] if refs else 800)), 1600)
            dist_txt = f"~{round(float(dist))} m" if dist is not None else "unknown distance"
            status_txt = status if status else "clear"
            obs = (
                f"A {name} is on the {pos} at {dist_txt}. The object is {status_txt} and clearly visible in the anchor list. "
                f"Nearby anchors reveal additional context for potential interactions and occlusions."
            )
            rea = (
                f"Given the {name}'s location ({pos}) and distance ({dist_txt}), its state ({status_txt}) influences "
                f"near-term safety margins and desired headway. We rely on anchors only, avoiding hallucination."
            )
            ans = f"Answer: The {name} is {status_txt} on the {pos} at {dist_txt}."
        else:
            obs = "Anchors indicate the relevant object near the referenced pixel; category and distance are provided in the candidates list."
            rea = "We rely on the matched anchor metadata for category, approximate distance, position, and motion state to form the description."
            ans = "Answer: The object is visible with its category, position, and distance grounded in the anchor metadata."
        return (obs, rea, ans)

    # ---------- LLM call with self-consistency ----------
    def call_with_self_consistency(
            self,
            messages: List[Dict[str, Any]],
            task_hint: str = "",
            question_text: str = "",
            prose_mode: bool = False,
            is_mcq: bool = False,
            refs: Optional[List[Dict[str, Any]]] = None,
            anchors: Optional[List[Dict[str, Any]]] = None,
        ) -> str:
        answers: List[str] = []

        temp = self.temperature
        top_p = self.top_p
        n_cons = self.n_consistency
        hint = (task_hint or "").lower()

        # category-based decoding
        if is_mcq or ("mcq" in hint) or ("prediction" in hint):
            temp, top_p, n_cons = 0.2, 0.2, max(1, min(3, n_cons))
        elif ("vqa" in hint) or ("description" in hint) or ("planning" in hint) or prose_mode:
            temp, top_p, n_cons = 0.7, 0.9, max(2, n_cons)

        if len(question_text.split()) > 40:
            temp, top_p, n_cons = max(temp,0.7), max(top_p,0.9), max(3, n_cons)

        # multiple samples
        for _ in range(n_cons):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": float(temp),
                "top_p": float(top_p),
                "max_tokens": int(self.max_tokens),
            }
            try:
                resp = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=1200
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                answers.append(content)
            except Exception as e:
                answers.append(f"Answer: (error) {e}")
            time.sleep(0.05)

        # pick best by quality score + majority + length
        scored = []
        norm_counts: Dict[str,int] = {}
        for a in answers:
            n = self._normalize_answer(a)
            norm_counts[n] = norm_counts.get(n,0)+1
        for a in answers:
            n = self._normalize_answer(a)
            score = self._score_answer_quality(n) + norm_counts[n]
            scored.append((score, len(n), n))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best = scored[0][2] if scored else (answers[0] if answers else "Answer: (no output)")

        # enforce structure/MCQ/Answer-line; enrich if too minimal
        final = self._ensure_structured_output(best, is_mcq, question_text, refs or [], anchors or [])

        # If prose mode and still short, softly enrich
        if (not is_mcq) and self.richness_boost and prose_mode:
            word_cnt = len(re.findall(r"\w+", re.sub(r"\s+"," ", final)))
            if word_cnt < 120:
                obs, rea, ans = self._synthesize_from_anchor(refs or [], anchors or [])
                final = f"1. Observations: {obs}\n2. Reasoning: {rea}\n3. Answer: {humanize_dotted_phrases(ans)}"

        return humanize_dotted_phrases(final)

    # ---------- Main per-sample ----------
    def process_sample(self, record: Dict[str, Any]) -> str:
        question = record.get("question", "")
        img_paths = record.get("img_paths", {}) or {}
        history   = record.get("history_frames", {}) or {}
        anchors   = record.get("anchors", []) or []
        category  = (record.get("category") or "").lower()

        # normalize bbox2d
        self._ensure_bbox2d(anchors)

        # parse refs
        coords = re.findall(r"<c(\d+),([^,>]+),([\d.]+),([\d.]+)>", question)
        refs = [{"id": int(cid), "cam": cam, "x": float(x), "y": float(y), "label": f"c{cid}"}
                for cid, cam, x, y in coords]

        recent_hist = self._take_recent_history(history, self.max_history_frames)

        # annotate & group
        annotated = self._annotate_images(img_paths, refs, anchors)
        anchors_by_cam = self._group_anchors_by_cam(anchors)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # broaden prose trigger
        is_visual_desc = bool(re.search(r"(visual\s+description|describe)\b", question, re.I))
        # single-object with no MCQ → prose mode
        if not is_visual_desc and len(refs) == 1 and not self._is_mcq_question(question):
            is_visual_desc = True

        if is_visual_desc:
            messages.append({"role": "system", "content": PROSE_MODE_SNIPPET})

        # category power-ups
        if category == "perception":
            messages.append({"role":"system","content":PERCEPTION_POWER_UP})
        elif category == "planning":
            messages.append({"role":"system","content":PLANNING_POWER_UP})

        is_mcq = self._is_mcq_question(question)
        if is_mcq:
            messages.append({"role":"system","content":MCQ_ELIMINATION_POWER_UP})

        user_content: List[Any] = []

        # (H) history images
        for _, paths in recent_hist:
            for cam in self.CAMERA_ORDER:
                p = paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))

        # (Q) question crops
        if self.include_question_crops:
            for r in refs:
                cam = r["cam"]
                p = img_paths.get(cam)
                if p and os.path.exists(p):
                    matched = self._match_anchor_by_bbox(cam, r["x"], r["y"], anchors)
                    bbox2d = matched.get("bbox2d") if matched else None
                    crop_path = self._crop_patch_with_optional_bbox(p, r["x"], r["y"], maybe_bbox2d=bbox2d, box=384)
                    if crop_path:
                        user_content.append(self._img_to_payload(crop_path))

        # (A) originals + raw anchors (normalized to evaluation vocab)
        if self.include_raw_anchor_text:
            for cam in self.CAMERA_ORDER:
                p = img_paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))
                    cam_raw = anchors_by_cam.get(cam) or []
                    if cam_raw:
                        try:
                            cam_no_id = []
                            for a in cam_raw:
                                x = float(a.get("x",0.0))
                                cam_no_id.append({
                                    "name": canon_name(a.get("name")),
                                    "cam": a.get("cam"),
                                    "x": x,
                                    "y": a.get("y"),
                                    "pos": pos_from_x(x, 1600),
                                    "distance": (round(float(a["distance"])) if a.get("distance") is not None else None),
                                    "state": canon_state(a.get("status",""))
                                })
                            raw_json = json.dumps(cam_no_id, ensure_ascii=False)
                        except Exception:
                            raw_json = "[]"
                        user_content.append({"type": "text", "text": f"ANCHORS_RAW[{cam}]:\n{raw_json}"})

        # (B) annotated images
        for cam in self.CAMERA_ORDER:
            p = annotated.get(cam)
            if p:
                user_content.append(self._img_to_payload(p))

        # (C) anchor quick summary + scene context + question anchor summary
        user_content.append({"type": "text", "text": self._anchor_quick_summary(anchors)})
        user_content.append({"type": "text", "text": self._build_context_text(record)})
        if refs:
            user_content.append({"type": "text", "text": self._build_question_anchor_text(refs, anchors)})

        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        return self.call_with_self_consistency(
            messages,
            task_hint=category,
            question_text=question,
            prose_mode=is_visual_desc,
            is_mcq=is_mcq,
            refs=refs,
            anchors=anchors,
        )


# ----------------------------- CLI & Runner -----------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Inference from Precomputed Context (Enhanced++ Long-Form)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1280)  # 768 → 1280
    parser.add_argument("--n_consistency", type=int, default=4)  # 3 → 4

    parser.add_argument("--max_history_frames", type=int, default=5,
                        help="Max number of history frames to include (most recent first)")

    parser.add_argument("--include_raw_anchor_text", type=int, default=1,
                        help="Include original image + raw anchor JSON per camera (1=yes, 0=no)")
    parser.add_argument("--include_question_crops", type=int, default=1,
                        help="Include high-res crops for question objects (1=yes, 0=no)")
    parser.add_argument("--max_anchors_per_cam", type=int, default=20,
                        help="Limit number of anchors per camera by nearest distance")

    parser.add_argument("--richness_boost", type=int, default=1,
                        help="If 1, softly enrich too-short PROSE answers using anchor facts")
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
        richness_boost=bool(args.richness_boost),
    )
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    process_all(data, vlm, args.output)
    print("Inference complete!")

if __name__ == "__main__":
    main()
