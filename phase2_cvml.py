#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Inference with Precomputed Context (Enhanced)

- 입력 동시 제공:
  (A) 원본 이미지 + 카메라별 RAW anchor JSON 텍스트  ← 항상 포함(옵션 on)
  (B) 주석(annotated) 이미지  ← 3D bbox wireframe + anchor 점/라벨
  (C) Question object crop(고해상도 패치)  ← bbox2d를 crop 위에 오버레이
  (D) SCENE CONTEXT 텍스트  ← anchor들을 사람이 읽기 쉬운 텍스트로
- Question object만 c~ 번호, 나머지(anchor)는 번호 없음(이름+거리만)
- Question 라벨 자동 비충돌 배치
- bbox 기반 앵커 매칭: (x,y)가 anchor의 2D bbox 내부면 매칭
- MCQ(객관식) 강제 포맷: 항상 "Answer: A." 형태로 반환
- 과제별 decoding 파라미터(Prediction/MCQ vs VQA/Planning)
- Self-Consistency: Answer 라인만 견고 추출·정규화·다수결
- 저장 dedup 키: (scene_token, frame_token, question)
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


SYSTEM_PROMPT = f"""
You are an autonomous driving assistant.

TASK DESCRIPTION:
In this phase, participants must answer high-level driving questions on possibly corrupted images.
You must rely on anchors and provided scene context.

Six camera views: [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT].
Coordinates: <id, camera_view, x, y> with resolution 1600×900.

ANCHORS:
- The provided ANCHORS are the *only valid set of objects* in the current scene.
- No additional objects exist beyond this anchor list; do not hallucinate.
- Always ground your reasoning and final answers explicitly in the anchor information
  (e.g., category, distance, position, status, occlusion).
- When <cID,CAM,x,y> tokens are referenced in the question, you must match them
  with the closest or overlapping anchor and describe reasoning based on that anchor’s metadata.

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
PROSE MODE (for single-object visual description)
- Write the answer as natural, fluent English prose (30–60 words).
- Always prefix with “Answer:”, no quotes or markdown.
- Use the anchor information (category, distance, position, status, occlusion) as the factual basis of your description.
- Explicitly mention distance (integer meters with optional ~), position (left, center, right), motion state, and visibility.
- Expand into a full sentence instead of a short phrase. You may add simple visible attributes such as color or lane relation.
- Do not use lists, bullet points, or telegraphic style. Avoid hedging words like “maybe” or “likely”.
- The description should sound like a human observer describing the scene in detail, grounded in the anchor data.
"""

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
                 n_consistency: int = 3,
                 max_history_frames: int = 5,
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

    # ---------- math / geometry utils ----------
    @staticmethod
    def _compute_3d_corners(center, size, quat):
        """
        center: [x,y,z], size: [w,l,h], quat: [qw,qx,qy,qz]
        Returns corners in shape (3,8) in world(cam) coords.
        """
        w, l, h = size
        # local corners
        x = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
        y = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
        z = np.array([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
        corners = np.vstack([x, y, z])  # (3,8)

        q = Quaternion(quat)  # [qw,qx,qy,qz]
        R = q.rotation_matrix  # (3,3)
        corners = R @ corners
        corners += np.array(center, dtype=float).reshape(3, 1)
        return corners

    @staticmethod
    def _project_points(K: np.ndarray, P3: np.ndarray) -> np.ndarray:
        """K (3,3), P3 (3,N) -> P2 (2,N)"""
        P = K @ P3
        z = P[2, :]
        z[z == 0] = 1e-6
        return P[:2, :] / z

    # ---------- small helpers ----------
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

    # ---------- NEW: 3D → 2D bbox 투영 사전 계산 ----------
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
                P3 = self._compute_3d_corners(center, size, quat)     # (3,8)
                P2 = self._project_points(Kmat, P3)                   # (2,8)
                x0, y0 = float(P2[0,:].min()), float(P2[1,:].min())
                x1, y1 = float(P2[0,:].max()), float(P2[1,:].max())
                a["bbox2d"] = [x0, y0, x1, y1]
            except Exception:
                a["bbox2d"] = None

    # ---------- NEW: bbox 기반 question-object 매칭 ----------
    @staticmethod
    def _is_point_in_bbox(px: float, py: float, bbox2d: Optional[List[float]]) -> bool:
        if not bbox2d:
            return False
        x0, y0, x1, y1 = bbox2d
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _match_anchor_by_bbox(self, cam: str, x: float, y: float, anchors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [a for a in anchors if a.get("cam")==cam]
        # 1) bbox 포함 검사 우선
        for a in candidates:
            if self._is_point_in_bbox(x, y, a.get("bbox2d")):
                return a
        # 2) 포인트-포인트 근접 (<=12px)
        best, best_d = None, 1e9
        for a in candidates:
            ax, ay = float(a.get("x", 1e9)), float(a.get("y", 1e9))
            d = (ax - x)**2 + (ay - y)**2
            if d < best_d:
                best, best_d = a, d
        if best is not None and best_d <= (12.0**2):
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
        """
        (x,y) 주변을 box×box로 crop하고, bbox2d가 있으면 crop 좌표계로 변환해 사각형을 오버레이.
        최종 이미지를 temp 파일로 저장 후 경로 반환.
        """
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

    # ---------- annotated images (3D bbox wireframe + anchor label) ----------
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
                    center = np.array(b["center"], dtype=float)
                    size   = np.array(b["size"], dtype=float)
                    quat   = np.array(b["orientation"], dtype=float)
                    Kmat   = np.array(K, dtype=float)

                    P3 = self._compute_3d_corners(center, size, quat)  # (3,8)
                    P2 = self._project_points(Kmat, P3)                # (2,8)
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

            # anchor 포인트 + 짧은 라벨
            for idx, a in enumerate(cam_anchors):
                x, y = float(a.get("x", 0)), float(a.get("y", 0))
                xi, yi = int(x), int(y)
                rgba = cmap(idx % cmap.N)
                color = tuple(int(c * 255) for c in rgba[:3])

                R = 6
                draw.ellipse([(xi - R, yi - R), (xi + R, yi + R)],
                             outline=color, width=2, fill=None)
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

            # 질문 포인트 + 자동 비충돌 라벨
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

    # ---------- TEXT: 사람 읽기 쉬운 SCENE CONTEXT ----------
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
                name = a.get('name','object')
                status = a.get('status','?')
                dist = a.get('distance')
                if dist is not None:
                    lines.append(f"<{cam},{x:.1f},{y:.1f}> {name} [{status}] (~{float(dist):.1f} m)")
                else:
                    lines.append(f"<{cam},{x:.1f},{y:.1f}> {name} [{status}]")
            if lines:
                parts.append("OBJECT CANDIDATES:\n" + "\n".join(lines))
        return "\n".join(parts)

    # ---------- TEXT: 질문 객체용 앵커 요약 ----------
    def _build_question_anchor_text(self, refs: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> str:
        lines = []
        for r in refs:
            cam, x, y = r["cam"], r["x"], r["y"]
            matched = self._match_anchor_by_bbox(cam, x, y, anchors)
            if matched:
                name = matched.get('name','object')
                status = matched.get('status','?')
                dist = matched.get('distance', None)
                tail = f"{name} [{status}] ~{float(dist):.1f}m" if dist is not None else f"{name} [{status}]"
                lines.append(f"{cam} ({x:.1f},{y:.1f}) → {tail}")
            else:
                lines.append(f"{cam} ({x:.1f},{y:.1f}) → (no anchor matched)")
        return "ANCHOR_INFO[Question Objects]:\n" + "\n".join(lines) if lines else ""

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
        s = re.sub(r"\s{2,}", " ", s).strip(",.; ").strip()
        return s

    def _post_format_description(self, s: str) -> str:
        s = self._normalize_answer(s)
        s = re.sub(r",\s*", ", ", s)
        s = re.sub(r",\s*,", ", ", s)
        return s

    # ---------- LLM call with self-consistency ----------
    def call_with_self_consistency(
            self,
            messages: List[Dict[str, Any]],
            task_hint: str = "",
            question_text: str = "",
            prose_mode: bool = False,
            is_mcq: bool = False,
        ) -> str:
        answers: List[str] = []

        temp = self.temperature
        top_p = self.top_p
        n_cons = self.n_consistency
        hint = (task_hint or "").lower()

        # 카테고리에 따라 샘플 수 및 탐색 파라미터 조정
        if is_mcq or ("mcq" in hint) or ("prediction" in hint):
            temp, top_p, n_cons = 0.2, 0.2, max(1, min(3, n_cons))
        elif ("vqa" in hint) or ("description" in hint) or ("planning" in hint):
            temp, top_p, n_cons = 0.5, 0.5, max(3, n_cons)

        # 긴 질문일 때는 temperature 살짝 올리기
        if len(question_text.split()) > 40:
            temp, top_p, n_cons = 0.6, 0.6, max(3, n_cons)

        # 여러 번 호출 (self-consistency)
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

        # 다수결로 best answer 선택
        counts: Dict[str, int] = {}
        best, best_cnt = None, -1
        for a in answers:
            norm = self._normalize_answer(a)
            counts[norm] = counts.get(norm, 0) + 1
            if counts[norm] > best_cnt:
                best_cnt, best = counts[norm], norm

        if not best:
            best = max(answers, key=len)

        # === 최종 포맷 제한하지 않고 모델 출력 그대로 반환 ===
        return best

    # ---------- Main per-sample ----------
    def process_sample(self, record: Dict[str, Any]) -> str:
        question = record.get("question", "")
        img_paths = record.get("img_paths", {}) or {}
        history   = record.get("history_frames", {}) or {}
        anchors   = record.get("anchors", []) or []
        category  = (record.get("category") or "").lower()

        # 0) bbox2d 사전계산 (3D bbox + intrinsic이 있을 때)
        self._ensure_bbox2d(anchors)


        # 질문 좌표 파싱
        coords = re.findall(r"<c(\d+),([^,>]+),([\d.]+),([\d.]+)>", question)
        refs = [{"id": int(cid), "cam": cam, "x": float(x), "y": float(y), "label": f"c{cid}"}
                for cid, cam, x, y in coords]

        recent_hist = self._take_recent_history(history, self.max_history_frames)

        # 주석 이미지(3D bbox + 점/라벨 + 질문)
        annotated = self._annotate_images(img_paths, refs, anchors)
        anchors_by_cam = self._group_anchors_by_cam(anchors)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        is_visual_desc = bool(re.search(r"(visual\s+description|describe)\b", question, re.I))
        if is_visual_desc:
            messages.append({"role": "system", "content": PROSE_MODE_SNIPPET})


        user_content: List[Any] = []

        # (H) history images
        for _, paths in recent_hist:
            for cam in self.CAMERA_ORDER:
                p = paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))

        # (Q) Question object crop images (bbox 오버레이 포함)
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

        # (A) 현재 원본 + 카메라별 RAW anchor JSON
        if self.include_raw_anchor_text:
            for cam in self.CAMERA_ORDER:
                p = img_paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append(self._img_to_payload(p))
                    cam_raw = anchors_by_cam.get(cam) or []
                    if cam_raw:
                        try:
                            # id 키 제거 → c1, c2 같은 번호는 제외
                            cam_no_id = [
                                {k: a.get(k) for k in ("name", "cam", "x", "y", "distance")}
                                for a in cam_raw
                            ]
                            raw_json = json.dumps(cam_no_id, ensure_ascii=False)
                        except Exception:
                            raw_json = "[]"
                        user_content.append({
                            "type": "text",
                            "text": f"ANCHORS_RAW[{cam}]:\n{raw_json}"
                        })

        # (B) annotated images
        for cam in self.CAMERA_ORDER:
            p = annotated.get(cam)
            if p:
                user_content.append(self._img_to_payload(p))

        # (C) 사람이 읽는 SCENE CONTEXT 텍스트 + (D) 질문 객체 앵커 요약
        user_content.append({"type": "text", "text": self._build_context_text(record)})
        if refs:
            user_content.append({"type": "text", "text": self._build_question_anchor_text(refs, anchors)})

        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        is_mcq = self._is_mcq_question(question)

        return self.call_with_self_consistency(
            messages,
            task_hint=category,
            question_text=question,
            prose_mode=is_visual_desc,
            is_mcq=is_mcq,
        )

# ----------------------------- CLI & Runner -----------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Inference from Precomputed Context (Enhanced)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--n_consistency", type=int, default=3)

    parser.add_argument("--max_history_frames", type=int, default=5,
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
