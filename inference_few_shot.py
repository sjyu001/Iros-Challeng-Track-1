import argparse
import json
import os
import time
import re
import tempfile
import random
from typing import Any, Dict, List
import requests

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from openai import OpenAI

SYSTEM_PROMPT = """\
You are an autonomous driving assistant.  
Six camera views are provided in this order:  
[CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT].  
Coordinates are given as `<id, camera_view, x, y>` with image resolution 1600×900.

**Please think through your reasoning step by step** (chain of thought) before giving the final answer.  
At the end, prefix your answer with “Answer: ” and give the concise result.

**Whenever I reference `<cX,CAM_Y,x,y>` tokens in my question,  
you must focus only on the object(s) marked by those coordinates**  
(e.g. c2 in CAM_FRONT_RIGHT at pixel 57.5,605.8)  
and describe or reason about that object alone.  
Do not describe the rest of the scene unless it directly impacts your answer.

OBJECTIVE:
Focus only on the object marked by each <cID,CAM,x,y> token.

TEMPLATE:
1. Observations: List what you see at each <cID,CAM,x,y>.
2. Reasoning: Explain step-by-step how you reach your conclusion.
3. Answer: Provide the final concise answer.

After drafting, critique your own answer and output a revised “answer” if needed.
"""

class VLMInference:
    CAMERA_ORDER = [
        'CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
        'CAM_BACK','CAM_BACK_RIGHT','CAM_BACK_LEFT'
    ]

    def __init__(
        self,
        model_name: str,
        api_base: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        shots_path: str,
        k_shot: int = 5
    ):
        self.api_base = api_base.rstrip('/')
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.k_shot = k_shot

        # Load few-shot pool
        with open(shots_path, 'r', encoding='utf-8') as f:
            self.shots: List[Dict[str, Any]] = json.load(f)

    def _get_few_shot(self, question: str) -> List[Dict[str, Any]]:
        # Random sampling fallback
        return random.sample(self.shots, min(self.k_shot, len(self.shots)))

    def call_with_self_consistency(self, messages: List[Dict[str, Any]]) -> str:
        answers: List[str] = []
        for _ in range(5):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }
            resp = requests.post(f"{self.api_base}/chat/completions", json=payload)
            resp.raise_for_status()
            resp_json = resp.json()
            content = resp_json["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                ans = parsed.get("answer", content)
            except:
                ans = content
            answers.append(ans)
            time.sleep(0.2)  # 짧은 딜레이
        return max(set(answers), key=answers.count)

    def process_sample(self, record: Dict[str, Any]) -> str:
        question  = record["question"]
        img_paths = record["img_paths"]
        history   = record.get("history_frames", {})

        # 1) Few-shot examples (random)
        shots = self._get_few_shot(question)

        # 2) Parse <cX,CAM,x,y>
        coords = re.findall(r"<c(\d+),([^,>]+),([\d.]+),([\d.]+)>", question)
        refs = [
            {"id":int(cid), "cam":cam, "x":float(x), "y":float(y), "label":f"c{cid}"}
            for cid, cam, x, y in coords
        ]

        # 3) Annotate images
        annotated: Dict[str, str] = {}
        for cam in self.CAMERA_ORDER:
            raw = img_paths.get(cam)
            if not raw or not os.path.exists(raw):
                continue
            img = Image.open(raw).convert("RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            for idx, r in enumerate([r for r in refs if r['cam'] == cam]):
                x, y = r['x'], r['y']
                color = ["yellow","cyan","magenta","lime","orange"][idx % 5]
                L, hs = 60, 10
                draw.line([(x+L, y-L), (x, y)], fill=color, width=3)
                draw.line([(x, y), (x+hs, y-hs)], fill=color, width=3)
                draw.line([(x, y), (x-hs, y-hs)], fill=color, width=3)
                draw.text((x+L+5, y-L-15), r['label'], fill=color, font=font)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp.name, format="JPEG")
            annotated[cam] = tmp.name

        # 4) Build messages
        messages: List[Dict[str, Any]] = []
        messages.append({"role":"system",   "content":SYSTEM_PROMPT})
        messages.append({"role":"system",   "content":f"Few-Shot Examples ({len(shots)}):"})
        for shot in shots:
            messages.append({"role":"user",      "content":shot['question']})
            messages.append({"role":"assistant", "content":shot['answer']})

        user_content: List[Any] = []
        for _, paths in history.items():
            for cam in self.CAMERA_ORDER:
                p = paths.get(cam)
                if p and os.path.exists(p):
                    user_content.append({
                        "type":"image_url",
                        "image_url":{"url":f"file://{os.path.abspath(p)}"}
                    })
        for cam in self.CAMERA_ORDER:
            p = annotated.get(cam)
            if p:
                user_content.append({
                    "type":"image_url",
                    "image_url":{"url":f"file://{os.path.abspath(p)}"}
                })
        user_content.append({"type":"text","text":question})
        messages.append({"role":"user","content":user_content})

        # 5) Self-consistency call
        return self.call_with_self_consistency(messages)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM Inference (requests-only, random few-shot)"
    )
    parser.add_argument("--model",       type=str, required=True, help="Model name")
    parser.add_argument("--data",        type=str, required=True, help="Input JSON path")
    parser.add_argument("--output",      type=str, required=True, help="Output JSON path")
    parser.add_argument("--shots",       type=str, required=True, help="Few-shot pool JSON path")
    parser.add_argument("--k",           type=int, default=5,   help="Number of Few-Shot Examples")
    parser.add_argument("--api_base",    type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p",       type=float, default=0.2)
    parser.add_argument("--max_tokens",  type=int,   default=512)
    return parser.parse_args()

def load_or_create_output(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            return json.load(open(path, 'r', encoding='utf-8'))
        except:
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
    seen_ids = {r.get('id') for r in results}
    to_run = [r for r in data if r.get('id') not in seen_ids]
    with tqdm(total=len(to_run), desc="Processing") as pbar:
        for rec in to_run:
            rec['answer'] = vlm.process_sample(rec)
            results.append(rec)
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
        shots_path=args.shots,
        k_shot=args.k
    )
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    process_all(data, vlm, args.output)
    print("Inference complete!")

if __name__ == "__main__":
    main()