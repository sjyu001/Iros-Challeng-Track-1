import json
import os

# ===== input root =====
input_dir = "your_results_file_per_category"
output_path = "result.json"

# ===== file list =====
files = [
    "perception_mcq.json",
    "other_perception.json",
    "prediction.json",
    "planning_corruption.json",
]

merged = []

for fname in files:
    fpath = os.path.join(input_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            merged.extend(json.load(f))

# ===== 병합 결과 저장 =====
with open(output_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Merged Complete! Total {len(merged)} QA saved")
print(f"→ {output_path}")
