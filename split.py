import json
import os
import re


# ===== input/output root =====
input_path = "final_nuscene_3_temporal.json"
output_dir = "track1/split_results_original_qa"
os.makedirs(output_dir, exist_ok=True)

# ===== JSON load =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== category split =====
perception_mcq = []
other_perception = []
prediction = []
planning_corruption = []

def is_mcq(question: str) -> bool:
    q = question.lower()
    # 1) select answer
    if "please select the correct answer" in q:
        return True
    # 2) A./B./C./D. 
    if re.search(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", question):
        return True
    return False

for item in data:
    cat = item.get("category", "").lower()
    q = item.get("question", "")

    if cat == "perception":
        if is_mcq(q):
            perception_mcq.append(item)
        else:
            other_perception.append(item)
    elif cat == "prediction":
        prediction.append(item)
    elif cat in ["planning", "corruption"]:
        planning_corruption.append(item)

# ===== save =====
outputs = {
    "perception_mcq_qa.json": perception_mcq,
    "other_perception_qa.json": other_perception,
    "prediction_qa.json": prediction,
    "planning_corruption_qa.json": planning_corruption,
}

for fname, content in outputs.items():
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(content, f, indent=2)
