MODEL_PATH="Qwen/Qwen2.5-VL-32B-Instruct"
INPUT_DATA="other_perception_qa.json"
OUTPUT_DIR="outputs"
PORT=8000

mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/other_perception.json"

python other_perception.py \
  --model "${MODEL_PATH}" \
  --data "${INPUT_DATA}" \
  --output "${OUTPUT_FILE}" \
  --api_base "http://localhost:${PORT}/v1" \
  --temperature 0.2 \
  --top_p 0.2 \
  --max_tokens 1536 \
  --n_consistency 1\
  --max_history_frames 0

