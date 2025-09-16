MODEL_PATH="Qwen/Qwen2.5-VL-32B-Instruct"
INPUT_DATA="perception_mcq_qa.json"
FEW_SHOT_DATA="test_qa.json"
OUTPUT_DIR="outputs"
K=10
TEMPERATURE=0.2
TOP_P=0.2
MAX_TOKENS=512
PORT=8000

mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/perception_mcq.json"

echo "Running inference..."
python perception_mcq.py \
    --model "${MODEL_PATH}" \
    --data "${INPUT_DATA}" \
    --shots "${FEW_SHOT_DATA}" \
    --k "${K}" \
    --output "${OUTPUT_FILE}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --api_base "http://localhost:${PORT}/v1"
