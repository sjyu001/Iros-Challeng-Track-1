MODEL_PATH="Qwen/Qwen2.5-VL-32B-Instruct"
INPUT_DATA="robosense_track1_realese_convert_temporal_10.json"
FEW_SHOT_DATA="test_qa.json"
OUTPUT_DIR="outputs"
K=5
TEMPERATURE=0.2
TOP_P=0.2
MAX_TOKENS=512
PORT=8000

mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results.json"

echo "Running inference..."
python inference_few_shot.py \
    --model "${MODEL_PATH}" \
    --data "${INPUT_DATA}" \
    --shots "${FEW_SHOT_DATA}" \
    --k "${K}" \
    --output "${OUTPUT_FILE}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --api_base "http://localhost:${PORT}/v1"
