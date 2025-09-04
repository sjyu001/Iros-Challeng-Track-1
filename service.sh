# tensor parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_NUM=4


vllm serve "Qwen/Qwen2.5-VL-32B-Instruct" \
   --allowed-local-media-path / \
   --tensor-parallel-size ${GPU_NUM} \
   --trust-remote-code \
   --gpu-memory-utilization 0.83 \
