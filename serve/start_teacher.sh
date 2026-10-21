#!/usr/bin/env bash
# Serve the TEACHER / JUDGE model (Qwen2.5-14B-Instruct) on port 8001 under the name "teacher".
# Used by distilabel (Part 5/6) and by the LLM judge. Stop it when you are done: docker rm -f medscriptai-teacher
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0-cu129}"
GPU_UTIL="${GPU_UTIL:-0.85}"                        # 0.85 alone on the GPU; 0.50 next to start_vllm.sh (0.35)

docker rm -f medscriptai-teacher 2>/dev/null || true
docker run -d --name medscriptai-teacher \
  --gpus all --ipc=host \
  -p 8001:8000 \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  "$VLLM_IMAGE" \
  --model Qwen/Qwen2.5-14B-Instruct \
  --served-model-name teacher \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_UTIL"

echo "Started. Follow the logs with:  docker logs -f medscriptai-teacher"
