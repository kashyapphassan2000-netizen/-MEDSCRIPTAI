#!/usr/bin/env bash
# Serve Llama-3.1-8B-Instruct + all MedScriptAI LoRA adapters from ONE GPU process (multi-LoRA).
# Usage:  bash serve/start_vllm.sh            (from the repo folder)
# Model names clients can use: base, qa, notes, summary, qa-dpo, notes-dpo
set -euo pipefail                                   # stop on the first error

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0-cu129}"   # cu129 build + compat for R570 drivers
GPU_UTIL="${GPU_UTIL:-0.85}"                        # share of GPU memory vLLM may use (0.35 when the teacher also runs)
ADAPTERS="$(pwd)/outputs/adapters"                  # trained adapters on the VM

# Only register adapters that exist, so the script works after Part 4 (qa only) and after Part 6 (all).
LORA_ARGS=()
for pair in qa=sft-qa notes=sft-notes summary=sft-summary qa-dpo=dpo-qa notes-dpo=dpo-notes; do
  name="${pair%%=*}"; dir="${pair#*=}"
  if [ -f "$ADAPTERS/$dir/adapter_config.json" ]; then LORA_ARGS+=("$name=/adapters/$dir"); fi
done
echo "Adapters: ${LORA_ARGS[*]:-none}"
EXTRA=()                                            # becomes: --lora-modules qa=/adapters/sft-qa ...
if [ "${#LORA_ARGS[@]}" -gt 0 ]; then EXTRA=(--lora-modules "${LORA_ARGS[@]}"); fi

docker rm -f medscriptai-vllm 2>/dev/null || true   # remove an old container with the same name
docker run -d --name medscriptai-vllm \
  --gpus all --ipc=host \
  -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "$ADAPTERS":/adapters:ro \
  "$VLLM_IMAGE" \
  --model unsloth/Llama-3.1-8B-Instruct \
  --served-model-name base \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --max-cpu-loras 8 \
  "${EXTRA[@]}"

echo "Started. Follow the logs with:  docker logs -f medscriptai-vllm"
