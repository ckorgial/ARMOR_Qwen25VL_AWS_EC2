import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
c=OpenAI(base_url=os.getenv('VLLM_BASE_URL','http://127.0.0.1:8000/v1'),api_key=os.getenv('VLLM_API_KEY','local'),timeout=20)
print([m.id for m in c.models.list().data])

#!/usr/bin/env bash
set -euo pipefail
: "${VLLM_API_KEY:=local}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${VLLM_HOST:=127.0.0.1}"
exec vllm serve "Qwen/Qwen2.5-VL-32B-Instruct-AWQ" \
  --host "$VLLM_HOST" --port 8000 --api-key "$VLLM_API_KEY" \
  --seed 42 --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image": 1}'
