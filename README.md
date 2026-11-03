# ARMOR — Qwen2.5-VL / vLLM / AWS EC2 research rebuild

A runnable **research reimplementation**, based on [ARMOR (arXiv:2601.18386v1)](https://arxiv.org/html/2601.18386v1) and the accompanying uploaded prototype. Not the authors' verified reference reproduction. Research and authorized robustness testing only.

## What this code implements

- **InfoAgent**: Qwen2.5-VL-32B-Instruct-AWQ describes image scene/regions (semantic suggestions, not authenticity ground truth).
- **Conductor**: explicit bounded target class, epsilon, SSIM constraint and baseline confidence; cannot change user-specified caps.
- **Advisor**: same Qwen model recommends numeric CW, sparse-JSMA and STA hyperparameters, clamped to safe configured ranges.
- **Method agents**: projected CW, gradient-saliency *approximate* JSMA, flow-field STA; each projects onto pixel / Linf constraints.
- **Mixer**: simplex weighting, randomized local search, evaluates surrogate confidence and SSIM.
- **Critique & Strategist**: iteration logs, surrogate confidence, quality, stagnation-triggered bounded epsilon/tau adjustments.
- **Blind ViT**: optional *evaluation-only* checkpoint. No ViT checkpoint is included in supplied files; its predictions are NEVER optimized against, avoiding target leakage.

**Important divergences from article**: published ARMOR uses Qwen3-32B-AWQ for reasoning agents, while this Qwen-only implementation uses Qwen2.5-VL for all LLM stages. The paper's JSMA uses pairs of pixels satisfying a two-logit saliency condition; this repo implements a simpler gradient-ranked sparse proxy and identifies it accurately. The mixer optimizes **surrogate confidence and SSIM**, not target black-box confidence: this avoids silently mislabeling a white-box attack as blind. Parallel *methods* are executed sequentially to fit a shared GPU; orchestration results are still combined per iteration. Consequently DO NOT treat published ASR numbers as expected outputs from this code.

## Repository layout

```
armor/                   Qwen client, detectors, CW/JSMA/STA, orchestration
run_attacker.py          command-line entry point
models/                  uploaded ResNet-50 and DenseNet-121 .pth (unmodified)
img_0006.png             uploaded example image
scripts/serve_vllm.sh    local/private-server startup
scripts/check_vllm.py    API health probe
legacy_original/         original uploaded source, preserved for audit
requirements.txt         attack-client dependencies (NOT vLLM)
.env.example             no public IP / no secrets
```

## AWS hardware and network: recommended topology

**Option A (recommended)**: `g6e.2xlarge`, NVIDIA L40S 48GB, AWS Deep Learning AMI GPU (Ubuntu), 200+ GiB gp3 storage, x86_64. This is a *starting recommendation*, not a validated 72k-context benchmark. AWQ weight memory is substantially lower than BF16, but multimodal encoder, runtime workspace, image tokens and KV cache add overhead. Start at **8192 tokens**, reduce to 4096 if OOM; increase only after measurements. 72,000 tokens may require a multi-GPU instance and testing. AWS quota, EC2 availability and regional pricing vary.

**Option B**: multi-GPU EC2 with GPUs of sufficient VRAM, set `TENSOR_PARALLEL_SIZE` appropriately; support for particular AWQ+tensor-parallel+kernel combinations must be confirmed on the actual installed vLLM version. A 24GB A10G `g5` is not a safe default for 32B VL + 72k context.

One EC2 instance can serve both vLLM and the attack client; if GPU contention or OOM occurs, use a *second instance* for the attack client, with access to the two CNN checkpoint files. The latter can be a smaller GPU instance if the two surrogate networks fit its VRAM.

Security group: SSH TCP/22 **only from your IP**, no inbound 8000 from Internet. Run vLLM on `127.0.0.1`; when on separate EC2 machines, bind to private VPC IP and allow TCP 8000 **only from the client security group**; still use an API token and preferably TLS/SSM. Do not use the hard-coded public IP in the older ZIP.

## 1. Launch EC2

1. EC2 Console → Launch instance → choose AWS Deep Learning AMI GPU (Ubuntu) for NVIDIA with compatible CUDA driver, x86_64.
2. Instance type: `g6e.2xlarge` where offered. Allocate at least **200 GiB gp3** root or dedicated EBS for model cache, sufficient RAM, and an IAM role limited to required resources. The exact need depends on disk cache and weight sizes.
3. Require IMDSv2; encrypt EBS; restrict SSH; avoid placing HF tokens or vLLM keys in user-data or shell history.
4. Connect via AWS SSM Session Manager (preferred) or `ssh -i key.pem ubuntu@<EC2_PUBLIC_DNS>`.
5. Confirm `nvidia-smi`. Verify NVIDIA driver/CUDA compatibility with your chosen vLLM wheel or container.

## 2. Transfer the zip and initialize server

Upload `armor-qwen-aws.zip` to EC2, then:

```bash
sudo apt update
sudo apt install -y unzip git curl python3-venv
unzip armor-qwen-aws.zip -d armor-project
cd armor-project/armor
nvidia-smi
```

**Conda vLLM server** (if Miniconda is installed):

```bash
conda create -n MultiAgenticResearch python=3.11 -y
conda activate MultiAgenticResearch
python -m pip install --upgrade pip
python -m pip install --upgrade vllm openai
# Only if required by a tested kernel/config: pip install flashinfer-python
export VLLM_API_KEY='replace-with-a-long-random-token'
export MAX_MODEL_LEN=8192
export VLLM_HOST=127.0.0.1
bash scripts/serve_vllm.sh
```

**Alternative with the official container (recommended for predictable deployment)**, after setting up Docker + NVIDIA Container Toolkit:

```bash
export VLLM_API_KEY='replace-with-a-long-random-token'
mkdir -p "$HOME/.cache/huggingface"
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --seed 42 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image": 1}'
```

For reproducibility, pin an **image digest** or known-good release tag after a successful test; `latest` can change. Hugging Face caching or model terms may require a `HF_TOKEN`. Do not send API token to untrusted services.

## 3. Install attack client (separate environment)

```bash
conda create -n mixingdeskadversarialattack python=3.11 -y
conda activate mixingdeskadversarialattack
python -m pip install -U pip
pip install -r requirements.txt
pip install pytest
cp .env.example .env
```

Edit `.env` to match `VLLM_API_KEY`, `VLLM_BASE_URL=http://127.0.0.1:8000/v1` and `VLLM_MODEL_ID`. **The client loads `.env` automatically with python-dotenv; to populate shell variables for other tools, use:** `set -a; source .env; set +a` before running. Never commit `.env`.

Verify:

```bash
set -a; source .env; set +a
python scripts/check_vllm.py
pytest -q
python run_attacker.py --image img_0006.png --output data/runs/img_0006
```

Quick non-vLLM, CPU-only test (slow, reduces optimization steps):

```bash
python run_attacker.py --image img_0006.png --no-llm --cpu --iterations 1 --method-steps 1 --mixer-trials 1 --output data/runs/smoke
```

Results: `data/runs/img_0006/iteration_XX.png`, `best.png`, `result.json` with per-iteration score, surrogate target confidence, SSIM, Linf bound and selected weights. `.pth` models are loaded as provided; **class index 0 = real** is assumed from the original prototype and must be verified against the actual checkpoint training label mapping. Checkpoint provenance and input normalization must also be validated before drawing scientific conclusions.

## 4. Optional blind ViT evaluation

Provide your separately trained **binary ViT-B/16** checkpoint; it must match TorchVision's `vit_b_16` architecture and have `heads.head = Linear(...,2)`. This archive does not contain one.

```bash
python run_attacker.py --image img_0006.png \
  --blind-checkpoint /secure/checkpoints/vit_b_16.pth \
  --evaluate-blind --output data/runs/vit_eval
```

Blind predictions are logged solely for evaluation after each candidate has been selected. They do not inform mixer weights, advisors, gradients, early stopping or stagnation. This is essential when reporting *transfer-based black-box* performance. The paper includes black-box confidence in its mixer score; that is a method deviation here and **must not be hidden in a paper reproduction claim**.

## 5. Troubleshooting

- **CUDA OOM on server:** use `MAX_MODEL_LEN=4096`, reduce `--gpu-memory-utilization` if processes compete, move attack client to another machine or use more GPUs. Beware large image inputs and batch/concurrency.
- **`/v1/models` unavailable:** inspect vLLM startup log, model access, driver and CUDA compatibility; check `.env` API key, and confirm localhost on the correct host.
- **vLLM works but image call fails:** confirm Qwen2.5-VL AWQ support of your vLLM version and image prompt restrictions, and use recent stable vLLM.
- **TorchVision / torch import error:** install matched torch+torchvision builds according to PyTorch's official selector for the machine CUDA runtime.
- **Checkpoint mismatch:** original checkpoint architecture may differ from assumed binary heads. Examine state keys and retrain/export matching checkpoints; don't force `strict=False` silently.
- **The `--max-model-len 72000` setting:** not automatically feasible merely because the model supports long context. GPU size, KV cache, image encoding, concurrency and versions determine whether it starts.
- **Slow attacks:** methods evaluate CNN gradients at each step, plus repeated mixer scoring. Reduce iterations/steps/trials for smoke tests; warm up models; profile before production evaluations.
- **Unexpected accuracy:** verify class order, original training resize/normalization, evaluation split, correct target semantics, consistent 224x224 resizing, and fixed seeds.

## Research methodology and limitations

AADD-LQ contains 710 fake samples in the paper's evaluation. The repo includes only the one example image; it cannot reproduce paper ASR/wASR, CIs or transfer conclusions on its own. The paper reports ResNet-50 / DenseNet-121 surrogates, ViT-B/16 blind target, 224x224 images, and ImageNet normalization. Sources: [paper](https://arxiv.org/html/2601.18386v1), [official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ), [vLLM Docker documentation](https://docs.vllm.ai/en/latest/deployment/docker/).

For a strict paper reproduction, replace simplified JSMA with exact pairwise saliency, establish the Qwen3 advisor service, use the paper's exact hill-climb/query budget and target-evaluation protocol, supply and validate all three detector checkpoints and the original AADD-LQ dataset/splits. Record package versions, hashes, hardware, and random seeds. Avoid reporting new results as the authors' published results without those checks.
