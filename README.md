# ARMOR — Qwen2.5-VL / vLLM / AWS EC2

A runnable **research reimplementation**, based on [ARMOR](https://ieeexplore.ieee.org/document/11460595/)

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
models/                  PLACE .pth files here after downloading separately; not tracked in Git
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

Upload the lightweight `ARMOR_Qwen25VL_AWS_EC2_Lightweight.zip` archive to EC2 (or clone your Git repository), then:

```bash
sudo apt update
sudo apt install -y unzip git curl python3-venv
unzip ARMOR_Qwen25VL_AWS_EC2_Lightweight.zip -d armor-project
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

## Model weights: required downloads (NOT included in GitHub release)

**The public repository / lightweight archive intentionally excludes the two trained surrogate `.pth` checkpoints** because binary weight files are large and cannot be reconstructed from architecture names. These are **custom 2-class classifier weights**, not TorchVision's pretrained ImageNet models. Do **not** download stock ResNet-50 / DenseNet-121 weights as substitutes: the classifier heads, label mapping and trained parameters differ. No public download URL for the supplied custom checkpoints has been established or verified, so this README does not invent one.

| File name | Size (bytes) | Model architecture | Required? | Where to obtain |
|---|---:|---|---|---|
| `models/resnet50.pth` | 94,366,454 | TorchVision ResNet-50, 2-output `fc` head | Yes | Project owner / original authorized checkpoint source / your private S3 bucket |
| `models/densenet121.pth` | 28,431,152 | TorchVision DenseNet-121, 2-output `classifier` head | Yes | Project owner / original authorized checkpoint source / your private S3 bucket |
| `vit_b_16.pth` | Not supplied | Optional binary ViT-B/16, 2-output head | No | Separate authorized trained checkpoint, for blind evaluation only |

**Expected project paths:**

```text
armor-project/armor/
├── models/
│   ├── resnet50.pth       # download separately; required
│   └── densenet121.pth    # download separately; required
├── README.md
└── run_attacker.py
```

### Method A — Download from a private Amazon S3 bucket (recommended on EC2)

First, an authorized holder of the original checkpoints uploads them to a **private**, encrypted S3 bucket (replace the placeholders; these are not working URLs):

```bash
# On the machine holding the ORIGINAL two trained .pth files:
aws s3 cp models/resnet50.pth s3://YOUR_PRIVATE_BUCKET/armor/checkpoints/resnet50.pth --sse AES256
aws s3 cp models/densenet121.pth s3://YOUR_PRIVATE_BUCKET/armor/checkpoints/densenet121.pth --sse AES256
```

Grant the EC2 instance's IAM role only the required `s3:GetObject` permissions for those exact object keys. Then, **from the `armor-project/armor` directory on EC2**:

```bash
mkdir -p models
aws s3 cp s3://YOUR_PRIVATE_BUCKET/armor/checkpoints/resnet50.pth models/resnet50.pth
aws s3 cp s3://YOUR_PRIVATE_BUCKET/armor/checkpoints/densenet121.pth models/densenet121.pth
```

The above requires the AWS CLI and an authenticated IAM principal with the necessary permissions. Prefer an EC2 instance profile over static access keys. AWS official instructions: https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html

### Method B — Copy directly from your workstation with SCP

From your workstation (where the checkpoints already exist), using a restricted EC2 SSH connection:

```bash
# Example: repository unpacked in /home/ubuntu/armor-project/armor
scp -i YOUR_EC2_KEY.pem models/resnet50.pth \
  ubuntu@YOUR_EC2_PUBLIC_DNS:/home/ubuntu/armor-project/armor/models/
scp -i YOUR_EC2_KEY.pem models/densenet121.pth \
  ubuntu@YOUR_EC2_PUBLIC_DNS:/home/ubuntu/armor-project/armor/models/
```

Create the destination `models` directory on the EC2 machine beforehand with `mkdir -p ~/armor-project/armor/models`. SSH access should be restricted to your IP, and use SSH host-key verification. SSM or private connectivity may be preferable.

### Method C — Authorized hosted HTTPS download

Only if the model owner publishes a **verified** URL or creates short-lived presigned HTTPS download links:

```bash
mkdir -p models
curl --fail --location --retry 3 --output models/resnet50.pth 'VERIFIED_RESNET50_CHECKPOINT_URL'
curl --fail --location --retry 3 --output models/densenet121.pth 'VERIFIED_DENSENET121_CHECKPOINT_URL'
```

The URL strings above are placeholders, **not actual download locations**. Treat presigned URLs as secrets and do not commit them or print them in shared logs. For long-term reproducibility, publish an access-controlled artifact registry or data availability statement, subject to redistribution rights.

### Verify checkpoint integrity before running

The following SHA-256 values were calculated from the checkpoint files in the supplied project archive. They allow you to verify that separately downloaded copies match those source files:

```text
6ed557fcc1a17afe877292566c10fb643e4e03566ee9c91bafb4c89a7a617a95  models/resnet50.pth
6f4e723ce63b13643bc533ce3594f0daacf7d97742972bf1b5fa202d7e625108  models/densenet121.pth
```

The same values are provided in `models/SHA256SUMS`. From the repository root:

```bash
sha256sum -c models/SHA256SUMS
ls -lh models/*.pth
```

Both entries must show `OK`. A checksum proves that the bytes match the source archive; it does **not** establish provenance, model accuracy, trustworthiness, or permission to redistribute. Never load untrusted `.pth` files. This project's loader uses `torch.load(..., weights_only=True)` and strict shape checking, but you should still obtain files from a trusted source.

**Validate the trained labels:** the runtime assumes class index `0` means **real** and index `1` means **fake**, as carried over from the uploaded prototype. Check that assumption, the preprocessing pipeline, and any dataset-specific normalization against your original training records. A stock pretrained checkpoint will not satisfy the binary heads or reproduce the experiment.

### Optional: storage policies for a GitHub release

- Keep `*.pth`, `*.pt`, `*.ckpt`, and large weight files out of normal Git history (`.gitignore` includes these patterns).
- Put small code/configuration files in Git; keep trained weights in private S3, a controlled research repository, or separately published release assets where redistribution is authorized.
- Git LFS is an **optional** alternative if the host's quotas and redistribution rules allow it; Git LFS files are not ordinary Git blobs and need their own download/access workflow.
- If you don't have the original trained `.pth` checkpoints, **the full adversarial attack pipeline cannot run as designed**. You can inspect code or run architecture-only tests, but must train/evaluate replacement checkpoints yourself rather than silently substituting ImageNet weights.

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

Verify after downloading both checkpoints:

```bash
sha256sum -c models/SHA256SUMS
set -a; source .env; set +a
python scripts/check_vllm.py
pytest -q
python run_attacker.py --image img_0006.png --output data/runs/img_0006
```

Quick non-vLLM, CPU-only test (still REQUIRES both custom checkpoints; slow, reduces optimization steps):

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
- **Missing checkpoint (`FileNotFoundError`):** download the two original `.pth` files into `models/` using one of the methods above, then run `sha256sum -c models/SHA256SUMS`.
- **Checkpoint mismatch:** original checkpoint architecture may differ from assumed binary heads. Examine state keys and retrain/export matching checkpoints; don't force `strict=False` silently.
- **The `--max-model-len 72000` setting:** not automatically feasible merely because the model supports long context. GPU size, KV cache, image encoding, concurrency and versions determine whether it starts.
- **Slow attacks:** methods evaluate CNN gradients at each step, plus repeated mixer scoring. Reduce iterations/steps/trials for smoke tests; warm up models; profile before production evaluations.
- **Unexpected accuracy:** verify class order, original training resize/normalization, evaluation split, correct target semantics, consistent 224x224 resizing, and fixed seeds.

## Research methodology and limitations

AADD-LQ contains 710 fake samples in the paper's evaluation. The lightweight repo includes only the one example image and no model checkpoints; it cannot reproduce paper ASR/wASR, CIs or transfer conclusions on its own. The paper reports ResNet-50 / DenseNet-121 surrogates, ViT-B/16 blind target, 224x224 images, and ImageNet normalization. Sources: [paper](https://arxiv.org/html/2601.18386v1), [official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ), [vLLM Docker documentation](https://docs.vllm.ai/en/latest/deployment/docker/).

For a strict paper reproduction, replace simplified JSMA with exact pairwise saliency, establish the Qwen3 advisor service, use the paper's exact hill-climb/query budget and target-evaluation protocol, supply and validate all three detector checkpoints and the original AADD-LQ dataset/splits. Record package versions, hashes, hardware, and random seeds. Avoid reporting new results as the authors' published results without those checks.

## License and third-party assets

No project-wide license was included in the provided prototype/rebuild. Do not assume that the original code, images, or supplied checkpoint files are redistributable. Before adding an Apache-2.0 or other LICENSE file, confirm ownership or permission for each component. Qwen model weights and third-party Python packages have their own licenses and terms; review them independently.

