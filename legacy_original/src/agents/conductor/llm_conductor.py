import json
import os
from typing import Optional, Dict, Any
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import ConductorObjective
from ...core.vllm_client import VLLMClient
from ...core.utils.images import image_to_data_url_resized
from ...core.utils.json_tools import extract_json_object
from ...core.logging_setup import setup_logging
from ...config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_ID

logger = setup_logging()

SYSTEM_PROMPT = """You are ConductorLLM. You set a realistic, measurable objective for an adversarial perturbation pipeline.
Output ONLY a JSON object with keys:
- target_class: int (0 => "real")
- ssim_min: float in [0.60, 0.90]
- conf_target: float in [0.70, 0.99]
- epsilon_max: float in [0.0039, 0.06275]  # ~[1/255, 16/255]
- aggregate: "mean" or "sum"
- note: short rationale.

Guidelines:
- If the image already looks benign and consistent, lean toward higher ssim_min (>=0.75) and moderate conf_target (~0.90).
- If InfoAgent finds many artifacts, allow slightly larger epsilon_max to compensate but keep ssim_min reasonable (>0.70).
- Prefer aggregate="mean" unless evaluating a small batch where "sum" is explicitly justified.
Return JSON only (no markdown)."""

class ConductorLLMAgent:
    def __init__(self, desk: FileMixingDesk, vllm: Optional[VLLMClient] = None):
        self.desk = desk
        self.vllm = vllm or VLLMClient(
            base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID
        )

    def build_messages(self, image_id: str) -> list[dict]:
        # Load InfoAgent summary if present
        info_path = os.path.join(self.desk._image_dir(image_id), "info_report_latest.json")
        info_json: Dict[str, Any] = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_json = json.load(f)
            except Exception:
                info_json = {}
        info_snippet = {
            "areas_count": len(info_json.get("areas", [])) if isinstance(info_json.get("areas", []), list) else 0,
            "sample_conclusions": [r.get("conclusion", "") for r in (info_json.get("results") or [])][:5],
        }

        # Image (thumbnail)
        orig = self.desk.path_original(image_id)
        content = [
            {"type": "text", "text": json.dumps({"image_id": image_id, "info_summary": info_snippet}, ensure_ascii=False)},
        ]
        if os.path.exists(orig):
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(orig)}})

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _parse_objective(self, txt: str) -> Dict[str, Any]:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            return extract_json_object(txt)
        except Exception:
            logger.warning("[ConductorLLM] Could not parse JSON. Using defaults.")
            return {
                "target_class": 0,
                "ssim_min": 0.70,
                "conf_target": 0.90,
                "epsilon_max": 12/255,
                "aggregate": "mean",
                "note": "Fallback defaults",
            }

    def run(self, image_id: str) -> ConductorObjective:
        messages = self.build_messages(image_id)
        txt = self.vllm.chat_vision(messages, temperature=0.2, max_tokens=2048, response_format={"type": "json_object"})
        data = self._parse_objective(txt)

        # sanitize ranges
        def clamp(v, lo, hi, default):
            try:
                return max(lo, min(hi, float(v)))
            except Exception:
                return default
        ssim_min = clamp(data.get("ssim_min", 0.70), 0.60, 0.90, 0.70)
        conf_target = clamp(data.get("conf_target", 0.90), 0.70, 0.99, 0.90)
        epsilon_max = clamp(data.get("epsilon_max", 12/255), 1/255, 16/255, 12/255)
        aggregate = "mean" if str(data.get("aggregate", "mean")).lower() not in {"mean", "sum"} else str(data.get("aggregate", "mean")).lower()
        target_class = int(data.get("target_class", 0))
        note = str(data.get("note", ""))[:200]

        obj = ConductorObjective(
            image_id=image_id,
            target_class=target_class,
            ssim_min=ssim_min,
            conf_target=conf_target,
            epsilon_max=epsilon_max,
            aggregate=aggregate,  # type: ignore
            note=note,
        )
        self.desk.save_objective(obj)
        logger.info(f"[ConductorLLM] Objective set: target_class={target_class} conf_target={conf_target:.2f} ssim_min={ssim_min:.2f} eps_max={epsilon_max:.5f} agg={aggregate}")
        return obj