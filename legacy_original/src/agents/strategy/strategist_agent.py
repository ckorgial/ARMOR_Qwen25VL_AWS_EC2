import json
import os
from typing import Dict, Any, List, Tuple
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import StrategyDoc
from ...core.utils.json_tools import extract_json_array, extract_json_object
from ...core.utils.images import image_to_data_url_resized
from ...core.logging_setup import setup_logging
from ...core.vllm_client import VLLMClient
from ...config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_ID

logger = setup_logging()

SYSTEM_PROMPT = """You are StrategistAgent. Read the current adversarial 'Mixing Desk' state and output a JSON object with keys:
- suggestions: per-agent dict, e.g.:
  "FGSM_Agent": {"epsilon": float, "frequency": "low"|"high"|"neutral", "note": "..."}
  "PGD_Agent": {"epsilon": float, "alpha": float, "steps": int, "frequency": "...", "note": "..."}
  Only include keys you want to change.
- mixer_weights: dict of {agent_name: weight >=0}, weights must sum to 1.0 (or leave empty to defer).
- rationale: short string.
Constraints:
- Obey epsilon_max from the objective.
- If SSIM is low but confidence is high, consider lower frequency and/or reduced epsilon.
- If confidence is below target but SSIM is high, allow slight increases in epsilon or 'high' frequency for one agent.
Output JSON only (no markdown)."""

def _safe_read(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[Strategist] Failed to read {path}: {e}")
        return default

class StrategistAgent:
    def __init__(self, desk: FileMixingDesk, vllm: VLLMClient | None = None, max_images: int = 2):
        self.desk = desk
        self.vllm = vllm or VLLMClient(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID)
        self.max_images = max_images

    def _context_json(self, image_id: str) -> Dict[str, Any]:
        obj = _safe_read(self.desk.path_objective(image_id), {})
        track_names = self.desk.list_tracks(image_id)
        tracks: Dict[str, Any] = {}
        for name in track_names:
            meta_path = os.path.join(self.desk.track_dir(image_id, name), "latest_meta.json")
            tracks[name] = _safe_read(meta_path, {})
        master_meta = _safe_read(os.path.join(self.desk.master_dir(image_id), "master_meta.json"), {})
        panel = _safe_read(os.path.join(self.desk.feedback_dir(image_id), "panel.json"), {"entries": []})
        info = _safe_read(os.path.join(self.desk._image_dir(image_id), "info_report_latest.json"), {})
        return {"objective": obj, "tracks": tracks, "master": master_meta, "feedback": panel, "info": info}

    def _image_parts(self, image_id: str) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        # Original
        orig = self.desk.path_original(image_id)
        if os.path.exists(orig):
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(orig)}})
        # Master
        master = os.path.join(self.desk.master_dir(image_id), "master.png")
        if os.path.exists(master):
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(master)}})
        # Up to N track images
        count = 0
        for name in self.desk.list_tracks(image_id):
            img_path = os.path.join(self.desk.track_dir(image_id, name), "latest.png")
            if os.path.exists(img_path):
                parts.append({"type": "text", "text": f"Track image: {name}"})
                parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(img_path)}})
                count += 1
                if count >= self.max_images:
                    break
        return parts

    def _messages(self, image_id: str) -> List[Dict[str, Any]]:
        ctx = self._context_json(image_id)
        text = json.dumps(
            {
                "image_id": image_id,
                "objective": ctx.get("objective", {}),
                "tracks": {k: {"metrics": v.get("metrics", {}), "params": v.get("params", {})} for k, v in ctx.get("tracks", {}).items()},
                "master": ctx.get("master", {}),
                "feedback": ctx.get("feedback", {}),
                "info": {"areas": ctx.get("info", {}).get("areas", [])},
                "instructions": {"epsilon_bound": ctx.get("objective", {}).get("epsilon_max", 12/255)},
            },
            ensure_ascii=False,
        )
        content = [{"type": "text", "text": text}] + self._image_parts(image_id)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    def _reformat_to_json(self, bad_text: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You output ONLY a valid JSON object. No extra text."},
            {"role": "user", "content": [{"type": "text", "text": "Reformat to a JSON object with keys: suggestions, mixer_weights, rationale.\n\n" + bad_text[:6000]}]},
        ]
        txt = self.vllm.chat_vision(messages, temperature=0.0, max_tokens=600, response_format={"type": "json_object"})
        try:
            return json.loads(txt)
        except Exception:
            try:
                return extract_json_object(txt)
            except Exception:
                return {"suggestions": {}, "mixer_weights": {}, "rationale": "Fallback after reformat."}

    def run_once(self, image_id: str):
        messages = self._messages(image_id)
        txt = self.vllm.chat_vision(messages, temperature=0.2, max_tokens=800, response_format={"type": "json_object"})
        # Parse
        try:
            out = json.loads(txt)
        except Exception:
            try:
                out = extract_json_object(txt)
            except Exception:
                logger.warning("[Strategist] Could not directly parse LLM JSON; attempting reformat.")
                out = self._reformat_to_json(txt)

        # Sanitize
        ctx = self._context_json(image_id)
        eps_max = float(ctx.get("objective", {}).get("epsilon_max", 12/255))
        suggestions = out.get("suggestions", {}) or {}
        clean_sug = {}
        for agent, cfg in suggestions.items():
            if not isinstance(cfg, dict):
                continue
            c = {}
            if "epsilon" in cfg:
                try:
                    c["epsilon"] = float(max(0.0, min(eps_max, float(cfg["epsilon"]))))
                except Exception:
                    pass
            if "alpha" in cfg:
                try:
                    c["alpha"] = float(max(0.0, min(1.0, float(cfg["alpha"]))))
                except Exception:
                    pass
            if "steps" in cfg:
                try:
                    c["steps"] = int(max(1, int(cfg["steps"])))
                except Exception:
                    pass
            if "frequency" in cfg and str(cfg["frequency"]) in ("low", "high", "neutral"):
                c["frequency"] = str(cfg["frequency"])
            if "note" in cfg:
                c["note"] = str(cfg["note"])[:200]
            if c:
                clean_sug[agent] = c

        weights = out.get("mixer_weights", {}) or {}
        present = list((ctx.get("tracks") or {}).keys())
        weights = {k: float(v) for k, v in weights.items() if k in present and isinstance(v, (int, float)) and v >= 0}
        s = sum(weights.values())
        if s > 0:
            weights = {k: v / s for k, v in weights.items()}
        else:
            weights = {}

        doc = StrategyDoc(
            image_id=image_id,
            suggestions=clean_sug,
            mixer_weights=weights,
            rationale=str(out.get("rationale", ""))[:500]
        )
        self.desk.save_strategy(doc)
        logger.info(f"[Strategist] strategy.json updated for {image_id}")