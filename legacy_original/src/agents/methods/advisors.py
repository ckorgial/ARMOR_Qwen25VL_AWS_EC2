import json
import os
from typing import Dict, Any, Optional, List
from ...core.mixing_desk import FileMixingDesk
from ...core.vllm_client import VLLMClient
from ...core.utils.images import image_to_data_url_resized
from ...core.utils.json_tools import extract_json_object
from ...core.logging_setup import setup_logging
from ...config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_ID

logger = setup_logging()

ADVISOR_SYSTEM = """You are a per-method Advisor. Output ONLY a JSON object "suggestions" with method-specific keys:
- For FGSM_Agent: {"epsilon": float, "frequency": "low"|"high"|"neutral", "note": "..."}
- For PGD_Agent: {"epsilon": float, "alpha": float, "steps": int, "frequency": "...", "note": "..."}
Guidelines:
- Respect epsilon_max from the objective.
- If SSIM is low while confidence is already high, reduce epsilon and set frequency="low".
- If confidence is below target and SSIM is high, slightly increase epsilon or set frequency="high".
- Only include keys you want to adjust.
Return JSON only (no markdown)."""

class MethodAdvisor:
    def __init__(self, agent_name: str, desk: FileMixingDesk, vllm: Optional[VLLMClient] = None, max_peer_images: int = 1):
        self.agent_name = agent_name
        self.desk = desk
        self.vllm = vllm or VLLMClient(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID)
        self.max_peer_images = max_peer_images

    def _build_context(self, image_id: str) -> Dict[str, Any]:
        obj_path = self.desk.path_objective(image_id)
        try:
            with open(obj_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            obj = {}
        master_meta = {}
        m_meta = os.path.join(self.desk.master_dir(image_id), "master_meta.json")
        if os.path.exists(m_meta):
            try:
                with open(m_meta, "r", encoding="utf-8") as f:
                    master_meta = json.load(f)
            except Exception:
                master_meta = {}
        # This agent meta
        my_meta = {}
        my_path = os.path.join(self.desk.track_dir(image_id, self.agent_name), "latest_meta.json")
        if os.path.exists(my_path):
            try:
                with open(my_path, "r", encoding="utf-8") as f:
                    my_meta = json.load(f)
            except Exception:
                my_meta = {}
        # one peer
        peer_meta = {}
        for name in self.desk.list_tracks(image_id):
            if name != self.agent_name:
                cand = os.path.join(self.desk.track_dir(image_id, name), "latest_meta.json")
                if os.path.exists(cand):
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            peer_meta = json.load(f)
                        break
                    except Exception:
                        pass
        panel = {}
        p = os.path.join(self.desk.feedback_dir(image_id), "panel.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    panel = json.load(f)
            except Exception:
                panel = {}

        return {"objective": obj, "master_meta": master_meta, "my_meta": my_meta, "peer_meta": peer_meta, "panel": panel}

    def _image_parts(self, image_id: str) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        # original
        orig = self.desk.path_original(image_id)
        if os.path.exists(orig):
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(orig)}})
        # master image
        master = os.path.join(self.desk.master_dir(image_id), "master.png")
        if os.path.exists(master):
            parts.append({"type": "text", "text": "Master image"})
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(master)}})
        # my track
        mine = os.path.join(self.desk.track_dir(image_id, self.agent_name), "latest.png")
        if os.path.exists(mine):
            parts.append({"type": "text", "text": f"Track image: {self.agent_name}"})
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(mine)}})
        # one peer
        for name in self.desk.list_tracks(image_id):
            if name == self.agent_name:
                continue
            peer_img = os.path.join(self.desk.track_dir(image_id, name), "latest.png")
            if os.path.exists(peer_img):
                parts.append({"type": "text", "text": f"Peer track: {name}"})
                parts.append({"type": "image_url", "image_url": {"url": image_to_data_url_resized(peer_img)}})
                break
        return parts

    def suggest(self, image_id: str) -> Dict[str, Any]:
        ctx = self._build_context(image_id)
        content = [{"type": "text", "text": json.dumps({"agent": self.agent_name, "context": ctx}, ensure_ascii=False)}] + self._image_parts(image_id)
        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM},
            {"role": "user", "content": content},
        ]
        txt = self.vllm.chat_vision(messages, temperature=0.2, max_tokens=400, response_format={"type": "json_object"})
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "suggestions" in obj and isinstance(obj["suggestions"], dict):
                return obj["suggestions"]
        except Exception:
            pass
        try:
            obj = extract_json_object(txt)
            if "suggestions" in obj and isinstance(obj["suggestions"], dict):
                return obj["suggestions"]
        except Exception:
            logger.warning(f"[Advisor:{self.agent_name}] Could not parse suggestions; returning empty.")
        return {}