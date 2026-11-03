import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import TrackMeta, TrackMetrics
from ...core.utils.images import (
    pil_read_rgb,
    pil_to_tensor_unit,
    tensor_to_pil_unit,
    save_png,
    save_delta_npy,
    pil_to_uint8,
    compute_ssim_rgb_uint8,
)
from ...critique.detectors import (
    load_classifier,
    get_device,
    build_transform_tensor,
)
from ...config import MODELS_DIR, DEVICE_CHOICE


def _merge_strategies(global_s: Optional[dict], advisor_s: Optional[dict]) -> Optional[dict]:
    if not advisor_s and not global_s:
        return None
    global_s = global_s or {}
    advisor_s = advisor_s or {}
    # Prefer global keys; fill missing from advisor
    merged = dict(advisor_s)
    merged.update(global_s)
    return merged


class MethodAgentBase(ABC):
    def __init__(self, agent_name: str, desk: FileMixingDesk, advisor=None, llm_every_k: int = 5):
        self.agent_name = agent_name
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        self.norm = build_transform_tensor()
        self.advisor = advisor
        self.llm_every_k = max(1, int(llm_every_k))
        from pathlib import Path

        models_dir = Path(MODELS_DIR)
        self.models = {
            "resnet50": load_classifier("resnet50", models_dir / "resnet50.pth", self.device),
            "densenet121": load_classifier("densenet121", models_dir / "densenet121.pth", self.device),
        }

    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        epsilon_max: float,
        strategy: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        ...

    def _read_strategy(self, image_id: str) -> Optional[dict]:
        doc = self.desk.load_strategy(image_id)
        if doc and isinstance(doc.suggestions, dict):
            return doc.suggestions.get(self.agent_name, {})
        return None

    def _predict_confs(self, x_unit: torch.Tensor, target_idx: int) -> Dict[str, float]:
        """
        Compute probability for target_idx for each surrogate model.
        x_unit is 1x3xHxW in [0,1].
        """
        x = x_unit.to(self.device)
        x_norm = self.norm(x.clone())
        out: Dict[str, float] = {}
        with torch.no_grad():
            for name, m in self.models.items():
                logits = m(x_norm)
                probs = F.softmax(logits, dim=1)
                out[name] = float(probs[0, target_idx].item())
        return out

    def run_once(self, image_id: str, step: int = 0):
        # Load original (256x256)
        orig_path = self.desk.path_original(image_id)
        img = pil_read_rgb(orig_path, size=None)
        x = pil_to_tensor_unit(img).to(self.device)
        objective = self.desk.load_objective(image_id)

        # Global strategy
        strat = self._read_strategy(image_id)

        # Per-agent advisor occasionally
        advisor_s = None
        if self.advisor is not None and (step % self.llm_every_k == 0):
            try:
                advisor_s = self.advisor.suggest(image_id)
            except Exception:
                advisor_s = None

        merged = _merge_strategies(strat, advisor_s)

        x_adv, params = self.generate(
            x=x.clone(),
            target_class=objective.target_class,
            epsilon_max=objective.epsilon_max,
            strategy=merged,
        )

        # Metrics
        adv_img = tensor_to_pil_unit(x_adv)
        adv_uint8 = pil_to_uint8(adv_img)
        orig_uint8 = pil_to_uint8(img)
        ssim_val = compute_ssim_rgb_uint8(orig_uint8, adv_uint8)

        confs = self._predict_confs(x_adv, target_idx=objective.target_class)
        avg_conf = sum(confs.values()) / len(confs)

        # Save track
        tr_dir = self.desk.track_dir(image_id, self.agent_name)
        img_path = os.path.join(tr_dir, "latest.png")
        delta = (x_adv - x).detach().cpu()
        delta_path = os.path.join(tr_dir, "latest_delta.npy")
        save_png(adv_img, img_path)
        save_delta_npy(delta, delta_path)

        meta = TrackMeta(
            agent=self.agent_name,
            method=params.get("method", self.agent_name),
            params=params,
            metrics=TrackMetrics(
                ssim=ssim_val,
                conf_resnet50=confs.get("resnet50"),
                conf_densenet121=confs.get("densenet121"),
                avg_conf=avg_conf,
            ),
            image_path=img_path,
            delta_path=delta_path,
            step=step,
        )
        self.desk.write_track(image_id, self.agent_name, meta)
        print(f"[{self.agent_name}] step={step} ssim={ssim_val:.4f} avg_conf={avg_conf:.4f} -> {img_path}")