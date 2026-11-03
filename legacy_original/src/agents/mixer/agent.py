import os
import itertools
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import MasterMeta, TrackMetrics
from ...core.utils.images import (
    pil_read_rgb,
    pil_to_tensor_unit,
    tensor_to_pil_unit,
    pil_to_uint8,
    compute_ssim_rgb_uint8,
    save_png,
)
from ...critique.detectors import get_device, load_classifier, build_transform_tensor
from ...config import MODELS_DIR, DEVICE_CHOICE


class MixerAgent:
    def __init__(self, desk: FileMixingDesk):
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        self.norm = build_transform_tensor()
        from pathlib import Path

        models_dir = Path(MODELS_DIR)
        self.models = {
            "resnet50": load_classifier("resnet50", models_dir / "resnet50.pth", self.device),
            "densenet121": load_classifier("densenet121", models_dir / "densenet121.pth", self.device),
        }

    def _load_track_candidate(self, track_dir: str):
        img_path = os.path.join(track_dir, "latest.png")
        if not os.path.exists(img_path):
            return None
        img = pil_read_rgb(img_path, size=None)
        x = pil_to_tensor_unit(img)
        return x, img_path

    def _blend(self, x_orig: torch.Tensor, xs: list[torch.Tensor], w: np.ndarray) -> torch.Tensor:
        x_mix = x_orig.clone()
        for wi, xi in zip(w, xs):
            x_mix = x_mix + wi * (xi - x_orig)
        return x_mix.clamp(0, 1)

    def _predict_confs(self, x_unit: torch.Tensor, target_idx: int) -> Dict[str, float]:
        x = x_unit.to(self.device)
        x_norm = self.norm(x.clone())
        out: Dict[str, float] = {}
        with torch.no_grad():
            for name, m in self.models.items():
                logits = m(x_norm)
                probs = F.softmax(logits, dim=1)
                out[name] = float(probs[0, target_idx].item())
        return out

    def run_once(self, image_id: str):
        obj = self.desk.load_objective(image_id)
        orig_img = pil_read_rgb(self.desk.path_original(image_id), size=None)
        x_orig = pil_to_tensor_unit(orig_img)

        track_names = self.desk.list_tracks(image_id)
        if not track_names:
            print("[Mixer] No tracks available yet.")
            return

        candidates: Dict[str, torch.Tensor] = {}
        for name in track_names:
            data = self._load_track_candidate(self.desk.track_dir(image_id, name))
            if data is not None:
                x, _ = data
                candidates[name] = x
        if not candidates:
            print("[Mixer] No valid candidate images.")
            return

        names = list(candidates.keys())
        xs = [candidates[n] for n in names]

        best = None
        best_meta = None

        # Try Strategist weights first
        strat = self.desk.load_strategy(image_id)
        if strat and strat.mixer_weights:
            w = np.array([float(strat.mixer_weights.get(n, 0.0)) for n in names], dtype=np.float32)
            if w.sum() > 0:
                w = w / w.sum()
                x_mix = self._blend(x_orig, xs, w)
                mix_img = tensor_to_pil_unit(x_mix)
                ssim_val = compute_ssim_rgb_uint8(pil_to_uint8(orig_img), pil_to_uint8(mix_img))
                if ssim_val >= obj.ssim_min:
                    confs = self._predict_confs(x_mix, target_idx=obj.target_class)
                    avg_conf = sum(confs.values()) / len(confs)
                    best = avg_conf
                    best_meta = (x_mix, ssim_val, confs, w)

        # If no valid strategist suggestion, perform grid search
        if best_meta is None:
            grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            for weights in itertools.product(grid, repeat=len(xs)):
                if sum(weights) == 0.0:
                    continue
                w = np.array(weights, dtype=np.float32)
                w = w / w.sum()
                x_mix = self._blend(x_orig, xs, w)
                mix_img = tensor_to_pil_unit(x_mix)
                ssim_val = compute_ssim_rgb_uint8(pil_to_uint8(orig_img), pil_to_uint8(mix_img))
                if ssim_val < obj.ssim_min:
                    continue
                confs = self._predict_confs(x_mix, target_idx=obj.target_class)
                avg_conf = sum(confs.values()) / len(confs)
                if best is None or avg_conf > best:
                    best = avg_conf
                    best_meta = (x_mix, ssim_val, confs, w)

        if best_meta is None:
            print("[Mixer] No blend met SSIM threshold; aborting write this cycle.")
            return

        x_mix, ssim_val, confs, chosen_w = best_meta
        mix_img = tensor_to_pil_unit(x_mix)
        mdir = self.desk.master_dir(image_id)
        out_path = os.path.join(mdir, "master.png")
        save_png(mix_img, out_path)
        weights_dict = {n: float(chosen_w[i]) for i, n in enumerate(names)}
        meta = MasterMeta(
            weights=weights_dict,
            metrics=TrackMetrics(
                ssim=ssim_val,
                conf_resnet50=confs.get("resnet50"),
                conf_densenet121=confs.get("densenet121"),
                avg_conf=sum(confs.values()) / len(confs),
            ),
            image_path=out_path,
        )
        self.desk.write_master(image_id, meta)
        print(f"[Mixer] Saved master with avg_conf={meta.metrics.avg_conf:.4f} ssim={meta.metrics.ssim:.4f} -> {out_path}")