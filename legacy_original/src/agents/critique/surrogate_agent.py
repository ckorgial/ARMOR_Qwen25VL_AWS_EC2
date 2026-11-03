import os
import torch
import torch.nn.functional as F
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import CritiqueEntry
from ...core.utils.images import pil_read_rgb, pil_to_tensor_unit
from ...critique.detectors import load_classifier, get_device, build_transform_tensor
from ...config import MODELS_DIR, DEVICE_CHOICE

class SurrogateCritiqueAgent:
    def __init__(self, model_name: str, desk: FileMixingDesk):
        self.model_name = model_name
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        from pathlib import Path
        self.model = load_classifier(model_name, Path(MODELS_DIR) / f"{model_name}.pth", self.device)
        self.norm = build_transform_tensor()

    def run_once(self, image_id: str):
        mdir = self.desk.master_dir(image_id)
        src = os.path.join(mdir, "master.png")
        if not os.path.exists(src):
            print(f"[SurrogateCritique:{self.model_name}] no master.png yet.")
            return
        obj = self.desk.load_objective(image_id)
        target_idx = int(getattr(obj, "target_class", 0))

        img = pil_read_rgb(src, size=None)
        x = pil_to_tensor_unit(img).to(self.device)
        with torch.no_grad():
            logits = self.model(self.norm(x))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            conf_target = float(probs[target_idx])

        entry = CritiqueEntry(
            name=f"SurrogateCritique_{self.model_name}",
            kind="surrogate",
            metrics={"target_idx": target_idx, "conf_target": conf_target, "probs": probs},
            source=src,
        )
        self.desk.append_feedback(image_id, entry)
        print(f"[SurrogateCritique:{self.model_name}] conf_target(idx={target_idx})={conf_target:.4f}")