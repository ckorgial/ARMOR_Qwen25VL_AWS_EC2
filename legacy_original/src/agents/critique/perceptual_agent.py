import os
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import CritiqueEntry
from ...core.utils.images import pil_read_rgb, pil_to_uint8, compute_ssim_rgb_uint8

class PerceptualCritiqueAgent:
    def __init__(self, desk: FileMixingDesk):
        self.desk = desk

    def run_once(self, image_id: str):
        orig = pil_read_rgb(self.desk.path_original(image_id), size=None)
        mdir = self.desk.master_dir(image_id)
        src = os.path.join(mdir, "master.png")
        if not os.path.exists(src):
            print("[PerceptualCritique] no master.png yet.")
            return
        adv = pil_read_rgb(src, size=None)
        ssim_val = compute_ssim_rgb_uint8(pil_to_uint8(orig), pil_to_uint8(adv))
        entry = CritiqueEntry(
            name="PerceptualCritique",
            kind="perceptual",
            metrics={"ssim": ssim_val},
            source=src
        )
        self.desk.append_feedback(image_id, entry)
        print(f"[PerceptualCritique] ssim={ssim_val:.4f}")