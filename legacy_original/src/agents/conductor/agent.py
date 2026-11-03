import os
from dataclasses import dataclass
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import ConductorObjective
from ...core.utils.images import file_sha1, pil_read_rgb, save_png

@dataclass
class ConductorConfig:
    ssim_min: float = 0.70
    conf_target: float = 0.90
    epsilon_max: float = 12/255
    target_class: int = 0
    aggregate: str = "mean"
    note: str | None = "Initial objective"

class ConductorAgent:
    def __init__(self, desk: FileMixingDesk):
        self.desk = desk

    def prepare_original(self, image_path: str, image_id: str | None = None) -> str:
        image_id = image_id or file_sha1(image_path)
        orig_dest = self.desk.path_original(image_id)
        if not os.path.exists(orig_dest):
            img = pil_read_rgb(image_path, size=(256, 256))
            save_png(img, orig_dest)
        return image_id

    def init_session(self, image_path: str, image_id: str | None = None, cfg: ConductorConfig | None = None) -> str:
        cfg = cfg or ConductorConfig()
        image_id = self.prepare_original(image_path, image_id=image_id)
        obj = ConductorObjective(
            image_id=image_id,
            target_class=cfg.target_class,
            ssim_min=cfg.ssim_min,
            conf_target=cfg.conf_target,
            epsilon_max=cfg.epsilon_max,
            aggregate=cfg.aggregate,  # type: ignore
            note=cfg.note,
        )
        self.desk.save_objective(obj)
        return image_id