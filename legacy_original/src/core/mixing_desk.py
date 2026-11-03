import os
from datetime import datetime
from .schemas import InfoAgentReport, ConductorObjective, TrackMeta, MasterMeta, CritiqueEntry, FeedbackPanel, StrategyDoc

class FileMixingDesk:
    def __init__(self, base_dir: str = "./data/mixing_desk"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _image_dir(self, image_id: str) -> str:
        d = os.path.join(self.base_dir, image_id)
        os.makedirs(d, exist_ok=True)
        return d

    def path_original(self, image_id: str) -> str:
        d = self._image_dir(image_id)
        p = os.path.join(d, "original", "image.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def path_objective(self, image_id: str) -> str:
        d = self._image_dir(image_id)
        return os.path.join(d, "conductor_objective.json")

    def tracks_dir(self, image_id: str) -> str:
        return os.path.join(self._image_dir(image_id), "tracks")

    def track_dir(self, image_id: str, agent_name: str) -> str:
        d = os.path.join(self.tracks_dir(image_id), agent_name)
        os.makedirs(d, exist_ok=True)
        return d

    def master_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "master")
        os.makedirs(d, exist_ok=True)
        return d

    def feedback_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "feedback")
        os.makedirs(d, exist_ok=True)
        return d

    def strategy_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "strategy")
        os.makedirs(d, exist_ok=True)
        return d

    # Info report
    def save_info_report(self, report: InfoAgentReport) -> str:
        d = self._image_dir(report.image_id)
        fname = f"info_report_{report.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(d, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        latest = os.path.join(d, "info_report_latest.json")
        with open(latest, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        return fpath

    # Objective
    def save_objective(self, obj: ConductorObjective):
        path = self.path_objective(obj.image_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(obj.model_dump_json(indent=2))
        return path

    def load_objective(self, image_id: str) -> ConductorObjective:
        path = self.path_objective(image_id)
        with open(path, "r", encoding="utf-8") as f:
            from json import load
            data = load(f)
        return ConductorObjective(**data)

    # Track IO
    def write_track(self, image_id: str, agent_name: str, meta: TrackMeta):
        d = self.track_dir(image_id, agent_name)
        latest_meta = os.path.join(d, "latest_meta.json")
        with open(latest_meta, "w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hist_meta = os.path.join(d, f"meta_{ts}_step{meta.step}.json")
        with open(hist_meta, "w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))

    def list_tracks(self, image_id: str):
        d = self.tracks_dir(image_id)
        if not os.path.exists(d):
            return []
        return [name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name))]

    # Master IO
    def write_master(self, image_id: str, meta: MasterMeta):
        d = self.master_dir(image_id)
        latest_meta = os.path.join(d, "master_meta.json")
        with open(latest_meta, "w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))

    # Feedback
    def append_feedback(self, image_id: str, entry: CritiqueEntry):
        d = self.feedback_dir(image_id)
        panel_path = os.path.join(d, "panel.json")
        from json import load, dump
        if os.path.exists(panel_path):
            with open(panel_path, "r", encoding="utf-8") as f:
                data = load(f)
            panel = FeedbackPanel(**data)
        else:
            panel = FeedbackPanel(image_id=image_id)
        panel.entries = [e for e in panel.entries if e.name != entry.name]
        panel.entries.append(entry)
        panel.updated_at = datetime.utcnow()
        with open(panel_path, "w", encoding="utf-8") as f:
            dump(panel.model_dump(), f, indent=2)

    def load_feedback_panel(self, image_id: str) -> FeedbackPanel:
        d = self.feedback_dir(image_id)
        panel_path = os.path.join(d, "panel.json")
        if not os.path.exists(panel_path):
            return FeedbackPanel(image_id=image_id)
        from json import load
        with open(panel_path, "r", encoding="utf-8") as f:
            data = load(f)
        return FeedbackPanel(**data)

    # Strategy IO
    def path_strategy(self, image_id: str) -> str:
        return os.path.join(self.strategy_dir(image_id), "strategy.json")

    def save_strategy(self, strategy: StrategyDoc):
        p = self.path_strategy(strategy.image_id)
        with open(p, "w", encoding="utf-8") as f:
            f.write(strategy.model_dump_json(indent=2))
        return p

    def load_strategy(self, image_id: str) -> StrategyDoc | None:
        p = self.path_strategy(image_id)
        if not os.path.exists(p):
            return None
        from json import load
        with open(p, "r", encoding="utf-8") as f:
            data = load(f)
        return StrategyDoc(**data)