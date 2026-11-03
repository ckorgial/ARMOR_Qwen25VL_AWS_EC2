import os
import sys
import json
import time
import traceback
from threading import Thread, Event

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.core.logging_setup import setup_logging
from src.core.mixing_desk import FileMixingDesk
from src.agents.conductor.agent import ConductorAgent
from src.agents.conductor.llm_conductor import ConductorLLMAgent
from src.agents.methods.fgsm_agent import FGSM_Agent
from src.agents.methods.pgd_agent import PGD_Agent
from src.agents.methods.advisors import MethodAdvisor
from src.agents.mixer.agent import MixerAgent
from src.agents.critique.surrogate_agent import SurrogateCritiqueAgent
from src.agents.critique.perceptual_agent import PerceptualCritiqueAgent
from src.agents.strategy.strategist_agent import StrategistAgent
from src.core.utils.images import file_sha1
from src.config import MODELS_DIR, DEVICE_CHOICE

from src.core.vllm_client import VLLMClient
from src.agents.info_agent.agent import InfoAgent
from src.agents.info_agent.prompt import DEFAULT_AREAS
from src.config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_ID, INFOAGENT_TEMPERATURE, INFOAGENT_MAX_TOKENS

logger = setup_logging()

def check_models():
    m1 = os.path.join(MODELS_DIR, "resnet50.pth")
    m2 = os.path.join(MODELS_DIR, "densenet121.pth")
    m3 = os.path.join(MODELS_DIR, "vit_b_16.pth")
    missing = [p for p in (m1, m2, m3) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model weight files: {missing}")
    logger.info(f"Models found under {MODELS_DIR}")


def safe_thread(target, name: str, *args, **kwargs) -> Thread:
    def runner():
        try:
            target(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thread '{name}' crashed: {e}\n{traceback.format_exc()}")
    t = Thread(target=runner, name=name, daemon=True)
    t.start()
    return t

def loop_method(agent, image_id: str, loops: int, interval: float, stop_event: Event):
    for i in range(loops):
        if stop_event.is_set():
            break
        agent.run_once(image_id, step=i)
        if stop_event.is_set():
            break
        time.sleep(interval)

def loop_mixer(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = MixerAgent(desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_resnet(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = SurrogateCritiqueAgent("resnet50", desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_densenet(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = SurrogateCritiqueAgent("densenet121", desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_perceptual(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = PerceptualCritiqueAgent(desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_vit(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = SurrogateCritiqueAgent("vit_b_16", desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_strategist(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk, vllm: VLLMClient):
    agent = StrategistAgent(desk, vllm=vllm)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def monitor_early_stop(image_id: str, desk: FileMixingDesk, stop_event: Event, interval: float = 2.0, no_early_stop: bool = False):
    # Objective is now set by LLM; read it to determine thresholds
    from json import load
    obj_path = desk.path_objective(image_id)
    with open(obj_path, "r", encoding="utf-8") as f:
        obj = load(f)
    conf_target = float(obj.get("conf_target", 0.90))
    ssim_min = float(obj.get("ssim_min", 0.70))

    start = time.time()
    last_print = 0.0
    while not stop_event.is_set():
        if (time.time() - start) > 900:  # 15 min safety
            logger.info("[Monitor] Max time exceeded; stopping.")
            stop_event.set()
            break
        meta_path = os.path.join(desk.master_dir(image_id), "master_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                metrics = meta.get("metrics", {})
                avg_conf = float(metrics.get("avg_conf") or 0.0)
                ssim_val = float(metrics.get("ssim") or 0.0)
                now = time.time()
                if now - last_print > 5.0:
                    logger.info(f"[Monitor] master avg_conf={avg_conf:.4f} ssim={ssim_val:.4f} (target_conf={conf_target:.2f}, ssim_min={ssim_min:.2f})")
                    last_print = now
                if not no_early_stop and avg_conf >= conf_target and ssim_val >= ssim_min:
                    logger.info("[Monitor] Objective met; triggering stop.")
                    stop_event.set()
                    break
            except Exception as e:
                logger.warning(f"[Monitor] Could not read master meta: {e}")
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def orchestrate(image_path: str, fgsm_loops: int = 20, pgd_loops: int = 20, interval: float = 2.0, no_early_stop: bool = False) -> str:
    check_models()
    desk = FileMixingDesk()
    vllm = VLLMClient(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID)

    # 1) Save original
    image_id = file_sha1(image_path)
    ConductorAgent(desk).prepare_original(image_path, image_id=image_id)
    logger.info(f"[Conductor] Prepared original | image_id={image_id} | DEVICE_CHOICE={DEVICE_CHOICE}")

    # 2) InfoAgent (LLM) first
    info = InfoAgent(
        vllm_client=vllm,
        mixing_desk=desk,
        temperature=INFOAGENT_TEMPERATURE,
        max_tokens=INFOAGENT_MAX_TOKENS,
    )
    info.run(image_source=image_path, image_id=image_id, save_to_mixing_desk=True)

    # 3) ConductorLLM sets the objective from image + InfoAgent output
    cond_llm = ConductorLLMAgent(desk, vllm=vllm)
    cond_llm.run(image_id)

    # Seed a first strategy immediately so methods can use it from step 0
    StrategistAgent(desk, vllm=vllm).run_once(image_id)

    # 4) Start Strategist (global LLM coordinator loop)
    stop_event = Event()
    t_strat = safe_thread(loop_strategist, "Strategist", image_id, interval, stop_event, desk, vllm)

    # 5) Method Agents with per-agent LLM advisors
    fgsm_advisor = MethodAdvisor("FGSM_Agent", desk, vllm=vllm)
    pgd_advisor = MethodAdvisor("PGD_Agent", desk, vllm=vllm)
    fgsm = FGSM_Agent(desk, advisor=fgsm_advisor, llm_every_k=3)
    pgd = PGD_Agent(desk, advisor=pgd_advisor, llm_every_k=3)

    t_fgsm = safe_thread(loop_method, "FGSM", fgsm, image_id, fgsm_loops, interval, stop_event)
    t_pgd = safe_thread(loop_method, "PGD", pgd, image_id, pgd_loops, interval, stop_event)

    # 6) Mixer and Critiques
    t_mixer = safe_thread(loop_mixer, "Mixer", image_id, interval, stop_event, desk)
    t_c_res = safe_thread(loop_critique_resnet, "Critique-ResNet50", image_id, interval, stop_event, desk)
    t_c_den = safe_thread(loop_critique_densenet, "Critique-DenseNet121", image_id, interval, stop_event, desk)
    t_c_per = safe_thread(loop_critique_perceptual, "Critique-Perceptual", image_id, interval, stop_event, desk)
    t_c_vit = safe_thread(loop_critique_vit, "Critique-ViT", image_id, interval, stop_event, desk)

    # 7) Monitor
    monitor_early_stop(image_id=image_id, desk=desk, stop_event=stop_event, interval=interval, no_early_stop=no_early_stop)

    time.sleep(0.5)
    logger.info("[Main] Joining threads...")
    for t in (t_fgsm, t_pgd, t_mixer, t_c_res, t_c_den, t_c_per, t_c_vit, t_strat):
        t.join(timeout=5.0)

    # Final summary
    master_meta_path = os.path.join(desk.master_dir(image_id), "master_meta.json")
    if os.path.exists(master_meta_path):
        try:
            with open(master_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})
            logger.info(
                f"[Final] avg_conf={float(metrics.get('avg_conf') or 0.0):.4f} "
                f"ssim={float(metrics.get('ssim') or 0.0):.4f} "
                f"master={meta.get('image_path')}"
            )
        except Exception as e:
            logger.warning(f"[Final] Could not read master meta: {e}")
    panel_path = os.path.join(desk.feedback_dir(image_id), "panel.json")
    if os.path.exists(panel_path):
        with open(panel_path, "r", encoding="utf-8") as f:
            panel = json.load(f)
        logger.info(f"[Final] Feedback panel entries: {[e.get('name') for e in panel.get('entries', [])]}")
    logger.info("[Main] Done.")
    return image_id

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run full LLM-coordinated attacker lineup on a single image.")
    p.add_argument("--image", required=True, help="Path to original image.")
    p.add_argument("--fgsm-loops", type=int, default=20)
    p.add_argument("--pgd-loops", type=int, default=20)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--no-early-stop", action="store_true")
    args = p.parse_args()

    try:
        orchestrate(
            image_path=args.image,
            fgsm_loops=args.fgsm_loops,
            pgd_loops=args.pgd_loops,
            interval=args.interval,
            no_early_stop=args.no_early_stop,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)