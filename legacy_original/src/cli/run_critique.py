import argparse
import time
from ..core.mixing_desk import FileMixingDesk
from ..agents.critique.surrogate_agent import SurrogateCritiqueAgent
from ..agents.critique.perceptual_agent import PerceptualCritiqueAgent

def main():
    p = argparse.ArgumentParser(description="Run Critique Agent(s) on master image.")
    p.add_argument("--image-id", required=True)
    p.add_argument("--agent", choices=["resnet50", "densenet121", "perceptual"], required=True)
    p.add_argument("--loop", type=int, default=0)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()

    desk = FileMixingDesk()
    if args.agent == "perceptual":
        agent = PerceptualCritiqueAgent(desk)
    else:
        agent = SurrogateCritiqueAgent(args.agent, desk)

    outer = max(args.loop, 1)
    for i in range(outer):
        agent.run_once(args.image_id)
        if i < outer - 1:
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()