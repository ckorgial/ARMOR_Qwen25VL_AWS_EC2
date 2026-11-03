import argparse
import time
from ..core.mixing_desk import FileMixingDesk
from ..agents.mixer.agent import MixerAgent

def main():
    p = argparse.ArgumentParser(description="Run Mixer Agent once or in a loop.")
    p.add_argument("--image-id", required=True)
    p.add_argument("--loop", type=int, default=0)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()

    desk = FileMixingDesk()
    agent = MixerAgent(desk)

    outer = max(args.loop, 1)
    for i in range(outer):
        agent.run_once(args.image_id)
        if i < outer - 1:
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()