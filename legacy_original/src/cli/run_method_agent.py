import argparse
import time
from ..core.mixing_desk import FileMixingDesk
from ..agents.methods.fgsm_agent import FGSM_Agent
from ..agents.methods.pgd_agent import PGD_Agent

def main():
    p = argparse.ArgumentParser(description="Run a Method Agent (FGSM/PGD) against a session image.")
    p.add_argument("--image-id", required=True, help="Target image_id (created by Conductor).")
    p.add_argument("--agent", choices=["fgsm", "pgd"], required=True)
    p.add_argument("--steps", type=int, default=10, help="Inner steps for PGD; ignored for FGSM.")
    p.add_argument("--loop", type=int, default=0, help="If >0, loop this many outer iterations with delay.")
    p.add_argument("--sleep", type=float, default=2.0, help="Sleep seconds between outer iterations.")
    args = p.parse_args()

    desk = FileMixingDesk()
    if args.agent == "fgsm":
        agent = FGSM_Agent(desk)
    else:
        agent = PGD_Agent(desk)

    outer = max(args.loop, 1)
    for i in range(outer):
        if args.agent == "pgd":
            agent.run_once(args.image_id, step=i)
        else:
            agent.run_once(args.image_id, step=i)
        if i < outer - 1:
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()