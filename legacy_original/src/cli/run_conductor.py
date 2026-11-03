import argparse
from ..core.mixing_desk import FileMixingDesk
from ..agents.conductor.agent import ConductorAgent, ConductorConfig

def main():
    p = argparse.ArgumentParser(description="Initialize a Mixing Desk session and objective.")
    p.add_argument("--image", required=True, help="Path to original image.")
    p.add_argument("--image-id", default=None, help="Optional image_id override.")
    p.add_argument("--ssim-min", type=float, default=0.70)
    p.add_argument("--conf-target", type=float, default=0.90)
    p.add_argument("--epsilon-max", type=float, default=12/255)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--note", default="Initial objective")
    args = p.parse_args()

    desk = FileMixingDesk()
    agent = ConductorAgent(desk)
    image_id = agent.init_session(
        image_path=args.image,
        image_id=args.image_id,
        cfg=ConductorConfig(
            ssim_min=args.ssim_min,
            conf_target=args.conf_target,
            epsilon_max=args.epsilon_max,
            target_class=args.target_class,
            note=args.note,
        ),
    )
    print(f"[CONDUCTOR] Session ready for image_id={image_id}")

if __name__ == "__main__":
    main()