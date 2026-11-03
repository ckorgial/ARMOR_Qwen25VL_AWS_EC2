import argparse
import json
from ..config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    VLLM_MODEL_ID,
    MIXING_DESK_DIR,
    INFOAGENT_TEMPERATURE,
    INFOAGENT_MAX_TOKENS,
)
from ..core.vllm_client import VLLMClient
from ..core.mixing_desk import FileMixingDesk
from ..agents.info_agent.agent import InfoAgent
from ..agents.info_agent.prompt import DEFAULT_AREAS

def main():
    parser = argparse.ArgumentParser(description="Run InfoAgent on an image.")
    parser.add_argument("--image", required=True, help="Local image path or URL.")
    parser.add_argument("--areas", nargs="*", default=None, help="Optional custom areas of interest.")
    parser.add_argument("--no-save", action="store_true", help="Do not write to Mixing Desk.")
    parser.add_argument("--print", action="store_true", help="Print parsed JSON results to stdout.")
    args = parser.parse_args()

    vllm = VLLMClient(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID)
    desk = FileMixingDesk(base_dir=MIXING_DESK_DIR)
    agent = InfoAgent(
        vllm_client=vllm,
        mixing_desk=desk,
        default_areas=DEFAULT_AREAS,
        temperature=INFOAGENT_TEMPERATURE,
        max_tokens=INFOAGENT_MAX_TOKENS,
    )

    report = agent.run(
        image_source=args.image,
        areas=args.areas or DEFAULT_AREAS,
        save_to_mixing_desk=not args.no_save,
    )

    if args.print:
        print(json.dumps([r.model_dump() for r in report.results], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()