import time
from datetime import datetime
from typing import List, Optional
from ...core.vllm_client import VLLMClient
from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import InfoAgentReport, AOIEntry
from ...core.utils.images import is_url, image_to_data_url, file_sha1
from ...core.utils.json_tools import extract_json_array
from ...core.logging_setup import setup_logging
from .prompt import SYSTEM_PROMPT, build_user_message, DEFAULT_AREAS

logger = setup_logging()

class InfoAgent:
    def __init__(
        self,
        vllm_client: VLLMClient,
        mixing_desk: FileMixingDesk,
        default_areas: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ):
        self.client = vllm_client
        self.desk = mixing_desk
        self.default_areas = default_areas or DEFAULT_AREAS
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_messages(self, image_source: str, areas: List[str]):
        content = [{"type": "text", "text": build_user_message(areas)}]
        if is_url(image_source):
            content.append({"type": "image_url", "image_url": {"url": image_source}})
        else:
            data_url = image_to_data_url(image_source)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return messages

    def _reformat_to_json_array(self, raw_text: str, areas: List[str]) -> str:
        # Ask the model to convert its own output to JSON-only, no image needed
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a formatter that outputs ONLY a valid JSON array. "
                    "No markdown, no code fences, no explanations."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Reformat the content below into a JSON array. "
                            'Each item must have keys: "area","reasoning","conclusion","isAI". '
                            "Ensure there is exactly one item per area listed. "
                            "If information is missing for any area, set reasoning/conclusion to 'Inconclusive' and isAI to 'false'.\n\n"
                            "Areas:\n" + "\n".join(f"- {a}" for a in areas)
                        ),
                    },
                    {
                        "type": "text",
                        "text": "Content to reformat:\n" + raw_text[:6000],
                    },
                ],
            },
        ]
        return self.client.chat_vision(messages, temperature=0.0, max_tokens=self.max_tokens)

    def run(
        self,
        image_source: str,
        areas: Optional[List[str]] = None,
        image_id: Optional[str] = None,
        save_to_mixing_desk: bool = True,
    ) -> InfoAgentReport:
        areas = areas or self.default_areas
        if image_id is None:
            image_id = file_sha1(image_source) if not is_url(image_source) else str(abs(hash(image_source)))

        messages = self._build_messages(image_source, areas)
        start = time.time()
        raw = self.client.chat_vision(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = int((time.time() - start) * 1000)
        logger.info(f"InfoAgent LLM call completed in {latency_ms} ms")

        # Try to parse the initial output
        second_raw = None
        try:
            parsed = extract_json_array(raw)
        except Exception as e:
            logger.error(f"Failed to parse model JSON output (first pass): {e}")
            # Second pass: ask model to reformat to JSON-only
            try:
                second_raw = self._reformat_to_json_array(raw, areas)
                parsed = extract_json_array(second_raw)
                logger.info("Recovered JSON via second-pass reformat.")
            except Exception as e2:
                logger.error(f"Second-pass JSON reformat failed: {e2}")
                parsed = [
                    {
                        "area": "general",
                        "reasoning": "Model output could not be parsed into JSON. Treating analysis as inconclusive.",
                        "conclusion": "Insufficient structured output to assess AI-generation indicators.",
                        "isAI": "false",
                    }
                ]

        entries: List[AOIEntry] = []
        for item in parsed:
            try:
                entry = AOIEntry(
                    area=str(item.get("area", "unknown")),
                    reasoning=str(item.get("reasoning", "No details provided.")),
                    conclusion=str(item.get("conclusion", "Inconclusive.")),
                    isAI="true" if str(item.get("isAI", "false")).lower() == "true" else "false",
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Skipping malformed AOI entry: {e}")

        # Keep both raw texts if we did a second pass (for debugging)
        combined_raw = raw if second_raw is None else (raw + "\n\n--- SECOND PASS ---\n" + second_raw)

        report = InfoAgentReport(
            agent="InfoAgent",
            image_id=image_id,
            model=self.client.model,
            created_at=datetime.utcnow(),
            areas=areas,
            results=entries,
            raw_text=combined_raw,
            meta={"latency_ms": latency_ms},
        )

        if save_to_mixing_desk:
            path = self.desk.save_info_report(report)
            logger.info(f"Info report saved to Mixing Desk: {path}")

        return report