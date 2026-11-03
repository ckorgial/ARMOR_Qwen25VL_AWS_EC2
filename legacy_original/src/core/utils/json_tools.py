import json
import re
from typing import Any, List, Dict

def extract_json_array(text: str) -> List[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output")
    candidate = text[start : end + 1]
    candidate = re.sub(r"^```(json)?", "", candidate.strip(), flags=re.IGNORECASE).strip("`").strip()
    return json.loads(candidate)

def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    candidate = text[start : end + 1]
    candidate = re.sub(r"^```(json)?", "", candidate.strip(), flags=re.IGNORECASE).strip("`").strip()
    return json.loads(candidate)