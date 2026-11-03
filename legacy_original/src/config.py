import os
from dotenv import load_dotenv

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://13.158.181.44:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY_KEY")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

MIXING_DESK_BACKEND = os.getenv("MIXING_DESK_BACKEND", "file")
MIXING_DESK_DIR = os.getenv("MIXING_DESK_DIR", "./data/mixing_desk")

MODELS_DIR = os.getenv("MODELS_DIR", "./models")

INFOAGENT_TEMPERATURE = float(os.getenv("INFOAGENT_TEMPERATURE", "0.2"))
INFOAGENT_MAX_TOKENS = int(os.getenv("INFOAGENT_MAX_TOKENS", "1500"))

# Default Conductor objectives
OBJ_TARGET_CLASS = int(os.getenv("OBJ_TARGET_CLASS", "0"))  # 0 => "real"
OBJ_TARGET_CONF = float(os.getenv("OBJ_TARGET_CONF", "0.90"))
OBJ_SSIM_MIN = float(os.getenv("OBJ_SSIM_MIN", "0.70"))
OBJ_EPS_MAX = float(os.getenv("OBJ_EPS_MAX", str(12/255)))  # L_inf bound in [0,1] scale
OBJ_AGGREGATE = os.getenv("OBJ_AGGREGATE", "mean")

# Device
DEVICE_CHOICE = os.getenv("DEVICE_CHOICE", "auto")  # "auto" or "cpu"