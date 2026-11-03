import base64
import mimetypes
import os
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO
import hashlib
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))

def image_to_data_url(path: str, force_format: Optional[str] = None) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if force_format:
        mime = f"image/{force_format.lower()}"
    if mime is None:
        mime = "image/png"
    desired_format = mime.split("/")[-1].upper()
    with Image.open(path) as img:
        buf = BytesIO()
        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(buf, format=desired_format if desired_format != "JPG" else "JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def image_to_data_url_resized(path: str, max_side: int = 192, fmt: str = "JPEG", quality: int = 70) -> str:
    """Create a small data URL for VL prompts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        if scale != 1.0:
            im = im.resize((new_w, new_h), Image.BILINEAR)
        buf = BytesIO()
        im.save(buf, format=fmt, quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def pil_read_rgb(path: str, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img

def pil_to_tensor_unit(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t

def tensor_to_pil_unit(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def save_png(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def compute_ssim_rgb_uint8(a: np.ndarray, b: np.ndarray) -> float:
    return sum(ssim(a[..., c], b[..., c], data_range=255) for c in range(3)) / 3.0

def pil_to_uint8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def save_delta_npy(delta: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as np
    np.save(path, delta.detach().cpu().numpy())