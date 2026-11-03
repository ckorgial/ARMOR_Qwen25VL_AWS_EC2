import torch
import torch.nn.functional as F

def box_blur(x: torch.Tensor, kernel_size: int = 3, iters: int = 1) -> torch.Tensor:
    """
    x: NCHW
    """
    pad = kernel_size // 2
    weight = torch.ones((x.shape[1], 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
    weight = weight / (kernel_size * kernel_size)
    out = x
    for _ in range(max(1, iters)):
        out = F.conv2d(out, weight, padding=pad, groups=x.shape[1])
    return out

def apply_frequency(delta: torch.Tensor, mode: str) -> torch.Tensor:
    """
    delta: NCHW in [-1,1] approx.
    mode: 'low' => smooth, 'high' => high-pass, 'neutral' => unchanged
    """
    if mode == "low":
        return box_blur(delta, kernel_size=5, iters=1)
    if mode == "high":
        smooth = box_blur(delta, kernel_size=5, iters=1)
        return delta - smooth
    return delta