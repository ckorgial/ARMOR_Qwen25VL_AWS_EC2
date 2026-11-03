import torch
import torch.nn.functional as F
from .base import MethodAgentBase
from .filters import apply_frequency

class FGSM_Agent(MethodAgentBase):
    def __init__(self, desk, advisor=None, llm_every_k: int = 5):
        super().__init__("FGSM_Agent", desk, advisor=advisor, llm_every_k=llm_every_k)

    def generate(self, x: torch.Tensor, target_class: int, epsilon_max: float, strategy: dict | None = None, **kwargs):
        # Defaults
        eps = min(epsilon_max, 8/255)
        freq = "neutral"
        if strategy:
            eps = float(strategy.get("epsilon", eps))
            freq = strategy.get("frequency", freq)

        x_orig = x.clone().detach()
        x_adv = x.clone().detach().requires_grad_(True)

        loss_total = 0.0
        for _, m in self.models.items():
            logits = m(self.norm(x_adv))
            loss_total = loss_total + F.cross_entropy(logits, torch.tensor([target_class], device=x.device))
        loss_total = loss_total / len(self.models)
        loss_total.backward()

        with torch.no_grad():
            delta = -eps * x_adv.grad.sign()  # targeted
            delta = apply_frequency(delta, freq)
            delta = delta.clamp(min=-eps, max=eps)
            x_new = (x_orig + delta).clamp(0.0, 1.0)

        params = {
            "method": "FGSM",
            "epsilon": float(eps),
            "target_class": target_class,
            "frequency": freq,
        }
        return x_new.detach(), params