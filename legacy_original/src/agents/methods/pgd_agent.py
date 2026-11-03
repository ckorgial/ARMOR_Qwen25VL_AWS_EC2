import torch
import torch.nn.functional as F
from .base import MethodAgentBase
from .filters import apply_frequency

class PGD_Agent(MethodAgentBase):
    def __init__(self, desk, advisor=None, llm_every_k: int = 5):
        super().__init__("PGD_Agent", desk, advisor=advisor, llm_every_k=llm_every_k)

    def generate(self, x: torch.Tensor, target_class: int, epsilon_max: float, strategy: dict | None = None, **kwargs):
        steps = 10
        eps = min(epsilon_max, 12/255)
        alpha = eps / 4.0
        freq = "neutral"
        if strategy:
            eps = float(min(epsilon_max, strategy.get("epsilon", eps)))
            alpha = float(strategy.get("alpha", alpha))
            steps = int(strategy.get("steps", steps))
            freq = strategy.get("frequency", freq)

        x_orig = x.clone().detach()
        x_adv = x.clone().detach()

        for _ in range(max(1, steps)):
            x_adv.requires_grad_(True)
            loss_total = 0.0
            for _, m in self.models.items():
                logits = m(self.norm(x_adv))
                loss_total = loss_total + F.cross_entropy(logits, torch.tensor([target_class], device=x.device))
            loss_total = loss_total / len(self.models)
            loss_total.backward()

            with torch.no_grad():
                grad = x_adv.grad.sign()
                delta = x_adv - x_orig
                delta = delta - alpha * grad  # targeted
                delta = apply_frequency(delta, freq)
                delta = delta.clamp(min=-eps, max=eps)
                x_adv = (x_orig + delta).clamp(0.0, 1.0)

        params = {
            "method": "PGD",
            "epsilon": float(eps),
            "alpha": float(alpha),
            "steps": int(steps),
            "target_class": target_class,
            "frequency": freq,
        }
        return x_adv.detach(), params