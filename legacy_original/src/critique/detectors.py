from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, resnet50
from torchvision import transforms as T

CLASSES = 2
CLASS_IDX_REAL = 0

def get_device(choice: str = "auto") -> torch.device:
    if choice is None or choice.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def build_transform_tensor() -> T.Normalize:
    # Assume resized to 256x256 and in [0,1] already; we only normalize.
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_classifier(name: str, weight_path: Path, device: torch.device) -> nn.Module:
    if name == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, CLASSES)
    elif name == "densenet121":
        model = densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, CLASSES)
    else:
        raise ValueError("Unsupported classifier: " + name)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model

def predict_real_conf(models: Dict[str, nn.Module], x_unit: torch.Tensor, device: torch.device) -> Dict[str, float]:
    """
    x_unit: 1x3xHxW in [0,1] (already resized 256x256).
    Returns confidence for class index 0 for each model.
    """
    x = x_unit.to(device)
    norm = build_transform_tensor()
    x_norm = norm(x.clone())
    out = {}
    with torch.no_grad():
        for name, m in models.items():
            logits = m(x_norm)
            probs = F.softmax(logits, dim=1)
            out[name] = float(probs[0, CLASS_IDX_REAL].item())
    return out