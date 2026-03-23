import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class SimpleCNN(nn.Module):
    """CIFAR-10 style CNN with 3 conv blocks followed by fully connected layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        self.last_activations: Dict[str, torch.Tensor] = {}
        self.gradcam_activation: Optional[torch.Tensor] = None
        self.gradcam_gradient: Optional[torch.Tensor] = None

    def _save_gradcam_gradient(self, grad: torch.Tensor) -> None:
        self.gradcam_gradient = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        self.last_activations["conv1"] = x.detach()
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        self.last_activations["conv2"] = x.detach()
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        self.gradcam_activation = x
        self.gradcam_gradient = None
        if x.requires_grad:
            x.register_hook(self._save_gradcam_gradient)
        self.last_activations["conv3"] = x.detach()
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model(model_path: Path, device: torch.device) -> SimpleCNN:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = SimpleCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a state_dict or dict containing 'model_state_dict'."
        )

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def compress_activation(activation: torch.Tensor) -> List[float]:
    # Reduce each channel to a single value using global average pooling.
    pooled = F.adaptive_avg_pool2d(activation, (1, 1)).squeeze(0).squeeze(-1).squeeze(-1)
    return pooled.cpu().tolist()


def generate_gradcam_heatmap(
    model: SimpleCNN,
    logits: torch.Tensor,
    class_index: int,
    output_size: Tuple[int, int],
) -> str:
    model.zero_grad(set_to_none=True)
    score = logits[:, class_index].sum()
    score.backward()

    if model.gradcam_activation is None or model.gradcam_gradient is None:
        raise RuntimeError("Grad-CAM tensors are not available from conv3.")

    gradients = model.gradcam_gradient
    activations = model.gradcam_activation

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)

    cam_2d = cam.squeeze(0).squeeze(0).detach().cpu().clamp(0.0, 1.0)
    heatmap_rgb = torch.stack(
        [cam_2d, torch.zeros_like(cam_2d), 1.0 - cam_2d],
        dim=0,
    )

    heatmap_image = transforms.ToPILImage()(heatmap_rgb)
    buffer = BytesIO()
    heatmap_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


app = FastAPI(title="Local AI Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "final_model.pth"

try:
    MODEL = load_model(MODEL_PATH, DEVICE)
except Exception as exc:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}") from exc

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Use POST /predict with an image file."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = await file.read()
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    logits = MODEL(input_tensor)
    probs = F.softmax(logits, dim=1)
    confidence, predicted_idx = torch.max(probs, dim=1)

    idx = int(predicted_idx.item())
    all_probabilities = probs.squeeze(0).detach().cpu().tolist()
    heatmap_b64 = generate_gradcam_heatmap(
        model=MODEL,
        logits=logits,
        class_index=idx,
        output_size=(image.height, image.width),
    )

    MODEL.zero_grad(set_to_none=True)
    response = {
        "prediction": CIFAR10_CLASSES[idx],
        "confidence": float(confidence.item()),
        "class_probabilities": [
            {"class": class_name, "probability": float(class_prob)}
            for class_name, class_prob in zip(CIFAR10_CLASSES, all_probabilities)
        ],
        "activations": {
            "conv1": compress_activation(MODEL.last_activations["conv1"]),
            "conv2": compress_activation(MODEL.last_activations["conv2"]),
            "conv3": compress_activation(MODEL.last_activations["conv3"]),
        },
        "heatmap": heatmap_b64,
    }
    return response
