import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self.debug_tensors: Dict[str, torch.Tensor] = {}
        self.gradcam_activation: Optional[torch.Tensor] = None
        self.gradcam_gradient: Optional[torch.Tensor] = None

    def _save_gradcam_gradient(self, grad: torch.Tensor) -> None:
        self.gradcam_gradient = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.debug_tensors["input"] = x.detach()

        conv1_pre = self.conv1(x)
        self.debug_tensors["conv1_pre"] = conv1_pre.detach()
        x = F.relu(conv1_pre)
        self.last_activations["conv1"] = x.detach()
        self.debug_tensors["conv1_post"] = x.detach()
        x = self.pool(x)
        self.debug_tensors["pool1"] = x.detach()

        conv2_pre = self.conv2(x)
        self.debug_tensors["conv2_pre"] = conv2_pre.detach()
        x = F.relu(conv2_pre)
        self.last_activations["conv2"] = x.detach()
        self.debug_tensors["conv2_post"] = x.detach()
        x = self.pool(x)
        self.debug_tensors["pool2"] = x.detach()

        conv3_pre = self.conv3(x)
        self.debug_tensors["conv3_pre"] = conv3_pre.detach()
        x = F.relu(conv3_pre)
        self.gradcam_activation = x
        self.gradcam_gradient = None
        if x.requires_grad:
            x.register_hook(self._save_gradcam_gradient)
        self.last_activations["conv3"] = x.detach()
        self.debug_tensors["conv3_post"] = x.detach()
        x = self.pool(x)
        self.debug_tensors["pool3"] = x.detach()

        x = x.view(x.size(0), -1)
        self.debug_tensors["flatten"] = x.detach()
        fc1_pre = self.fc1(x)
        self.debug_tensors["fc1_pre"] = fc1_pre.detach()
        x = F.relu(fc1_pre)
        self.debug_tensors["fc1_post"] = x.detach()
        logits = self.fc2(x)
        self.debug_tensors["logits"] = logits.detach()
        return logits


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


def extract_feature_maps(
    activation: torch.Tensor,
    pre_activation: Optional[torch.Tensor] = None,
    max_maps: int = 12,
) -> List[Dict[str, Any]]:
    """Return a small, representative set of normalized feature maps for visualization."""
    feature_tensor = activation.squeeze(0).detach().cpu()  # [C, H, W]
    pre_tensor = pre_activation.squeeze(0).detach().cpu() if pre_activation is not None else None
    channels = feature_tensor.shape[0]
    k = min(max_maps, channels)

    # Pick channels with highest mean activation to keep maps informative.
    channel_scores = feature_tensor.mean(dim=(1, 2))
    top_indices = torch.topk(channel_scores, k=k).indices

    maps: List[Dict[str, Any]] = []
    for channel_index in top_indices:
        fmap = feature_tensor[int(channel_index)]
        fmap_min = fmap.min()
        fmap_max = fmap.max()
        normalized = (fmap - fmap_min) / (fmap_max - fmap_min + 1e-8)
        maps.append(
            {
                "channel": int(channel_index.item()),
                "mean_activation": float(fmap.mean().item()),
                "signed_mean": float(pre_tensor[int(channel_index)].mean().item()) if pre_tensor is not None else float(fmap.mean().item()),
                "values": normalized.tolist(),
                "preview_image": _tensor_to_png_b64(normalized),
            }
        )

    return maps


def _round_matrix(values: torch.Tensor, digits: int = 4) -> List[List[float]]:
    return [[round(float(v), digits) for v in row] for row in values.tolist()]


def _tensor_to_png_b64(tensor_2d: torch.Tensor) -> str:
    tensor_2d = tensor_2d.detach().cpu()
    t_min = tensor_2d.min()
    t_max = tensor_2d.max()
    normalized = (tensor_2d - t_min) / (t_max - t_min + 1e-8)
    image = transforms.ToPILImage()(normalized.unsqueeze(0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_conv_example(
    layer_name: str,
    conv: nn.Conv2d,
    conv_input: torch.Tensor,
    conv_pre: torch.Tensor,
    conv_post: torch.Tensor,
    pooled: torch.Tensor,
) -> Dict[str, Any]:
    post = conv_post.squeeze(0)
    channel_means = post.mean(dim=(1, 2))
    out_channel = int(torch.argmax(channel_means).item())

    out_h = conv_pre.shape[2]
    out_w = conv_pre.shape[3]
    y = min(2, out_h - 1)
    x = min(2, out_w - 1)

    kernel_weights = conv.weight.detach()[out_channel]  # [C, K, K]
    in_channel = int(torch.argmax(kernel_weights.abs().mean(dim=(1, 2))).item())
    kernel = kernel_weights[in_channel]

    padded = F.pad(conv_input.detach(), (1, 1, 1, 1), mode="constant", value=0)
    patch = padded[0, in_channel, y : y + 3, x : x + 3]
    elementwise = kernel * patch
    single_channel_sum = elementwise.sum()

    pre_value = conv_pre[0, out_channel, y, x]
    post_value = conv_post[0, out_channel, y, x]
    bias_value = conv.bias.detach()[out_channel]

    py = min(y // 2, pooled.shape[2] - 1)
    px = min(x // 2, pooled.shape[3] - 1)
    post_slice = conv_post[0, out_channel]
    pool_window = post_slice[py * 2 : py * 2 + 2, px * 2 : px * 2 + 2]
    pooled_value = pooled[0, out_channel, py, px]

    return {
        "layer": layer_name,
        "selected_output_channel": out_channel,
        "selected_input_channel": in_channel,
        "location": {"x": int(x), "y": int(y)},
        "equation": "Output(x,y) = sum(kernel(i,j) * input(x+i,y+j)) + bias",
        "input_patch": _round_matrix(patch),
        "kernel": _round_matrix(kernel),
        "elementwise_product": _round_matrix(elementwise),
        "single_channel_sum": round(float(single_channel_sum.item()), 4),
        "bias": round(float(bias_value.item()), 4),
        "pre_activation": round(float(pre_value.item()), 4),
        "relu_output": round(float(post_value.item()), 4),
        "pool_window": _round_matrix(pool_window),
        "pooled_value": round(float(pooled_value.item()), 4),
        "feature_map_preview": _tensor_to_png_b64(conv_post[0, out_channel]),
    }


def build_explainability_payload(
    model: SimpleCNN,
    input_tensor: torch.Tensor,
    resized_image: Image.Image,
    logits: torch.Tensor,
    probs: torch.Tensor,
    predicted_idx: int,
    original_size: Tuple[int, int],
) -> Dict[str, Any]:
    debug = model.debug_tensors

    conv1_input = debug["input"]
    conv2_input = debug["pool1"]
    conv3_input = debug["pool2"]

    conv1_pre = debug["conv1_pre"]
    conv2_pre = debug["conv2_pre"]
    conv3_pre = debug["conv3_pre"]

    conv1_post = debug["conv1_post"]
    conv2_post = debug["conv2_post"]
    conv3_post = debug["conv3_post"]

    pool1 = debug["pool1"]
    pool2 = debug["pool2"]
    pool3 = debug["pool3"]
    flatten = debug["flatten"].squeeze(0)

    fc1_features = debug["fc1_post"].squeeze(0)
    fc2_weights = model.fc2.weight.detach()[predicted_idx]
    fc2_bias = model.fc2.bias.detach()[predicted_idx]
    fc_terms = fc1_features * fc2_weights

    top_term_indices = torch.topk(fc_terms.abs(), k=min(10, fc_terms.numel())).indices
    top_terms = [
        {
            "feature_index": int(idx.item()),
            "feature_value": round(float(fc1_features[idx].item()), 4),
            "weight": round(float(fc2_weights[idx].item()), 4),
            "product": round(float(fc_terms[idx].item()), 4),
        }
        for idx in top_term_indices
    ]

    flatten_len = flatten.numel()
    top_flat_indices = torch.topk(flatten.abs(), k=min(10, flatten_len)).indices
    linked_features: List[Dict[str, Any]] = []
    for idx in top_flat_indices:
        index = int(idx.item())
        channel = index // 16
        rem = index % 16
        row = rem // 4
        col = rem % 4
        linked_features.append(
            {
                "flatten_index": index,
                "value": round(float(flatten[index].item()), 4),
                "conv3_channel": int(channel),
                "conv3_position": {"row": int(row), "col": int(col)},
                "heatmap_anchor": {
                    "x": round((col + 0.5) / 4, 4),
                    "y": round((row + 0.5) / 4, 4),
                },
            }
        )

    max_logit = logits.max(dim=1, keepdim=True).values
    stabilized = logits - max_logit
    exp_scores = torch.exp(stabilized)
    denom = exp_scores.sum(dim=1)
    numerator = exp_scores[0, predicted_idx]

    resized_buffer = BytesIO()
    resized_image.save(resized_buffer, format="PNG")
    resized_b64 = base64.b64encode(resized_buffer.getvalue()).decode("utf-8")

    input_avg = input_tensor.squeeze(0).mean(dim=0)
    pixel_matrix = input_avg[:8, :8]

    return {
        "input_representation": {
            "resized_image": resized_b64,
            "shape": [3, 32, 32],
            "normalized_range": {
                "min": round(float(input_tensor.min().item()), 4),
                "max": round(float(input_tensor.max().item()), 4),
            },
            "pixel_matrix": _round_matrix(pixel_matrix),
            "sample_pixels": [round(float(v), 4) for v in input_tensor.flatten()[:16].detach().cpu().tolist()],
        },
        "convolution_examples": {
            "conv1": _build_conv_example("conv1", model.conv1, conv1_input, conv1_pre, conv1_post, pool1),
            "conv2": _build_conv_example("conv2", model.conv2, conv2_input, conv2_pre, conv2_post, pool2),
            "conv3": _build_conv_example("conv3", model.conv3, conv3_input, conv3_pre, conv3_post, pool3),
        },
        "relu_examples": {
            "conv1": {
                "before": round(float(conv1_pre[0, 0, 1, 1].item()), 4),
                "after": round(float(conv1_post[0, 0, 1, 1].item()), 4),
            },
            "conv2": {
                "before": round(float(conv2_pre[0, 0, 1, 1].item()), 4),
                "after": round(float(conv2_post[0, 0, 1, 1].item()), 4),
            },
            "conv3": {
                "before": round(float(conv3_pre[0, 0, 1, 1].item()), 4),
                "after": round(float(conv3_post[0, 0, 1, 1].item()), 4),
            },
        },
        "pooling": {
            "input_shape": [int(conv3_post.shape[1]), int(conv3_post.shape[2]), int(conv3_post.shape[3])],
            "output_shape": [int(pool3.shape[1]), int(pool3.shape[2]), int(pool3.shape[3])],
            "example_window": _round_matrix(conv3_post[0, 0, 0:2, 0:2]),
            "example_max": round(float(pool3[0, 0, 0, 0].item()), 4),
        },
        "flattening": {
            "from_shape": [int(pool3.shape[1]), int(pool3.shape[2]), int(pool3.shape[3])],
            "to_length": int(flatten_len),
            "sample_values": [round(float(v), 4) for v in flatten[:20].detach().cpu().tolist()],
        },
        "fully_connected": {
            "equation": "Score_class = sum(feature_i * weight_i) + bias",
            "target_class": CIFAR10_CLASSES[predicted_idx],
            "top_contributions": top_terms,
            "sum_products": round(float(fc_terms.sum().item()), 4),
            "bias": round(float(fc2_bias.item()), 4),
            "score": round(float(logits[0, predicted_idx].item()), 4),
            "linked_features": linked_features,
        },
        "softmax": {
            "equation": "P(class) = exp(score_class) / sum(exp(all_scores))",
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "numerator": round(float(numerator.item()), 6),
            "denominator": round(float(denom.item()), 6),
            "probability": round(float(probs[0, predicted_idx].item()), 6),
        },
        "final_output": {
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "probability": round(float(probs[0, predicted_idx].item()), 6),
            "max_probability_class": CIFAR10_CLASSES[int(torch.argmax(probs, dim=1).item())],
        },
    }


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

    resized_for_model = image.resize((32, 32))
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
    explainability = build_explainability_payload(
        model=MODEL,
        input_tensor=input_tensor,
        resized_image=resized_for_model,
        logits=logits,
        probs=probs,
        predicted_idx=idx,
        original_size=(image.width, image.height),
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
        "feature_maps": {
            "conv1": extract_feature_maps(
                MODEL.last_activations["conv1"],
                pre_activation=MODEL.debug_tensors.get("conv1_pre"),
                max_maps=12,
            ),
            "conv2": extract_feature_maps(
                MODEL.last_activations["conv2"],
                pre_activation=MODEL.debug_tensors.get("conv2_pre"),
                max_maps=12,
            ),
            "conv3": extract_feature_maps(
                MODEL.last_activations["conv3"],
                pre_activation=MODEL.debug_tensors.get("conv3_pre"),
                max_maps=12,
            ),
        },
        "explainability": explainability,
        "heatmap": heatmap_b64,
    }
    return response
