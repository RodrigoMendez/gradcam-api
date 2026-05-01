import io
import os
import base64

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

app = FastAPI()

# CORS
origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model and labels at startup
weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
model = squeezenet1_1(weights=weights)
model.eval()
class_names = weights.meta["categories"]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(227),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Target layer for Grad-CAM — last Fire module
target_layer = model.features[-1]


def compute_gradcam(input_tensor, class_idx):
    """Compute Grad-CAM heatmap using PyTorch hooks."""
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        inp = input_tensor.clone().requires_grad_(True)
        output = model(inp)
        model.zero_grad()
        output[0, class_idx].backward()

        act = activations[0]
        grad = gradients[0]

        # Global average pooling of gradients -> channel weights
        w = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = (w * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # Normalize to 0-1
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.numpy()
    finally:
        fh.remove()
        bh.remove()


def apply_heatmap(rgb_img, cam, alpha=0.5):
    """Overlay jet colormap heatmap on RGB image (both as numpy arrays)."""
    # Resize cam to image size
    h, w = rgb_img.shape[:2]
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))) / 255.0

    # Create jet colormap manually
    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    # Blue to cyan to green to yellow to red
    heatmap[:, :, 0] = np.clip(1.5 - abs(cam_resized * 4 - 3), 0, 1)  # R
    heatmap[:, :, 1] = np.clip(1.5 - abs(cam_resized * 4 - 2), 0, 1)  # G
    heatmap[:, :, 2] = np.clip(1.5 - abs(cam_resized * 4 - 1), 0, 1)  # B

    overlay = rgb_img * (1 - alpha) + heatmap * alpha
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    contents = await image.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(413, "Imagen demasiado grande (max 5 MB)")

    try:
        img = Image.open(io.BytesIO(contents))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "No se pudo procesar la imagen")
    del contents

    # Resize early to save memory (phone photos can be 12MP+)
    img = img.resize((227, 227), Image.LANCZOS)
    rgb_img = np.array(img).astype(np.float32) / 255.0

    # Preprocess for model
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = normalize(img).unsqueeze(0)

    # Forward pass — top 5 predictions
    with torch.no_grad():
        output = model(input_tensor)
    probs = F.softmax(output, dim=1)[0]
    top5_scores, top5_indices = torch.topk(probs, 5)

    predictions = [
        {"label": class_names[idx.item()], "score": round(score.item(), 4)}
        for score, idx in zip(top5_scores, top5_indices)
    ]

    # Grad-CAM (needs gradients, so separate from the no_grad block)
    top1_idx = top5_indices[0].item()
    input_cam = normalize(img).unsqueeze(0)
    cam = compute_gradcam(input_cam, top1_idx)

    # Create overlay
    overlay = apply_heatmap(rgb_img, cam)

    # Encode overlay as base64 PNG
    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "predictions": predictions,
        "overlay": f"data:image/png;base64,{b64}",
    }
