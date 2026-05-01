import io
import os
import base64

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

# GradCAM setup — last Fire module's expand3x3 conv
target_layers = [model.features[-1]]


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
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "No se pudo procesar la imagen")

    # Prepare image for display (227x227 RGB normalized to 0-1)
    img_resized = img.resize((227, 227))
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0

    # Preprocess for model
    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass — top 5 predictions
    with torch.no_grad():
        output = model(input_tensor)
    probs = F.softmax(output, dim=1)[0]
    top5_scores, top5_indices = torch.topk(probs, 5)

    predictions = [
        {"label": class_names[idx.item()], "score": round(score.item(), 4)}
        for score, idx in zip(top5_scores, top5_indices)
    ]

    # Grad-CAM
    top1_idx = top5_indices[0].item()
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(top1_idx)])[0]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Encode overlay as base64 PNG
    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "predictions": predictions,
        "overlay": f"data:image/png;base64,{b64}",
    }
