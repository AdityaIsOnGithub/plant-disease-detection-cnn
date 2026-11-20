# Automated Plant Disease Detection via CNNs

An end-to-end agricultural pathology diagnostics pipeline combining a **38-Class PlantVillage MobileNetV2 Deep Convolutional Network**, **Conv_1 Spatial Feature Activation Heatmaps**, **Out-of-Distribution (OOD) Foliage Gatekeeping**, and an optimized FastAPI backend executing in **<150ms**.

## Architecture & Tech Stack
- **Deep Learning**: PyTorch, TensorFlow / Keras, Hugging Face Transformers (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`)
- **Backend API & Testing**: FastAPI, Uvicorn, Pydantic, Pytest
- **Frontend**: Tailwind CSS, Space Grotesk / JetBrains Mono, Grad-CAM neural activation overlays
- **Containerization**: Docker (WSL 2 backend)

## Key Features
- **38-Class PlantVillage Inference**: Identifies bacterial, fungal, viral, and pest pathologies across major crop species with agronomic treatment protocols.
- **OOD Deep Vision Gatekeeper**: Rejects non-botanical inputs using ImageNet classification and chlorophyll density thresholds.
- **Real-Time Grad-CAM Attribution**: Extracts neural activation heatmaps directly from convolutional feature maps to highlight lesion focus areas.
- **Sub-150ms Edge Latency SLA**: Pre-warmed tensor graphs optimized for real-time local inference.

## Quick Start

```bash
git clone https://github.com/AdityaIsOnGithub/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt

pytest -v
uvicorn main:app --reload