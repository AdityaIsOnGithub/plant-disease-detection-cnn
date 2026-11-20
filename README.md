# Automated Plant Disease Detection via CNNs

End-to-end agricultural pathology diagnostics pipeline combining a **38-Class PlantVillage MobileNetV2 Deep Convolutional Network**, **Sequential Keras CNN Architecture Benchmarking (IJSRT Study)**, **Conv_1 Spatial Feature Activation Heatmaps**, and an **Out-of-Distribution (OOD) Foliage Gatekeeper** executing in **<150ms**.

## Key Features
- **38-Class Deep Plant Pathology Inference**: Identifies bacterial, fungal, viral, and pest pathologies across major crop species with real agronomic treatment protocols.
- **Sub-150ms Local Tensor Inference**: Pre-warmed PyTorch/TensorFlow tensor graphs with deterministic OOD guardrails against non-botanical inputs.
- **Conv_1 Spatial Activation & Tissue Segmentation**: Real-time neural activation overlay paired with HSV/RGB chlorophyll, necrosis, chlorosis, rust, and mycelium lamina segmentation.
- **Edge CLI & Automated Pytest Suite**: Includes `cli.py` for headless edge diagnostics and `test_app.py` enforcing schema and latency SLAs.

## Quick Start
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_and_benchmark.py --epochs 10
pytest -v
uvicorn main:app --reload