# 🌿 PhytoScan Edge — Automated Plant Disease Detection via CNNs

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-0f3d2e?style=for-the-badge&logo=vercel&logoColor=white)](https://plant-disease-detection-cnn-three.vercel.app/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime-WASM_Edge-18181b?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![React + Vite](https://img.shields.io/badge/React_18-Vite_+_Tailwind_v4-0f766e?style=for-the-badge&logo=react&logoColor=white)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_+_Pytest-059669?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A dual-architecture plant pathology platform combining a **zero-server, client-side ONNX Runtime Web application** (sub-50ms browser inference with live neural CAM lesion heatmaps) and a **production FastAPI + Pytest microservice** trained across **38 PlantVillage crop disease classes**.

🔗 **Live Edge Demo:** [https://plant-disease-detection-cnn-three.vercel.app/](https://plant-disease-detection-cnn-three.vercel.app/)

---

## ✨ Key Capabilities

- **Client-Side Edge Inference (ONNX Runtime Web):** Runs an **8.97 MB MobileNetV2** graph (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) directly in the browser via WebAssembly — zero server cold starts, ~39 ms local latency, and 100% image privacy.
- **Real-Time Neural CAM Attribution:** Extracts convolutional Class Activation Maps directly from the exported ONNX graph and renders an adjustable thermal heatmap overlay aligned 1:1 with center-cropped leaf specimens.
- **OOD Botanical Gatekeeper:** Combines pixel-level chlorophyll/autumn foliage heuristics with softmax confidence thresholds to flag non-foliage or low-confidence uploads.
- **Pathogen-Aware Agronomic Protocols & PDF Export:** Routes bacterial, fungal, viral, oomycete, and pest detections to tailored treatment protocols with one-click printable PDF diagnostic reports and `Ctrl+V` clipboard image support.
- **Production Python Backend & Export Pipeline:** Includes a FastAPI REST API (`main.py`), CLI diagnostic tool (`cli.py`), automated Pytest suite (`test_app.py`), and a reproducible PyTorch-to-ONNX exporter (`export_onnx.py`).

---

## 🏗 Architecture & Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Deep Learning & Export** | PyTorch, Hugging Face Transformers (MobileNetV2, 38-Class PlantVillage), ONNX, Keras/TensorFlow |
| **Edge Web Frontend (`web/`)** | React 18, Vite, Tailwind CSS v4, `onnxruntime-web` (WASM), HTML5 Canvas CAM Renderer |
| **Backend Microservice (Root)** | FastAPI, Uvicorn, Pydantic, Pytest, CLI (`cli.py`), Docker |

---

## 🚀 Quick Start

### 1. Run the Edge Web App (`web/`)
```bash
git clone [https://github.com/AdityaIsOnGithub/plant-disease-detection-cnn.git](https://github.com/AdityaIsOnGithub/plant-disease-detection-cnn.git)
cd plant-disease-detection-cnn/web
npm install
npm run dev
```

### 2. Re-Export the ONNX Model (Optional)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt onnx==1.17.0
python export_onnx.py
```

### 3. Run the FastAPI Microservice & Pytest Suite
```bash
pytest
uvicorn main:app --reload
```

---

## 📂 Repository Structure

```text
plant-disease-detection-cnn/
├── web/                       # Client-side React + ONNX Runtime Web app (Vercel root)
│   ├── public/models/         # Exported 8.97 MB MobileNetV2 .onnx + metadata.json
│   ├── public/samples/        # 1-click leaf test bench specimens
│   └── src/App.jsx            # ONNX inference engine, CAM renderer & diagnostic UI
├── export_onnx.py             # PyTorch -> ONNX exporter with embedded CAM output head
├── main.py                    # FastAPI REST microservice
├── model_engine.py            # Python inference & Grad-CAM engine
├── cli.py                     # Local CLI diagnostic tool
├── test_app.py                # Automated Pytest validation suite
├── train_and_benchmark.py     # Training & evaluation benchmark script
└── models/                    # Benchmark reports & local model artifacts
```

---

## 👤 Author

**Aditya Tiwari** — [GitHub](https://github.com/AdityaIsOnGithub)