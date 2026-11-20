import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from model_engine import PlantDiseaseEngine
from cli import generate_sample_images

app = FastAPI(
    title="Automated Plant Disease Detection CNN API",
    description="Sub-150ms Keras CNN & Spatial Feature Extraction Pipeline",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PlantDiseaseEngine()
generate_sample_images("sample_leaves")


class SpatialFeaturesSchema(BaseModel):
    healthy_chlorophyll_pct: float
    necrotic_lesion_pct: float
    chlorotic_halo_pct: float
    rust_pustule_pct: float
    powdery_mycelium_pct: float
    lesion_edge_density: float


class DiagnosticResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted plant leaf pathology class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., gt=0.0, description="Local tensor inference latency in milliseconds")
    fallback_triggered: bool
    diagnostic_note: str
    treatment_recommendation: str
    tensor_shape: List[int]
    spatial_features: SpatialFeaturesSchema
    class_probabilities: Dict[str, float]
    heatmap_base64: str


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "architecture": "Edge_Sequential_CNN (Keras/TensorFlow)"
    }


@app.get("/benchmarks")
async def get_architecture_benchmarks():
    return engine.get_benchmarks()


@app.get("/samples/{filename}")
async def get_sample_leaf(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join("sample_leaves", safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/predict", response_model=DiagnosticResponse)
async def predict_leaf_disease(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file (JPEG/PNG).")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image payload is empty.")

    try:
        result = engine.predict(image_bytes)
        return DiagnosticResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process image tensor: {str(exc)}")