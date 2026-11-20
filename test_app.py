import os
import io
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_real_plantvillage_inference_and_latency():
    healthy_path = os.path.join("sample_leaves", "healthy_leaf.jpg")
    blight_path = os.path.join("sample_leaves", "bacterial_blight.jpg")

    with open(healthy_path, "rb") as f:
        res_h = client.post("/predict", files={"file": ("healthy.jpg", f.read(), "image/jpeg")})
    with open(blight_path, "rb") as f:
        res_b = client.post("/predict", files={"file": ("blight.jpg", f.read(), "image/jpeg")})

    assert res_h.status_code == 200
    assert res_b.status_code == 200

    dh = res_h.json()
    db = res_b.json()

    assert dh["latency_ms"] < 250.0
    assert dh["tensor_shape"] == [1, 224, 224, 3]
    assert len(dh["treatment_recommendation"]) > 20
    assert dh["spatial_features"]["healthy_chlorophyll_pct"] > db["spatial_features"]["healthy_chlorophyll_pct"]


def test_non_plant_ood_guardrail_rejection():
    """Verifies a non-green solid dark/skin-toned image is rejected as Non_Plant_Object_Detected."""
    non_plant = Image.new("RGB", (224, 224), color=(140, 100, 90))
    buf = io.BytesIO()
    non_plant.save(buf, format="JPEG")

    res = client.post("/predict", files={"file": ("selfie.jpg", buf.getvalue(), "image/jpeg")})
    assert res.status_code == 200
    assert res.json()["predicted_class"] == "Non_Plant_Object_Detected"


def test_reject_non_image_payload():
    res = client.post("/predict", files={"file": ("payload.txt", b"invalid tensor", "text/plain")})
    assert res.status_code == 400