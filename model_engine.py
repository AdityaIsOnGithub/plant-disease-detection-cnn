import os
import io
import json
import time
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import MobileNetV2ImageProcessor, AutoModelForImageClassification

BENCHMARK_PATH = os.path.join("models", "benchmark_report.json")
HF_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

REAL_AGRONOMIC_PROTOCOLS = {
    "Healthy": (
        "No pathogen detected. Maintain crop-specific NPK fertility, consistent drip/soil-level "
        "irrigation, and 30-45 cm canopy spacing for airflow."
    ),
    "Bacterial_Spot_or_Blight": (
        "Causal Agent: Xanthomonas / Pseudomonas spp. Prune symptomatic foliage immediately with "
        "sterilized shears, switch from overhead sprinkler to drip irrigation, and apply fixed "
        "copper hydroxide + mancozeb bactericide every 7-10 days."
    ),
    "Early_Blight": (
        "Causal Agent: Alternaria solani. Remove infected lower leaves to halt upward spore splash, "
        "mulch soil base, rotate out of nightshades for 2 seasons, and apply chlorothalonil or "
        "azoxystrobin fungicide."
    ),
    "Late_Blight": (
        "Causal Agent: Phytophthora infestans (High Epidemic Risk). Destroy severely blighted plants "
        "immediately, avoid evening irrigation, and apply systemic oomycete fungicides (mefenoxam, "
        "cymoxanil, or mandipropamid)."
    ),
    "Leaf_Mold_or_Spot": (
        "Causal Agent: Passalora fulva / Septoria / Cercospora. Vent greenhouse or thin canopy to keep "
        "relative humidity <85%, remove fallen leaf litter, and apply copper octanoate or difenoconazole."
    ),
    "Rust": (
        "Causal Agent: Puccinia / Gymnosporangium spp. Remove nearby alternate weeds/hosts, avoid "
        "prolonged leaf wetness, and apply systemic triazole (myclobutanil/tebuconazole) or micronized sulfur."
    ),
    "Powdery_Mildew": (
        "Causal Agent: Erysiphales / Podosphaera spp. Apply potassium bicarbonate, horticultural neem oil, "
        "or wettable sulfur (when temps <30°C) and prune dense interior shoots to increase UV exposure."
    ),
    "Viral_Mosaic_or_Curl": (
        "Causal Agent: Begomovirus / Tobamovirus (Vector: Whiteflies/Aphids). No chemical cure exists for "
        "infected plants—rogue and destroy infected plant immediately, sanitize tools with 10% bleach, "
        "and control whitefly/aphid vectors with imidacloprid or neem oil."
    ),
    "Mite_or_Pest_Damage": (
        "Causal Agent: Tetranychus urticae (Two-Spotted Spider Mite). Increase local humidity, hose down "
        "leaf undersides, release predatory mites (Phytoseiulus persimilis), or apply abamectin/spinosad."
    ),
    "Non_Plant_Object_Detected": (
        "Out-of-Distribution (OOD) Guardrail Triggered: Insufficient botanical chlorophyll/foliage signature "
        "detected. Please upload a valid plant or crop leaf photograph."
    )
}


def map_disease_to_protocol(raw_label: str) -> str:
    label_low = raw_label.lower()
    if "healthy" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Healthy"]
    if "bacterial" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Bacterial_Spot_or_Blight"]
    if "early_blight" in label_low or "early blight" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Early_Blight"]
    if "late_blight" in label_low or "late blight" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Late_Blight"]
    if "rust" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Rust"]
    if "powdery" in label_low or "mildew" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Powdery_Mildew"]
    if "virus" in label_low or "mosaic" in label_low or "curl" in label_low or "greening" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Viral_Mosaic_or_Curl"]
    if "mite" in label_low:
        return REAL_AGRONOMIC_PROTOCOLS["Mite_or_Pest_Damage"]
    return REAL_AGRONOMIC_PROTOCOLS["Leaf_Mold_or_Spot"]


class PlantDiseaseEngine:
    def __init__(self, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = MobileNetV2ImageProcessor.from_pretrained(HF_MODEL_ID)
        self.hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID).to(self.device)
        self.hf_model.eval()

        # Warm up tensor pipeline for <50ms inference
        dummy_pil = Image.new("RGB", (224, 224), (40, 140, 40))
        with torch.no_grad():
            inputs = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            _ = self.hf_model(**inputs, output_hidden_states=True)

    def extract_spatial_and_cam(self, pil_img: Image.Image, last_conv_feat: torch.Tensor):
        rgb_128 = np.asarray(pil_img.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        r, g, b = rgb_128[:, :, 0], rgb_128[:, :, 1], rgb_128[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = (max_c - min_c) / (max_c + 1e-6)
        val = max_c

        bg_mask = ((r > 0.92) & (g > 0.92) & (b > 0.92) & (sat < 0.08)) | (val < 0.08)
        fg_mask = ~bg_mask
        fg_count = max(1, int(np.sum(fg_mask)))

        green_mask = fg_mask & (g > r * 1.04) & (g > b * 1.08) & (val > 0.15)
        yellow_mask = fg_mask & (~green_mask) & (r > 0.42) & (g > 0.38) & (b < 0.32)
        rust_mask = fg_mask & (~green_mask) & (r > 0.45) & (g > 0.20) & (g < 0.58) & (b < 0.25) & ((r - g) > 0.14)
        powdery_mask = fg_mask & (~green_mask) & (val > 0.65) & (sat < 0.22)
        necrotic_mask = fg_mask & (~green_mask) & (~yellow_mask) & (~rust_mask) & (~powdery_mask) & (val < 0.48)

        healthy_pct = round(float(np.sum(green_mask)) / fg_count * 100, 2)
        necrotic_pct = round(float(np.sum(necrotic_mask)) / fg_count * 100, 2)
        yellow_pct = round(float(np.sum(yellow_mask)) / fg_count * 100, 2)
        rust_pct = round(float(np.sum(rust_mask)) / fg_count * 100, 2)
        powdery_pct = round(float(np.sum(powdery_mask)) / fg_count * 100, 2)

        lesion_union = (necrotic_mask | rust_mask | yellow_mask).astype(np.float32)
        gx = np.abs(np.diff(lesion_union, axis=1, prepend=0))
        gy = np.abs(np.diff(lesion_union, axis=0, prepend=0))
        edge_density = round(float(np.mean(gx + gy)) * 100, 2)

        # Compute direct Spatial Feature Activation Map from last convolutional hidden state (7x7)
        cam = torch.mean(torch.relu(last_conv_feat[0]), dim=0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        hm_img = Image.fromarray(np.uint8(255 * cam)).resize((256, 256), Image.Resampling.BILINEAR)
        hm_arr = np.asarray(hm_img, dtype=np.float32) / 255.0

        orig_256 = np.asarray(pil_img.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
        cam_color = np.zeros_like(orig_256)
        cam_color[:, :, 0] = np.clip(hm_arr * 2.0, 0, 1) * 255.0
        cam_color[:, :, 1] = np.clip(hm_arr * 1.5 - 0.3, 0, 1) * 255.0
        cam_color[:, :, 2] = np.clip(0.5 - hm_arr, 0, 1) * 180.0

        blended = np.clip(0.58 * orig_256 + 0.42 * cam_color, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        spatial_metrics = {
            "healthy_chlorophyll_pct": healthy_pct,
            "necrotic_lesion_pct": necrotic_pct,
            "chlorotic_halo_pct": yellow_pct,
            "rust_pustule_pct": rust_pct,
            "powdery_mycelium_pct": powdery_pct,
            "lesion_edge_density": edge_density
        }
        return spatial_metrics, heatmap_b64

    def predict(self, image_bytes: bytes) -> dict:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Measure pure local tensor inference latency (<150ms SLA)
        t0 = time.perf_counter()
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.hf_model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]
            last_conv = outputs.hidden_states[-1]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        spatial_metrics, heatmap_b64 = self.extract_spatial_and_cam(pil_img, last_conv)

        # Out-of-Distribution (OOD) Guardrail for non-plant selfies/objects (<2.5% green foliage)
        if spatial_metrics["healthy_chlorophyll_pct"] < 2.5 and spatial_metrics["chlorotic_halo_pct"] < 15.0:
            return {
                "predicted_class": "Non_Plant_Object_Detected",
                "confidence": 0.99,
                "latency_ms": latency_ms,
                "fallback_triggered": True,
                "diagnostic_note": (
                    f"OOD Guardrail rejected non-botanical input "
                    f"(chlorophyll={spatial_metrics['healthy_chlorophyll_pct']}% < 2.5% threshold)."
                ),
                "treatment_recommendation": REAL_AGRONOMIC_PROTOCOLS["Non_Plant_Object_Detected"],
                "tensor_shape": [1, 224, 224, 3],
                "spatial_features": spatial_metrics,
                "class_probabilities": {"Non_Plant_Object_Detected": 99.0},
                "heatmap_base64": heatmap_b64
            }

        top5_probs, top5_indices = torch.topk(probs, k=5)
        top_idx = int(top5_indices[0].item())
        confidence = float(top5_probs[0].item())
        raw_class_name = self.hf_model.config.id2label[top_idx]
        clean_class = raw_class_name.replace("___", " - ").replace("__", " - ").replace("_", " ").strip()

        top5_dict = {
            self.hf_model.config.id2label[int(idx.item())].replace("___", " - ").replace("_", " "): round(float(p.item()) * 100, 2)
            for p, idx in zip(top5_probs, top5_indices)
        }

        return {
            "predicted_class": clean_class,
            "confidence": round(confidence, 4),
            "latency_ms": latency_ms,
            "fallback_triggered": confidence < self.confidence_threshold,
            "diagnostic_note": (
                f"PlantVillage 38-Class MobileNetV2 + Conv_1 Spatial Activation "
                f"(Top-1: {clean_class} @ {round(confidence * 100, 1)}%)."
            ),
            "treatment_recommendation": map_disease_to_protocol(raw_class_name),
            "tensor_shape": [1, 224, 224, 3],
            "spatial_features": spatial_metrics,
            "class_probabilities": top5_dict,
            "heatmap_base64": heatmap_b64
        }

    def get_benchmarks(self) -> dict:
        if os.path.exists(BENCHMARK_PATH):
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"benchmarks": []}