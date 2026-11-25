import inspect
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import MobileNetV2ForImageClassification

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
OUT_DIR = Path("web/public/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUT_DIR / "plant_disease_mobilenetv2.onnx"
META_PATH = OUT_DIR / "metadata.json"


class PlantDiseaseONNXWrapper(nn.Module):
    def __init__(self, base_model: MobileNetV2ForImageClassification):
        super().__init__()
        self.backbone = base_model.mobilenet_v2
        self.classifier = base_model.classifier

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(pixel_values, return_dict=True)
        features = outputs.last_hidden_state
        logits = self.classifier(outputs.pooler_output)
        weights = self.classifier.weight.unsqueeze(-1).unsqueeze(-1)
        cam = F.relu(F.conv2d(features, weights))
        return logits, cam


def load_preprocess_config():
    try:
        cfg_path = hf_hub_download(MODEL_ID, "preprocessor_config.json")
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        size = cfg.get("size", 224)
        input_size = size.get("shortest_edge", size.get("height", 224)) if isinstance(size, dict) else int(size)
        return input_size, cfg.get("image_mean", [0.5, 0.5, 0.5]), cfg.get("image_std", [0.5, 0.5, 0.5])
    except Exception:
        return 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def main():
    input_size, image_mean, image_std = load_preprocess_config()
    base_model = MobileNetV2ForImageClassification.from_pretrained(MODEL_ID).eval()
    wrapper = PlantDiseaseONNXWrapper(base_model).eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    export_kwargs = {
        "opset_version": 17,
        "input_names": ["pixel_values"],
        "output_names": ["logits", "cam"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(wrapper, (dummy,), str(ONNX_PATH), **export_kwargs)

    labels = [base_model.config.id2label[i] for i in range(base_model.config.num_labels)]
    metadata = {
        "model_id": MODEL_ID,
        "input_size": input_size,
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "num_labels": len(labels),
        "labels": labels,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported: {ONNX_PATH} ({size_mb:.2f} MB)")
    print(f"Metadata: {META_PATH} ({len(labels)} classes)")


if __name__ == "__main__":
    main()