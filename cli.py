import os
import io
import json
import argparse
import numpy as np
from PIL import Image
from model_engine import PlantDiseaseEngine
from train_and_benchmark import CLASSES, synthesize_pathology_leaf


def generate_sample_images(output_dir: str = "sample_leaves"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(101)
    saved_files = []
    for idx, class_name in enumerate(CLASSES):
        arr = (synthesize_pathology_leaf(idx, rng) * 255.0).astype(np.uint8)
        img = Image.fromarray(arr).resize((256, 256), Image.Resampling.BICUBIC)
        file_path = os.path.join(output_dir, f"{class_name.lower()}.jpg")
        img.save(file_path, format="JPEG", quality=95)
        saved_files.append(file_path)
    print(f"[SUCCESS] Generated {len(saved_files)} test leaf images in ./{output_dir}/:")
    for f in saved_files:
        print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description="Automated Plant Disease Detection Edge CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to leaf image file to diagnose")
    parser.add_argument("--generate-samples", action="store_true", help="Generate 5 test leaf images in ./sample_leaves/")
    args = parser.parse_args()

    if args.generate_samples:
        generate_sample_images()
        if not args.image:
            return

    engine = PlantDiseaseEngine()

    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        with open(args.image, "rb") as f:
            img_bytes = f.read()
        source_label = args.image
    else:
        generate_sample_images()
        default_sample = os.path.join("sample_leaves", "bacterial_blight.jpg")
        with open(default_sample, "rb") as f:
            img_bytes = f.read()
        source_label = default_sample

    result = engine.predict(img_bytes)
    cli_output = {k: v for k, v in result.items() if k != "heatmap_base64"}
    cli_output["source_image"] = source_label
    print("\n=== Automated Plant Disease CNN Diagnostics Report ===")
    print(json.dumps(cli_output, indent=2))


if __name__ == "__main__":
    main()