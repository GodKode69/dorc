import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from model import build_model


def export_onnx():
    device = torch.device("cpu")

    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    onnx_path = config.BASE_DIR / "api" / "model.onnx"
    onnx_path.parent.mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(num_classes)]

    classes_path = config.BASE_DIR / "api" / "classes.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {onnx_path} ({size_mb:.1f} MB)")
    print(f"Classes: {classes_path} ({num_classes} classes)")
    print(f"Input shape: (1, 3, {config.IMG_SIZE}, {config.IMG_SIZE})")


if __name__ == "__main__":
    export_onnx()
