import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from model import buildModel


def exportOnnx():
    device = torch.device("cpu")

    checkpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
    classToIdx = checkpoint["class_to_idx"]
    numClasses = len(classToIdx)

    stateDict = checkpoint["model_state_dict"]
    stateDict = {k.replace("_orig_mod.", ""): v for k, v in stateDict.items()}

    model = buildModel(numClasses=numClasses).to(device)
    model.load_state_dict(stateDict)
    model.eval()

    dummy = torch.randn(1, 3, config.imgSize, config.imgSize)

    onnxPath = config.basePath / "frontend" / "public" / "model.onnx"
    onnxPath.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnxPath),
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    idxToClass = {v: k for k, v in classToIdx.items()}
    classNames = [idxToClass[i] for i in range(numClasses)]

    classesPath = config.basePath / "frontend" / "public" / "classes.json"
    with open(classesPath, "w") as f:
        json.dump(classNames, f)

    sizeMb = onnxPath.stat().st_size / (1024 * 1024)
    print(f"Exported: {onnxPath} ({sizeMb:.1f} MB)")
    print(f"Classes: {classesPath} ({numClasses} classes)")
    print(f"Input shape: (1, 3, {config.imgSize}, {config.imgSize})")


if __name__ == "__main__":
    exportOnnx()
