import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import getDataloaders, getFullDataloader
from model import buildModel


def loadModel(device):
    checkpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
    classToIdx = checkpoint["class_to_idx"]
    idxToClass = {v: k for k, v in classToIdx.items()}

    model = buildModel(numClasses=len(classToIdx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classToIdx, idxToClass


def evaluate(model, loader, device):
    allPreds = []
    allLabels = []
    allConfs = []
    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with torch.inference_mode():
        for images, labels, _weights in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=useBf16):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            confs, preds = probs.max(1)

            allPreds.extend(preds.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
            allConfs.extend(confs.float().cpu().numpy())

    return np.array(allLabels), np.array(allPreds), np.array(allConfs)


def plotConfusionMatrix(labels, preds, classNames, path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(40, 36))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=classNames, yticklabels=classNames,
        ax=ax, square=True, linewidths=0.1,
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")
    return cm


def findConfusedPairs(cm, classNames, topN=20):
    pairs = []
    n = len(classNames)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((classNames[i], classNames[j], int(cm[i][j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:topN]


def perClassAccuracy(labels, preds, classNames):
    cm = confusion_matrix(labels, preds, labels=list(range(len(classNames))))
    perClass = {}
    for i, name in enumerate(classNames):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        perClass[name] = {"correct": int(correct), "total": int(total), "accuracy": round(acc * 100, 2)}
    return perClass


def main():
    fullMode = "--full" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, classToIdx, idxToClass = loadModel(device)

    if fullMode:
        loader, classToIdxLoader = getFullDataloader()
        classToIdx = classToIdxLoader
        idxToClass = {v: k for k, v in classToIdx.items()}
        print(f"Full dataset: {len(loader.dataset)} images (--full mode)")
    else:
        _, loader, _ = getDataloaders()
        print(f"Validation set: {len(loader.dataset)} images")

    print("Running inference...")
    labels, preds, confs = evaluate(model, loader, device)

    overallAcc = (labels == preds).mean() * 100
    avgConf = confs.mean() * 100
    print(f"\nOverall accuracy: {overallAcc:.2f}%")
    print(f"Average confidence: {avgConf:.2f}%")

    if fullMode:
        print("(Note: includes training data — accuracy may be slightly inflated)")

    classNames = [idxToClass[i] for i in range(len(idxToClass))]

    suffix = "_full" if fullMode else ""
    cm = plotConfusionMatrix(labels, preds, classNames, path=f"confusion_matrix{suffix}.png")

    report = classification_report(labels, preds, target_names=classNames, digits=3)
    reportPath = f"classification_report{suffix}.txt"
    with open(reportPath, "w") as f:
        f.write(report)
    print(f"Classification report saved to {reportPath}")

    confused = findConfusedPairs(cm, classNames)
    confusedPath = f"confused_pairs{suffix}.txt"
    with open(confusedPath, "w") as f:
        f.write("Top Confused Pairs (True -> Predicted: Count)\n")
        f.write("=" * 50 + "\n")
        for true, pred, count in confused:
            f.write(f"{true} -> {pred}: {count}\n")
    print(f"Confused pairs saved to {confusedPath}")

    perClass = perClassAccuracy(labels, preds, classNames)
    sortedClasses = sorted(perClass.items(), key=lambda x: x[1]["accuracy"])

    print("\n--- Worst 10 Classes ---")
    for name, stats in sortedClasses[:10]:
        print(f"  {name:20s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n--- Best 10 Classes ---")
    for name, stats in sortedClasses[-10:]:
        print(f"  {name:20s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Top 5 Confused Pairs ---")
    for true, pred, count in confused[:5]:
        print(f"  {true} -> {pred}: {count} times")


if __name__ == "__main__":
    main()
