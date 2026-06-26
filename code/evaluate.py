import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import get_dataloaders, get_full_dataloader
from model import build_model


def load_model(device):
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx, idx_to_class


def evaluate(model, loader, device):
    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_confs)


def plot_confusion_matrix(labels, preds, class_names, path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(40, 36))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True, linewidths=0.1,
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix (90 Classes)", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")
    return cm


def find_confused_pairs(cm, class_names, top_n=20):
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i][j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def per_class_accuracy(labels, preds, class_names):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    per_class = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        per_class[name] = {"correct": int(correct), "total": int(total), "accuracy": round(acc * 100, 2)}
    return per_class


def main():
    full_mode = "--full" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, class_to_idx, idx_to_class = load_model(device)

    if full_mode:
        loader, class_to_idx_loader = get_full_dataloader()
        class_to_idx = class_to_idx_loader
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Full dataset: {len(loader.dataset)} images (--full mode)")
    else:
        _, loader, _ = get_dataloaders()
        print(f"Validation set: {len(loader.dataset)} images")

    print("Running inference...")
    labels, preds, confs = evaluate(model, loader, device)

    overall_acc = (labels == preds).mean() * 100
    avg_conf = confs.mean() * 100
    print(f"\nOverall accuracy: {overall_acc:.2f}%")
    print(f"Average confidence: {avg_conf:.2f}%")

    if full_mode:
        print("(Note: includes training data — accuracy may be slightly inflated)")

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    suffix = "_full" if full_mode else ""
    cm = plot_confusion_matrix(labels, preds, class_names, path=f"confusion_matrix{suffix}.png")

    report = classification_report(labels, preds, target_names=class_names, digits=3)
    report_path = f"classification_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    confused = find_confused_pairs(cm, class_names)
    confused_path = f"confused_pairs{suffix}.txt"
    with open(confused_path, "w") as f:
        f.write("Top Confused Pairs (True -> Predicted: Count)\n")
        f.write("=" * 50 + "\n")
        for true, pred, count in confused:
            f.write(f"{true} -> {pred}: {count}\n")
    print(f"Confused pairs saved to {confused_path}")

    per_class = per_class_accuracy(labels, preds, class_names)
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])

    print("\n--- Worst 10 Classes ---")
    for name, stats in sorted_classes[:10]:
        print(f"  {name:20s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n--- Best 10 Classes ---")
    for name, stats in sorted_classes[-10:]:
        print(f"  {name:20s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Top 5 Confused Pairs ---")
    for true, pred, count in confused[:5]:
        print(f"  {true} -> {pred}: {count} times")


if __name__ == "__main__":
    main()
