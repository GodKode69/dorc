import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from model import build_model, load_backbone_weights

INCREMENT_LR = 0.0001
INCREMENT_EPOCHS = 15
INCREMENT_BATCH_SIZE = 32


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    incremental = "--increment" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if incremental:
        config.BATCH_SIZE = INCREMENT_BATCH_SIZE
        print(f"Incremental batch size: {INCREMENT_BATCH_SIZE}")

    train_loader, val_loader, class_to_idx = get_dataloaders()
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    model = build_model(num_classes=num_classes).to(device)

    if incremental:
        load_backbone_weights(model, config.MODEL_SAVE_PATH)
        epochs = INCREMENT_EPOCHS
        lr = INCREMENT_LR
    else:
        epochs = config.EPOCHS
        lr = config.LR

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {lr_now:.6f} | Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
            }, config.MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
