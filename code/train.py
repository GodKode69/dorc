import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import config
from dataset import getDataloaders
from model import buildModel, loadBackboneWeights

incrementLr = 0.0001
incrementEpochs = 15
incrementBatchSize = 32


def trainOneEpoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    runningLoss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if images.dim() == 4:
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        runningLoss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return runningLoss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    runningLoss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)
            outputs = model(images)
            loss = criterion(outputs, labels)

            runningLoss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return runningLoss / total, 100.0 * correct / total


def main():
    incremental = "--increment" in sys.argv
    trainingStart = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    if incremental:
        config.batchSize = incrementBatchSize
        print(f"Incremental batch size: {incrementBatchSize}")

    trainLoader, valLoader, classToIdx = getDataloaders()
    numClasses = len(classToIdx)
    print(f"Classes: {numClasses}")
    print(f"Train: {len(trainLoader.dataset)} images, Val: {len(valLoader.dataset)} images")

    trainTargets = [trainLoader.dataset.dataset.samples[i][1] for i in trainLoader.dataset.indices]
    classCounts = Counter(trainTargets)
    classWeights = torch.tensor(
        [len(trainTargets) / (numClasses * classCounts[i]) for i in range(numClasses)],
        dtype=torch.float,
    )

    model = buildModel(numClasses=numClasses).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if incremental:
        loadBackboneWeights(model, config.modelSavePath)
        numEpochs = incrementEpochs
        currentLr = incrementLr
    else:
        numEpochs = config.epochs
        currentLr = config.lr

    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile failed, continuing without: {e}")

    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainableParams:,} trainable / {totalParams:,} total")

    criterion = nn.CrossEntropyLoss(weight=classWeights.to(device), label_smoothing=config.labelSmoothing)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=currentLr, weight_decay=config.weightDecay, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("AMP enabled")

    bestValAcc = 0.0
    patienceCounter = 0

    for epoch in range(1, numEpochs + 1):
        start = time.time()

        trainLoss, trainAcc = trainOneEpoch(model, trainLoader, criterion, optimizer, device, scaler)
        valLoss, valAcc = validate(model, valLoader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lrNow = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{numEpochs}] "
            f"Train Loss: {trainLoss:.4f} Acc: {trainAcc:.2f}% | "
            f"Val Loss: {valLoss:.4f} Acc: {valAcc:.2f}% | "
            f"LR: {lrNow:.6f} | Time: {elapsed:.1f}s"
        )

        if valAcc > bestValAcc:
            bestValAcc = valAcc
            patienceCounter = 0
            rawState = model.state_dict()
            cleanState = {k.replace("_orig_mod.", ""): v for k, v in rawState.items()}
            torch.save({
                "model_state_dict": cleanState,
                "class_to_idx": classToIdx,
            }, config.modelSavePath)
            print(f"  -> Saved best model (val_acc: {valAcc:.2f}%)")
        else:
            patienceCounter += 1
            if patienceCounter >= config.earlyStopPatience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.earlyStopPatience} epochs)")
                break

    totalTime = time.time() - trainingStart
    hours, remainder = divmod(totalTime, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining complete. Best val accuracy: {bestValAcc:.2f}%")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Model saved to {config.modelSavePath}")


if __name__ == "__main__":
    main()
