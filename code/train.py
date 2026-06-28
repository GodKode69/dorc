import time
import sys
import os
import warnings
warnings.filterwarnings("ignore", message="Palette images")

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


def trainOneEpoch(model, loader, criterion, optimizer, device, useBf16):
    model.train()
    runningLoss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if images.dim() == 4:
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=useBf16):
            outputs = model(images)
            loss = criterion(outputs, labels)
            del outputs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradientClipNorm)
        optimizer.step()

        runningLoss += loss.detach() * images.size(0)
        total += labels.size(0)

    return runningLoss.item() / total


def validate(model, loader, criterion, device, useBf16):
    model.eval()
    runningLoss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=useBf16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            runningLoss += loss * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return runningLoss.item() / total, 100.0 * correct / total


def saveCheckpoint(model, optimizer, scheduler, classToIdx, epoch, bestValAcc, patienceCounter):
    rawState = model.state_dict()
    cleanState = {k.replace("_orig_mod.", ""): v for k, v in rawState.items()}
    torch.save({
        "model_state_dict": cleanState,
        "class_to_idx": classToIdx,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_acc": bestValAcc,
        "patience_counter": patienceCounter,
    }, config.modelSavePath)


def main():
    incremental = "--increment" in sys.argv
    resume = "--resume" in sys.argv
    trainingStart = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    useBf16 = False
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if torch.cuda.is_bf16_supported():
            useBf16 = True
            print("AMP enabled (bfloat16)")

        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.shape_padding = True
        torch._inductor.config.triton.cudagraphs = True

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if incremental:
        config.batchSize = incrementBatchSize
        print(f"Incremental batch size: {incrementBatchSize}")

    trainLoader, valLoader, classToIdx = getDataloaders()
    numClasses = len(classToIdx)
    print(f"Classes: {numClasses}")
    print(f"Train: {len(trainLoader.dataset)} images, Val: {len(valLoader.dataset)} images")

    trainTargets = [trainLoader.dataset.samples[i][1] for i in trainLoader.dataset.indices]
    classCounts = Counter(trainTargets)
    classWeights = torch.tensor(
        [len(trainTargets) / (numClasses * classCounts[i]) for i in range(numClasses)],
        dtype=torch.float,
    )

    model = buildModel(numClasses=numClasses).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    startEpoch = 1
    bestValAcc = 0.0
    patienceCounter = 0
    resumeCheckpoint = None

    if resume:
        resumeCheckpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
        model.load_state_dict(resumeCheckpoint["model_state_dict"])
        startEpoch = resumeCheckpoint.get("epoch", 0) + 1
        bestValAcc = resumeCheckpoint.get("best_val_acc", 0.0)
        patienceCounter = resumeCheckpoint.get("patience_counter", 0)
        print(f"Resumed from epoch {resumeCheckpoint.get('epoch', 0)} (best_val_acc: {bestValAcc:.2f}%, patience: {patienceCounter})")

    if incremental:
        loadBackboneWeights(model, config.modelSavePath)
        numEpochs = incrementEpochs
        currentLr = incrementLr
    else:
        numEpochs = config.epochs
        currentLr = config.lr

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (reduce-overhead)")
    except Exception as e:
        print(f"torch.compile failed, continuing without: {e}")

    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainableParams:,} trainable / {totalParams:,} total")

    criterion = nn.CrossEntropyLoss(weight=classWeights.to(device), label_smoothing=config.labelSmoothing)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=currentLr, weight_decay=config.weightDecay, fused=True)

    schedulerWarmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config.warmupEpochs)
    schedulerCosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs - config.warmupEpochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [schedulerWarmup, schedulerCosine], milestones=[config.warmupEpochs])

    if resumeCheckpoint is not None:
        if "optimizer_state_dict" in resumeCheckpoint:
            optimizer.load_state_dict(resumeCheckpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resumeCheckpoint:
            scheduler.load_state_dict(resumeCheckpoint["scheduler_state_dict"])

    for epoch in range(startEpoch, numEpochs + 1):
        start = time.time()

        trainLoss = trainOneEpoch(model, trainLoader, criterion, optimizer, device, useBf16)
        valLoss, valAcc = validate(model, valLoader, criterion, device, useBf16)
        scheduler.step()

        elapsed = time.time() - start
        lrNow = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{numEpochs}] "
            f"Train Loss: {trainLoss:.4f} | "
            f"Val Loss: {valLoss:.4f} Acc: {valAcc:.2f}% | "
            f"LR: {lrNow:.6f} | Time: {elapsed:.1f}s"
        )

        if valAcc > bestValAcc:
            bestValAcc = valAcc
            patienceCounter = 0
            saveCheckpoint(model, optimizer, scheduler, classToIdx, epoch, bestValAcc, patienceCounter)
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
