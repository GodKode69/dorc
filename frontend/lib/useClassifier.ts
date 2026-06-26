"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/";

interface Prediction {
  class: string;
  confidence: number;
  top5: { class: string; confidence: number }[];
}

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const IMG_SIZE = 224;

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let classNamesPromise: Promise<string[]> | null = null;

function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create("/model.onnx", {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

function getClassNames(): Promise<string[]> {
  if (!classNamesPromise) {
    classNamesPromise = fetch("/classes.json").then((r) => r.json());
  }
  return classNamesPromise;
}

function preprocessImage(imageData: ImageData): ort.Tensor {
  const { width, height, data } = imageData;

  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d")!;

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext("2d")!;
  srcCtx.putImageData(imageData, 0, 0);

  const scale = Math.max(width, height) / IMG_SIZE;
  const sw = IMG_SIZE * scale;
  const sh = IMG_SIZE * scale;
  const sx = (width - sw) / 2;
  const sy = (height - sh) / 2;
  ctx.drawImage(srcCanvas, sx, sy, sw, sh, 0, 0, IMG_SIZE, IMG_SIZE);

  const resized = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const pixels = resized.data;

  const nchw = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
  for (let h = 0; h < IMG_SIZE; h++) {
    for (let w = 0; w < IMG_SIZE; w++) {
      const idx = (h * IMG_SIZE + w) * 4;
      for (let c = 0; c < 3; c++) {
        const pixel = pixels[idx + c] / 255.0;
        nchw[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] =
          (pixel - MEAN[c]) / STD[c];
      }
    }
  }

  return new ort.Tensor("float32", nchw, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

async function loadImageToImageData(file: File): Promise<ImageData> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = reject;
    });

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function useClassifier() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    Promise.all([getSession(), getClassNames()]).then(() => setModelReady(true));
  }, []);

  const classify = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const imageData = await loadImageToImageData(file);
      const tensor = preprocessImage(imageData);

      const [session, classNames] = await Promise.all([
        getSession(),
        getClassNames(),
      ]);

      const inputName = session.inputNames[0];
      const results = await session.run({ [inputName]: tensor });
      const outputName = session.outputNames[0];
      const logits = results[outputName].data as Float32Array;

      const probs = softmax(logits);
      const indexed: { prob: number; idx: number }[] = [];
      for (let i = 0; i < probs.length; i++) {
        indexed.push({ prob: probs[i], idx: i });
      }
      indexed.sort((a, b) => b.prob - a.prob);

      const top5 = indexed.slice(0, 5).map((item) => ({
        class: classNames[item.idx],
        confidence: Math.round(item.prob * 10000) / 100,
      }));

      setPrediction({
        class: top5[0].class,
        confidence: top5[0].confidence,
        top5,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setPrediction(null);
    setError(null);
  }, []);

  return { prediction, loading, error, modelReady, classify, reset };
}
