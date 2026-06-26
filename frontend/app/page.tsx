"use client";

import { useState, useRef, useCallback } from "react";
import ImageUpload from "@/components/ImageUpload";
import PredictionResult from "@/components/PredictionResult";

interface Prediction {
  class: string;
  confidence: number;
  top5: { class: string; confidence: number }[];
}

export default function Home() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleImageSelect = useCallback(async (file: File) => {
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("https://api.godkode.xyz/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">DORC</h1>
          <p className="text-lg text-gray-600">
            Animal Image Classifier - 108 Species
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Powered by EfficientNet-B0 with ONNX Runtime
          </p>
        </header>

        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <ImageUpload onImageSelect={handleImageSelect} disabled={loading} />

          {preview && (
            <div className="mt-6 flex justify-center">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 rounded-lg shadow-md"
              />
            </div>
          )}
        </div>

        {loading && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              <span className="ml-3 text-gray-600">Analyzing image...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-6 mb-8">
            <p className="text-red-600 text-center">{error}</p>
          </div>
        )}

        {prediction && <PredictionResult prediction={prediction} />}

        <footer className="text-center text-sm text-gray-500 mt-12">
          <p>Model: EfficientNet-B0 | Classes: 108 | Input: 224x224 RGB</p>
        </footer>
      </div>
    </main>
  );
}
