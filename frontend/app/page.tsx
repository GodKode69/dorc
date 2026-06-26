"use client";

import { useState, useCallback } from "react";
import ImageUpload from "@/components/ImageUpload";
import PredictionResult from "@/components/PredictionResult";
import { useClassifier } from "@/lib/useClassifier";

export default function Home() {
  const { prediction, loading, error, modelReady, classify, reset } =
    useClassifier();
  const [preview, setPreview] = useState<string | null>(null);

  const handleImageSelect = useCallback(
    async (file: File) => {
      setPreview(URL.createObjectURL(file));
      classify(file);
    },
    [classify]
  );

  const handleReset = useCallback(() => {
    reset();
    setPreview(null);
  }, [reset]);

  const hasResult = prediction || loading || error;

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        position: "relative",
        zIndex: 1,
      }}
    >
      {/* Top bar */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 2.5rem",
          height: "52px",
          borderBottom: "1px solid var(--border)",
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontWeight: 700,
            fontSize: "0.9rem",
            letterSpacing: "-0.02em",
          }}
        >
          DORC
        </span>
        <span
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.68rem",
            color: "var(--text-dim)",
            letterSpacing: "0.08em",
          }}
        >
          108 CLASSES &middot; EFFICIENTNET-B0 &middot;{" "}
          {modelReady ? "READY" : "LOADING MODEL..."}
        </span>
      </header>

      {/* Main */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Left: Upload */}
        <div
          style={{
            flex: hasResult ? "0 0 48%" : "1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "2rem",
            transition: "flex 0.4s cubic-bezier(0.22, 1, 0.36, 1)",
            position: "relative",
          }}
        >
          <div style={{ width: "100%", maxWidth: "420px" }}>
            {/* Label */}
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.62rem",
                color: "var(--text-dim)",
                textTransform: "uppercase",
                letterSpacing: "0.15em",
                marginBottom: "1rem",
              }}
            >
              {hasResult ? "Input" : "Classify"}
            </div>

            <ImageUpload onImageSelect={handleImageSelect} disabled={loading} />

            {preview && (
              <div
                className="fade-in"
                style={{
                  marginTop: "1rem",
                  borderRadius: "12px",
                  overflow: "hidden",
                  border: "1px solid var(--border)",
                  background: "var(--bg-surface)",
                  aspectRatio: "16/10",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <img
                  src={preview}
                  alt="Preview"
                  style={{ width: "100%", height: "100%", objectFit: "contain" }}
                />
              </div>
            )}

            {!hasResult && (
              <div
                style={{
                  marginTop: "2rem",
                  textAlign: "center",
                  fontFamily: '"Inter", sans-serif',
                  fontSize: "0.8rem",
                  color: "var(--text-dim)",
                  lineHeight: 1.6,
                }}
              >
                Drop any image to classify it into one of 108 species.
              </div>
            )}
          </div>
        </div>

        {/* Right: Results */}
        {hasResult && (
          <div
            className="slide-in"
            style={{
              flex: "0 0 52%",
              borderLeft: "1px solid var(--border)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "1.5rem 2rem",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "0.62rem",
                  color: "var(--text-dim)",
                  textTransform: "uppercase",
                  letterSpacing: "0.15em",
                }}
              >
                Output
              </span>
              {loading && (
                <div
                  style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                >
                  <div
                    style={{
                      width: 12,
                      height: 12,
                      border: "1.5px solid var(--border)",
                      borderTopColor: "var(--accent)",
                      borderRadius: "50%",
                    }}
                    className="animate-spin"
                  />
                  <span
                    style={{
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: "0.65rem",
                      color: "var(--accent)",
                    }}
                  >
                    Processing
                  </span>
                </div>
              )}
            </div>

            <div
              style={{ flex: 1, overflow: "auto", padding: "1.5rem 2rem" }}
            >
              {error && (
                <div
                  className="fade-in"
                  style={{
                    padding: "1rem 1.25rem",
                    borderRadius: "10px",
                    background: "rgba(248, 113, 113, 0.08)",
                    border: "1px solid rgba(248, 113, 113, 0.15)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.78rem",
                    color: "var(--red)",
                  }}
                >
                  {error}
                </div>
              )}
              {prediction && <PredictionResult prediction={prediction} />}

              {(prediction || error) && (
                <button
                  onClick={handleReset}
                  className="fade-in"
                  style={{
                    marginTop: "1.5rem",
                    width: "100%",
                    padding: "0.75rem 1rem",
                    borderRadius: "10px",
                    border: "1px solid var(--border)",
                    background: "var(--bg-surface)",
                    color: "var(--text)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.78rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.5rem",
                    transition: "border-color 0.2s ease, background 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                    e.currentTarget.style.background = "var(--accent-dim)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                    e.currentTarget.style.background = "var(--bg-surface)";
                  }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  Upload Another
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
