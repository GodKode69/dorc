"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import ImageUpload, { type ImageUploadHandle } from "@/components/ImageUpload";
import PredictionResult from "@/components/PredictionResult";
import { useClassifier } from "@/lib/useClassifier";

interface ClassData {
  name: string;
  image: string;
}

export default function Home() {
  const { prediction, error, status, progress, statusLabel, isLoading, classify, reset } =
    useClassifier();
  const [preview, setPreview] = useState<string | null>(null);
  const [view, setView] = useState<"classify" | "classes">("classify");
  const [classList, setClassList] = useState<ClassData[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/class_data.json").then((r) => r.json()).then(setClassList);
  }, []);

  const handleImageSelect = useCallback(
    async (file: File) => {
      setPreview(URL.createObjectURL(file));
      classify(file);
    },
    [classify]
  );

  const handleReset = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const hasResult = prediction || error;

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
      {/* Hidden file input for "Upload Another" */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleImageSelect(file);
          e.target.value = "";
        }}
        style={{ display: "none" }}
      />
      {/* Hidden camera input for "Click Another" */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleImageSelect(file);
          e.target.value = "";
        }}
        style={{ display: "none" }}
      />

      {/* Dock navbars */}
      <div
        style={{
          position: "fixed",
          top: "16px",
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "space-between",
          padding: "0 24px",
          zIndex: 10,
          pointerEvents: "none",
        }}
      >
        {/* Left dock */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "0.5rem 1.25rem",
            borderRadius: "999px",
            background: "rgba(18, 18, 26, 0.8)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border)",
            pointerEvents: "auto",
          }}
        >
          <span
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontWeight: 700,
              fontSize: "0.85rem",
              letterSpacing: "-0.02em",
            }}
          >
            DORC
          </span>
          <span
            style={{
              width: "1px",
              height: "14px",
              background: "var(--border)",
            }}
          />
          <span
            style={{
              fontFamily: '"Inter", sans-serif',
              fontSize: "0.7rem",
              color: "var(--text-dim)",
            }}
          >
            Based on EfficientNet-B0
          </span>
        </div>

        {/* Right dock */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.4rem 0.5rem",
            borderRadius: "999px",
            background: "rgba(18, 18, 26, 0.8)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border)",
            pointerEvents: "auto",
          }}
        >
          <button
            onClick={() => setView("classify")}
            style={{
              padding: "0.35rem 1rem",
              borderRadius: "999px",
              border: "none",
              background: view === "classify" ? "var(--accent-dim)" : "transparent",
              color: view === "classify" ? "var(--accent)" : "var(--text-dim)",
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.7rem",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            Classify
          </button>
          <button
            onClick={() => setView("classes")}
            style={{
              padding: "0.35rem 1rem",
              borderRadius: "999px",
              border: "none",
              background: view === "classes" ? "var(--accent-dim)" : "transparent",
              color: view === "classes" ? "var(--accent)" : "var(--text-dim)",
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.7rem",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            Classes
          </button>
        </div>
      </div>

      {/* Classify view */}
      {view === "classify" && (
        <div
          className="classify-view"
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "5rem 1rem 2rem",
          }}
        >
          <div
            className="classify-layout"
            style={{
              display: "flex",
              gap: "1.5rem",
              width: "100%",
              maxWidth: "900px",
              alignItems: "stretch",
            }}
          >
            {/* Left popup: Upload / Image */}
            <div
              className="classify-panel"
              style={{
                flex: 1,
                minWidth: 0,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {!hasResult ? (
                <>
                  <ImageUpload
                    onImageSelect={handleImageSelect}
                    disabled={isLoading}
                    statusLabel={statusLabel}
                    progress={progress}
                  />
                  <div
                    style={{
                      marginTop: "1.5rem",
                      textAlign: "center",
                      fontFamily: '"Inter", sans-serif',
                      fontSize: "0.8rem",
                      color: "var(--text-dim)",
                      lineHeight: 1.6,
                    }}
                  >
                    Drop any image to classify it into one of 108 species.
                  </div>
                </>
              ) : (
                <div
                  className="pop-in"
                  style={{
                    borderRadius: "14px",
                    background: "rgba(18, 18, 26, 0.95)",
                    backdropFilter: "blur(16px)",
                    border: "1px solid var(--border)",
                    overflow: "hidden",
                    flex: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    position: "relative",
                  }}
                >
                  {preview && (
                    <img
                      src={preview}
                      alt="Uploaded"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  )}
                  {isLoading && (
                    <div style={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      padding: "0.75rem 1.5rem 1rem",
                      background: "linear-gradient(transparent, rgba(10, 10, 15, 0.9))",
                    }}>
                      <div style={{
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: "0.62rem",
                        color: "var(--accent)",
                        marginBottom: "0.4rem",
                      }}>
                        {statusLabel}
                      </div>
                      <div style={{
                        height: "3px",
                        borderRadius: "999px",
                        background: "rgba(255,255,255,0.06)",
                        overflow: "hidden",
                      }}>
                        <div style={{
                          height: "100%",
                          width: `${progress}%`,
                          borderRadius: "999px",
                          background: "var(--accent)",
                          transition: "width 0.4s cubic-bezier(0.22, 1, 0.36, 1)",
                        }} />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right popup: Results */}
            {hasResult && (
              <div
                className="pop-in classify-panel"
                style={{
                  flex: 1,
                  minWidth: 0,
                  borderRadius: "14px",
                  background: "rgba(18, 18, 26, 0.95)",
                  backdropFilter: "blur(16px)",
                  border: "1px solid var(--border)",
                  padding: "1.75rem",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                {error && (
                  <div
                    style={{
                      padding: "0.75rem 1rem",
                      borderRadius: "10px",
                      background: "rgba(248, 113, 113, 0.08)",
                      border: "1px solid rgba(248, 113, 113, 0.15)",
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: "0.78rem",
                      color: "var(--red)",
                      marginBottom: "0.75rem",
                    }}
                  >
                    {error}
                  </div>
                )}
                {prediction && <PredictionResult prediction={prediction} />}

                <button
                  onClick={handleReset}
                  style={{
                    marginTop: "1.25rem",
                    width: "100%",
                    padding: "0.65rem 1rem",
                    borderRadius: "10px",
                    border: "1px solid var(--border)",
                    background: "var(--bg-surface)",
                    color: "var(--text)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.75rem",
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
                    width="13"
                    height="13"
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

                <button
                  onClick={() => cameraInputRef.current?.click()}
                  style={{
                    marginTop: "0.5rem",
                    width: "100%",
                    padding: "0.65rem 1rem",
                    borderRadius: "10px",
                    border: "1px solid var(--border)",
                    background: "var(--bg-surface)",
                    color: "var(--text)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.75rem",
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
                    width="13"
                    height="13"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                  Click Another
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Classes view */}
      {view === "classes" && (
        <div
          className="fade-in"
          style={{
            flex: 1,
            overflow: "auto",
            padding: "5rem 1rem 2rem",
          }}
        >
          <div
            style={{
              maxWidth: "1100px",
              margin: "0 auto",
            }}
          >
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.62rem",
                color: "var(--text-dim)",
                textTransform: "uppercase",
                letterSpacing: "0.15em",
                marginBottom: "1.5rem",
              }}
            >
              {classList.length} Classes
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                gap: "0.75rem",
              }}
            >
              {classList.map((cls) => (
                <div
                  key={cls.name}
                  style={{
                    borderRadius: "10px",
                    border: "1px solid var(--border)",
                    background: "var(--bg-surface)",
                    overflow: "hidden",
                    cursor: "default",
                    transition: "border-color 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border)";
                  }}
                >
                  <div
                    style={{
                      aspectRatio: "1",
                      background: "var(--bg-elevated)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      overflow: "hidden",
                    }}
                  >
                    <img
                      src={cls.image}
                      alt={cls.name}
                      loading="lazy"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                      }}
                    />
                  </div>
                  <div
                    style={{
                      padding: "0.5rem 0.6rem",
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: "0.65rem",
                      fontWeight: 500,
                      textTransform: "capitalize",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {cls.name.replace(/_/g, " ")}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* First-load notice */}
      {status === "loading_runtime" && (
        <div
          className="fade-in"
          style={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            maxWidth: "280px",
            padding: "0.75rem 1rem",
            borderRadius: "10px",
            background: "rgba(18, 18, 26, 0.9)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border)",
            fontFamily: '"Inter", sans-serif',
            fontSize: "0.72rem",
            color: "var(--text-dim)",
            lineHeight: 1.5,
            zIndex: 10,
          }}
        >
          <span style={{ color: "var(--accent)", fontWeight: 500 }}>First visit?</span>{" "}
          The model (~29MB) is being downloaded and compiled. This only happens once — it will be cached for future visits.
        </div>
      )}
    </div>
  );
}
