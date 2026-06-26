"use client";

import { useRef, useCallback, useState, useImperativeHandle, forwardRef, useEffect } from "react";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  preview?: string | null;
  disabled?: boolean;
  statusLabel?: string;
  progress?: number;
}

export interface ImageUploadHandle {
  triggerUpload: () => void;
  triggerCamera: () => void;
}

const ImageUpload = forwardRef<ImageUploadHandle, ImageUploadProps>(
  function ImageUpload({ onImageSelect, preview, disabled = false, statusLabel, progress = 0 }, ref) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);
    const [cameraOpen, setCameraOpen] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const stopCamera = useCallback(() => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      setCameraOpen(false);
    }, []);

    useEffect(() => {
      if (!cameraOpen) return;
      let active = true;
      (async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false,
          });
          if (!active) { stream.getTracks().forEach((t) => t.stop()); return; }
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
          }
        } catch {
          setCameraOpen(false);
        }
      })();
      return () => { active = false; stopCamera(); };
    }, [cameraOpen, stopCamera]);

    const captureFrame = useCallback(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File([blob], `photo_${Date.now()}.jpg`, { type: "image/jpeg" });
            stopCamera();
            onImageSelect(file);
          }
        },
        "image/jpeg",
        0.92
      );
    }, [onImageSelect, stopCamera]);

    useImperativeHandle(ref, () => ({
      triggerUpload: () => {
        if (!disabled) fileInputRef.current?.click();
      },
      triggerCamera: () => {
        if (!disabled) setCameraOpen(true);
      },
    }));

    const handleDrop = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) onImageSelect(file);
    }, [onImageSelect, disabled]);

    const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onImageSelect(file);
      e.target.value = "";
    }, [onImageSelect]);

    const showProgress = disabled && statusLabel;

    return (
      <>
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragActive(true); }}
          onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
          onClick={() => !disabled && !preview && fileInputRef.current?.click()}
          className={dragActive ? "drag-active" : ""}
          style={{
            border: `1.5px dashed ${dragActive ? "var(--accent)" : "var(--border)"}`,
            borderRadius: "14px",
            padding: preview ? "0" : "2.5rem 1.5rem",
            textAlign: "center",
            cursor: disabled || preview ? "default" : "pointer",
            transition: "all 0.2s ease",
            background: dragActive ? "var(--accent-dim)" : "var(--bg-surface)",
            height: "100%",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            position: "relative",
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFile}
            style={{ display: "none" }}
            disabled={disabled}
          />

          {preview ? (
            <img
              src={preview}
              alt="Uploaded"
              style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: "13px" }}
            />
          ) : (
            <>
              <div style={{
                width: "48px",
                height: "48px",
                margin: "0 auto 1rem",
                borderRadius: "12px",
                background: "var(--accent-dim)",
                border: "1px solid rgba(167, 139, 250, 0.15)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p style={{
                fontFamily: '"Inter", sans-serif',
                fontSize: "0.92rem",
                fontWeight: 500,
                color: dragActive ? "var(--accent)" : "var(--text)",
                marginBottom: "0.3rem",
              }}>
                {disabled ? statusLabel || "Processing..." : dragActive ? "Drop to classify" : "Drop image or click to upload"}
              </p>
              <p style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.68rem",
                color: "var(--text-dim)",
                marginBottom: "0.75rem",
              }}>
                JPG &middot; PNG &middot; WebP
              </p>

              {!disabled && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setCameraOpen(true);
                  }}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    padding: "0.4rem 0.9rem",
                    borderRadius: "999px",
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--text-dim)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.65rem",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                    e.currentTarget.style.color = "var(--text-dim)";
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                  Take Photo
                </button>
              )}
            </>
          )}

          {showProgress && (
            <div style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              padding: "0.75rem 1.5rem 1rem",
              background: "linear-gradient(transparent, rgba(10, 10, 15, 0.9))",
            }}>
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

        {cameraOpen && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 9999,
              background: "#000",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />

            <div style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              padding: "1.5rem",
              display: "flex",
              justifyContent: "center",
              gap: "1rem",
              background: "linear-gradient(transparent, rgba(0,0,0,0.7))",
            }}>
              <button
                onClick={stopCamera}
                style={{
                  padding: "0.6rem 1.2rem",
                  borderRadius: "999px",
                  border: "1px solid rgba(255,255,255,0.3)",
                  background: "rgba(255,255,255,0.1)",
                  color: "#fff",
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "0.75rem",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={captureFrame}
                style={{
                  width: "64px",
                  height: "64px",
                  borderRadius: "50%",
                  border: "3px solid #fff",
                  background: "rgba(255,255,255,0.2)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: 0,
                }}
              >
                <div style={{
                  width: "52px",
                  height: "52px",
                  borderRadius: "50%",
                  background: "#fff",
                  transition: "transform 0.1s ease",
                }} />
              </button>
              <button
                onClick={stopCamera}
                style={{
                  padding: "0.6rem 1.2rem",
                  borderRadius: "999px",
                  border: "none",
                  background: "transparent",
                  color: "transparent",
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "0.75rem",
                  cursor: "default",
                  pointerEvents: "none",
                }}
              >
                spacer
              </button>
            </div>

            <button
              onClick={stopCamera}
              style={{
                position: "absolute",
                top: "1rem",
                left: "1rem",
                width: "36px",
                height: "36px",
                borderRadius: "50%",
                border: "none",
                background: "rgba(0,0,0,0.5)",
                color: "#fff",
                fontSize: "1.1rem",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              &times;
            </button>
          </div>
        )}
      </>
    );
  }
);

export default ImageUpload;
