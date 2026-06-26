"use client";

import { useRef, useCallback, useState, useImperativeHandle, forwardRef } from "react";

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
    const cameraInputRef = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);

    useImperativeHandle(ref, () => ({
      triggerUpload: () => {
        if (!disabled) fileInputRef.current?.click();
      },
      triggerCamera: () => {
        if (!disabled) cameraInputRef.current?.click();
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
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
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

            {/* Camera button */}
            {!disabled && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  cameraInputRef.current?.click();
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

        {/* Progress bar */}
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
    );
  }
);

export default ImageUpload;
