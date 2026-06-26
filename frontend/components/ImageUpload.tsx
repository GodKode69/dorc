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
}

const ImageUpload = forwardRef<ImageUploadHandle, ImageUploadProps>(
  function ImageUpload({ onImageSelect, preview, disabled = false, statusLabel, progress = 0 }, ref) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);

    useImperativeHandle(ref, () => ({
      triggerUpload: () => {
        if (!disabled) inputRef.current?.click();
      },
    }));

    const handleDrop = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) onImageSelect(file);
    }, [onImageSelect, disabled]);

    const showProgress = disabled && statusLabel;

    return (
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragActive(true); }}
        onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
        onClick={() => !disabled && !preview && inputRef.current?.click()}
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
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onImageSelect(file);
          }}
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
            }}>
              JPG &middot; PNG &middot; WebP
            </p>
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
