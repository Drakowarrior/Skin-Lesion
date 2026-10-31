import React, { useState, useRef } from "react";
import { Upload, AlertCircle } from "lucide-react";

export function ImageUploader({ onImageSelected, error }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onImageSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onImageSelected(e.target.files[0]);
    }
  };

  return (
    <div className="uploader-wrapper">
      <div
        className={`uploader-dropzone ${isDragging ? "dragging" : ""} ${error ? "has-error" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
        aria-label="Upload Skin Image"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
          className="file-input-hidden"
        />

        <div className="dropzone-content">
          <div className="upload-icon-container">
            <Upload className="upload-icon" size={30} />
          </div>

          <h3 className="upload-title">Upload a skin image</h3>
          <p className="upload-subtitle">
            Drag & drop an image here, or <span className="upload-browse">browse files</span>
          </p>

          <div className="upload-pills">
            <span className="format-pill">JPG, PNG or WEBP · Up to 15 MB</span>
          </div>

          <div className="sample-loader-row" style={{ marginTop: "0.75rem" }}>
            <button
              type="button"
              id="btn-try-sample-image"
              className="btn-try-sample"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.35rem",
                padding: "0.35rem 0.85rem",
                fontSize: "0.8rem",
                fontWeight: "500",
                color: "#2563eb",
                background: "#eff6ff",
                border: "1px solid #bfdbfe",
                borderRadius: "9999px",
                cursor: "pointer"
              }}
              onClick={async (e) => {
                e.stopPropagation();
                try {
                  const res = await fetch("/test_sample.jpg");
                  const blob = await res.blob();
                  const file = new File([blob], "test_sample.jpg", { type: "image/jpeg" });
                  onImageSelected(file);
                } catch (err) {
                  console.error("Failed to load sample image", err);
                }
              }}
            >
              Try Clinical Sample Image
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="uploader-error-banner">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
