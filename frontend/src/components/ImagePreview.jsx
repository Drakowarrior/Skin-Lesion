import React from "react";
import { Trash2, Play } from "lucide-react";

export function ImagePreview({ imagePreview, imageMeta, onRemove, onAnalyze, isAnalyzing }) {
  return (
    <div className="preview-card">
      <h3 className="preview-card-title">Image Preview</h3>
      
      <div className="preview-body">
        {/* Image Thumbnail Display */}
        <div className="preview-image-wrapper">
          <img
            src={imagePreview}
            alt="Selected skin lesion preview"
            className="preview-image"
          />
        </div>

        {/* Metadata Details */}
        <div className="preview-meta-container">
          <div className="preview-meta-details">
            <h4 className="preview-filename" title={imageMeta?.name || "Uploaded Image"}>
              {imageMeta?.name || "lesion_image.jpg"}
            </h4>

            <div className="preview-stats-row">
              {imageMeta?.width && imageMeta?.height && (
                <div className="meta-item">
                  <span className="meta-value">{imageMeta.width} × {imageMeta.height}</span>
                </div>
              )}
              <div className="meta-item">
                <span className="meta-value">{imageMeta?.sizeStr || "Unknown size"}</span>
              </div>
            </div>
          </div>

          <div className="preview-status-note">
            <span>Image selected. Click <strong>Analyze Image</strong> to run prediction.</span>
          </div>

          {/* Action Buttons */}
          <div className="preview-actions">
            <button
              className="btn btn-secondary btn-md"
              onClick={onRemove}
              disabled={isAnalyzing}
            >
              <Trash2 size={16} />
              <span>Remove</span>
            </button>

            <button
              className="btn btn-primary btn-md flex-grow"
              onClick={onAnalyze}
              disabled={isAnalyzing}
            >
              <Play size={16} />
              <span>{isAnalyzing ? "Analyzing Image..." : "Analyze Image"}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

