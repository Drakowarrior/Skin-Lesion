import React, { useState, useRef } from "react";
import { Sliders, RefreshCw, Loader2, Info, Eye } from "lucide-react";
import { formatProbability } from "../utils/formatProbability";

/**
 * Normalizes an image representation into a valid URL or data URL.
 * Handles both complete data URLs, http/blob URLs, and raw base64 strings.
 */
function normalizeImageUrl(val, fallbackMime = "image/jpeg") {
  if (!val || typeof val !== "string") return null;
  const trimmed = val.trim();
  if (!trimmed) return null;
  if (
    trimmed.startsWith("data:") ||
    trimmed.startsWith("http:") ||
    trimmed.startsWith("https:") ||
    trimmed.startsWith("blob:")
  ) {
    return trimmed;
  }
  return `data:${fallbackMime};base64,${trimmed}`;
}

export function ExplainabilityPanel({
  explainResult,
  predictionResult,
  imagePreview,
  isExplaining,
  explainError,
  onRetry,
}) {
  const [overlayOpacity, setOverlayOpacity] = useState(40);
  const [imageLoadError, setImageLoadError] = useState(false);
  const [hoverInfo, setHoverInfo] = useState({ visible: false, text: "", x: 0, y: 0 });
  const containerRef = useRef(null);

  // Reset errors and hover state when analysis image or explainResult updates
  React.useEffect(() => {
    setImageLoadError(false);
    setHoverInfo({ visible: false, text: "", x: 0, y: 0 });
  }, [explainResult, imagePreview]);

  // 1. Loading State
  if (isExplaining) {
    return (
      <div className="explainability-card loading-panel">
        <div className="explainability-header">
          <h3 className="card-title">Explainable AI (Grad-CAM)</h3>
        </div>
        <div className="explainability-loading-box">
          <Loader2 size={24} className="animate-spin spinner-icon text-primary" />
          <p className="loading-exp-text">Generating Grad-CAM spatial activation map...</p>
        </div>
      </div>
    );
  }

  // Extract images and normalize URLs
  const images = explainResult?.images || {};
  const baseImgRaw = images.original_preview || images.original;
  const overlayImgRaw =
    images.overlay_blended ||
    images.overlay ||
    images.heatmap_colored ||
    images.heatmap;

  const baseImgUrl = normalizeImageUrl(baseImgRaw) || imagePreview;
  const overlayImgUrl = normalizeImageUrl(overlayImgRaw);

  // 2. Error / Unavailable State
  const hasValidOverlay = Boolean(overlayImgUrl) && !imageLoadError;
  if (explainError || !hasValidOverlay) {
    if (!explainResult && !explainError) {
      return null;
    }

    return (
      <div className="explainability-card error-panel">
        <div className="explainability-header">
          <h3 className="card-title">Explainable AI (Grad-CAM)</h3>
        </div>
        <div className="explain-error-body">
          <p className="explain-error-msg">
            {explainError ||
              "Grad-CAM spatial activation visualization is currently unavailable for this image."}
          </p>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => {
              setImageLoadError(false);
              if (onRetry) onRetry();
            }}
          >
            <RefreshCw size={14} />
            <span>Try Again</span>
          </button>
        </div>
      </div>
    );
  }

  // Metadata Extraction
  const explainability = explainResult?.explainability || {};
  const predictedClass =
    explainability.target_class ||
    explainResult?.prediction?.class_name ||
    predictionResult?.prediction?.class ||
    "Predicted Class";

  const rawConfidence =
    predictionResult?.prediction?.confidence ??
    explainResult?.prediction?.confidence ??
    explainability.target_class_score;
  const confidencePct =
    rawConfidence !== undefined && rawConfidence !== null
      ? formatProbability(rawConfidence)
      : "N/A";

  const targetLayer = explainability.target_layer || "features.7 (ConvNeXtBlock Stage 4)";
  const heatmapGrid = explainability.heatmap_grid || explainResult?.heatmap || null;

  const handleMouseMove = (e) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
      setHoverInfo({ visible: false, text: "", x: 0, y: 0 });
      return;
    }

    if (heatmapGrid && Array.isArray(heatmapGrid) && heatmapGrid.length > 0) {
      const gridH = heatmapGrid.length;
      const gridW = heatmapGrid[0]?.length || 0;

      if (gridH > 0 && gridW > 0) {
        const normX = x / rect.width;
        const normY = y / rect.height;
        const cellY = Math.min(Math.floor(normY * gridH), gridH - 1);
        const cellX = Math.min(Math.floor(normX * gridW), gridW - 1);
        const val = heatmapGrid[cellY]?.[cellX];

        if (typeof val === "number" && !isNaN(val)) {
          const pct = (val * 100).toFixed(1);
          setHoverInfo({
            visible: true,
            text: `Relative Activation: ${pct}%`,
            x,
            y,
          });
          return;
        }
      }
    }

    // When grid array is omitted but activation is valid
    if (explainability.peak_activation !== undefined) {
      setHoverInfo({
        visible: true,
        text: `Salient Feature Activation (Layer: ${targetLayer})`,
        x,
        y,
      });
    } else {
      setHoverInfo({ visible: false, text: "", x: 0, y: 0 });
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo({ visible: false, text: "", x: 0, y: 0 });
  };

  return (
    <div className="explainability-card active-panel">
      {/* ── Header ───────────────────────────────────────────── */}
      <div className="explainability-header">
        <div>
          <h3 className="card-title">Explainable AI</h3>
          <p className="card-subtitle">
            Model attention overlay for <strong>{predictedClass}</strong> ({confidencePct}%) · Layer: <code>{targetLayer}</code>
          </p>
        </div>
      </div>

      {/* ── Opacity Control ─────────────────────────────────── */}
      <div className="opacity-slider-container">
        <div className="opacity-slider-header">
          <span className="slider-label">
            <Sliders size={15} />
            <span>Overlay Opacity: <strong>{overlayOpacity}%</strong></span>
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={overlayOpacity}
          onChange={(e) => setOverlayOpacity(Number(e.target.value))}
          className="opacity-slider-input"
          aria-label="Heatmap overlay opacity"
        />
      </div>

      {/* ── Visual Viewport: Base Image + Heatmap Overlay ──── */}
      <div className="explainability-image-container">
        <div
          ref={containerRef}
          className="layered-image-wrapper"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {baseImgUrl && (
            <img
              src={baseImgUrl}
              alt="Original skin lesion"
              className="explain-display-image base-layer"
            />
          )}
          <img
            src={overlayImgUrl}
            alt={`Grad-CAM heatmap overlay for ${predictedClass}`}
            className="explain-display-image heatmap-overlay-layer"
            style={{ opacity: overlayOpacity / 100 }}
            onError={() => setImageLoadError(true)}
          />

          {/* Interactive Hover Tooltip */}
          {hoverInfo.visible && hoverInfo.text && (
            <div
              className="gradcam-hover-tooltip"
              style={{
                top: Math.max(0, hoverInfo.y - 35),
                left: hoverInfo.x + 15,
              }}
            >
              <Eye size={12} />
              <span>{hoverInfo.text}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Model Attention Color Scale Legend ───────────────── */}
      <div className="gradcam-compact-legend">
        <div className="legend-header-row">
          <h4 className="compact-legend-title">Model Relative Activation Scale</h4>
        </div>

        <div className="gradient-bar"></div>

        <div className="gradient-endpoints">
          <span className="endpoint-low">Low</span>
          <span className="endpoint-labels">Blue → Cyan → Green → Yellow → Orange → Red</span>
          <span className="endpoint-high">High</span>
        </div>

        <p className="legend-explanation">
          Red regions indicate image areas receiving stronger relative contribution to the selected model prediction.
        </p>

        <div className="disclaimer-callout">
          <Info size={14} className="disclaimer-icon" />
          <span>
            Grad-CAM visualizes convolutional network feature weights for explainability. It does not identify disease severity, tissue type, or histopathology.
          </span>
        </div>
      </div>
    </div>
  );
}
