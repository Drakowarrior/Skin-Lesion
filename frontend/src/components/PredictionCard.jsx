import React, { useState } from "react";
import { Info, ChevronDown, ChevronUp, AlertTriangle, AlertCircle, CheckCircle2 } from "lucide-react";
import { formatProbability, formatProbabilityDetailed } from "../utils/formatProbability";

export function PredictionCard({
  prediction,
  uncertainty,
  quality,
}) {
  const [showTechnical, setShowTechnical] = useState(false);

  if (!prediction) return null;

  // 1. Handle Image Quality Failure Short-Circuit
  if (quality && quality.state === "QUALITY_FAIL") {
    return (
      <div className="prediction-main-card quality-fail-card">
        <div className="card-header-group">
          <h3 className="results-card-heading">Image Quality Assessment</h3>
        </div>
        <div className="abstain-alert quality-fail-box">
          <AlertTriangle size={24} className="text-warning" />
          <div>
            <h4 className="abstain-title">Image Quality Insufficient</h4>
            <p className="abstain-desc">
              {quality.guidance_message ||
                "Image quality is insufficient for reliable analysis. Please capture a clear, well-lit, in-focus clinical photograph."}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 2. Resolve Decision State
  const state = (
    prediction.uncertainty_state ||
    (uncertainty && uncertainty.state) ||
    "CONFIDENT"
  ).toUpperCase();

  const isAbstain = state === "ABSTAIN" || state === "HARD_ABSTAIN";

  // 3. Calibrated Model Probability
  const rawProb =
    prediction.calibrated_probability !== undefined
      ? prediction.calibrated_probability
      : prediction.probability !== undefined
      ? prediction.probability
      : prediction.confidence;

  const calibratedProb = typeof rawProb === "number" ? rawProb : 0;
  const formattedPercent = formatProbability(calibratedProb);

  const rawPercent =
    prediction.raw_confidence !== undefined
      ? formatProbability(prediction.raw_confidence)
      : null;

  const temperatureUsed =
    prediction.temperature_used !== undefined
      ? prediction.temperature_used
      : 1.0;

  // Uncertainty Metrics
  const uncMetrics = uncertainty?.metrics || {};
  const rawMargin =
    uncMetrics.prediction_margin !== undefined
      ? uncMetrics.prediction_margin
      : prediction.margin !== undefined
      ? prediction.margin
      : null;
  const marginPercent =
    rawMargin !== null ? (rawMargin * 100).toFixed(4) + "%" : "N/A";

  const rawEntropy =
    uncMetrics.normalized_entropy !== undefined
      ? uncMetrics.normalized_entropy
      : prediction.normalized_entropy !== undefined
      ? prediction.normalized_entropy
      : null;
  const normEntropy =
    rawEntropy !== null ? rawEntropy.toFixed(4) : "N/A";

  // State Badge Rendering
  const renderStateBadge = () => {
    if (isAbstain) {
      return (
        <span className="state-badge badge-abstain">
          <AlertTriangle size={14} />
          <span>Abstained / Low Reliability</span>
        </span>
      );
    }
    if (state === "CONFIDENT") {
      return (
        <span className="state-badge badge-confident">
          <CheckCircle2 size={14} />
          <span>Validated Confident Prediction</span>
        </span>
      );
    }
    return (
      <span className="state-badge badge-uncertain">
        <AlertCircle size={14} />
        <span>Borderline / Moderate Uncertainty</span>
      </span>
    );
  };

  return (
    <div className="prediction-main-card">
      {/* 1. Header & Primary Classification */}
      <div className="card-header-group">
        <span className="card-label">Model prediction</span>
        <div className="prediction-class-row">
          <h2 className="prediction-class-name">{prediction.class}</h2>
          {renderStateBadge()}
        </div>
      </div>

      {/* 2. Hard Abstain State Warning */}
      {isAbstain ? (
        <div className="abstain-notice-box">
          <AlertTriangle size={20} className="text-warning" />
          <div className="abstain-text-group">
            <strong>Insufficient Discriminatory Evidence</strong>
            <p className="abstain-subtext">
              {uncertainty?.clinical_message ||
                prediction.clinical_message ||
                "This image does not provide sufficient discriminatory evidence for a reliable classification. The prediction has been abstained to prevent potential diagnostic error. Please consult a qualified dermatologist for clinical evaluation."}
            </p>
          </div>
        </div>
      ) : (
        /* 3. Scientifically Honest Calibrated Probability Display */
        <div className="probability-display-section">
          <div className="probability-header">
            <div className="probability-title-group">
              <span className="probability-label-text">Calibrated model probability</span>
              <span className="probability-subtext-note">
                {prediction.is_calibrated
                  ? `(Temperature Scaling, T = ${temperatureUsed})`
                  : `(Calibrated Softmax, T = ${temperatureUsed})`}
              </span>
            </div>
            <span className="probability-value-text">{formattedPercent}</span>
          </div>

          {/* Progress Bar Visualizer for Calibrated Probability */}
          <div className="probability-bar-track">
            <div
              className={`probability-bar-fill ${state === "CONFIDENT" ? "fill-confident" : "fill-uncertain"}`}
              style={{ width: `${Math.min(100, Math.max(0.5, calibratedProb * 100))}%` }}
            />
          </div>

          {/* Mandatory Scientific Disclaimer */}
          <p className="probability-disclaimer-text">
            Model-assigned probability from the active calibrated classifier. This is not a guarantee of correctness or a medical diagnosis; individual predictions can be incorrect.
          </p>
        </div>
      )}

      {/* 4. Expandable Technical Model Output Section */}
      <div className="technical-details-toggle">
        <button
          type="button"
          className="technical-toggle-btn"
          onClick={() => setShowTechnical(!showTechnical)}
          aria-expanded={showTechnical}
        >
          <div className="toggle-btn-left">
            <Info size={15} />
            <span>Technical Model Output & Audit Metrics</span>
          </div>
          {showTechnical ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showTechnical && (
          <div className="technical-details-content">
            <div className="tech-metric-row">
              <span className="tech-label">Calibrated Probability:</span>
              <span className="tech-value highlight">
                {formatProbabilityDetailed(calibratedProb)}
              </span>
            </div>
            {rawPercent && (
              <div className="tech-metric-row">
                <span className="tech-label">Uncalibrated Softmax (Raw):</span>
                <span className="tech-value">{rawPercent}</span>
              </div>
            )}
            <div className="tech-metric-row">
              <span className="tech-label">Uncertainty Decision State:</span>
              <span className={`tech-value state-${state.toLowerCase()}`}>
                {state}
              </span>
            </div>
            <div className="tech-metric-row">
              <span className="tech-label">Prediction Margin (Top1 - Top2):</span>
              <span className="tech-value">{marginPercent}</span>
            </div>
            <div className="tech-metric-row">
              <span className="tech-label">Normalized Entropy (0 to 1):</span>
              <span className="tech-value">{normEntropy}</span>
            </div>
            <div className="tech-metric-row">
              <span className="tech-label">Calibration Engine:</span>
              <span className="tech-value">
                {prediction.is_calibrated
                  ? `Temperature Scaling (T = ${temperatureUsed})`
                  : `Calibrated Softmax (T = ${temperatureUsed})`}
              </span>
            </div>
            <div className="tech-metric-row">
              <span className="tech-label">Active Model:</span>
              <span className="tech-value">ConvNeXt-Small (Production v1)</span>
            </div>
            <div className="tech-metric-row">
              <span className="tech-label">Quality Gate State:</span>
              <span className="tech-value state-confident">
                {quality?.state || "QUALITY_PASS"}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
