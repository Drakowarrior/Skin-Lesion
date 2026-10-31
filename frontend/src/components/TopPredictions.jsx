import React from "react";
import { formatProbability } from "../utils/formatProbability";

export function TopPredictions({ top3 }) {
  if (!top3 || top3.length === 0) return null;

  return (
    <div className="top3-card">
      <div className="top3-header-row">
        <h4 className="top3-title">Differential Probabilities (Calibrated Top-3)</h4>
        <span className="top3-subtitle-note">Calibrated distribution</span>
      </div>

      <div className="top3-list">
        {top3.map((item, index) => {
          const prob =
            item.calibrated_probability !== undefined
              ? item.calibrated_probability
              : item.probability !== undefined
              ? item.probability
              : item.confidence;
          const numProb = typeof prob === "number" ? prob : parseFloat(prob) || 0;
          const displayPct = formatProbability(numProb);

          return (
            <div key={item.class_index !== undefined ? item.class_index : index} className="top3-item">
              <div className="top3-item-meta">
                <span className="top3-rank">{index + 1}.</span>
                <span className="top3-class-name">{item.class}</span>
                <span className="top3-confidence-value">{displayPct}</span>
              </div>

              <div className="top3-bar-track">
                <div
                  className={`top3-bar-fill rank-${index + 1}`}
                  style={{ width: `${Math.min(100, Math.max(0.5, numProb * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
