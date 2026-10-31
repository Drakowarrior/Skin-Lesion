import React from "react";
import { BookOpen, ExternalLink, Info } from "lucide-react";

export function ResearchSection({ predictedClass }) {
  if (!predictedClass) return null;

  const scholarUrl = `https://scholar.google.com/scholar?q=${encodeURIComponent(`"${predictedClass}"`)}`;

  return (
    <div className="research-section-container">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="research-header-card">
        <div className="research-header-left">
          <div className="research-icon-badge">
            <BookOpen size={20} />
          </div>
          <div>
            <div className="assistant-title-row">
              <h3 className="research-title">Scientific Literature Discovery</h3>
            </div>
            <p className="research-subtitle">
              Dedicated external research literature search for peer-reviewed studies regarding {predictedClass}.
            </p>
          </div>
        </div>
      </div>

      {/* ── Content Body ───────────────────────────────────── */}
      <div className="research-content-body">
        {/* Predicted Condition Display */}
        <div className="predicted-condition-card">
          <span className="condition-label">Predicted condition:</span>
          <strong className="condition-value">{predictedClass}</strong>
        </div>

        {/* Dedicated Google Scholar Action Card */}
        <div className="scholar-action-box">
          <a
            href={scholarUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-scholar-primary"
            title={`Search Google Scholar for "${predictedClass}"`}
          >
            <span>Search Google Scholar for "{predictedClass}"</span>
            <ExternalLink size={16} />
          </a>
          <p className="scholar-explanation">
            Opens Google Scholar in an external tab to explore published clinical research and scientific studies. Google Scholar search results are independent external sources and are not automatically verified by this application.
          </p>
        </div>

        {/* Educational / Medical Disclaimer */}
        <p className="research-inline-disclaimer">
          External research literature is provided for educational and research context. Literature queries do not validate or confirm the model prediction. This system is not a medical diagnostic tool.
        </p>
      </div>
    </div>
  );
}

