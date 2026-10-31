import React from "react";
import { AlertCircle, RefreshCw, Upload } from "lucide-react";

export function ErrorState({ error, onRetry, onReset }) {
  if (!error) return null;

  return (
    <div className="error-card">
      <div className="error-icon-box">
        <AlertCircle size={28} />
      </div>

      <h3 className="error-title">Analysis Unable to Complete</h3>
      <p className="error-message">{error}</p>

      <div className="error-actions">
        <button className="btn btn-secondary btn-md" onClick={onRetry}>
          <RefreshCw size={16} />
          <span>Try Again</span>
        </button>

        <button className="btn btn-primary btn-md" onClick={onReset}>
          <Upload size={16} />
          <span>Analyze Another Image</span>
        </button>
      </div>
    </div>
  );
}

