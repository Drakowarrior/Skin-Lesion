import React from "react";
import { Upload, HelpCircle, Shield } from "lucide-react";

export function Hero({ onUploadClick, onHowItWorksClick }) {
  return (
    <section className="landing-hero-section">
      <div className="landing-hero-container">
        <h1 className="landing-brand-title">Skin Lesion Classifier</h1>
        <h2 className="landing-brand-subtitle">Skin lesion image analysis</h2>
        
        <p className="landing-description">
          Upload an image to see the model's prediction and visual explanation.
        </p>

        <div className="landing-actions">
          <button className="btn btn-primary btn-lg" onClick={onUploadClick}>
            <Upload size={18} />
            <span>Upload image</span>
          </button>

          <button className="btn-subtle-link" onClick={onHowItWorksClick}>
            <HelpCircle size={16} />
            <span>How it works</span>
          </button>
        </div>

        <div className="landing-safety-statement">
          <Shield size={14} className="safety-shield-icon" />
          <span>Research and educational use only. This tool does not provide a medical diagnosis.</span>
        </div>
      </div>
    </section>
  );
}
