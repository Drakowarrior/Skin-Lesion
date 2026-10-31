import React from "react";
import { Upload, Eye, Sparkles, ArrowRight } from "lucide-react";

export function HowItWorks({ onGoToAnalyze }) {
  const steps = [
    {
      num: "1",
      icon: <Upload size={22} />,
      title: "1. Upload Skin Image",
      desc: "Select or drop a clear skin image (JPG, PNG, or WEBP up to 15 MB). Your image is processed securely for analysis.",
    },
    {
      num: "2",
      icon: <Eye size={22} />,
      title: "2. Model Prediction & Grad-CAM",
      desc: "Inspect top predictions with confidence percentages and visual Grad-CAM heatmaps showing where the model focused.",
    },
    {
      num: "3",
      icon: <Sparkles size={22} />,
      title: "3. AI Patient Educational Guidance",
      desc: "Ask our patient education assistant about your result, symptoms to watch for, triage guidance levels, and safe self-care tips.",
    },
  ];

  return (
    <section className="how-it-works-page container">
      <div className="section-header text-center">
        <h1 className="how-title">How It Works</h1>
        <p className="how-subtitle">
          A simple, transparent 3-step workflow combining computer vision classification with patient educational assistance.
        </p>
      </div>

      <div className="steps-grid mt-8">
        {steps.map((step) => (
          <div key={step.num} className="step-card">
            <div className="step-header">
              <span className="step-number">{step.num}</span>
              <div className="step-icon-box">{step.icon}</div>
            </div>
            <h3 className="step-title">{step.title}</h3>
            <p className="step-desc">{step.desc}</p>
          </div>
        ))}
      </div>

      <div className="how-actions mt-8 text-center">
        <button className="btn btn-primary btn-lg" onClick={onGoToAnalyze}>
          <span>Start Image Analysis</span>
          <ArrowRight size={18} />
        </button>
      </div>
    </section>
  );
}
