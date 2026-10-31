import React from "react";

export function AboutAndHowItWorks() {
  const steps = [
    {
      title: "Upload Skin Image",
      desc: "Upload a clear skin photograph (JPG, PNG, or WEBP up to 15 MB). The image is processed securely for analysis.",
    },
    {
      title: "Image Classification & Grad-CAM",
      desc: "The neural network evaluates visual patterns across 14 trained categories and generates spatial feature activation heatmaps.",
    },
    {
      title: "Educational Guidance, Treatment & Precautions",
      desc: "Receive AI-powered patient educational guidance, general precautions, self-care tips, warning symptoms, and triage advice.",
    },
  ];

  const features = [
    {
      title: "ConvNeXt-Small Classification",
      desc: "Evaluates visual lesion patterns across 14 canonical categories, providing top prediction confidence percentages and probability distributions.",
    },
    {
      title: "Grad-CAM Visual Explanations",
      desc: "Inspect spatial feature activation maps highlighting regions in the image that contributed most to the model's output.",
    },
    {
      title: "Treatment & Precautions Guidance",
      desc: "Server-side clinical assistant provides clear educational information on symptoms, general precautions, self-care, and triage levels grounded in peer-reviewed PubMed literature.",
    },
  ];

  const supportedClasses = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD (Hand, Foot & Mouth)",
    "Healthy skin",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
  ];

  return (
    <div className="about-page-container container">
      {/* 1. Header Overview */}
      <div className="about-hero text-center">
        <h1 className="about-hero-title">Skin Lesion Research & Patient Education Platform</h1>
        <p className="about-hero-desc">
          Combining deep learning image classification with Grad-CAM visual explainability and AI patient educational guidance.
        </p>
      </div>

      {/* 2. How It Works Section */}
      <div className="section-block">
        <div className="section-header text-center mb-6">
          <h2 className="block-title">How It Works</h2>
          <p className="block-desc">A transparent workflow for skin image analysis and patient education.</p>
        </div>

        <div className="steps-grid">
          {steps.map((step, idx) => (
            <div key={idx} className="step-card plain-step-card">
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Core Features Overview */}
      <div className="section-block">
        <h2 className="block-title mb-4 text-center">Core Features & Capabilities</h2>

        <div className="info-features-grid">
          {features.map((feat, idx) => (
            <div key={idx} className="info-card">
              <h3 className="info-card-title">{feat.title}</h3>
              <p className="info-card-text">{feat.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. Supported Image Categories with Pointers */}
      <div className="section-block">
        <h2 className="block-title mb-2">Supported Image Categories (14 Classes)</h2>
        <p className="block-desc mb-4">
          The model is trained on a fixed set of 14 skin-lesion categories:
        </p>

        <div className="category-pointers-grid">
          {supportedClasses.map((cls, idx) => (
            <div key={idx} className="pointer-item">
              <span className="pointer-bullet">•</span>
              <span className="pointer-text">{cls}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
