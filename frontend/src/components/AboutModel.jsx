import React from "react";
import { ShieldCheck, Cpu, Layers, CheckCircle2, Eye, Info, Database } from "lucide-react";

export function AboutModel() {
  const supportedClasses = [
    { name: "Actinic keratoses", type: "Pre-malignant", desc: "Dry, scaly patches caused by long-term sun exposure." },
    { name: "Basal cell carcinoma", type: "Malignant", desc: "Common non-melanoma skin cancer originating in basal cells." },
    { name: "Benign keratosis-like lesions", type: "Benign", desc: "Non-cancerous skin growths such as seborrheic keratoses." },
    { name: "Chickenpox", type: "Infectious", desc: "Viral infection causing itchy rash with fluid-filled blisters." },
    { name: "Cowpox", type: "Infectious", desc: "Rare viral skin condition caused by orthopoxvirus." },
    { name: "Dermatofibroma", type: "Benign", desc: "Harmless firm nodule commonly found on lower legs." },
    { name: "HFMD", type: "Infectious", desc: "Hand, foot, and mouth disease caused by enteroviruses." },
    { name: "Healthy", type: "Normal", desc: "Unaffected normal skin texture without visible lesions." },
    { name: "Measles", type: "Infectious", desc: "Viral illness causing characteristic widespread erythematous maculopapular rash." },
    { name: "Melanocytic nevi", type: "Benign", desc: "Common moles formed by clusters of melanocyte cells." },
    { name: "Melanoma", type: "Malignant", desc: "Serious skin cancer arising from pigment-producing melanocytes." },
    { name: "Monkeypox", type: "Infectious", desc: "Viral zoonotic lesion presentation characterized by vesicular/pustular rash." },
    { name: "Squamous cell carcinoma", type: "Malignant", desc: "Second most common form of skin cancer arising in squamous cells." },
    { name: "Vascular lesions", type: "Benign/Vascular", desc: "Skin growths formed by blood vessels, such as cherry angiomas." },
  ];

  return (
    <div className="about-model-page container">
      {/* Header Section */}
      <div className="about-hero">
        <h1 className="about-title">Model Specifications & Technical Details</h1>
        <p className="about-subtitle">
          This platform utilizes a frozen, production-verified ConvNeXt-Small neural network combined with Grad-CAM explainability for skin lesion classification research.
        </p>
      </div>

      {/* Model Key Specs Grid */}
      <div className="specs-grid">
        <div className="spec-card">
          <Cpu className="spec-icon" size={24} />
          <h3 className="spec-title">Architecture</h3>
          <p className="spec-value">ConvNeXt-Small (Modern ConvNet)</p>
          <span className="spec-sub">50.2M parameters · 384×384 input resolution</span>
        </div>

        <div className="spec-card">
          <Database className="spec-icon" size={24} />
          <h3 className="spec-title">Trained Categories</h3>
          <p className="spec-value">14 Canonical Classes</p>
          <span className="spec-sub">Supervised classification taxonomy</span>
        </div>

        <div className="spec-card">
          <Eye className="spec-icon" size={24} />
          <h3 className="spec-title">Explainability</h3>
          <p className="spec-value">Grad-CAM</p>
          <span className="spec-sub">Target layer: features.7 (Stage 4)</span>
        </div>

        <div className="spec-card">
          <ShieldCheck className="spec-icon" size={24} />
          <h3 className="spec-title">Model Status</h3>
          <p className="spec-value">Active Production Champion</p>
          <span className="spec-sub font-mono">SHA256: 67f4d8d04e...</span>
        </div>
      </div>

      {/* 14 Canonical Supported Classes Table */}
      <div className="section-block">
        <h2 className="block-title">Canonical Trained Categories (14 Classes)</h2>
        <p className="block-desc">
          The image classification model is strictly trained on the following 14 categories. Images showing conditions outside this list cannot be confirmed by this tool.
        </p>

        <div className="classes-table-container">
          <table className="classes-table">
            <thead>
              <tr>
                <th>Index</th>
                <th>Condition Category</th>
                <th>Category Classification</th>
                <th>Clinical Overview</th>
              </tr>
            </thead>
            <tbody>
              {supportedClasses.map((item, idx) => (
                <tr key={idx}>
                  <td className="font-mono">{idx}</td>
                  <td className="font-semibold">{item.name}</td>
                  <td>
                    <span className={`type-tag ${item.type.toLowerCase().replace('/', '-')}`}>
                      {item.type}
                    </span>
                  </td>
                  <td>{item.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Grad-CAM Methodology Section */}
      <div className="section-block">
        <h2 className="block-title">Grad-CAM Spatial Explainability</h2>
        <div className="gradcam-info-card">
          <p>
            Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of the target class logit flowing into the final convolutional stage (<code className="font-mono">features.7</code>) to produce a coarse localization map highlighting the regions in the image most relevant for the prediction.
          </p>
          <p className="mt-2">
            The generated activation maps are processed with a Jet colormap and superimposed over the original image, allowing researchers to verify whether the neural network focuses on clinically relevant lesion structures or background artifacts.
          </p>
        </div>
      </div>

      {/* Research Disclaimer & Limitations */}
      <div className="section-block disclaimer-block">
        <div className="disclaimer-header">
          <Info size={20} />
          <h3>Research & Educational Disclaimer</h3>
        </div>
        <p>
          This system is developed for computer vision research and educational demonstration only. The automated predictions and Grad-CAM spatial heatmaps do NOT constitute medical diagnoses, clinical evaluations, or treatment advice. Always consult a board-certified dermatologist for medical evaluation of skin lesions.
        </p>
      </div>
    </div>
  );
}
