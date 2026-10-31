import React from "react";
import { ImageUploader } from "./ImageUploader";
import { ImagePreview } from "./ImagePreview";
import { LoadingState } from "./LoadingState";
import { PredictionCard } from "./PredictionCard";
import { TopPredictions } from "./TopPredictions";
import { ExplainabilityPanel } from "./ExplainabilityPanel";
import { PatientGuidanceSection } from "./PatientGuidanceSection";
import { ResearchSection } from "./ResearchSection";
import { ErrorState } from "./ErrorState";
import { RefreshCw, Sparkles } from "lucide-react";

export function AnalysisWorkspace({
  imageFile,
  imagePreview,
  imageMeta,
  isAnalyzing,
  predictionResult,
  error,
  isExplaining,
  explainResult,
  explainError,
  selectImage,
  clearImage,
  runAnalysis,
  fetchGradCAM,
}) {
  return (
    <section id="workspace" className="workspace-section">
      <div className="container">
        <div className="workspace-card">

          {/* Header Banner for Analyzing Feature */}
          {!predictionResult && (
            <div className="workspace-header text-center mb-6">
              <h2 className="workspace-title">Skin Lesion Image Analysis</h2>
              <p className="workspace-subtitle">
                Upload a skin image to evaluate model predictions, Grad-CAM feature heatmaps, and AI treatment & precautions guidance.
              </p>
            </div>
          )}

          {/* 1. Loading State */}
          {isAnalyzing && <LoadingState />}

          {/* 2. Error State */}
          {!isAnalyzing && error && (
            <ErrorState
              error={error}
              onRetry={runAnalysis}
              onReset={clearImage}
            />
          )}

          {/* 3. Image Selected (Ready to Analyze) */}
          {!isAnalyzing && !error && !predictionResult && imagePreview && (
            <ImagePreview
              imagePreview={imagePreview}
              imageMeta={imageMeta}
              onRemove={clearImage}
              onAnalyze={runAnalysis}
              isAnalyzing={isAnalyzing}
            />
          )}

          {/* 4. Initial Dropzone Upload State */}
          {!isAnalyzing && !error && !predictionResult && !imagePreview && (
            <ImageUploader onImageSelected={selectImage} error={null} />
          )}

          {/* 5. Successful Result State */}
          {!isAnalyzing && !error && predictionResult && (
            <div className="results-container">
              <div className="results-grid">
                {/* Uploaded Image Preview Column */}
                <div className="results-image-col">
                  <div className="results-image-card">
                    <h4 className="results-section-heading">Uploaded Image</h4>
                    <div className="results-image-box">
                      <img src={imagePreview} alt="Uploaded skin lesion" />
                    </div>
                    <p className="image-caption">{imageMeta?.name || "Uploaded image"}</p>
                  </div>
                </div>

                {/* Main Prediction & Top 3 Column */}
                <div className="results-main-col">
                  <PredictionCard
                    prediction={predictionResult.prediction}
                    uncertainty={predictionResult.uncertainty}
                    quality={predictionResult.quality}
                  />
                  <TopPredictions top3={predictionResult.top3} />
                </div>
              </div>

              {/* Grad-CAM Explainability Section */}
              <ExplainabilityPanel
                explainResult={explainResult}
                predictionResult={predictionResult}
                imagePreview={imagePreview}
                isExplaining={isExplaining}
                explainError={explainError}
                onRetry={() => fetchGradCAM(imageFile)}
              />

              {/* AI Patient Guidance Layer (Treatment, Precautions & Education via Groq API) */}
              <PatientGuidanceSection predictionResult={predictionResult} />

              {/* Related Scientific Research Literature Section */}
              <ResearchSection predictedClass={predictionResult?.prediction?.class} />

              {/* Reset Flow Action Button */}
              <div className="results-actions-row">
                <button className="btn btn-primary btn-lg" onClick={clearImage}>
                  <RefreshCw size={18} />
                  <span>Analyze another image</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
