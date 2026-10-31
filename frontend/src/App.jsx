import React, { useState } from "react";
import { usePrediction } from "./hooks/usePrediction";
import { Navbar } from "./components/Navbar";
import { AboutAndHowItWorks } from "./components/AboutAndHowItWorks";
import { AnalysisWorkspace } from "./components/AnalysisWorkspace";

import "./styles/globals.css";
import "./styles/components.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("about"); // Default Page 1: "about" | Page 2: "analyze"

  const {
    imageFile,
    imagePreview,
    imageMeta,
    isAnalyzing,
    predictionResult,
    error,
    isExplaining,
    explainResult,
    explainError,
    backendStatus,
    selectImage,
    clearImage,
    runAnalysis,
    fetchGradCAM,
  } = usePrediction();

  return (
    <div className="app-root">
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        backendOnline={backendStatus.online}
      />

      <main className="main-content-container py-8">
        {/* Page 1: About the Website & How It Works */}
        {activeTab === "about" && (
          <AboutAndHowItWorks onStartAnalysis={() => setActiveTab("analyze")} />
        )}

        {/* Page 2: Analyzing Feature (Image Upload, Prediction, Grad-CAM, Treatment & Precautions API) */}
        {activeTab === "analyze" && (
          <AnalysisWorkspace
            imageFile={imageFile}
            imagePreview={imagePreview}
            imageMeta={imageMeta}
            isAnalyzing={isAnalyzing}
            predictionResult={predictionResult}
            error={error}
            isExplaining={isExplaining}
            explainResult={explainResult}
            explainError={explainError}
            selectImage={selectImage}
            clearImage={clearImage}
            runAnalysis={runAnalysis}
            fetchGradCAM={fetchGradCAM}
          />
        )}
      </main>
    </div>
  );
}
