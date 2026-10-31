import { useState, useEffect, useCallback } from "react";
import { fetchHealth, fetchModelInfo, predictImage, explainImage } from "../services/api";

const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15MB

export function usePrediction() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageMeta, setImageMeta] = useState(null);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);

  // Grad-CAM Explainability state
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainResult, setExplainResult] = useState(null);
  const [explainError, setExplainError] = useState(null);

  const [backendStatus, setBackendStatus] = useState({
    online: false,
    checking: true,
  });
  const [modelInfo, setModelInfo] = useState(null);

  // Check health and model metadata on mount
  useEffect(() => {
    let isMounted = true;
    async function checkBackend() {
      const health = await fetchHealth();
      const info = await fetchModelInfo();
      if (isMounted) {
        setBackendStatus({
          online: health.status === "healthy" && health.model_loaded,
          checking: false,
        });
        if (info) {
          setModelInfo(info);
        }
      }
    }
    checkBackend();
    return () => { isMounted = false; };
  }, []);

  const formatFileSize = (bytes) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const selectImage = useCallback((file) => {
    setError(null);
    setPredictionResult(null);
    setExplainResult(null);
    setExplainError(null);

    if (!file) {
      setError("No file selected.");
      return false;
    }

    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError("Please upload a supported image format (JPG, PNG, or WEBP).");
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError("The selected image file is too large (max 15MB).");
      return false;
    }

    const objectUrl = URL.createObjectURL(file);
    setImageFile(file);
    setImagePreview(objectUrl);

    // Extract image dimensions asynchronously
    const img = new Image();
    img.onload = () => {
      setImageMeta({
        name: file.name,
        sizeStr: formatFileSize(file.size),
        width: img.width,
        height: img.height,
      });
    };
    img.onerror = () => {
      setImageMeta({
        name: file.name,
        sizeStr: formatFileSize(file.size),
        width: null,
        height: null,
      });
    };
    img.src = objectUrl;

    return true;
  }, []);

  const clearImage = useCallback(() => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImageFile(null);
    setImagePreview(null);
    setImageMeta(null);
    setPredictionResult(null);
    setExplainResult(null);
    setExplainError(null);
    setError(null);
  }, [imagePreview]);

  const fetchGradCAM = useCallback(async (file) => {
    const targetFile = file || imageFile;
    if (!targetFile) return;

    setIsExplaining(true);
    setExplainError(null);

    try {
      const expRes = await explainImage(targetFile);
      setExplainResult(expRes);
    } catch (err) {
      console.warn("Grad-CAM generation failed:", err.message);
      setExplainError("Grad-CAM visualization temporarily unavailable.");
    } finally {
      setIsExplaining(false);
    }
  }, [imageFile]);

  const runAnalysis = useCallback(async () => {
    if (!imageFile) {
      setError("Please select a skin lesion image to analyze.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setExplainResult(null);
    setExplainError(null);

    try {
      const result = await predictImage(imageFile);
      setPredictionResult(result);
      setIsAnalyzing(false);

      // Trigger Grad-CAM explainability after prediction
      fetchGradCAM(imageFile);
    } catch (err) {
      setError(err.message || "An unexpected error occurred during analysis.");
      setIsAnalyzing(false);
    }
  }, [imageFile, fetchGradCAM]);

  return {
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
    modelInfo,
    selectImage,
    clearImage,
    runAnalysis,
    fetchGradCAM,
  };
}
