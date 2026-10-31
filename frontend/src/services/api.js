/**
 * Centralized API Service for DermaVision AI Backend (Phase 33)
 * Communicates with FastAPI server via configurable VITE_API_BASE_URL.
 */

const rawApiUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";
// Normalize localhost to 127.0.0.1 to avoid Safari WebKit IPv6 ::1 connection refusal on macOS
const API_BASE_URL = rawApiUrl.replace("localhost", "127.0.0.1").replace(/\/$/, "");


/**
 * Check Backend Liveness & Readiness Health Status
 * @returns {Promise<{status: string, model_loaded: boolean, readiness: string}>}
 */
export async function fetchHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Health check returned status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn("Backend health check failed:", error.message);
    return {
      status: "offline",
      model_loaded: false,
      readiness: "not_ready",
    };
  }
}

/**
 * Fetch Frozen Production Model Metadata
 * @returns {Promise<Object>}
 */
export async function fetchModelInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/model`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Model info fetch returned status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn("Model info fetch failed:", error.message);
    return null;
  }
}

/**
 * Check AI Guidance Service Status & Readiness
 * @returns {Promise<Object>}
 */
export async function fetchAiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/ai/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`AI status fetch returned status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn("AI status fetch failed:", error.message);
    return {
      available: false,
      provider: "unavailable",
      mode: "none",
    };
  }
}

/**
 * Submit Image File for Skin Lesion Classification
 * @param {File} imageFile - Validated Image file object
 * @returns {Promise<Object>} Structured prediction JSON
 */
export async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || "An unexpected error occurred.";
      if (response.status === 415) {
        throw new Error("The uploaded file format is not supported. Please use JPG, JPEG, PNG, or WEBP.");
      } else if (response.status === 413) {
        throw new Error("The selected image file is too large. Maximum allowed size is 15 MB.");
      } else if (response.status === 400) {
        throw new Error(detail.includes("empty") ? "The uploaded file is empty." : "The selected file could not be read as an image.");
      } else if (response.status === 503) {
        throw new Error("Analysis service is currently initializing or unavailable. Please try again in a moment.");
      } else {
        throw new Error(detail);
      }
    }

    if (!data || !data.success || !data.prediction) {
      throw new Error("Received an unexpected response structure from the analysis server.");
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError || error.name === "TypeError") {
      throw new Error("Unable to connect to the analysis service. Please make sure the FastAPI backend is running.");
    }
    throw error;
  }
}

/**
 * Submit Image File for Grad-CAM Explainability Generation
 * @param {File} imageFile - Validated Image file object
 * @returns {Promise<Object>} Structured explainability JSON with base64 Data URLs
 */
export async function explainImage(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/explain`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || "Failed to generate Grad-CAM explanation.";
      throw new Error(detail);
    }

    if (!data || !data.success || !data.images) {
      throw new Error("Received an invalid response structure from the explainability service.");
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError || error.name === "TypeError") {
      throw new Error("The prediction succeeded, but the visual explanation service is currently unavailable.");
    }
    throw error;
  }
}

/**
 * Submit Structured Patient Guidance Request to FastAPI Backend
 * @param {Object} payload - Guidance request containing prediction, question, context
 * @returns {Promise<Object>} Structured patient guidance response
 */
export async function fetchPatientGuidance(payload) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/guidance`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || "Educational assistant is temporarily unavailable. Your image analysis is still available.";
      throw new Error(detail);
    }

    if (!data || !data.success || !data.answer_markdown) {
      throw new Error("Received an invalid response from the educational guidance service.");
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError || error.name === "TypeError") {
      throw new Error("Educational assistant is temporarily unavailable. Your image analysis is still available.");
    }
    throw error;
  }
}

/**
 * Fetch Related Peer-Reviewed Biomedical Scientific Literature for Predicted Condition
 * @param {string} condition - Predicted class name
 * @returns {Promise<Object>} Structured research JSON
 */
export async function fetchResearchArticles(condition) {
  try {
    const encoded = encodeURIComponent(condition || "Melanoma");
    const response = await fetch(`${API_BASE_URL}/api/v1/research/${encoded}`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`Research fetch returned status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.warn("Scientific literature fetch failed:", error.message);
    return {
      success: false,
      condition: condition || "Melanoma",
      articles: [],
      message: "Scientific literature is temporarily unavailable. Analysis results remain available.",
    };
  }
}
