import React, { useState, useEffect } from "react";
import { Loader2, CheckCircle2 } from "lucide-react";

export function LoadingState() {
  const [stepIndex, setStepIndex] = useState(0);

  const steps = [
    "Preparing image",
    "Running classification model",
    "Generating explanation"
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setStepIndex(1), 500);
    const timer2 = setTimeout(() => setStepIndex(2), 1200);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <div className="loading-card">
      <div className="loading-spinner-container">
        <Loader2 className="loading-spinner animate-spin" size={36} />
      </div>

      <h3 className="loading-title">Analyzing image...</h3>

      <div className="loading-steps-container">
        {steps.map((stepText, idx) => {
          const isDone = idx < stepIndex;
          const isCurrent = idx === stepIndex;
          return (
            <div
              key={idx}
              className={`loading-step-item ${isDone ? "done" : ""} ${isCurrent ? "current" : ""}`}
            >
              <div className="step-icon-col">
                {isDone ? (
                  <CheckCircle2 size={16} className="step-check" />
                ) : isCurrent ? (
                  <Loader2 size={16} className="step-loader animate-spin" />
                ) : (
                  <div className="step-dot" />
                )}
              </div>
              <span className="step-text">{idx + 1}. {stepText}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

