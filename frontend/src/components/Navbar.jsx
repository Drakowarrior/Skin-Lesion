import React from "react";

export function Navbar({ activeTab, onTabChange, backendOnline }) {
  return (
    <header className="apple-navbar-header">
      <div className="container apple-navbar-container">
        {/* Centered Segmented Pill Selector */}
        <div className="apple-segmented-pill" role="tablist" aria-label="Main Navigation">
          <button
            role="tab"
            aria-selected={activeTab === "about"}
            className={`apple-tab-btn ${activeTab === "about" ? "active" : ""}`}
            onClick={() => onTabChange("about")}
          >
            About & How It Works
          </button>

          <button
            role="tab"
            aria-selected={activeTab === "analyze"}
            className={`apple-tab-btn ${activeTab === "analyze" ? "active" : ""}`}
            onClick={() => onTabChange("analyze")}
          >
            Skin Lesion Analysis
          </button>
        </div>

        {/* Right-aligned Service Readiness Indicator */}
        <div className="apple-navbar-status">
          <div
            className={`status-badge-minimal ${backendOnline ? "online" : "offline"}`}
            title={backendOnline ? "Analysis & Guidance Service Ready" : "Backend Offline"}
          >
            <span className="status-dot"></span>
            <span className="status-text">{backendOnline ? "Service Ready" : "Service Offline"}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
