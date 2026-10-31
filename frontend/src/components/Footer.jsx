import React from "react";
import { Activity, ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-container">
        <div className="footer-brand">
          <div className="brand-icon-sm">
            <Activity size={18} />
          </div>
          <span className="footer-brand-title">Skin Lesion Classifier</span>
        </div>

        <p className="footer-copyright">
          © 2026 Skin Lesion Classification Platform (ConvNeXt-Small). Designed for research & educational demonstration.
        </p>

        <div className="footer-links">
          <a href="#hero">Back to Top</a>
          <a href="#disclaimer">Safety Note</a>
        </div>
      </div>
    </footer>
  );
}
