import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  HelpCircle,
  MessageSquare,
  AlertTriangle,
  Info,
  ShieldAlert,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Loader2,
  Stethoscope,
  BookOpen,
  ExternalLink,
  FileText,
  Search,
  Layers,
  Award,
} from "lucide-react";
import { fetchPatientGuidance } from "../services/api";
import { deduplicateSources } from "../utils/deduplicateSources";

import { formatProbability } from "../utils/formatProbability";

const cleanMarkdown = (rawText) => {
  if (!rawText) return "";
  let text = rawText;
  text = text.replace(/\\([*_#\-+.!()\[\]{}])/g, "$1");
  text = text.replace(/\\\\/g, "\\");
  text = text.replace(/^\s*[*•+]\s+/gm, "- ");
  return text;
};

export function PatientGuidanceSection({ predictionResult }) {
  const [isOpen, setIsOpen] = useState(false);
  const [showContextForm, setShowContextForm] = useState(false);
  
  // Form State
  const [customQuestion, setCustomQuestion] = useState("");
  const [activeAction, setActiveAction] = useState("");
  const [ageRange, setAgeRange] = useState("");
  const [duration, setDuration] = useState("");
  const [isChanging, setIsChanging] = useState(false);
  const [isPainful, setIsPainful] = useState(false);
  const [isBleeding, setIsBleeding] = useState(false);
  const [isItching, setIsItching] = useState(false);

  // Response State
  const [isLoading, setIsLoading] = useState(false);
  const [guidanceData, setGuidanceData] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [expandedSourceId, setExpandedSourceId] = useState(null);
  const [highlightedSourceId, setHighlightedSourceId] = useState(null);

  // Reset guidance state when a new prediction arrives
  useEffect(() => {
    setGuidanceData(null);
    setErrorMsg(null);
    setCustomQuestion("");
    setActiveAction("");
    setExpandedSourceId(null);
    setHighlightedSourceId(null);
  }, [predictionResult]);

  if (!predictionResult || !predictionResult.prediction) return null;

  const pred = predictionResult.prediction;
  const top3 = predictionResult.top3 || [];

  const handleAsk = async (actionKey = null, overrideQuestion = null) => {
    setIsLoading(true);
    setErrorMsg(null);
    setIsOpen(true);
    if (actionKey) setActiveAction(actionKey);

    const questionText =
      overrideQuestion ||
      customQuestion.trim() ||
      (actionKey === "what_is"
        ? `What is ${pred.class}?`
        : actionKey === "risk_factors"
        ? `What are the risk factors for ${pred.class}?`
        : actionKey === "gradcam_explain"
        ? "What does this Grad-CAM heatmap mean?"
        : actionKey === "why_predicted"
        ? `Why did the model predict ${pred.class}?`
        : actionKey === "research_says"
        ? `What does scientific research say about ${pred.class}?`
        : actionKey === "when_doctor"
        ? "When should I see a doctor?"
        : `What does this result mean?`);

    const payload = {
      prediction: {
        predicted_class: pred.class,
        confidence_percent: pred.confidence ? pred.confidence * 100 : 0,
        top3: top3.map((t) => ({ class: t.class, confidence: t.confidence })),
      },
      user_question: questionText,
      selected_quick_action: actionKey,
      user_context: {
        age_range: ageRange || null,
        duration: duration || null,
        is_changing: isChanging,
        is_painful: isPainful,
        is_bleeding: isBleeding,
        is_itching: isItching,
      },
    };

    try {
      const response = await fetchPatientGuidance(payload);
      setGuidanceData(response);
    } catch (err) {
      console.warn("Research chatbot request failed:", err.message);
      setErrorMsg(err.message || "Research assistant service is temporarily unavailable.");
    } finally {
      setIsLoading(false);
    }
  };

  const scrollToSource = (sourceId) => {
    setHighlightedSourceId(sourceId);
    const element = document.getElementById(`source-card-${sourceId}`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    setTimeout(() => {
      setHighlightedSourceId(null);
    }, 2500);
  };

  const getSourceBadgeClass = (sourceType) => {
    const st = (sourceType || "").toLowerCase();
    if (st.includes("nih") | st.includes("nci") || st.includes("guideline")) return "badge-tier-1";
    if (st.includes("review") || st.includes("systematic")) return "badge-tier-review";
    return "badge-tier-peer";
  };

  const renderTextWithCitations = (contentStr) => {
    if (!contentStr) return null;
    const parts = contentStr.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const num = parseInt(match[1], 10);
        return (
          <button
            key={idx}
            className="inline-citation-badge"
            onClick={() => scrollToSource(num)}
            title={`Jump to verified source [${num}]`}
          >
            [{num}]
          </button>
        );
      }
      return part;
    });
  };

  return (
    <div className="guidance-section-container">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="guidance-header-card">
        <div className="guidance-header-left">
          <div className="guidance-icon-badge research-badge">
            <BookOpen size={22} className="guidance-sparkle-icon" />
          </div>
          <div>
            <div className="assistant-title-row">
              <h3 className="guidance-title">Authenticated Research Assistant</h3>
              <span className="research-status-pill">
                <Award size={13} />
                <span>Evidence-Grounded</span>
              </span>
            </div>
            <p className="guidance-subtitle">
              Evidence-based biomedical question answering for skin lesion analysis backed by verified PubMed literature.
            </p>
          </div>
        </div>
      </div>

      {/* ── Quick Action Research Prompts ───────────────────── */}
      <div className="quick-actions-grid" role="group" aria-label="Research Prompts">
        <button
          className={`quick-action-btn ${activeAction === "what_is" ? "active" : ""}`}
          onClick={() => handleAsk("what_is", `What is ${pred.class}?`)}
        >
          <BookOpen size={15} />
          <span>What is {pred.class}?</span>
        </button>

        <button
          className={`quick-action-btn ${activeAction === "risk_factors" ? "active" : ""}`}
          onClick={() => handleAsk("risk_factors", `What are the risk factors for ${pred.class}?`)}
        >
          <Info size={15} />
          <span>Risk factors & etiology</span>
        </button>

        <button
          className={`quick-action-btn ${activeAction === "gradcam_explain" ? "active" : ""}`}
          onClick={() => handleAsk("gradcam_explain", "What does this Grad-CAM heatmap mean?")}
        >
          <Layers size={15} />
          <span>Explain Grad-CAM heatmap</span>
        </button>

        <button
          className={`quick-action-btn ${activeAction === "why_predicted" ? "active" : ""}`}
          onClick={() => handleAsk("why_predicted", `Why did the model predict ${pred.class}?`)}
        >
          <Sparkles size={15} />
          <span>Why did model predict this?</span>
        </button>

        <button
          className={`quick-action-btn ${activeAction === "when_doctor" ? "active" : ""}`}
          onClick={() => handleAsk("when_doctor", "When should I see a doctor?")}
        >
          <Stethoscope size={15} />
          <span>When to see a doctor</span>
        </button>
      </div>

      {/* ── Custom Question Input Box ──────────────────────── */}
      <div className="guidance-input-box">
        <div className="custom-question-row">
          <div className="input-search-wrapper">
            <Search size={18} className="search-input-icon" />
            <input
              type="text"
              className="custom-question-input"
              placeholder={`Ask a scientific or research question about ${pred.class}...`}
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && customQuestion.trim()) {
                  handleAsk("custom", customQuestion);
                }
              }}
            />
          </div>
          <button
            className="btn btn-primary btn-ask"
            onClick={() => handleAsk("custom", customQuestion)}
            disabled={isLoading}
          >
            {isLoading ? <Loader2 size={16} className="animate-spin" /> : <span>Ask Assistant</span>}
          </button>
        </div>

        {/* Optional Context Toggle */}
        <div className="context-toggle-row">
          <button
            className="context-toggle-btn"
            onClick={() => setShowContextForm(!showContextForm)}
          >
            <span>Provide optional context for tailored literature inquiry</span>
            {showContextForm ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {/* Optional Context Form */}
        {showContextForm && (
          <div className="context-form-panel">
            <p className="context-form-note">
              All fields are optional and used strictly to refine evidence inquiry parameters.
            </p>

            <div className="context-grid">
              <div className="context-field">
                <label className="context-label">Age range:</label>
                <select
                  className="context-select"
                  value={ageRange}
                  onChange={(e) => setAgeRange(e.target.value)}
                >
                  <option value="">Select age (optional)</option>
                  <option value="under_18">Under 18</option>
                  <option value="18_35">18 – 35</option>
                  <option value="36_50">36 – 50</option>
                  <option value="51_65">51 – 65</option>
                  <option value="65_plus">65+</option>
                </select>
              </div>

              <div className="context-field">
                <label className="context-label">Duration present:</label>
                <select
                  className="context-select"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                >
                  <option value="">How long present? (optional)</option>
                  <option value="few_days">Few days</option>
                  <option value="1_4_weeks">1 – 4 weeks</option>
                  <option value="months">Several months</option>
                  <option value="years">Years / As long as I remember</option>
                </select>
              </div>
            </div>

            <div className="context-checkboxes-row">
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={isChanging}
                  onChange={(e) => setIsChanging(e.target.checked)}
                />
                <span>Changing in size, shape, or color</span>
              </label>

              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={isPainful}
                  onChange={(e) => setIsPainful(e.target.checked)}
                />
                <span>Painful or tender</span>
              </label>

              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={isBleeding}
                  onChange={(e) => setIsBleeding(e.target.checked)}
                />
                <span>Bleeding or oozing</span>
              </label>

              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={isItching}
                  onChange={(e) => setIsItching(e.target.checked)}
                />
                <span>Itching</span>
              </label>
            </div>
          </div>
        )}
      </div>

      {/* ── Research Assistant Response View ────────────────── */}
      {isOpen && (
        <div className="guidance-result-panel">
          {isLoading && (
            <div className="guidance-loading-state">
              <Loader2 size={26} className="animate-spin text-primary" />
              <p>Retrieving verified PubMed sources and synthesizing evidence...</p>
            </div>
          )}

          {!isLoading && errorMsg && (
            <div className="guidance-error-banner">
              <AlertTriangle size={18} />
              <span>{errorMsg}</span>
            </div>
          )}

          {!isLoading && guidanceData && (
            <div className="guidance-content-body">
              {/* Triage Urgency Header */}
              {guidanceData.urgency_level && guidanceData.urgency_level !== "general_info" && (
                <div className={`urgency-header-box ${guidanceData.urgency_level}`}>
                  <div className="urgency-title-row">
                    <AlertTriangle size={18} className="urgency-icon prompt" />
                    <h4 className="urgency-title">{guidanceData.urgency_title}</h4>
                  </div>
                  <p className="urgency-desc">{guidanceData.urgency_description}</p>
                </div>
              )}

              {/* Red Flags Warning Banner */}
              {guidanceData.red_flags_detected && guidanceData.red_flags_detected.length > 0 && (
                <div className="red-flags-warning-box">
                  <div className="warning-box-header">
                    <ShieldAlert size={20} className="warning-icon" />
                    <h5>Noticeable Warning Symptoms Reported</h5>
                  </div>
                  <ul className="warning-list">
                    {guidanceData.red_flags_detected.map((flag, idx) => (
                      <li key={idx}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Class Limitation Note */}
              {guidanceData.class_limitation_note && (
                <div className="limitation-note-box">
                  <Info size={18} className="info-icon" />
                  <p>{guidanceData.class_limitation_note}</p>
                </div>
              )}

              {/* ── Model Evidence vs. Scientific Evidence Card ───── */}
              <div className="evidence-dual-breakdown">
                {/* Left Card: Model Evidence */}
                <div className="evidence-card model-evidence-card">
                  <div className="evidence-card-header">
                    <Sparkles size={18} className="icon-model" />
                    <h4>Model Evidence</h4>
                  </div>
                  <div className="evidence-card-body">
                    <div className="evidence-stat-row">
                      <span>Predicted Condition:</span>
                      <strong>{pred.class}</strong>
                    </div>
                    <div className="evidence-stat-row">
                      <span>Calibrated Probability:</span>
                      <strong>{formatProbability(pred.confidence)}</strong>
                    </div>
                    <div className="evidence-desc-box">
                      <p>
                        <strong>Grad-CAM Interpretation:</strong> Visualizes relative convolutional neural network feature weighting. Heatmap intensity indicates image pixel regions receiving higher model weight.
                      </p>
                      <p className="gradcam-disclaimer-note">
                        ⚠️ Grad-CAM represents model attention; it does NOT establish histological diagnosis or medical proof.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Right Card: Scientific Literature */}
                <div className="evidence-card scientific-evidence-card">
                  <div className="evidence-card-header">
                    <BookOpen size={18} className="icon-scientific" />
                    <h4>Scientific Literature</h4>
                  </div>
                  <div className="evidence-card-body">
                    <div className="evidence-stat-row">
                      <span>Verified Sources:</span>
                      <strong>{guidanceData.sources?.length || 0} Sources</strong>
                    </div>
                    <div className="evidence-stat-row">
                      <span>Evidence Strength:</span>
                      <strong className="strength-badge">{guidanceData.evidence_strength || "High"}</strong>
                    </div>
                    <div className="evidence-desc-box">
                      <p>
                        Peer-reviewed publications indexed in PubMed and official clinical guidelines (NIH/NCI).
                      </p>
                      <p className="source-grounding-note">
                        ✓ Answers are synthesized strictly from retrieved verified sources with inline citations.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Formatted Answer with Interactive Inline Citations ── */}
              <div className="guidance-markdown-content evidence-answer-box">
                <ReactMarkdown
                  components={{
                    h2: ({ ...props }) => <h3 className="guidance-section-heading" {...props} />,
                    h3: ({ ...props }) => <h4 className="guidance-section-subheading" {...props} />,
                    p: ({ children }) => (
                      <p className="guidance-text-p">
                        {typeof children === "string" ? renderTextWithCitations(children) : children}
                      </p>
                    ),
                    li: ({ children }) => (
                      <li className="guidance-bullet-item">
                        {typeof children === "string" ? renderTextWithCitations(children) : children}
                      </li>
                    ),
                    strong: ({ ...props }) => <strong className="guidance-strong" {...props} />,
                  }}
                >
                  {cleanMarkdown(guidanceData.answer_markdown)}
                </ReactMarkdown>
              </div>

              {/* ── Verified Evidence & Sources Section ───────────── */}
              {(() => {
                const uniqueSources = deduplicateSources(guidanceData.sources || []);
                return (
                  <div className="verified-sources-section">
                    <div className="sources-section-header">
                      <div className="sources-header-title">
                        <FileText size={20} className="sources-header-icon" />
                        <h4>Evidence & Verified Sources</h4>
                      </div>
                      <span className="sources-count-badge">
                        {uniqueSources.length} Verified
                      </span>
                    </div>

                    {uniqueSources.length > 0 ? (
                      <div className="source-cards-stack">
                        {uniqueSources.map((src, sIdx) => {
                          const cardKey = src.id || `src-${sIdx + 1}`;
                          return (
                            <div
                              key={cardKey}
                              id={`source-card-${src.id || sIdx + 1}`}
                              className={`source-item-card ${highlightedSourceId === (src.id || sIdx + 1) ? "highlighted" : ""}`}
                            >
                              <div className="source-card-top">
                                <span className="source-number-badge">[{src.id || sIdx + 1}]</span>
                                <span className={`source-type-pill ${getSourceBadgeClass(src.source_type)}`}>
                                  {src.source_type}
                                </span>
                              </div>

                              <h5 className="source-card-title">{src.title}</h5>

                              <div className="source-card-meta">
                                <span className="source-authors">{src.authors}</span>
                                <span className="meta-sep">•</span>
                                <span className="source-journal">{src.journal}</span>
                                <span className="meta-sep">•</span>
                                <span className="source-year">{src.year}</span>
                              </div>

                              {/* Expandable Why This Source */}
                              <div className="why-source-box">
                                <button
                                  className="btn-why-source"
                                  onClick={() => setExpandedSourceId(expandedSourceId === cardKey ? null : cardKey)}
                                >
                                  <Info size={13} />
                                  <span>Why this source is relevant?</span>
                                  {expandedSourceId === cardKey ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                                </button>
                                {expandedSourceId === cardKey && (
                                  <div className="why-source-explanation">
                                    <p>{src.relevance_explanation}</p>
                                    {src.claims_supported && src.claims_supported.length > 0 && (
                                      <ul className="claims-supported-list">
                                        {src.claims_supported.map((claim, cIdx) => (
                                          <li key={cIdx}>✓ {claim}</li>
                                        ))}
                                      </ul>
                                    )}
                                  </div>
                                )}
                              </div>

                              {/* Verified External Action Links */}
                              <div className="source-action-buttons">
                                {src.pmid && (
                                  <a
                                    href={`https://pubmed.ncbi.nlm.nih.gov/${src.pmid}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="btn-source-action btn-pubmed"
                                  >
                                    <span>Open PubMed</span>
                                    <ExternalLink size={13} />
                                  </a>
                                )}
                                {src.doi && (
                                  <a
                                    href={`https://doi.org/${src.doi}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="btn-source-action btn-doi"
                                  >
                                    <span>Open DOI</span>
                                    <ExternalLink size={13} />
                                  </a>
                                )}
                                {src.url && (
                                  <a
                                    href={src.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="btn-source-action btn-fulltext"
                                  >
                                    <span>Open Source</span>
                                    <ExternalLink size={13} />
                                  </a>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="no-sources-fallback-box">
                        <Info size={18} />
                        <p>
                          No verified research sources were retrieved for this specific question.
                        </p>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Safety Disclaimer Footer */}
              <div className="guidance-disclaimer-footer">
                <p>
                  <strong>Research & Educational Disclaimer:</strong> {guidanceData.disclaimer || "AI-generated educational & research assistance. This tool does not establish a clinical diagnosis."}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
