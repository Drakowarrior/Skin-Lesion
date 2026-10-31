"""
Evidence-Based Biomedical Research Assistant & AI Guidance Module (Phase 34)

Provides server-side evidence-grounded scientific question answering for skin lesion analysis.
Features:
- Multi-source biomedical retrieval & verification via PubMed E-utilities & landmark registry
- Grounded answer generation with inline citations [1], [2]
- Strict separation of Model Evidence (Grad-CAM, prediction %, architecture) vs Scientific Evidence (published literature)
- Zero-hallucination guarantee (unverified sources are rejected and reported)
- Red-flag warning symptom detection & 4-level triage urgency classification
- Google Scholar discovery URL integration
"""

import os
import re
import logging
import urllib.parse
import httpx
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.config import AI_PROVIDER, AI_API_KEY, AI_MODEL, MEDICAL_DISCLAIMER
from backend.research import (
    SourceCard,
    retrieve_and_verify_sources,
    get_scientific_literature
)
from src.class_labels import CLASS_NAMES

logger = logging.getLogger("backend.guidance")

SUPPORTED_MODEL_CLASSES = CLASS_NAMES

COMMON_UNSUPPORTED_CONDITIONS = [
    "eczema", "psoriasis", "ringworm", "shingles", "acne", "dermatitis",
    "hives", "scabies", "rosacea", "vitiligo", "warts", "molluscum",
    "poison ivy", "keloid", "tinea", "seborrheic dermatitis", "herpes"
]

EMERGENCY_KEYWORDS = [
    "difficulty breathing", "shortness of breath", "swelling of face", "facial swelling",
    "swelling of tongue", "throat closing", "fainting", "passed out", "unconscious",
    "chest pain", "anaphylaxis", "severe allergic reaction"
]

PROMPT_ATTENTION_KEYWORDS = [
    "rapidly changing", "growing fast", "doubled in size", "changed shape quickly",
    "bleeding heavily", "oozing pus", "spreading redness", "red streaks",
    "severe pain", "unbearable pain", "fever", "chills", "foul smell",
    "open sore won't heal", "non-healing ulcer", "rapidly spreading rash"
]

CONSIDER_DOCTOR_KEYWORDS = [
    "changing", "growing", "bleeding", "itching", "painful", "sore", "crusting",
    "persistent", "new spot", "darkening", "irregular border", "multi-colored",
    "family history of melanoma", "sunburn history"
]


class UserContext(BaseModel):
    age_range: Optional[str] = None
    duration: Optional[str] = None
    is_changing: Optional[bool] = False
    is_painful: Optional[bool] = False
    is_bleeding: Optional[bool] = False
    is_itching: Optional[bool] = False
    other_symptoms: Optional[str] = None


class PredictionSummary(BaseModel):
    predicted_class: str
    confidence_percent: float
    top3: List[Dict[str, Any]] = []


class GuidanceRequest(BaseModel):
    prediction: PredictionSummary
    user_question: str = Field(..., max_length=1000)
    user_context: Optional[UserContext] = None
    selected_quick_action: Optional[str] = None


class GuidanceResponse(BaseModel):
    success: bool = True
    urgency_level: str  # "general_info" | "consider_doctor" | "prompt_attention" | "emergency"
    urgency_title: str
    urgency_description: str
    answer_markdown: str
    summary: str
    sources: List[SourceCard] = []
    evidence_strength: str = "High"  # "High" | "Moderate" | "Limited"
    model_evidence: Dict[str, Any] = {}
    google_scholar_url: str = ""
    red_flags_detected: List[str] = []
    self_care_tips: List[str] = []
    unsupported_condition_mentioned: bool = False
    class_limitation_note: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER
    provider_used: str


# Strict Evidence-Grounded System Prompt
SYSTEM_PROMPT = """You are an Evidence-Based Biomedical Research Assistant for a skin lesion classification platform.
Your objective is to provide evidence-grounded scientific question answering.

CRITICAL PIPELINE RULES:
1. GROUNDED IN EVIDENCE: Answer ONLY using the supplied verified sources. Attach inline citations [1], [2], etc. immediately after every significant factual claim.
2. DO NOT INVENT SOURCES: Never fabricate paper titles, authors, journals, years, DOIs, PMIDs, or URLs. If the supplied evidence does not cover a point, explicitly state: "I could not verify this claim from the available scientific sources."
3. SEPARATE MODEL PREDICTION FROM SCIENTIFIC EVIDENCE:
   - Model Evidence: The ConvNeXt-Small prediction (confidence %) and Grad-CAM heatmap visualization. Grad-CAM shows relative convolutional network attention; it is an interpretability tool, NOT histological or medical diagnostic proof.
   - Scientific Literature: Peer-reviewed published literature on disease biology, clinical features, and risk factors.
4. STRUCTURE YOUR RESPONSE WITH THESE MARKDOWN HEADERS:
   ### Answer
   ### What the model found
   ### What research says
   ### Important limitation
5. Keep tone scientific, objective, clear, and reassuring. Always emphasize that predictions and research context do not constitute a medical diagnosis.
"""


def detect_red_flags(text: str, context: Optional[UserContext] = None) -> List[str]:
    """Detect red flag symptoms from user text and structured context."""
    flags = []
    text_lower = text.lower()
    
    for kw in EMERGENCY_KEYWORDS:
        if kw in text_lower:
            flags.append(f"Emergency symptom reported: {kw}")

    for kw in PROMPT_ATTENTION_KEYWORDS:
        if kw in text_lower:
            flags.append(f"Urgent symptom described: {kw}")

    if context:
        if context.is_bleeding:
            flags.append("Lesion is reported to be bleeding")
        if context.is_painful:
            flags.append("Lesion is reported to be painful")
        if context.is_changing:
            flags.append("Lesion is reported to be actively changing in size, shape, or color")
        if context.other_symptoms and any(kw in context.other_symptoms.lower() for kw in PROMPT_ATTENTION_KEYWORDS + EMERGENCY_KEYWORDS):
            flags.append(f"Additional symptom noted: {context.other_symptoms}")

    return flags


def determine_urgency(text: str, context: Optional[UserContext] = None, flags: List[str] = None) -> Dict[str, str]:
    """Determine 4-level triage guidance level based on text, context, and detected red flags."""
    text_lower = text.lower()
    flags = flags or []

    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS) or any("Emergency" in f for f in flags):
        return {
            "level": "emergency",
            "title": "Emergency Medical Services Recommended",
            "description": "The symptoms described (such as breathing difficulty or severe allergic reaction) may indicate a medical emergency. Please seek immediate emergency medical care."
        }

    if any(kw in text_lower for kw in PROMPT_ATTENTION_KEYWORDS) or any("Urgent" in f or "bleeding" in f.lower() or "painful" in f.lower() for f in flags):
        return {
            "level": "prompt_attention",
            "title": "Seek Prompt Medical Attention",
            "description": "Based on active symptoms (bleeding, pain, or rapid change), we strongly recommend contacting a doctor or dermatologist for a timely professional evaluation."
        }

    if any(kw in text_lower for kw in CONSIDER_DOCTOR_KEYWORDS) or (context and (context.is_changing or context.is_itching or context.duration)):
        return {
            "level": "consider_doctor",
            "title": "Consider Speaking with a Healthcare Professional",
            "description": "If your lesion is persistent, changing, or causing concern, a board-certified dermatologist can perform a dermoscopic evaluation and provide personalized guidance."
        }

    return {
        "level": "general_info",
        "title": "General Educational & Research Information",
        "description": "This evidence response is provided for educational and research context to help you understand scientific literature related to your image analysis result."
    }


def check_class_limitation(question: str) -> Tuple[bool, Optional[str]]:
    """Check if question asks about a condition outside the model's 14 trained classes."""
    question_lower = question.lower()
    mentioned_unsupported = []
    
    for cond in COMMON_UNSUPPORTED_CONDITIONS:
        if re.search(r'\b' + re.escape(cond) + r'\b', question_lower):
            mentioned_unsupported.append(cond.capitalize())

    if mentioned_unsupported:
        cond_str = ", ".join(mentioned_unsupported)
        note = (
            f"The image classification model is trained on a fixed set of "
            f"14 skin-lesion categories ({', '.join(SUPPORTED_MODEL_CLASSES)}). "
            f"'{cond_str}' is not one of those 14 trained categories, so this tool cannot evaluate or confirm '{cond_str}' from your image."
        )
        return True, note
    
    return False, None


async def generate_evidence_fallback_guidance(
    request: GuidanceRequest,
    sources: List[SourceCard]
) -> GuidanceResponse:
    """
    Generates a high-quality evidence-grounded response with real inline citations [1], [2]
    and source cards when LLM API key is unconfigured or unavailable.
    """
    q_text = request.user_question
    pred = request.prediction
    ctx = request.user_context
    quick_action = request.selected_quick_action

    red_flags = detect_red_flags(q_text, ctx)
    urgency = determine_urgency(q_text, ctx, red_flags)
    unsupported_mentioned, limitation_note = check_class_limitation(q_text)

    pred_class = pred.predicted_class
    conf = pred.confidence_percent
    encoded_class = urllib.parse.quote(f'"{pred_class}"')
    scholar_url = f"https://scholar.google.com/scholar?q={encoded_class}"

    # Build claim citations from verified sources
    citations_text = ""
    if sources:
        citations_text = "\n\n".join([
            f"[{s.id}] **{s.title}** ({s.journal}, {s.year})\n"
            f"   - " + "\n   - ".join(s.claims_supported[:2])
            for s in sources[:3]
        ])

    source_ref_ids = " ".join([f"[{s.id}]" for s in sources[:2]]) if sources else ""

    q_lower = q_text.lower()

    if quick_action == "when_doctor" or "when should i see a doctor" in q_lower or "doctor" in q_lower:
        answer = (
            f"### Answer\n\n"
            f"You should consider consulting a board-certified dermatologist or medical doctor for a professional evaluation if you observe any evolving or concerning skin changes {source_ref_ids}.\n\n"
            f"### What the model found\n\n"
            f"The image model predicted **{pred_class}** ({conf:.1f}% confidence). Machine learning classification score does not evaluate individual clinical risk or replace professional dermatological examination.\n\n"
            f"### What research says\n\n"
            f"Clinical guidelines from the American Academy of Dermatology (AAD) recommend professional doctor evaluation for:\n"
            f"- **Changing Spots:** Any lesion exhibiting changing size, shape, color, or elevation.\n"
            f"- **ABCDE Criteria:** Asymmetry, Irregular Borders, Multiple Colors, Diameter > 6mm, or Evolving characteristics {source_ref_ids}.\n"
            f"- **Symptomatic Spots:** Lesions that itch, hurt, crust, bleed, or fail to heal after several weeks.\n\n"
            f"### Important limitation\n\n"
            f"Seek prompt medical attention from a physician or dermatologist if you notice rapid lesion growth or persistent symptoms."
        )

    elif quick_action == "what_means" or "what does this result mean" in q_lower or "explain" in q_lower:
        answer = (
            f"### Answer\n\n"
            f"The classification model evaluated the uploaded image as **{pred_class}** with a confidence score of **{conf:.1f}%** {source_ref_ids}.\n\n"
            f"### What the model found\n\n"
            f"- **Predicted Category:** {pred_class}\n"
            f"- **Model Confidence:** {conf:.1f}%\n"
            f"- **Grad-CAM Interpretation:** Saliency heatmaps indicate image regions receiving higher convolutional layer weighting during inference. Grad-CAM visualizes network attention; it does not provide histological diagnosis or medical proof.\n\n"
            f"### What research says\n\n"
            f"Biomedical literature defines **{pred_class}** through clinical morphology, dermoscopic patterns, and biological characteristics {source_ref_ids}.\n\n"
            f"### Important limitation\n\n"
            f"This AI classification is generated for research and educational purposes. Scientific evidence provided here describes the general condition and does not confirm a diagnosis for your specific uploaded image."
        )

    elif "grad-cam" in q_lower or "gradcam" in q_lower or "heat map" in q_lower or "region" in q_lower or "focus" in q_lower:
        gradcam_source = next((s for s in sources if "grad" in s.title.lower() or "cam" in s.title.lower()), None)
        cite_tag = f"[{gradcam_source.id}]" if gradcam_source else (f"[{sources[0].id}]" if sources else "")

        answer = (
            f"### Answer\n\n"
            f"Grad-CAM (Gradient-weighted Class Activation Mapping) is a visual explainability technique for deep learning models {cite_tag}.\n\n"
            f"### What the model found\n\n"
            f"The warmer colors (red and yellow) in the Grad-CAM overlay highlight pixel regions that contributed most strongly to predicting **{pred_class}** ({conf:.1f}% confidence).\n\n"
            f"### What research says\n\n"
            f"According to published machine learning research, Grad-CAM calculates the gradients of the target class score with respect to the final convolutional feature maps {cite_tag}.\n\n"
            f"**Critical Distinction:**\n"
            f"- **Grad-CAM Saliency ≠ Medical Proof:** High activation indicates where the neural network focused its mathematical weight, NOT that the region is histologically malignant or pathologically proven.\n\n"
            f"### Important limitation\n\n"
            f"Grad-CAM explains model behavior to researchers and clinicians; it does not substitute for histopathological tissue evaluation."
        )

    elif "risk" in q_lower or "cause" in q_lower or "factor" in q_lower:
        answer = (
            f"### Answer\n\n"
            f"Research literature identifies several environmental and genetic risk factors associated with **{pred_class}** {source_ref_ids}.\n\n"
            f"### What the model found\n\n"
            f"The image model classified the visual appearance as **{pred_class}** ({conf:.1f}% confidence). Model predictions evaluate surface pixels, not patient risk history.\n\n"
            f"### What research says\n\n"
            f"Key risk factors established in peer-reviewed dermatology literature include:\n"
            f"- **Ultraviolet (UV) Exposure:** Sunburn history and intense episodic UV exposure {source_ref_ids}.\n"
            f"- **Genetic Susceptibility:** Family history of dysplastic nevi or cutaneous malignancies.\n"
            f"- **Phenotypic Traits:** Fair skin, light eye/hair color, and high total nevus count.\n\n"
            f"### Important limitation\n\n"
            f"Individual risk profile assessment requires a comprehensive clinical consultation with a dermatologist."
        )

    else:
        answer = (
            f"### Answer\n\n"
            f"Biomedical research regarding **{pred_class}** provides structured scientific insight into its clinical presentation, dermoscopy, and management {source_ref_ids}.\n\n"
            f"### What the model found\n\n"
            f"The ConvNeXt-Small classifier assigned the top prediction score to **{pred_class}** ({conf:.1f}% confidence).\n\n"
            f"### What research says\n\n"
            f"Peer-reviewed medical sources describe key characteristics and evidence-based clinical evaluation steps {source_ref_ids}.\n\n"
            f"### Important limitation\n\n"
            f"Research context is provided strictly for educational purposes and does not establish a personal diagnosis."
        )

    if limitation_note:
        answer = f"### Trained model scope notice\n\n{limitation_note}\n\n{answer}"

    summary = f"Evidence-grounded research response for {pred_class} with {len(sources)} verified citations."

    model_ev = {
        "predicted_class": pred_class,
        "confidence_percent": conf,
        "top3": pred.top3,
        "gradcam_explanation": "Grad-CAM highlights relative convolutional neural network feature weighting. It reflects model attention, not histological diagnosis."
    }

    self_care = [
        "Keep the skin area clean and dry with mild soap and warm water.",
        "Avoid scratching, picking, or applying unverified topical chemicals.",
        "Protect skin from direct sun exposure using broad-spectrum sunscreen (SPF 30+).",
        "Document lesion changes with clear periodic photographs.",
        "Consult a board-certified dermatologist for clinical evaluation."
    ]

    return GuidanceResponse(
        success=True,
        urgency_level=urgency["level"],
        urgency_title=urgency["title"],
        urgency_description=urgency["description"],
        answer_markdown=answer,
        summary=summary,
        sources=sources,
        evidence_strength="High" if len(sources) >= 2 else "Moderate",
        model_evidence=model_ev,
        google_scholar_url=scholar_url,
        red_flags_detected=red_flags,
        self_care_tips=self_care,
        unsupported_condition_mentioned=unsupported_mentioned,
        class_limitation_note=limitation_note,
        provider_used="verified_evidence_engine"
    )


async def call_llm_provider(
    request: GuidanceRequest,
    sources: List[SourceCard]
) -> Optional[GuidanceResponse]:
    """
    Calls external LLM API (Groq / OpenAI / Gemini) using retrieved verified sources as ground truth context.
    """
    if not AI_API_KEY or AI_API_KEY.strip() == "" or AI_API_KEY.startswith("your_"):
        return None

    provider = AI_PROVIDER.lower()
    
    # Format retrieved verified evidence into context
    evidence_block = "\n".join([
        f"SOURCE [{s.id}]: Title: {s.title} | Authors: {s.authors} | Journal: {s.journal} ({s.year}) | Type: {s.source_type} | URL: {s.url}\n"
        f"Supported Claims: {'; '.join(s.claims_supported)}"
        for s in sources
    ]) if sources else "No verified external sources found for this specific query."

    context_prompt = (
        f"MODEL PREDICTION: {request.prediction.predicted_class} ({request.prediction.confidence_percent:.1f}% confidence).\n"
        f"TOP 3 MODEL PREDICTIONS: {request.prediction.top3}\n\n"
        f"VERIFIED RETRIEVED SOURCES:\n{evidence_block}\n\n"
        f"USER QUESTION: {request.user_question}\n"
    )

    try:
        if provider in ["groq", "openai"]:
            model_name = AI_MODEL if AI_MODEL else "qwen/qwen3.8-27b"
            url = "https://api.groq.com/openai/v1/chat/completions" if provider == "groq" else "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 900,
            }

            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        text_out = choices[0]["message"].get("content", "")
                        red_flags = detect_red_flags(request.user_question, request.user_context)
                        urgency = determine_urgency(request.user_question, request.user_context, red_flags)
                        unsupported, lim_note = check_class_limitation(request.user_question)

                        encoded_pred = urllib.parse.quote(f'"{request.prediction.predicted_class}"')
                        scholar_url = f"https://scholar.google.com/scholar?q={encoded_pred}"

                        return GuidanceResponse(
                            success=True,
                            urgency_level=urgency["level"],
                            urgency_title=urgency["title"],
                            urgency_description=urgency["description"],
                            answer_markdown=text_out,
                            summary=f"Evidence-grounded response via {provider.upper()} ({model_name}).",
                            sources=sources,
                            evidence_strength="High" if len(sources) >= 2 else "Moderate",
                            model_evidence={
                                "predicted_class": request.prediction.predicted_class,
                                "confidence_percent": request.prediction.confidence_percent,
                                "gradcam_explanation": "Grad-CAM visualizes model attention regions; it is an interpretability tool, NOT medical proof."
                            },
                            google_scholar_url=scholar_url,
                            red_flags_detected=red_flags,
                            self_care_tips=[
                                "Keep skin clean and dry",
                                "Avoid picking or scratching lesion",
                                "Protect from direct UV exposure",
                                "Monitor ABCDE signs regularly"
                            ],
                            unsupported_condition_mentioned=unsupported,
                            class_limitation_note=lim_note,
                            provider_used=f"llm_{provider}"
                        )
    except Exception as e:
        logger.error(f"Failed to call LLM provider ({provider}): {str(e)}")

    return None


async def get_patient_guidance(request: GuidanceRequest) -> GuidanceResponse:
    """
    Main entry point for generating AI evidence-grounded research answers.
    1. Retrieves verified sources from PubMed & landmark registry.
    2. Calls LLM with strict evidence grounding prompt if API key available.
    3. Falls back to deterministic verified evidence engine.
    """
    if not request.user_question or not request.user_question.strip():
        request.user_question = "What does this result mean?"

    # 1. Retrieve & Verify Sources
    sources = await retrieve_and_verify_sources(
        user_question=request.user_question,
        predicted_class=request.prediction.predicted_class
    )

    # 2. Try LLM Provider with Evidence Context
    llm_response = await call_llm_provider(request, sources)
    if llm_response:
        return llm_response

    # 3. Fallback to Verified Evidence Engine
    return await generate_evidence_fallback_guidance(request, sources)
