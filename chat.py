"""
Agricultural AI assistant using the Google Gemini API.

Environment variables:
  GEMINI_API_KEY  - required for the Gemini API
  GEMINI_MODEL    - optional, defaults to gemini-2.5-flash
"""

import os
import logging
from typing import Optional, Dict, Any

from google import genai
from google.genai import types as genai_types
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["CHAT"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_INSTRUCTION = """You are an expert agricultural assistant for farmers in India.
Your role is to:
- Answer questions about crop diseases, pests, irrigation, fertilizers, and farming practices.
- Support multilingual conversations — respond in the same language the farmer uses.
- Explain AI crop analysis results in simple terms when context is provided.
- Always recommend consulting a certified agricultural expert or laboratory for:
  - Uncertain diagnoses (AI confidence is low)
  - High-risk disease situations
  - Exact pesticide dosage decisions
- Never fabricate or invent specific pesticide dosages or laboratory-confirmed diagnoses.
- Never claim a disease is confirmed unless a laboratory has confirmed it.
- Be compassionate and practical — farmers need actionable guidance, not jargon.
- If the farmer's crop analysis context is provided, use it to give relevant advice.
"""


class ChatRequest(BaseModel):
    message: str
    analysis_context: Optional[Dict[str, Any]] = None  # Full or partial predict result


class ChatResponse(BaseModel):
    reply: str
    model_used: str


def _build_context_block(context: Optional[Dict[str, Any]]) -> str:
    """Convert the analysis context into a human-readable block for the prompt."""
    if not context:
        return ""

    lines = ["[Current Crop Analysis Context]"]
    if context.get("crop"):
        lines.append(f"- Crop: {context['crop']}")
    if context.get("disease"):
        lines.append(f"- Detected disease: {context['disease']}")
    if context.get("disease_confidence") is not None:
        pct = round(context["disease_confidence"] * 100, 1)
        lines.append(f"- AI confidence: {pct}%")
    if context.get("pest_detected"):
        lines.append(f"- Pest detected: {context['pest_detected']}")
    if context.get("pest_confidence") and context["pest_confidence"] > 0:
        pct = round(context["pest_confidence"] * 100, 1)
        lines.append(f"- Pest confidence: {pct}%")
    if context.get("warning"):
        lines.append(f"- Warning: {context['warning']}")
    if context.get("recommended_action"):
        lines.append(f"- Recommended action: {context['recommended_action']}")
    if context.get("risk_assessment"):
        ra = context["risk_assessment"]
        lines.append(f"- Risk level: {ra.get('risk_level', 'Unknown')}")
        if ra.get("risk_reason"):
            lines.append(f"- Risk reason: {ra['risk_reason']}")
        if ra.get("weather"):
            w = ra["weather"]
            lines.append(f"- Weather data source: {w.get('source', 'unknown')}")
    return "\n".join(lines)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Multilingual agricultural AI assistant endpoint.
    Accepts a farmer's question and optional crop analysis context.
    Returns a helpful response from Gemini.
    """
    api_key = GEMINI_API_KEY
    model_name = GEMINI_MODEL

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Please set it in your environment variables."
        )

    context_block = _build_context_block(request.analysis_context)

    if context_block:
        full_prompt = f"{context_block}\n\n[Farmer's Question]\n{request.message}"
    else:
        full_prompt = request.message

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )
        reply_text = response.text
    except Exception as exc:
        logger.error(f"Gemini API error ({model_name}): {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error ({model_name}): {exc}"
        )

    return ChatResponse(reply=reply_text, model_used=model_name)
