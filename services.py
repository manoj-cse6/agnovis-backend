"""
Business logic services:
- Expert referral evaluation
- Follow-up schedule generation
- Alert condition checking
- Email notification dispatch (SMTP, configurable via env vars)

Environment variables for email:
  SMTP_HOST        - SMTP server hostname (e.g. smtp.gmail.com)
  SMTP_PORT        - SMTP server port (default 587)
  SMTP_USERNAME    - SMTP login username
  SMTP_PASSWORD    - SMTP login password
  SMTP_SENDER      - Sender email address
  ALERT_EMAILS     - Comma-separated list of team recipient emails
"""

import os
import smtplib
import logging
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# --- Email configuration from environment ---
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_SENDER = os.environ.get("SMTP_SENDER", "")
ALERT_EMAILS_RAW = os.environ.get("ALERT_EMAILS", "")

# Alert trigger thresholds (configurable via env vars)
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("ALERT_HIGH_CONFIDENCE", "0.80"))
REPEATED_DETECTION_DAYS = int(os.environ.get("ALERT_REPEATED_DAYS", "30"))


def get_alert_recipients() -> List[str]:
    """Return parsed team email list from ALERT_EMAILS env var."""
    if not ALERT_EMAILS_RAW:
        return []
    return [e.strip() for e in ALERT_EMAILS_RAW.split(",") if e.strip()]


# ─── Expert Referral ──────────────────────────────────────────────────────────

def evaluate_expert_referral(
    disease_confidence: float,
    warning: Optional[str]
) -> Dict[str, Any]:
    """
    Expert referral decision service.
    Recommends expert or lab confirmation when AI confidence is low
    or a crop-mismatch/low-confidence warning was triggered.
    """
    if warning or disease_confidence < 0.65:
        return {
            "recommended": True,
            "reason": "AI confidence is low or crop mismatch detected. Expert or laboratory confirmation is recommended.",
            "referral_options": [
                "Agricultural extension officer",
                "Plant diagnostic laboratory"
            ]
        }
    return {
        "recommended": False,
        "reason": None,
        "referral_options": None
    }


# ─── Follow-up scheduling ─────────────────────────────────────────────────────

def generate_follow_up(disease: str, disease_confidence: float) -> Dict[str, Any]:
    """
    Recommend a follow-up period based on disease type.
    """
    disease_lower = disease.lower()

    if "healthy" in disease_lower:
        return {
            "recommended": False,
            "recommended_in_days": None,
            "recommended_date": None,
            "status": "pending",
            "message": "Crop appears healthy. Regular monitoring is sufficient."
        }

    if "blight" in disease_lower or "rot" in disease_lower:
        days = 3
    elif "rust" in disease_lower or "scab" in disease_lower:
        days = 5
    else:
        days = 7

    recommended_date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")

    return {
        "recommended": True,
        "recommended_in_days": days,
        "recommended_date": recommended_date,
        "status": "pending",
        "message": f"Monitor the affected crop and upload a follow-up image in {days} days."
    }


# ─── Alert logic ─────────────────────────────────────────────────────────────

def check_alert_conditions(
    db: Session,
    analysis_id: int,
    crop: str,
    disease: str,
    disease_confidence: float,
    pest_detected: Optional[str],
    risk_level: str,
    expert_referral_recommended: bool,
    field_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Evaluate whether an alert should be triggered for a new analysis.
    Returns an alert dict if triggered, else None.

    Rules:
    1. Disease confidence >= HIGH_CONFIDENCE_THRESHOLD
    2. Risk level is High
    3. Expert referral was triggered
    4. Repeated detection of same disease in same field in last N days
    """
    import models  # local import to avoid circular dependency

    triggers = []

    # Rule 1: High confidence detection
    if disease_confidence >= HIGH_CONFIDENCE_THRESHOLD and "healthy" not in disease.lower():
        triggers.append({
            "reason": "high_confidence",
            "detail": (
                f"High-confidence detection: {disease} on {crop} "
                f"(AI confidence: {round(disease_confidence * 100, 1)}%). "
                f"This is classified as a definitive AI detection, not a laboratory confirmation."
            )
        })

    # Rule 2: High weather risk
    if risk_level == "High":
        triggers.append({
            "reason": "high_risk",
            "detail": (
                f"Weather and disease combination classified as High Risk for {crop}. "
                f"Conditions may accelerate disease spread if untreated."
            )
        })

    # Rule 3: Expert referral triggered
    if expert_referral_recommended:
        triggers.append({
            "reason": "expert_referral",
            "detail": (
                f"AI confidence or mismatch conditions triggered an automatic expert referral recommendation "
                f"for {disease} on {crop}."
            )
        })

    # Rule 4: Repeated detection in same field
    if field_id and "healthy" not in disease.lower():
        cutoff = datetime.utcnow() - timedelta(days=REPEATED_DETECTION_DAYS)
        prior_count = (
            db.query(models.Analysis)
            .filter(
                models.Analysis.field_id == field_id,
                models.Analysis.disease == disease,
                models.Analysis.created_at >= cutoff,
                models.Analysis.id != analysis_id,
            )
            .count()
        )
        if prior_count > 0:
            triggers.append({
                "reason": "repeated_detection",
                "detail": (
                    f"Repeated detection: {disease} has been detected {prior_count} additional time(s) "
                    f"in this field in the past {REPEATED_DETECTION_DAYS} days. "
                    f"This indicates possible ongoing disease pressure — NOT confirmed spread."
                )
            })

    if not triggers:
        return None

    primary = triggers[0]
    return {
        "analysis_id": analysis_id,
        "trigger_reason": primary["reason"],
        "trigger_detail": " | ".join(t["detail"] for t in triggers),
        "crop": crop,
        "disease": disease,
        "disease_confidence": disease_confidence,
        "pest_detected": pest_detected,
        "risk_level": risk_level,
        "_all_triggers": triggers,
    }


# ─── Email dispatch ───────────────────────────────────────────────────────────

def build_alert_email_body(alert_data: Dict[str, Any], analysis_data: Optional[Dict] = None) -> str:
    """Build a plain-text email body for the team alert."""
    lines = [
        "SIH 26131 — Crop Health Alert",
        "=" * 50,
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Analysis ID: {alert_data.get('analysis_id', 'N/A')}",
        "",
        "DETECTION SUMMARY",
        "-" * 30,
        f"Crop:               {alert_data.get('crop', 'N/A')}",
        f"Disease:            {alert_data.get('disease', 'N/A')}",
        f"AI Confidence:      {round((alert_data.get('disease_confidence') or 0) * 100, 1)}%",
        f"Pest Detected:      {alert_data.get('pest_detected') or 'None'}",
        f"Field:              {alert_data.get('field_name') or 'N/A'}",
        f"Location:           {alert_data.get('location_name') or 'N/A'}",
        f"Coordinates:        {alert_data.get('latitude') or 'N/A'}, {alert_data.get('longitude') or 'N/A'}",
        f"Risk Level:         {alert_data.get('risk_level', 'N/A')}",
        "",
        "ALERT REASON",
        "-" * 30,
        alert_data.get("trigger_detail", "N/A"),
        "",
        "IMPORTANT DISCLAIMER",
        "-" * 30,
        "This alert is generated by an AI detection system.",
        "Confidence scores represent AI model output — NOT laboratory confirmation.",
        "For definitive diagnosis, consult a certified agricultural expert or laboratory.",
        "",
        "This email was sent automatically by the SIH 26131 Crop Health backend.",
    ]
    return "\n".join(lines)


def send_team_alert_email(alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an alert email to the configured team recipients.
    Returns a dict with success/failure status and any error message.
    """
    recipients = get_alert_recipients()
    if not recipients:
        logger.warning("ALERT_EMAILS not configured. Skipping email send.")
        return {"sent": False, "error": "ALERT_EMAILS not configured"}

    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER]):
        logger.warning("SMTP credentials incomplete. Skipping email send.")
        return {"sent": False, "error": "SMTP configuration incomplete"}

    subject = (
        f"[CROP ALERT] {alert_data.get('risk_level', '')} Risk — "
        f"{alert_data.get('disease', 'Unknown disease')} on {alert_data.get('crop', 'Unknown crop')}"
    )
    body = build_alert_email_body(alert_data)

    msg = EmailMessage()
    msg["From"] = SMTP_SENDER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Alert email sent to {recipients}")
        return {"sent": True, "recipients": recipients, "error": None}
    except Exception as exc:
        logger.error(f"Failed to send alert email: {exc}")
        return {"sent": False, "recipients": recipients, "error": str(exc)}
