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


# ─── Disease Cluster Detection ────────────────────────────────────────────────
#
# Configuration (all via environment variables):
#   CLUSTER_CASE_THRESHOLD     - min reports to trigger a cluster (default: 5)
#   CLUSTER_TIME_WINDOW_HOURS  - lookback window in hours (default: 24)
#   CLUSTER_GRID_SIZE          - coarse grid cell size in degrees (default: 1.0)
#                                1.0° ≈ 111 km — change to 0.5 for ~55 km
#
CLUSTER_CASE_THRESHOLD = int(os.environ.get("CLUSTER_CASE_THRESHOLD", "5"))
CLUSTER_TIME_WINDOW_HOURS = int(os.environ.get("CLUSTER_TIME_WINDOW_HOURS", "24"))
CLUSTER_GRID_SIZE = float(os.environ.get("CLUSTER_GRID_SIZE", "0.1"))

# Authority alert email (separate from team ALERT_EMAILS)
# Set AGRI_ALERT_EMAIL to notify an Agriculture Officer / Agronomist
AGRI_ALERT_EMAIL_RAW = os.environ.get("AGRI_ALERT_EMAIL", "")


def _coarse_grid(value: float, grid_size: float) -> float:
    """Round a coordinate to the nearest grid cell center."""
    return round(round(value / grid_size) * grid_size, 6)


def check_cluster_conditions(
    db: "Session",
    crop: str,
    disease: str,
    lat_grid: Optional[float],
    lon_grid: Optional[float],
    current_analysis_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Check whether the current analysis triggers a community disease cluster.

    Rules:
    - Same crop + same disease + same coarse grid cell
    - Within CLUSTER_TIME_WINDOW_HOURS
    - At least CLUSTER_CASE_THRESHOLD unique analyses (excluding the current one)
    - Healthy crops are never counted

    Returns a cluster payload dict if threshold is reached, else None.

    PRIVACY: Only coarse grid values are used. Individual farmer identities
    and precise coordinates are never accessed or returned here.
    """
    import models  # local import to avoid circular dependency

    if "healthy" in disease.lower():
        return None  # never cluster healthy detections

    if lat_grid is None or lon_grid is None:
        return None  # no location data — cannot cluster

    cutoff = datetime.utcnow() - timedelta(hours=CLUSTER_TIME_WINDOW_HOURS)

    # Count distinct analyses (excluding current) matching crop+disease+grid+window
    count = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.crop == crop,
            models.Analysis.disease == disease,
            models.Analysis.lat_grid == lat_grid,
            models.Analysis.lon_grid == lon_grid,
            models.Analysis.created_at >= cutoff,
            models.Analysis.id != current_analysis_id,
        )
        .count()
    )

    total = count + 1  # include current analysis
    if total < CLUSTER_CASE_THRESHOLD:
        return None

    return {
        "crop": crop,
        "disease": disease,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "case_count": total,
        "time_window_hours": CLUSTER_TIME_WINDOW_HOURS,
    }


def create_cluster_alert_and_record(
    db_session_factory,
    cluster_payload: Dict[str, Any],
) -> None:
    """
    Background task: persist/update the DiseaseCluster record and create an Alert.
    Called only when check_cluster_conditions() returns a non-None payload.

    PRIVACY: The Alert message contains only crop, disease, case count, and
    time window — no individual farmer names, IDs, or precise coordinates.
    """
    import models  # local import

    try:
        db = db_session_factory()
        crop = cluster_payload["crop"]
        disease = cluster_payload["disease"]
        lat_grid = cluster_payload["lat_grid"]
        lon_grid = cluster_payload["lon_grid"]
        case_count = cluster_payload["case_count"]
        time_window = cluster_payload["time_window_hours"]
        now = datetime.utcnow()

        # Upsert cluster record
        cluster = (
            db.query(models.DiseaseCluster)
            .filter(
                models.DiseaseCluster.crop == crop,
                models.DiseaseCluster.disease == disease,
                models.DiseaseCluster.lat_grid == lat_grid,
                models.DiseaseCluster.lon_grid == lon_grid,
                models.DiseaseCluster.status == "active",
            )
            .first()
        )

        if cluster:
            already_notified = cluster.alert_id is not None
            cluster.case_count = case_count
            cluster.last_detected = now
        else:
            already_notified = False
            cluster = models.DiseaseCluster(
                crop=crop,
                disease=disease,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                case_count=case_count,
                first_detected=now,
                last_detected=now,
                status="active",
            )
            db.add(cluster)
            db.flush()  # get cluster.id

        # Build farmer-friendly alert message (no individual identities)
        title = f"Possible Disease Cluster: {disease} on {crop}"
        message = (
            f"Multiple recent reports ({case_count}) of {disease} have been detected "
            f"in your area within the past {time_window} hours. "
            f"Increased monitoring of your {crop} crop is recommended. "
            f"This is an early-warning signal, NOT a confirmed outbreak."
        )

        alert = models.Alert(
            trigger_reason="disease_cluster",
            trigger_detail=message,
            crop=crop,
            disease=disease,
            # Store only coarse grid — not precise individual location
            latitude=lat_grid,
            longitude=lon_grid,
            risk_level="High",
            status="new",
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        if cluster.alert_id is None:
            cluster.alert_id = alert.id
            db.commit()

        # Notify agriculture officer ONLY on initial cluster escalation (deduplicated)
        if not already_notified:
            _notify_agri_authority(cluster_payload, alert.id)
        db.close()

    except Exception as exc:
        logger.error(f"Cluster alert creation failed: {exc}")


def _notify_agri_authority(cluster_payload: Dict[str, Any], alert_id: int) -> None:
    """
    Send a notification to the configured agriculture officer / agronomist.

    Configuration:
      AGRI_ALERT_EMAIL - authority contact email (NOT team recipients)

    If AGRI_ALERT_EMAIL is not set, this is a no-op and the in-app alert
    still works normally.
    """
    authority_email = AGRI_ALERT_EMAIL_RAW.strip()
    if not authority_email:
        logger.info("AGRI_ALERT_EMAIL not configured. Skipping authority notification.")
        return

    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER]):
        logger.warning("SMTP credentials incomplete. Cannot notify agri authority.")
        return

    crop = cluster_payload["crop"]
    disease = cluster_payload["disease"]
    count = cluster_payload["case_count"]
    hours = cluster_payload["time_window_hours"]

    subject = f"[AGNOVIS CLUSTER ALERT] Possible {disease} cluster on {crop}"
    body = "\n".join([
        "Agnovis — Possible Community Disease Cluster Detected",
        "=" * 55,
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Alert ID: {alert_id}",
        "",
        "CLUSTER SUMMARY",
        "-" * 30,
        f"Crop:               {crop}",
        f"Disease:            {disease}",
        f"Reports in window:  {count} (past {hours} hours)",
        f"Detection method:   Community aggregation (coarse geographic grid)",
        "",
        "IMPORTANT",
        "-" * 30,
        "This is an early-warning signal from the Agnovis AI system.",
        "It indicates multiple independent reports of the same disease",
        "in the same general area — NOT a laboratory-confirmed outbreak.",
        "Please investigate and advise farmers in the affected region.",
        "",
        "Individual farmer identities and precise locations are NOT included",
        "in this notification to protect farmer privacy.",
        "",
        "This email was sent automatically by the Agnovis Community Alert system.",
    ])

    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = SMTP_SENDER
        msg["To"] = authority_email
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Agri authority cluster alert sent to {authority_email}")
    except Exception as exc:
        logger.error(f"Failed to notify agri authority: {exc}")


# ─── Water Advisor ────────────────────────────────────────────────────────────
#
# Transparent rule-based irrigation advisory using weather forecast + disease.
# NO soil moisture sensors. NO invented data. NO exact litres.
# This is decision SUPPORT — not an automated irrigation controller.
#
# Disease-moisture association list (conservative — only well-documented links):
_MOISTURE_SENSITIVE_DISEASES = {
    "blight": "Blight diseases are worsened by wet and humid conditions.",
    "mildew": "Mildew spreads rapidly in high-humidity environments.",
    "rot": "Rot pathogens thrive in waterlogged or overly moist conditions.",
    "leaf spot": "Leaf spots can spread faster under wet foliage conditions.",
    "anthracnose": "Anthracnose is favored by warm, wet weather.",
}


def _get_disease_moisture_note(disease: str) -> Optional[str]:
    """Return a moisture note if the disease is known to be moisture-sensitive."""
    dl = disease.lower()
    for key, note in _MOISTURE_SENSITIVE_DISEASES.items():
        if key in dl:
            return note
    return None


def compute_water_advice(
    crop: str,
    disease: str,
    weather_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute a farmer-friendly irrigation recommendation.

    Inputs:
      crop         - crop name (for context)
      disease      - detected disease name
      weather_data - result from get_7day_forecast() or None

    Returns a dict matching WaterAdvisorResponse schema.

    Rules (applied in priority order):
    1. If significant rain is expected (>=60% probability in next 2 days)
         → Do Not Irrigate
    2. If disease is moisture-sensitive AND humidity/rain is elevated
         → Avoid excess watering (Light Irrigation at most)
    3. If conditions are hot/dry (temp>35, rain<1mm, precip_prob<20%)
         → Irrigation Recommended
    4. Otherwise → Moderate / Monitor

    No soil sensor data is used or fabricated.
    """
    is_healthy = "healthy" in disease.lower()
    has_weather = (
        weather_data is not None
        and weather_data.get("source") == "live"
        and weather_data.get("forecast")
    )

    # Extract weather signals
    today = weather_data["forecast"][0] if has_weather else {}
    tomorrow = weather_data["forecast"][1] if (has_weather and len(weather_data["forecast"]) > 1) else {}

    today_precip_prob = today.get("precipitation_probability_pct", 0) or 0
    today_temp_max = today.get("temperature_max_c") or 25
    today_rain = today.get("rainfall_mm", 0) or 0
    tomorrow_precip_prob = tomorrow.get("precipitation_probability_pct", 0) or 0
    current_humidity = (weather_data.get("current", {}) or {}).get("relative_humidity_2m") if has_weather else None
    weather_source = weather_data.get("source", "fallback/unavailable") if weather_data else "unavailable"

    # Max rain probability over today + tomorrow
    max_precip_prob = max(today_precip_prob, tomorrow_precip_prob)

    # Disease moisture consideration
    disease_moisture_note = None if is_healthy else _get_disease_moisture_note(disease)

    # Apply rules
    if not has_weather:
        # No weather data — give generic moderate advice
        return {
            "decision": "Monitor — check conditions",
            "water_need": "Moderate",
            "reason": (
                "Weather data is currently unavailable for your location. "
                "Monitor soil moisture and irrigate based on local conditions."
            ),
            "disease_consideration": (
                f"Your crop has {disease}. {disease_moisture_note}"
                if disease_moisture_note else None
            ),
            "next_check": "Check again tomorrow after weather data is available.",
            "weather_source": weather_source,
        }

    # Rule 1: Significant rain expected
    if max_precip_prob >= 60:
        disease_note = None
        if disease_moisture_note:
            disease_note = (
                f"Your crop has {disease}. {disease_moisture_note} "
                f"With rain expected, additional irrigation would increase disease pressure."
            )
        return {
            "decision": "Do Not Irrigate",
            "water_need": "Low",
            "reason": (
                f"Rain is expected soon (probability: {max_precip_prob}%). "
                f"Delay irrigation to avoid waterlogging and unnecessary moisture."
            ),
            "disease_consideration": disease_note,
            "next_check": "Recheck after the rain passes.",
            "weather_source": weather_source,
        }

    # Rule 2: Disease moisture-sensitive + elevated humidity/rain
    high_humidity = current_humidity is not None and current_humidity >= 75
    recent_rain = today_rain >= 3  # 3mm already fell today
    if disease_moisture_note and (high_humidity or recent_rain or max_precip_prob >= 40):
        return {
            "decision": "Light Irrigation Only",
            "water_need": "Low",
            "reason": (
                f"Humidity is elevated" if high_humidity else
                f"Some rain is expected ({max_precip_prob}%) or has recently fallen ({today_rain} mm)."
            ) + " Avoid excess watering to reduce disease pressure.",
            "disease_consideration": (
                f"Because your {crop} has {disease}: {disease_moisture_note} "
                f"Unnecessary irrigation can worsen conditions."
            ),
            "next_check": "Monitor disease progression. Recheck in 1–2 days.",
            "weather_source": weather_source,
        }

    # Rule 3: Hot and dry
    if today_temp_max >= 35 and today_rain <= 1 and max_precip_prob <= 20:
        return {
            "decision": "Irrigation Recommended",
            "water_need": "High",
            "reason": (
                f"Hot and dry conditions are expected — temperature up to {today_temp_max}°C "
                f"with little rainfall ({today_rain} mm) and low rain probability ({max_precip_prob}%). "
                f"Crops may experience water stress."
            ),
            "disease_consideration": (
                f"Your crop has {disease}. {disease_moisture_note} "
                f"Water carefully — avoid wetting foliage if possible."
                if disease_moisture_note else None
            ),
            "next_check": "Monitor crop stress daily in hot conditions.",
            "weather_source": weather_source,
        }

    # Rule 4: Moderate conditions
    return {
        "decision": "Moderate Irrigation — Monitor",
        "water_need": "Moderate",
        "reason": (
            f"Conditions are moderate — temperature around {today_temp_max}°C, "
            f"rainfall {today_rain} mm, rain probability {max_precip_prob}%. "
            f"Irrigate as needed based on crop appearance and soil feel."
        ),
        "disease_consideration": (
            f"Your crop has {disease}. {disease_moisture_note} "
            f"Avoid overwatering."
            if disease_moisture_note else None
        ),
        "next_check": "Recheck tomorrow.",
        "weather_source": weather_source,
    }
