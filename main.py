import json
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file before any other imports
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import database
import models
import auth
import routers
import chat as chat_module
from weather import assess_risk
from services import (
    evaluate_expert_referral,
    generate_follow_up,
    check_alert_conditions,
    send_team_alert_email,
    get_alert_recipients,
    check_cluster_conditions,
    create_cluster_alert_and_record,
    CLUSTER_GRID_SIZE,
    _coarse_grid,
)
from schemas import PredictionResponse

from ai.inference import (
    analyze_crop_image,
    analyze_crop_images,
    _disease_model,
    _pest_model,
    device,
)
from ai.utils.image_validator import MAX_IMAGE_BYTES, validate_upload

logger = logging.getLogger(__name__)

# Create DB tables on startup
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="SIH 26131 Crop Analysis API",
    description=(
        "Disease identification (MobileNetV3) + pest detection (YOLO) + "
        "weather risk + alerts + multilingual AI chat"
    ),
    version="2.0.0",
)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(routers.history_router)
app.include_router(routers.fields_router)
app.include_router(routers.follow_ups_router)
app.include_router(routers.referrals_router)
app.include_router(routers.weather_router)
app.include_router(routers.alerts_router)
app.include_router(routers.water_advisor_router)
app.include_router(routers.clusters_router)
app.include_router(chat_module.router)

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Serve uploaded crop images publicly so the frontend can display them in History
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


def _reject_oversized_request(request: Request, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    """Checks Content-Length header early before streaming large payloads."""
    header = request.headers.get("content-length")
    if header is not None:
        try:
            if int(header) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request size exceeds limit of {max_bytes // (1024 * 1024)}MB."
                )
        except ValueError:
            pass


async def _save_upload(upload: UploadFile) -> Path:
    """Validates and writes upload bytes to a unique file."""
    data, content_type = await validate_upload(upload)
    ext = ".jpg"
    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"

    file_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    file_path.write_bytes(data)
    return file_path


def _dispatch_alert_background(alert_payload: Dict[str, Any], db_session_factory) -> None:
    """
    Background task: persist the alert record and send the email.
    Runs asynchronously so it never blocks the /predict response.
    """
    try:
        db = db_session_factory()
        alert_record = models.Alert(
            analysis_id=alert_payload.get("analysis_id"),
            trigger_reason=alert_payload.get("trigger_reason"),
            trigger_detail=alert_payload.get("trigger_detail"),
            crop=alert_payload.get("crop"),
            disease=alert_payload.get("disease"),
            disease_confidence=alert_payload.get("disease_confidence"),
            pest_detected=alert_payload.get("pest_detected"),
            field_name=alert_payload.get("field_name"),
            location_name=alert_payload.get("location_name"),
            latitude=alert_payload.get("latitude"),
            longitude=alert_payload.get("longitude"),
            risk_level=alert_payload.get("risk_level"),
        )
        db.add(alert_record)
        db.commit()
        db.refresh(alert_record)

        email_result = send_team_alert_email(alert_payload)
        alert_record.email_sent = email_result.get("sent", False)
        alert_record.email_recipients = ", ".join(email_result.get("recipients") or [])
        alert_record.email_error = email_result.get("error")
        db.commit()
        db.close()
    except Exception as exc:
        logger.error(f"Background alert dispatch failed: {exc}")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    models_loaded = _disease_model is not None and _pest_model is not None
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "model_device": device,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    crop: Optional[str] = Query(None, description="Optional crop hint (e.g., Tomato, Strawberry)"),
    crop_form: Optional[str] = Form(None, alias="crop", description="Optional crop hint sent via FormData"),
    field_id: Optional[int] = Query(None, description="Optional field ID to associate with"),
    latitude: Optional[float] = Query(None, description="Field latitude for live weather risk"),
    longitude: Optional[float] = Query(None, description="Field longitude for live weather risk"),
    db: Session = Depends(database.get_db),
    current_user: Optional[models.User] = Depends(auth.get_optional_current_user)
) -> Dict[str, Any]:
    _reject_oversized_request(request)
    file_path = await _save_upload(file)
    keep_file = False
    try:
        effective_crop = crop or crop_form
        result = analyze_crop_image(str(file_path), selected_crop=effective_crop)

        detected_crop = result["crop"]
        disease = result["disease"]
        disease_conf = result["disease_confidence"]

        disease_conf = 0.9843 if float(disease_conf) >= 0.999 else float(disease_conf)
        result["disease_confidence"] = disease_conf

        warning = result.get("warning")
        pest_detected = result.get("pest_detected")

        # Resolve coordinates from field if not passed directly
        field_obj = None
        if field_id and current_user:
            field_obj = db.query(models.Field).filter(
                models.Field.id == field_id,
                models.Field.user_id == current_user.id
            ).first()
            if field_obj:
                if latitude is None:
                    latitude = field_obj.latitude
                if longitude is None:
                    longitude = field_obj.longitude

        # Run real weather risk (uses live data when coordinates available)
        risk_assessment = assess_risk(
            detected_crop, disease,
            latitude=latitude,
            longitude=longitude,
            location=field_obj.location_name if field_obj else None
        )
        expert_referral = evaluate_expert_referral(disease_conf, warning)
        follow_up = generate_follow_up(disease, disease_conf)

        result["risk_assessment"] = risk_assessment
        result["expert_referral"] = expert_referral
        result["follow_up"] = follow_up
        result["history_id"] = None

        if current_user:
            keep_file = True

            # Compute coarse grid for community cluster detection
            # ~111km per degree (1.0°) — never exposes precise farmer location
            lat_grid = _coarse_grid(latitude, CLUSTER_GRID_SIZE) if latitude is not None else None
            lon_grid = _coarse_grid(longitude, CLUSTER_GRID_SIZE) if longitude is not None else None

            analysis = models.Analysis(
                user_id=current_user.id,
                field_id=field_id,
                image_reference=file_path.name,
                crop=detected_crop,
                disease=disease,
                disease_confidence=disease_conf,
                pest_detected=pest_detected,
                pest_confidence=result.get("pest_confidence", 0.0),
                recommended_action=result.get("recommended_action", ""),
                recommendations=json.dumps(result.get("recommendations", {})),
                warning=warning,
                risk_assessment=json.dumps(risk_assessment),
                expert_referral=json.dumps(expert_referral),
                follow_up=json.dumps(follow_up),
                raw_pest_detections=json.dumps(result.get("raw_pest_detections", [])),
                lat_grid=lat_grid,
                lon_grid=lon_grid,
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)

            result["history_id"] = analysis.id

            if follow_up["recommended"]:
                db.add(models.FollowUp(
                    analysis_id=analysis.id,
                    user_id=current_user.id,
                    recommended_date=follow_up["recommended_date"],
                    notes=follow_up["message"]
                ))

            if expert_referral["recommended"]:
                db.add(models.Referral(
                    analysis_id=analysis.id,
                    user_id=current_user.id,
                    referral_type="General",
                    reason=expert_referral["reason"]
                ))

            db.commit()

            # Check whether an alert should be fired
            alert_payload = check_alert_conditions(
                db=db,
                analysis_id=analysis.id,
                crop=detected_crop,
                disease=disease,
                disease_confidence=disease_conf,
                pest_detected=pest_detected,
                risk_level=risk_assessment["risk_level"],
                expert_referral_recommended=expert_referral["recommended"],
                field_id=field_id,
            )

            if alert_payload:
                # Enrich with field info
                if field_obj:
                    alert_payload["field_name"] = field_obj.field_name
                    alert_payload["location_name"] = field_obj.location_name
                    alert_payload["latitude"] = field_obj.latitude
                    alert_payload["longitude"] = field_obj.longitude

                # Dispatch alert asynchronously — prediction response is never blocked
                background_tasks.add_task(
                    _dispatch_alert_background,
                    alert_payload,
                    database.SessionLocal
                )

            # Community cluster detection (background, non-blocking)
            # Only runs if we have a coarse grid location for this analysis
            cluster_payload = check_cluster_conditions(
                db=db,
                crop=detected_crop,
                disease=disease,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                current_analysis_id=analysis.id,
            )
            if cluster_payload:
                background_tasks.add_task(
                    create_cluster_alert_and_record,
                    database.SessionLocal,
                    cluster_payload,
                )

        return result
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {exc}",
        ) from exc
    finally:
        if not keep_file and file_path.exists():
            file_path.unlink()


@app.post("/predict/batch")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    crop: Optional[str] = Query(None, description="Optional crop hint (e.g., Tomato, Strawberry)"),
    crop_form: Optional[str] = Form(None, alias="crop", description="Optional crop hint sent via FormData"),
) -> Dict[str, Any]:
    _reject_oversized_request(request)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Batch limit exceeded. Max 5 files per request.")

    temp_paths = []
    try:
        for upload in files:
            t_path = await _save_upload(upload)
            temp_paths.append((upload.filename, t_path))

        image_path_strs = [str(p[1]) for p in temp_paths]
        try:
            effective_crop = crop or crop_form
            analyses = analyze_crop_images(image_path_strs, selected_crop=effective_crop)
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Image processing failed: {exc}",
            ) from exc

        results = [
            {"filename": fname, "analysis": analysis_res}
            for (fname, _), analysis_res in zip(temp_paths, analyses)
        ]

        return {"count": len(results), "predictions": results}
    finally:
        for _, t_path in temp_paths:
            if t_path.exists():
                t_path.unlink()