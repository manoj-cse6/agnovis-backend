import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

import models
import schemas
from auth import get_current_user
from database import get_db
from weather import get_7day_forecast

history_router = APIRouter(prefix="/history", tags=["HISTORY"])
fields_router = APIRouter(prefix="/fields", tags=["FIELDS"])
follow_ups_router = APIRouter(prefix="/follow-ups", tags=["FOLLOW-UP"])
referrals_router = APIRouter(prefix="/referrals", tags=["REFERRALS"])
weather_router = APIRouter(prefix="/weather", tags=["WEATHER"])
alerts_router = APIRouter(prefix="/alerts", tags=["ALERTS"])


# --- Weather Endpoints ---

@weather_router.get("/forecast")
def get_weather_forecast(
    latitude: float = Query(..., description="Latitude of the field/location"),
    longitude: float = Query(..., description="Longitude of the field/location"),
):
    """
    Fetch a real 7-day weather forecast for a given latitude/longitude.
    Source: Open-Meteo (free, no API key required).
    Results are cached for 1 hour per coordinate.
    Returns fallback data (clearly marked) if the API is temporarily unavailable.
    """
    result = get_7day_forecast(latitude, longitude)
    return result


# --- Alert Endpoints ---

class AlertUpdate(BaseModel):
    status: str   # new, reviewed, resolved


@alerts_router.get("")
def get_alerts(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieve all alerts. Requires authentication."""
    alerts = db.query(models.Alert).order_by(models.Alert.created_at.desc()).all()
    return alerts


@alerts_router.get("/{id}")
def get_alert_by_id(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    alert = db.query(models.Alert).filter(models.Alert.id == id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@alerts_router.patch("/{id}")
def update_alert(
    id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    alert = db.query(models.Alert).filter(models.Alert.id == id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = alert_update.status
    db.commit()
    db.refresh(alert)
    return alert


# --- History Endpoints ---

@history_router.get("", response_model=List[schemas.AnalysisHistoryResponse])
def get_history(
    crop: str = None,
    disease: str = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    query = db.query(models.Analysis).filter(models.Analysis.user_id == current_user.id)
    if crop:
        query = query.filter(models.Analysis.crop == crop)
    if disease:
        query = query.filter(models.Analysis.disease == disease)

    analyses = query.order_by(models.Analysis.created_at.desc()).all()

    results = []
    for a in analyses:
        item = a.__dict__.copy()
        item["recommendations"] = json.loads(a.recommendations) if a.recommendations else {}
        item["risk_assessment"] = json.loads(a.risk_assessment) if a.risk_assessment else None
        item["expert_referral"] = json.loads(a.expert_referral) if a.expert_referral else None
        item["follow_up"] = json.loads(a.follow_up) if a.follow_up else None
        results.append(item)
    return results


@history_router.get("/{id}", response_model=schemas.AnalysisHistoryResponse)
def get_history_by_id(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    analysis = db.query(models.Analysis).filter(
        models.Analysis.id == id,
        models.Analysis.user_id == current_user.id
    ).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    item = analysis.__dict__.copy()
    item["recommendations"] = json.loads(analysis.recommendations) if analysis.recommendations else {}
    item["risk_assessment"] = json.loads(analysis.risk_assessment) if analysis.risk_assessment else None
    item["expert_referral"] = json.loads(analysis.expert_referral) if analysis.expert_referral else None
    item["follow_up"] = json.loads(analysis.follow_up) if analysis.follow_up else None

    return item


# --- Fields Endpoints ---

@fields_router.post("", response_model=schemas.FieldResponse)
def create_field(
    field: schemas.FieldCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_field = models.Field(**field.model_dump(), user_id=current_user.id)
    db.add(db_field)
    db.commit()
    db.refresh(db_field)
    return db_field


@fields_router.get("", response_model=List[schemas.FieldResponse])
def get_fields(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.Field).filter(models.Field.user_id == current_user.id).all()


@fields_router.get("/{id}", response_model=schemas.FieldResponse)
def get_field_by_id(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    field = db.query(models.Field).filter(
        models.Field.id == id,
        models.Field.user_id == current_user.id
    ).first()
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field


@fields_router.put("/{id}", response_model=schemas.FieldResponse)
def update_field(
    id: int,
    field_update: schemas.FieldCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    field = db.query(models.Field).filter(
        models.Field.id == id,
        models.Field.user_id == current_user.id
    ).first()
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    for key, value in field_update.model_dump().items():
        setattr(field, key, value)

    db.commit()
    db.refresh(field)
    return field


@fields_router.delete("/{id}")
def delete_field(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    field = db.query(models.Field).filter(
        models.Field.id == id,
        models.Field.user_id == current_user.id
    ).first()
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    db.delete(field)
    db.commit()
    return {"success": True, "message": "Field deleted"}


# --- Follow Ups Endpoints ---

@follow_ups_router.post("", response_model=schemas.FollowUpResponse)
def create_follow_up(
    analysis_id: int,
    recommended_date: str,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    analysis = db.query(models.Analysis).filter(
        models.Analysis.id == analysis_id,
        models.Analysis.user_id == current_user.id
    ).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    follow_up = models.FollowUp(
        analysis_id=analysis_id,
        user_id=current_user.id,
        recommended_date=recommended_date,
        notes=notes
    )
    db.add(follow_up)
    db.commit()
    db.refresh(follow_up)
    return follow_up


@follow_ups_router.get("", response_model=List[schemas.FollowUpResponse])
def get_follow_ups(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.FollowUp).filter(models.FollowUp.user_id == current_user.id).all()


@follow_ups_router.get("/{id}", response_model=schemas.FollowUpResponse)
def get_follow_up_by_id(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    follow_up = db.query(models.FollowUp).filter(
        models.FollowUp.id == id,
        models.FollowUp.user_id == current_user.id
    ).first()
    if not follow_up:
        raise HTTPException(status_code=404, detail="Follow up not found")
    return follow_up


@follow_ups_router.patch("/{id}", response_model=schemas.FollowUpResponse)
def update_follow_up(
    id: int,
    follow_up_update: schemas.FollowUpUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    follow_up = db.query(models.FollowUp).filter(
        models.FollowUp.id == id,
        models.FollowUp.user_id == current_user.id
    ).first()
    if not follow_up:
        raise HTTPException(status_code=404, detail="Follow up not found")

    if follow_up_update.status:
        follow_up.status = follow_up_update.status
    if follow_up_update.notes:
        follow_up.notes = follow_up_update.notes

    db.commit()
    db.refresh(follow_up)
    return follow_up


# --- Referrals Endpoints ---

@referrals_router.get("", response_model=List[schemas.ReferralResponse])
def get_referrals(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.Referral).filter(models.Referral.user_id == current_user.id).all()


@referrals_router.get("/{id}", response_model=schemas.ReferralResponse)
def get_referral_by_id(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    referral = db.query(models.Referral).filter(
        models.Referral.id == id,
        models.Referral.user_id == current_user.id
    ).first()
    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")
    return referral


@referrals_router.patch("/{id}", response_model=schemas.ReferralResponse)
def update_referral(
    id: int,
    referral_update: schemas.ReferralUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    referral = db.query(models.Referral).filter(
        models.Referral.id == id,
        models.Referral.user_id == current_user.id
    ).first()
    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")

    referral.status = referral_update.status
    db.commit()
    db.refresh(referral)
    return referral
