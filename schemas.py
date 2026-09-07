from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any, Dict
from datetime import datetime

# --- Auth Schemas ---
class UserCreate(BaseModel):
    email: Optional[EmailStr] = None
    mobile: Optional[str] = None
    password: str

class UserLogin(BaseModel):
    identifier: str # Email or mobile
    password: str

class UserResponse(BaseModel):
    id: int
    email: Optional[EmailStr]
    mobile: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True

class TokenResponse(BaseModel):
    success: bool
    message: str
    user: UserResponse
    access_token: str
    token_type: str

# --- Field Schemas ---
class FieldCreate(BaseModel):
    field_name: str
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class FieldResponse(BaseModel):
    id: int
    field_name: str
    location_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    created_at: datetime

    class Config:
        orm_mode = True

# --- Prediction and History Schemas ---
class RiskAssessment(BaseModel):
    risk_level: str
    risk_reason: str
    risk_advisory: str
    weather: Dict[str, Any]

class ExpertReferral(BaseModel):
    recommended: bool
    reason: Optional[str] = None
    referral_options: Optional[List[str]] = None

class FollowUpRecommendation(BaseModel):
    recommended: bool
    recommended_in_days: Optional[int] = None
    recommended_date: Optional[str] = None
    status: str = "pending"
    message: Optional[str] = None

class PredictionResponse(BaseModel):
    crop: str
    disease: str
    disease_confidence: float
    pest_detected: Optional[str] = None
    pest_confidence: float = 0.0
    recommended_action: str
    raw_pest_detections: List[Any] = []
    recommendations: Dict[str, Any]
    warning: Optional[str] = None
    risk_assessment: Optional[RiskAssessment] = None
    expert_referral: Optional[ExpertReferral] = None
    follow_up: Optional[FollowUpRecommendation] = None
    history_id: Optional[int] = None

class AnalysisHistoryResponse(BaseModel):
    id: int
    created_at: datetime
    field: Optional[FieldResponse] = None
    crop: str
    disease: str
    disease_confidence: float
    pest_detected: Optional[str] = None
    pest_confidence: float
    recommended_action: str
    recommendations: Dict[str, Any]
    warning: Optional[str] = None
    risk_assessment: Optional[RiskAssessment] = None
    expert_referral: Optional[ExpertReferral] = None
    follow_up: Optional[FollowUpRecommendation] = None

    class Config:
        orm_mode = True

# --- Follow Up Schemas ---
class FollowUpResponse(BaseModel):
    id: int
    analysis_id: int
    recommended_date: str
    status: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class FollowUpUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None

# --- Referral Schemas ---
class ReferralResponse(BaseModel):
    id: int
    analysis_id: int
    referral_type: str
    reason: str
    status: str
    created_at: datetime

    class Config:
        orm_mode = True

class ReferralUpdate(BaseModel):
    status: str
