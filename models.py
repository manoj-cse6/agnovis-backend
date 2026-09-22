from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
import datetime
import json
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    mobile = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    fields = relationship("Field", back_populates="owner")
    analyses = relationship("Analysis", back_populates="owner")

class Field(Base):
    __tablename__ = "fields"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    field_name = Column(String, index=True)
    location_name = Column(String)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="fields")
    analyses = relationship("Analysis", back_populates="field")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    field_id = Column(Integer, ForeignKey("fields.id"), nullable=True)
    image_reference = Column(String)
    crop = Column(String)
    disease = Column(String)
    disease_confidence = Column(Float)
    pest_detected = Column(String, nullable=True)
    pest_confidence = Column(Float, default=0.0)
    recommended_action = Column(String)
    recommendations = Column(String) # JSON string
    warning = Column(String, nullable=True)
    risk_assessment = Column(String, nullable=True) # JSON string
    expert_referral = Column(String, nullable=True) # JSON string
    follow_up = Column(String, nullable=True) # JSON string
    raw_pest_detections = Column(String, nullable=True) # JSON string
    # Coarse geographic grid for community cluster detection (~111km per degree)
    # Never used to expose precise farmer location — only for aggregation
    lat_grid = Column(Float, nullable=True)   # rounded to CLUSTER_GRID_SIZE degrees
    lon_grid = Column(Float, nullable=True)   # rounded to CLUSTER_GRID_SIZE degrees
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="analyses")
    field = relationship("Field", back_populates="analyses")
    follow_ups = relationship("FollowUp", back_populates="analysis")
    referrals = relationship("Referral", back_populates="analysis")

class FollowUp(Base):
    __tablename__ = "follow_ups"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    recommended_date = Column(String) # YYYY-MM-DD
    status = Column(String, default="pending") # pending, completed, skipped
    notes = Column(String, nullable=True)
    followup_image_reference = Column(String, nullable=True)
    followup_result = Column(String, nullable=True) # JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    analysis = relationship("Analysis", back_populates="follow_ups")

class Referral(Base):
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    referral_type = Column(String)
    reason = Column(String)
    status = Column(String, default="recommended") # recommended, requested, in_review, resolved
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    analysis = relationship("Analysis", back_populates="referrals")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=True)
    # What triggered the alert
    trigger_reason = Column(String)    # e.g. "high_confidence", "repeated_detection", "high_risk"
    trigger_detail = Column(String)    # Human-readable explanation
    # Detection snapshot
    crop = Column(String, nullable=True)
    disease = Column(String, nullable=True)
    disease_confidence = Column(Float, nullable=True)
    pest_detected = Column(String, nullable=True)
    field_name = Column(String, nullable=True)
    location_name = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    # Email delivery status
    email_sent = Column(Boolean, default=False)
    email_recipients = Column(String, nullable=True)   # comma-separated list
    email_error = Column(String, nullable=True)
    # Status for dashboard review
    status = Column(String, default="new")   # new, reviewed, resolved
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class DiseaseCluster(Base):
    """
    Tracks community disease clusters detected when multiple farmers
    in the same coarse geographic region report the same disease on
    the same crop within the configured time window.

    PRIVACY: Only coarse grid coordinates are stored — never individual
    farmer coordinates. Individual farmer identities are never stored here.
    """
    __tablename__ = "disease_clusters"

    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String, index=True)
    disease = Column(String, index=True)
    # Coarse grid — NOT precise farmer coordinates
    lat_grid = Column(Float, nullable=True)
    lon_grid = Column(Float, nullable=True)
    case_count = Column(Integer, default=0)
    # Linked alert ID for the cluster notification
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    first_detected = Column(DateTime, default=datetime.datetime.utcnow)
    last_detected = Column(DateTime, default=datetime.datetime.utcnow)
    # status: active, monitoring, resolved
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
