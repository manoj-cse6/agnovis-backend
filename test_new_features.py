"""
Unit and integration tests for new Agnovis features:
1. Smart Water Advisor (/water-advisor)
2. Community Disease Cluster detection (/clusters) & Agri Authority notification deduplication
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
import models
from main import app
from services import (
    compute_water_advice,
    _coarse_grid,
    check_cluster_conditions,
    create_cluster_alert_and_record,
)

# Test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_new_features.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    app.dependency_overrides[database.get_db] = override_get_db
    models.Base.metadata.create_all(bind=engine)
    yield
    app.dependency_overrides.clear()
    models.Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists("./test_new_features.db"):
        os.remove("./test_new_features.db")


client = TestClient(app)


# ─── Water Advisor Tests ──────────────────────────────────────────────────────

def test_water_advisor_rain_expected():
    weather_data = {
        "source": "live",
        "forecast": [
            {"precipitation_probability_pct": 75, "temperature_max_c": 30, "rainfall_mm": 5},
            {"precipitation_probability_pct": 80, "temperature_max_c": 28, "rainfall_mm": 10},
        ],
        "current": {"relative_humidity_2m": 85},
    }
    advice = compute_water_advice(crop="Tomato", disease="Tomato_Early_blight", weather_data=weather_data)
    assert advice["decision"] == "Do Not Irrigate"
    assert advice["water_need"] == "Low"
    assert "Rain is expected soon" in advice["reason"]


def test_water_advisor_hot_and_dry():
    weather_data = {
        "source": "live",
        "forecast": [
            {"precipitation_probability_pct": 5, "temperature_max_c": 38, "rainfall_mm": 0},
            {"precipitation_probability_pct": 10, "temperature_max_c": 37, "rainfall_mm": 0},
        ],
        "current": {"relative_humidity_2m": 35},
    }
    advice = compute_water_advice(crop="Cotton", disease="healthy", weather_data=weather_data)
    assert advice["decision"] == "Irrigation Recommended"
    assert advice["water_need"] == "High"


def test_water_advisor_moisture_sensitive_disease():
    weather_data = {
        "source": "live",
        "forecast": [
            {"precipitation_probability_pct": 45, "temperature_max_c": 26, "rainfall_mm": 2},
        ],
        "current": {"relative_humidity_2m": 80},
    }
    advice = compute_water_advice(crop="Potato", disease="Late_blight", weather_data=weather_data)
    assert advice["decision"] == "Light Irrigation Only"
    assert advice["water_need"] == "Low"
    assert advice["disease_consideration"] is not None


def test_water_advisor_endpoint():
    response = client.get("/water-advisor?crop=Rice&disease=healthy&lat=17.3850&lon=78.4867")
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "water_need" in data
    assert "reason" in data


# ─── Coarse Grid & Cluster Tests ─────────────────────────────────────────────

def test_coarse_grid_0_1_deg():
    # 0.1 degree grid (~11 km)
    assert _coarse_grid(17.3850, 0.1) == 17.4
    assert _coarse_grid(78.4867, 0.1) == 78.5


def test_cluster_detection_threshold_and_deduplication(monkeypatch):
    db = TestingSessionLocal()
    lat_g = _coarse_grid(17.3850, 0.1)
    lon_g = _coarse_grid(78.4867, 0.1)

    # Insert 4 historical analyses for same crop/disease/grid
    for i in range(4):
        analysis = models.Analysis(
            user_id=1,
            crop="Tomato",
            disease="Tomato_Early_blight",
            disease_confidence=0.9,
            recommended_action="Spray fungicide",
            lat_grid=lat_g,
            lon_grid=lon_g,
            image_reference=f"test_{i}.jpg",
        )
        db.add(analysis)
    db.commit()

    # Case 1: Before threshold (4 cases total) — check_cluster_conditions returns None
    result = check_cluster_conditions(
        db=db,
        crop="Tomato",
        disease="Tomato_Early_blight",
        lat_grid=lat_g,
        lon_grid=lon_g,
        current_analysis_id=999,
    )
    # Total count = 4 + 1 = 5 (default threshold is 5)
    assert result is not None
    assert result["case_count"] == 5

    # Track notifications
    notified_calls = []
    def mock_notify(payload, alert_id):
        notified_calls.append(alert_id)

    import services
    monkeypatch.setattr(services, "_notify_agri_authority", mock_notify)

    # First escalation — creates cluster record + alert + calls _notify_agri_authority
    create_cluster_alert_and_record(TestingSessionLocal, result)
    assert len(notified_calls) == 1

    # Second report on same cluster — updates case count, but DOES NOT duplicate authority notification
    result["case_count"] = 6
    create_cluster_alert_and_record(TestingSessionLocal, result)
    assert len(notified_calls) == 1  # Deduplicated! Still 1 call

    db.close()
