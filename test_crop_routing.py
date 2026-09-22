"""
Automated tests for Crop-Specific Model Routing & Registry (SIH 26131).
Verifies:
1. Each of the 7 crops routes to its exact crop-specific model.
2. Alias normalization (e.g. Sorghum -> Jowar, Pearl Millet -> Bajra).
3. Fallback behavior (Tomato, Corn/Maize, and omitted crop use the 28-class model).
4. No cross-crop model invocation (only the selected model activates).
5. Pest detection runs independently via YOLO.
6. HTTP /predict accepts crop via Query or FormData.
7. HTTP /predict/batch functions properly.
"""

import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
import ai.inference as inf

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_leaf_image_bytes():
    """Generate a clean synthetic leaf image in memory."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def sample_leaf_file(tmp_path_factory):
    """Save a temporary sample leaf image on disk."""
    fn = tmp_path_factory.mktemp("data") / "sample_leaf.jpg"
    img = Image.new("RGB", (224, 224), color=(46, 139, 87))
    img.save(str(fn), format="JPEG")
    return str(fn)


# ─── 1. Core Model Routing Tests ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "crop_input,expected_model,expected_crop",
    [
        ("Rice", "rice_mobilenetv3", "Rice"),
        ("rice", "rice_mobilenetv3", "Rice"),
        ("Sugarcane", "sugarcane_mobilenetv3", "Sugarcane"),
        ("Sugar Cane", "sugarcane_mobilenetv3", "Sugarcane"),
        ("Cotton", "cotton_mobilenetv3", "Cotton"),
        ("cotton", "cotton_mobilenetv3", "Cotton"),
        ("Soybean", "soybean_mobilenetv3", "Soybean"),
        ("soya", "soybean_mobilenetv3", "Soybean"),
        ("Wheat", "wheat_mobilenetv3", "Wheat"),
        ("wheat", "wheat_mobilenetv3", "Wheat"),
        ("Jowar", "jowar_mobilenetv3", "Jowar"),
        ("Sorghum", "jowar_mobilenetv3", "Jowar"),
        ("Bajra", "bajra_mobilenetv3", "Bajra"),
        ("Pearl Millet", "bajra_mobilenetv3", "Bajra"),
    ],
)
def test_crop_specific_model_routing(sample_leaf_file, crop_input, expected_model, expected_crop):
    """Verify each crop activates strictly its corresponding model."""
    result = inf.analyze_crop_image(sample_leaf_file, selected_crop=crop_input)
    assert result["model_used"] == expected_model, f"Expected {expected_model}, got {result['model_used']}"
    assert result["crop"] == expected_crop
    assert "disease" in result
    assert 0.0 <= result["disease_confidence"] <= 1.0
    assert "recommended_action" in result
    assert "recommendations" in result


# ─── 2. Fallback Behavior Tests ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "crop_input,expected_crop_substring",
    [
        ("Tomato", "tomato"),
        ("Corn", "corn"),
        ("Maize", "corn"),
        (None, None),
    ],
)
def test_fallback_to_28_class_model(sample_leaf_file, crop_input, expected_crop_substring):
    """Verify crops without a crop-specific model (and Maize/Corn) fallback to the 28-class model."""
    result = inf.analyze_crop_image(sample_leaf_file, selected_crop=crop_input)
    assert result["model_used"] == "plant_disease_mobilenetv3_28"
    if expected_crop_substring:
        assert expected_crop_substring in result["crop"].lower() or expected_crop_substring in result["disease"].lower()
    assert 0.0 <= result["disease_confidence"] <= 1.0


# ─── 3. Pest Detection Independence ─────────────────────────────────────────

def test_pest_detection_independence(sample_leaf_file):
    """Verify YOLO pest detection operates independently alongside crop-specific models."""
    result = inf.analyze_crop_image(sample_leaf_file, selected_crop="Rice")
    assert "pest_detected" in result
    assert "pest_confidence" in result
    assert "raw_pest_detections" in result
    assert isinstance(result["raw_pest_detections"], list)


# ─── 4. HTTP /predict API Endpoint Tests ────────────────────────────────────

def test_predict_endpoint_with_query_param(sample_leaf_image_bytes):
    """Verify POST /predict with crop passed as query parameter."""
    files = {"file": ("leaf.jpg", sample_leaf_image_bytes, "image/jpeg")}
    response = client.post("/predict?crop=Wheat", files=files)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["crop"] == "Wheat"
    assert data["model_used"] == "wheat_mobilenetv3"
    assert "disease" in data
    assert "risk_assessment" in data


def test_predict_endpoint_with_form_data(sample_leaf_image_bytes):
    """Verify POST /predict with crop passed inside FormData."""
    files = {"file": ("leaf.jpg", sample_leaf_image_bytes, "image/jpeg")}
    data_payload = {"crop": "Sugarcane"}
    response = client.post("/predict", files=files, data=data_payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["crop"] == "Sugarcane"
    assert data["model_used"] == "sugarcane_mobilenetv3"


def test_predict_endpoint_fallback(sample_leaf_image_bytes):
    """Verify POST /predict with no crop hint uses fallback model."""
    files = {"file": ("leaf.jpg", sample_leaf_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_used"] == "plant_disease_mobilenetv3_28"


def test_predict_batch_endpoint(sample_leaf_image_bytes):
    """Verify POST /predict/batch uses selected crop routing."""
    files = [
        ("files", ("leaf1.jpg", sample_leaf_image_bytes, "image/jpeg")),
        ("files", ("leaf2.jpg", sample_leaf_image_bytes, "image/jpeg")),
    ]
    response = client.post("/predict/batch?crop=Cotton", files=files)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["count"] == 2
    for item in data["predictions"]:
        analysis = item["analysis"]
        assert analysis["crop"] == "Cotton"
        assert analysis["model_used"] == "cotton_mobilenetv3"
