"""Standalone local verification of the SIH 26131 inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ai.inference import LOW_CONFIDENCE_THRESHOLD, analyze_crop_image

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REQUIRED_KEYS = {
    "crop",
    "disease",
    "disease_confidence",
    "pest_detected",
    "pest_confidence",
    "recommended_action",
    "raw_pest_detections",
    "recommendations",
    "warning",  # always present; None when no warning conditions are triggered
}


def _find_or_create_test_image() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        path
        for path in UPLOADS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if existing:
        return existing[0]

    sample_path = UPLOADS_DIR / "sample_test.jpg"
    Image.new("RGB", (224, 224), color=(34, 139, 34)).save(sample_path, format="JPEG")
    return sample_path


def _assert_result(result: dict, label: str) -> None:
    """Run structural and type assertions on a single inference result."""
    # All required keys must be present
    missing_keys = REQUIRED_KEYS - result.keys()
    assert not missing_keys, f"[{label}] Missing keys in result: {missing_keys}"

    # disease_confidence must be a float in [0, 1]
    dc = result["disease_confidence"]
    assert isinstance(dc, float), f"[{label}] disease_confidence must be float, got {type(dc)}"
    assert 0.0 <= dc <= 1.0, f"[{label}] disease_confidence out of range: {dc}"

    # pest_confidence must be a non-negative float
    pc = result["pest_confidence"]
    assert isinstance(pc, float), f"[{label}] pest_confidence must be float, got {type(pc)}"
    assert pc >= 0.0, f"[{label}] pest_confidence must be >= 0.0, got {pc}"

    # raw_pest_detections must be a list
    rpd = result["raw_pest_detections"]
    assert isinstance(rpd, list), f"[{label}] raw_pest_detections must be list, got {type(rpd)}"

    # crop and disease must be non-empty strings
    assert isinstance(result["crop"], str) and result["crop"], \
        f"[{label}] crop must be a non-empty string"
    assert isinstance(result["disease"], str) and result["disease"], \
        f"[{label}] disease must be a non-empty string"

    # recommended_action must be a non-empty string
    assert isinstance(result["recommended_action"], str) and result["recommended_action"], \
        f"[{label}] recommended_action must be a non-empty string"

    # recommendations must be a dict with a 'disease' key
    recs = result["recommendations"]
    assert isinstance(recs, dict), f"[{label}] recommendations must be a dict, got {type(recs)}"
    assert "disease" in recs, f"[{label}] recommendations must contain 'disease' key"

    # pest_detected must be None or a non-empty string
    pd = result["pest_detected"]
    assert pd is None or (isinstance(pd, str) and pd), \
        f"[{label}] pest_detected must be None or non-empty string, got {pd!r}"

    # warning must be None or a non-empty string — never an empty string
    w = result["warning"]
    assert w is None or (isinstance(w, str) and w.strip()), \
        f"[{label}] warning must be None or a non-empty string, got {w!r}"

    # Low-confidence invariant: if confidence < threshold, warning must be set
    if dc < LOW_CONFIDENCE_THRESHOLD:
        assert w is not None, (
            f"[{label}] disease_confidence={dc} is below threshold "
            f"{LOW_CONFIDENCE_THRESHOLD}, but warning is None"
        )


def main() -> None:
    image_path = _find_or_create_test_image()

    # ── Test 1: Standard inference (no crop hint) ────────────────────────────
    print(f"\nTest 1 — Standard inference\nUsing image: {image_path}")
    result1 = analyze_crop_image(str(image_path))
    print(json.dumps(result1, indent=2))
    _assert_result(result1, "Test 1 (no crop hint)")

    # ── Test 2: Inference with matching selected_crop ────────────────────────
    # Passing the crop that the model already predicts should produce no mismatch
    # warning (though a low-confidence warning may still fire).
    detected_crop = result1["crop"]
    print(f"\nTest 2 — selected_crop='{detected_crop}' (matching) path\nUsing image: {image_path}")
    result2 = analyze_crop_image(str(image_path), selected_crop=detected_crop)
    print(json.dumps(result2, indent=2))
    _assert_result(result2, f"Test 2 (selected_crop={detected_crop!r})")
    assert result2["crop"] == detected_crop, (
        f"Test 2: expected crop={detected_crop!r}, got {result2['crop']!r}"
    )
    # A matching crop hint must NOT produce a mismatch warning.
    w2 = result2["warning"]
    assert w2 is None or "mismatch" not in w2.lower(), (
        f"Test 2: unexpected mismatch warning for a matching crop hint: {w2!r}"
    )

    # ── Test 3: Cross-crop mismatch warning ──────────────────────────────────
    # Force a crop that differs from the model's global top prediction.
    # The response MUST NOT raise any exception, crop must equal the forced hint,
    # and the warning field must describe the mismatch.
    mismatch_crop = "Apple" if detected_crop != "Apple" else "Strawberry"
    print(f"\nTest 3 — Cross-crop mismatch: selected_crop='{mismatch_crop}' on a '{detected_crop}' image")
    result3 = analyze_crop_image(str(image_path), selected_crop=mismatch_crop)
    print(json.dumps(result3, indent=2))
    _assert_result(result3, f"Test 3 (selected_crop={mismatch_crop!r})")
    assert result3["crop"] == mismatch_crop, (
        f"Test 3: expected crop={mismatch_crop!r}, got {result3['crop']!r}"
    )
    w3 = result3["warning"]
    assert w3 is not None and "resemble" in w3.lower(), (
        f"Test 3: expected a mismatch warning, got {w3!r}"
    )
    print(f"  ✓ Mismatch warning present: {w3}")

    print("\n\u2705 All assertions passed.")


if __name__ == "__main__":
    main()
