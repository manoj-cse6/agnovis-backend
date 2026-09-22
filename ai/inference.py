"""Production crop-disease and pest-detection inference pipeline with diagnostic warnings."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

logger = logging.getLogger("sih26131.inference")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "ai", "models", "plant_disease_mobilenetv3_28.pth")
PEST_MODEL_PATH = os.path.join(BASE_DIR, "ai", "models", "pest_best.pt")

DISEASE_CLASSES_PATH = os.path.join(BASE_DIR, "ai", "configs", "disease_class_names.json")
PEST_CLASSES_PATH = os.path.join(BASE_DIR, "ai", "configs", "pest_class_names.json")
ACTIONS_PATH = os.path.join(BASE_DIR, "ai", "configs", "actions.json")
CROP_PESTS_PATH = os.path.join(BASE_DIR, "ai", "configs", "crop_pests.json")
REMEDIES_PATH = os.path.join(BASE_DIR, "ai", "data", "treatment_remedies.json")

NUM_DISEASE_CLASSES = 28

REQUIRED_FILES = (
    DISEASE_MODEL_PATH,
    PEST_MODEL_PATH,
    DISEASE_CLASSES_PATH,
    PEST_CLASSES_PATH,
    ACTIONS_PATH,
    CROP_PESTS_PATH,
    REMEDIES_PATH,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
_torch_device = torch.device(device)

_INFERENCE_LOCK = threading.Lock()

_disease_model: nn.Module | None = None
_pest_model: YOLO | None = None
_disease_class_names: dict[int, str] = {}
_pest_class_names: dict[int, str] = {}
_actions: dict[str, Any] = {}
_crop_pests: dict[str, list[str]] = {}
_remedies: dict[str, Any] = {}
_loaded = False

RECOMMENDATION_KEYS = (
    "chemical_control",
    "organic_control",
    "preventative_measures",
    "severity_level",
)

_DEFAULT_DISEASE_RECOMMENDATION = {
    "chemical_control": "Consult a local agronomist before applying chemical treatments.",
    "organic_control": "Improve field sanitation and monitor the crop daily.",
    "preventative_measures": "Use certified planting material, crop rotation, and balanced irrigation.",
    "severity_level": "Low",
}

disease_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not os.path.isfile(path)]
    if missing:
        listed = "\n".join(f"   - {path}" for path in missing)
        raise FileNotFoundError(
            "Required model or config file(s) not found:\n"
            f"{listed}\n"
            "Expected files:\n"
            f"  {DISEASE_MODEL_PATH}\n"
            f"  {PEST_MODEL_PATH}\n"
            f"  {DISEASE_CLASSES_PATH}\n"
            f"  {PEST_CLASSES_PATH}\n"
            f"  {ACTIONS_PATH}\n"
            f"  {CROP_PESTS_PATH}\n"
            f"  {REMEDIES_PATH}"
        )


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _index_map(data: Any, source_path: str) -> dict[int, str]:
    """Load class names from a list or a dict with integer or string keys."""
    if isinstance(data, list):
        return {index: str(name) for index, name in enumerate(data)}
    if isinstance(data, dict):
        mapped: dict[int, str] = {}
        for key, value in data.items():
            try:
                mapped[int(key)] = str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Class-name key {key!r} in {source_path} is not an integer index."
                ) from exc
        return mapped
    raise ValueError(f"{source_path} must be a JSON list or object of class names.")


def _string_key_map(data: dict[Any, Any]) -> dict[str, Any]:
    """Normalize JSON object keys so integer or string keys both resolve."""
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        normalized[str(key)] = value
        try:
            normalized[str(int(key))] = value
        except (TypeError, ValueError):
            pass
    return normalized


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported disease checkpoint format: {type(checkpoint)!r}")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, dict) and value:
            sample_keys = list(value.keys())[:8]
            if all(isinstance(item, str) for item in sample_keys):
                checkpoint = value
                break

    return {
        key.removeprefix("module."): tensor
        for key, tensor in checkpoint.items()
        if hasattr(tensor, "shape")
    }


def _build_mobilenetv3_small(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v3_small(weights=None)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _parse_disease_label(label: str) -> tuple[str, str]:
    if "___" in label:
        crop_raw, disease_raw = label.split("___", 1)
    else:
        crop_raw, disease_raw = label, "unknown"

    crop_token = crop_raw.split("(")[0].split(",")[0].replace("_", " ").strip()
    crop = crop_token.title() if crop_token else "Unknown"

    for known_crop in _crop_pests:
        if _normalize_text(known_crop) == _normalize_text(crop) or _normalize_text(
            crop_raw
        ).startswith(_normalize_text(known_crop)):
            crop = known_crop
            break

    disease = disease_raw.replace("_", " ").strip() or "unknown"
    return crop, disease


def _lookup_mapping(mapping: dict[str, Any], key: str) -> Any | None:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping[key]
    key_norm = _normalize_text(str(key))
    for stored_key, value in mapping.items():
        if _normalize_text(str(stored_key)) == key_norm:
            return value
    return None


def _pest_allowed_for_crop(crop: str, pest_name: str) -> bool:
    allowed: list[str] = []
    crop_entry = _lookup_mapping(_crop_pests, crop)
    if isinstance(crop_entry, list):
        allowed = [str(item) for item in crop_entry]
    target = _normalize_text(pest_name)
    return any(_normalize_text(item) == target for item in allowed)


def _resolve_pest_name(class_id: int, model_names: dict[Any, Any]) -> str:
    name = ""
    if class_id in model_names:
        name = str(model_names[class_id]).strip()
    elif str(class_id) in model_names:
        name = str(model_names[str(class_id)]).strip()

    generic = not name or _normalize_text(name).startswith("class")
    if not generic:
        return name
    if class_id in _pest_class_names:
        return _pest_class_names[class_id]
    return f"class_{class_id}"



CROP_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rice": {
        "model_file": "rice_mobilenetv3_4.pth",
        "class_json": "rice_class_names.json",
        "canonical_name": "Rice",
        "num_classes": 4,
    },
    "sugarcane": {
        "model_file": "sugarcane_mobilenetv3_5.pth",
        "class_json": "sugarcane_class_names.json",
        "canonical_name": "Sugarcane",
        "num_classes": 5,
    },
    "cotton": {
        "model_file": "cotton_mobilenetv3_5.pth",
        "class_json": "cotton_class_names.json",
        "canonical_name": "Cotton",
        "num_classes": 5,
    },
    "soybean": {
        "model_file": "soybean_mobilenetv3_4.pth",
        "class_json": "soybean_class_names.json",
        "canonical_name": "Soybean",
        "num_classes": 4,
    },
    "wheat": {
        "model_file": "wheat_mobilenetv3_4.pth",
        "class_json": "wheat_class_names.json",
        "canonical_name": "Wheat",
        "num_classes": 4,
    },
    "jowar": {
        "model_file": "jowar_mobilenetv3_6.pth",
        "class_json": "jowar_class_names.json",
        "canonical_name": "Jowar",
        "num_classes": 6,
    },
    "bajra": {
        "model_file": "bajra_mobilenetv3_5.pth",
        "class_json": "bajra_class_names.json",
        "canonical_name": "Bajra",
        "num_classes": 5,
    },
}

CROP_ALIASES: dict[str, str] = {
    "sorghum": "jowar",
    "jowar": "jowar",
    "pearl millet": "bajra",
    "pearlmillet": "bajra",
    "bajra": "bajra",
    "rice": "rice",
    "paddy": "rice",
    "sugarcane": "sugarcane",
    "sugar cane": "sugarcane",
    "cotton": "cotton",
    "soybean": "soybean",
    "soya": "soybean",
    "soy": "soybean",
    "wheat": "wheat",
}

_crop_models_cache: dict[str, tuple[nn.Module, dict[int, str], str]] = {}


def _normalize_crop_name(name: Optional[str]) -> Optional[str]:
    """Normalize input crop names and aliases to internal registry keys."""
    if not name or not isinstance(name, str):
        return None
    cleaned = _normalize_text(name.strip())
    if cleaned in CROP_ALIASES:
        return CROP_ALIASES[cleaned]
    for alias, canonical in CROP_ALIASES.items():
        if alias in cleaned:
            return canonical
    return cleaned


def _get_crop_model(crop_key: str) -> tuple[nn.Module, dict[int, str], str]:
    """
    Lazy-loads and caches the specific crop disease model.
    Must be called under _INFERENCE_LOCK.
    """
    if crop_key in _crop_models_cache:
        return _crop_models_cache[crop_key]

    if crop_key not in CROP_MODEL_REGISTRY:
        raise KeyError(f"No crop-specific model registered for {crop_key!r}")

    info = CROP_MODEL_REGISTRY[crop_key]
    model_path = os.path.join(BASE_DIR, "ai", "models", info["model_file"])
    class_path = os.path.join(BASE_DIR, "ai", "models", info["class_json"])

    if not os.path.isfile(model_path) or not os.path.isfile(class_path):
        raise FileNotFoundError(f"Missing crop model files: {model_path} or {class_path}")

    class_names = _index_map(_load_json(class_path), class_path)
    num_classes = len(class_names)

    try:
        checkpoint = torch.load(model_path, map_location=_torch_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=_torch_device)

    model = _build_mobilenetv3_small(num_classes)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.to(_torch_device)
    model.eval()

    canonical_name = info["canonical_name"]
    _crop_models_cache[crop_key] = (model, class_names, canonical_name)
    logger.info("Loaded crop-specific disease model for %s (%d classes)", canonical_name, num_classes)
    return model, class_names, canonical_name


def _parse_crop_disease(raw_label: str, canonical_crop: str) -> str:
    """Extract clean disease label from crop-specific model output."""
    if "___" in raw_label:
        _, disease_part = raw_label.split("___", 1)
        return disease_part.replace("_", " ").strip()

    norm_label = _normalize_text(raw_label)
    norm_crop = _normalize_text(canonical_crop)
    if norm_label.startswith(norm_crop):
        remainder = raw_label[len(canonical_crop):].lstrip(" _-")
        if remainder:
            return remainder.replace("_", " ").strip()

    return raw_label.replace("_", " ").strip()


def _recommended_action(raw_label: str, crop: str, disease: str, pest_name: str | None) -> str:
    disease_actions = _actions.get("disease_actions", _actions)
    pest_actions = _actions.get("pest_actions", {})
    if not isinstance(disease_actions, dict):
        disease_actions = {}
    if not isinstance(pest_actions, dict):
        pest_actions = {}

    action = _lookup_mapping(disease_actions, raw_label)
    if not isinstance(action, str) or not action.strip():
        action = _lookup_mapping(disease_actions, f"{crop}___{disease.replace(' ', '_')}")
    if not isinstance(action, str) or not action.strip():
        action = _lookup_mapping(disease_actions, f"{crop} {disease}")
    if not isinstance(action, str) or not action.strip():
        action = _lookup_mapping(disease_actions, disease)

    is_healthy = "healthy" in _normalize_text(disease)
    if not isinstance(action, str) or not action.strip():
        if is_healthy:
            action = (
                "Plant appears healthy. Continue regular monitoring and proper watering."
            )
        else:
            action = (
                "No specific disease action is listed. Continue monitoring and consult "
                "local agricultural guidance."
            )

    if pest_name:
        pest_action = _lookup_mapping(pest_actions, pest_name)
        if isinstance(pest_action, str) and pest_action.strip():
            action = f"{action} {pest_action.strip()}"

    return action.strip()


def _coerce_recommendation(raw: Any, fallback_severity: str = "Low") -> dict[str, str]:
    payload = dict(_DEFAULT_DISEASE_RECOMMENDATION)
    payload["severity_level"] = fallback_severity
    if isinstance(raw, dict):
        for key in RECOMMENDATION_KEYS:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()
    return payload


def _lookup_recommendations(raw_label: str, disease: str, pest_name: str | None) -> dict[str, Any]:
    disease_table = _remedies.get("diseases", {})
    pest_table = _remedies.get("pests", {})
    if not isinstance(disease_table, dict):
        disease_table = {}
    if not isinstance(pest_table, dict):
        pest_table = {}

    disease_raw = _lookup_mapping(disease_table, raw_label)
    if not disease_raw:
        disease_raw = _lookup_mapping(disease_table, disease)
    is_healthy = "healthy" in _normalize_text(disease)
    disease_rec = _coerce_recommendation(
        disease_raw,
        fallback_severity="Low" if is_healthy else "Medium",
    )

    pest_rec = None
    if pest_name:
        pest_raw = _lookup_mapping(pest_table, pest_name)
        pest_rec = _coerce_recommendation(pest_raw, fallback_severity="Medium")

    return {"disease": disease_rec, "pest": pest_rec}


# ── Warning flags ──────────────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD: float = 0.40


def _build_warnings(
    selected_crop: Optional[str],
    filtered_crop: str,
    filtered_confidence: float,
    global_crop: str,
    global_confidence: float,
) -> Optional[str]:
    """Compose a human-readable warning string when diagnostic conditions are met.

    Two conditions are checked independently and their messages combined:

    1. **Cross-crop mismatch** — ``selected_crop`` was supplied AND the global
       (unfiltered) best prediction disagrees with the forced crop.  This signals
       that the image likely shows a different plant species than the caller
       assumed.

    2. **Low confidence** — The final ``filtered_confidence`` is below
       ``LOW_CONFIDENCE_THRESHOLD`` (0.40), indicating that the visual evidence
       is ambiguous regardless of any crop hint.

    Returns ``None`` when no warning conditions are triggered.
    """
    messages: List[str] = []

    if (
        selected_crop
        and isinstance(selected_crop, str)
        and selected_crop.strip()
        and _normalize_text(global_crop) != _normalize_text(filtered_crop)
    ):
        messages.append(
            f"Visual features closely resemble {global_crop}, but classification was restricted to {filtered_crop} based on your selection."
        )

    if filtered_confidence < LOW_CONFIDENCE_THRESHOLD and not messages:
        messages.append(
            f"Low classification confidence ({filtered_confidence:.2%}). Diagnosis may be uncertain."
        )

    return " | ".join(messages) if messages else None



def load_models() -> None:
    """Validate files, load JSON configs, and initialize both models."""
    global _disease_model, _pest_model, _disease_class_names, _pest_class_names
    global _actions, _crop_pests, _remedies, _loaded

    if _loaded and _disease_model is not None and _pest_model is not None:
        return

    with _INFERENCE_LOCK:
        # Re-check inside the lock to prevent double-loading under concurrency.
        if _loaded and _disease_model is not None and _pest_model is not None:
            return

        _validate_required_files()

        _disease_class_names = _index_map(_load_json(DISEASE_CLASSES_PATH), DISEASE_CLASSES_PATH)
        if len(_disease_class_names) != NUM_DISEASE_CLASSES:
            raise ValueError(
                f"{DISEASE_CLASSES_PATH} must contain {NUM_DISEASE_CLASSES} class names, "
                f"found {len(_disease_class_names)}."
            )

        _pest_class_names = _index_map(_load_json(PEST_CLASSES_PATH), PEST_CLASSES_PATH)

        actions_raw = _load_json(ACTIONS_PATH)
        if not isinstance(actions_raw, dict):
            raise ValueError(f"{ACTIONS_PATH} must be a JSON object.")
        _actions = {
            str(key): _string_key_map(value) if isinstance(value, dict) else value
            for key, value in actions_raw.items()
        }

        crop_pests_raw = _load_json(CROP_PESTS_PATH)
        if not isinstance(crop_pests_raw, dict):
            raise ValueError(f"{CROP_PESTS_PATH} must be a JSON object.")
        _crop_pests = {}
        for key, value in crop_pests_raw.items():
            if not isinstance(value, list):
                raise ValueError(f"{CROP_PESTS_PATH} value for {key!r} must be a list of pest names.")
            _crop_pests[str(key)] = [str(item) for item in value]

        remedies_raw = _load_json(REMEDIES_PATH)
        if not isinstance(remedies_raw, dict):
            raise ValueError(f"{REMEDIES_PATH} must be a JSON object.")
        _remedies = {
            str(section): _string_key_map(value) if isinstance(value, dict) else value
            for section, value in remedies_raw.items()
        }

        try:
            checkpoint = torch.load(DISEASE_MODEL_PATH, map_location=_torch_device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(DISEASE_MODEL_PATH, map_location=_torch_device)

        disease_model = _build_mobilenetv3_small(NUM_DISEASE_CLASSES)
        disease_model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        disease_model.to(_torch_device)
        disease_model.eval()
        _disease_model = disease_model

        _pest_model = YOLO(PEST_MODEL_PATH)
        _loaded = True
        logger.info("Models successfully loaded on device %s", device)



def _classify_disease(
    image: Image.Image,
    selected_crop: Optional[str] = None,
) -> Tuple[str, str, str, float, str, float, str]:
    """Classify leaf disease using either a crop-specific model or the original 28-class model.

    Returns a 7-tuple:
        (crop, disease, raw_label, disease_confidence, global_crop, global_confidence, model_used)
    """
    input_tensor = disease_transform(image).unsqueeze(0).to(_torch_device)
    norm_crop = _normalize_crop_name(selected_crop)

    # 1. Route to crop-specific model if registered for this crop
    if norm_crop and norm_crop in CROP_MODEL_REGISTRY:
        crop_model, class_names, canonical_crop = _get_crop_model(norm_crop)
        with torch.no_grad():
            output = crop_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        num_classes = len(class_names)
        top_idx = int(torch.argmax(probabilities[:num_classes]).item())
        raw_label = class_names.get(top_idx, f"unknown_{top_idx}")
        disease_confidence = float(probabilities[top_idx].cpu().item())
        disease = _parse_crop_disease(raw_label, canonical_crop)
        model_used = f"{norm_crop}_mobilenetv3"
        logger.info("Using %s crop-specific disease model (%s)", canonical_crop, model_used)

        return canonical_crop, disease, raw_label, disease_confidence, canonical_crop, disease_confidence, model_used

    # 2. Fallback to original 28-class disease model
    if _disease_model is None:
        raise RuntimeError("Disease model is not loaded.")

    with torch.no_grad():
        output = _disease_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    num_classes = len(_disease_class_names)
    if num_classes == 0:
        raise ValueError("_disease_class_names is empty.")

    # Always compute the global (unfiltered) best — needed for mismatch detection.
    global_idx = int(torch.argmax(probabilities[:num_classes]).item())
    global_label = _disease_class_names.get(global_idx, f"unknown_{global_idx}")
    global_crop, _ = _parse_disease_label(global_label)
    global_confidence = float(probabilities[global_idx].cpu().item())

    top_idx: Optional[int] = None

    # Filter to only crop-matching indices when selected_crop is provided.
    if selected_crop and isinstance(selected_crop, str) and selected_crop.strip():
        crop_hint = selected_crop.strip().lower()
        matching_indices = [
            idx for idx, name in _disease_class_names.items()
            if crop_hint in name.lower() and idx < len(probabilities)
        ]

        if matching_indices:
            # Zero-out non-matching classes; do NOT fill with -1.0 (causes index shifts).
            masked_probs = torch.zeros_like(probabilities)
            for idx in matching_indices:
                masked_probs[idx] = probabilities[idx]
            top_idx = int(torch.argmax(masked_probs).item())

    if top_idx is None or top_idx >= num_classes:
        top_idx = global_idx  # reuse already-computed global argmax

    disease_confidence = float(probabilities[top_idx].cpu().item())
    raw_label = _disease_class_names.get(top_idx, f"unknown_{top_idx}")
    crop, disease = _parse_disease_label(raw_label)
    model_used = "plant_disease_mobilenetv3_28"
    logger.info("Using original 28-class disease model (fallback)")

    return crop, disease, raw_label, disease_confidence, global_crop, global_confidence, model_used



def _detect_pests(image_path: str, crop: str) -> tuple[str | None, float, list[dict[str, Any]]]:
    if _pest_model is None:
        raise RuntimeError("Pest model is not loaded.")

    results = _pest_model.predict(
        source=image_path,
        verbose=False,
        device=device,
    )

    raw_pest_detections: list[dict[str, Any]] = []
    for result in results:
        names = result.names or {}
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            pest_name = _resolve_pest_name(class_id, names)
            raw_pest_detections.append(
                {
                    "class_id": class_id,
                    "name": pest_name,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": xyxy,
                }
            )

    raw_pest_detections.sort(key=lambda item: item["confidence"], reverse=True)

    relevant = [
        item
        for item in raw_pest_detections
        if _pest_allowed_for_crop(crop, item["name"]) and item["confidence"] >= 0.30
    ]
    if not relevant:
        return None, 0.0, raw_pest_detections

    top = relevant[0]
    return str(top["name"]), float(top["confidence"]), raw_pest_detections


def analyze_crop_image(image_path: str, selected_crop: Optional[str] = None) -> Dict[str, Any]:
    """Run disease classification and pest detection on a single image."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    load_models()

    try:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
    except OSError as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc

    with _INFERENCE_LOCK:
        crop, disease, raw_label, disease_confidence, global_crop, global_confidence, model_used = (
            _classify_disease(image, selected_crop=selected_crop)
        )
        pest_detected, pest_confidence, raw_pest_detections = _detect_pests(image_path, crop)

    warning = _build_warnings(
        selected_crop=selected_crop,
        filtered_crop=crop,
        filtered_confidence=disease_confidence,
        global_crop=global_crop,
        global_confidence=global_confidence,
    )

    return {
        "crop": crop,
        "disease": disease,
        "disease_confidence": round(max(0.0, float(disease_confidence)), 4),
        "pest_detected": pest_detected,
        "pest_confidence": round(pest_confidence, 4),
        "recommended_action": _recommended_action(raw_label, crop, disease, pest_detected),
        "raw_pest_detections": raw_pest_detections,
        "recommendations": _lookup_recommendations(raw_label, disease, pest_detected),
        "warning": warning,
        "model_used": model_used,
    }


def analyze_crop_images(image_paths: list[str], selected_crop: Optional[str] = None) -> list[dict[str, Any]]:
    """Run inference on multiple images sequentially under the model lock."""
    return [analyze_crop_image(path, selected_crop=selected_crop) for path in image_paths]


load_models()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python inference.py <image_path>")
    print(json.dumps(analyze_crop_image(sys.argv[1]), indent=2))