"""Production crop-disease and pest-detection inference pipeline.

Memory strategy:
- No ML model is loaded at application startup.
- No ML model is permanently cached between requests.
- Disease model is loaded only when needed.
- Disease model is fully released before YOLO is loaded.
- YOLO is fully released before the request finishes.
- Crop-specific models are used when a supported crop is selected.
- Original 28-class model remains the fallback.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

# Keep CPU inference memory usage low on Render's 512 MB instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


logger = logging.getLogger("sih26131.inference")


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISEASE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "models",
    "plant_disease_mobilenetv3_28.pth",
)

PEST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "models",
    "pest_best.pt",
)

DISEASE_CLASSES_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "configs",
    "disease_class_names.json",
)

PEST_CLASSES_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "configs",
    "pest_class_names.json",
)

ACTIONS_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "configs",
    "actions.json",
)

CROP_PESTS_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "configs",
    "crop_pests.json",
)

REMEDIES_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "data",
    "treatment_remedies.json",
)


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


# ============================================================================
# DEVICE
# ============================================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)

_torch_device = torch.device(device)


# ============================================================================
# THREAD SAFETY
# ============================================================================

# One inference at a time.
#
# This is intentional for the 512 MB Render instance. It prevents concurrent
# requests from loading multiple large ML models simultaneously.
_INFERENCE_LOCK = threading.Lock()


# ============================================================================
# BACKWARD-COMPATIBILITY MODEL REFERENCES
# ============================================================================
#
# These names are kept because main.py imports them.
#
# IMPORTANT:
# They are intentionally NEVER populated.
#
# Models are loaded locally inside inference functions and released immediately
# after use. This prevents resident model accumulation between requests.

_disease_model: nn.Module | None = None
_pest_model: Any = None

# Kept for backward compatibility with older code.
# It is intentionally never populated.
_crop_models_cache: dict[
    str,
    tuple[nn.Module, dict[int, str], str],
] = {}


# ============================================================================
# LIGHTWEIGHT CONFIGURATION
# ============================================================================

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
    "chemical_control": (
        "Consult a local agronomist before applying chemical treatments."
    ),
    "organic_control": (
        "Improve field sanitation and monitor the crop daily."
    ),
    "preventative_measures": (
        "Use certified planting material, crop rotation, and balanced irrigation."
    ),
    "severity_level": "Low",
}


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

disease_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def _release_memory() -> None:
    """Best-effort cleanup after ML inference/model loading."""

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()


# ============================================================================
# FILE / JSON HELPERS
# ============================================================================

def _validate_required_files() -> None:
    missing = [
        path
        for path in REQUIRED_FILES
        if not os.path.isfile(path)
    ]

    if missing:
        listed = "\n".join(
            f"   - {path}"
            for path in missing
        )

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


def _index_map(
    data: Any,
    source_path: str,
) -> dict[int, str]:
    """Load class names from a list or dict."""

    if isinstance(data, list):
        return {
            index: str(name)
            for index, name in enumerate(data)
        }

    if isinstance(data, dict):
        mapped: dict[int, str] = {}

        for key, value in data.items():
            try:
                mapped[int(key)] = str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Class-name key {key!r} in "
                    f"{source_path} is not an integer index."
                ) from exc

        return mapped

    raise ValueError(
        f"{source_path} must be a JSON list or object of class names."
    )


def _string_key_map(
    data: dict[Any, Any],
) -> dict[str, Any]:
    """Normalize JSON object keys."""

    normalized: dict[str, Any] = {}

    for key, value in data.items():
        normalized[str(key)] = value

        try:
            normalized[str(int(key))] = value
        except (TypeError, ValueError):
            pass

    return normalized


# ============================================================================
# MODEL HELPERS
# ============================================================================

def _extract_state_dict(
    checkpoint: Any,
) -> dict[str, Any]:

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported disease checkpoint format: "
            f"{type(checkpoint)!r}"
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(value, nn.Module):
            return value.state_dict()

        if isinstance(value, dict) and value:
            sample_keys = list(value.keys())[:8]

            if all(
                isinstance(item, str)
                for item in sample_keys
            ):
                checkpoint = value
                break

    return {
        key.removeprefix("module."): tensor
        for key, tensor in checkpoint.items()
        if hasattr(tensor, "shape")
    }


def _build_mobilenetv3_small(
    num_classes: int,
) -> nn.Module:

    try:
        model = models.mobilenet_v3_small(
            weights=None
        )
    except TypeError:
        model = models.mobilenet_v3_small(
            pretrained=False
        )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


# ============================================================================
# TEXT / CROP HELPERS
# ============================================================================

def _normalize_text(value: str) -> str:
    return " ".join(
        value
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _parse_disease_label(
    label: str,
) -> tuple[str, str]:

    if "___" in label:
        crop_raw, disease_raw = label.split(
            "___",
            1,
        )
    else:
        crop_raw = label
        disease_raw = "unknown"

    crop_token = (
        crop_raw
        .split("(")[0]
        .split(",")[0]
        .replace("_", " ")
        .strip()
    )

    crop = (
        crop_token.title()
        if crop_token
        else "Unknown"
    )

    for known_crop in _crop_pests:
        if (
            _normalize_text(known_crop)
            == _normalize_text(crop)
            or _normalize_text(crop_raw).startswith(
                _normalize_text(known_crop)
            )
        ):
            crop = known_crop
            break

    disease = (
        disease_raw
        .replace("_", " ")
        .strip()
        or "unknown"
    )

    return crop, disease


def _lookup_mapping(
    mapping: dict[str, Any],
    key: str,
) -> Any | None:

    if not isinstance(mapping, dict):
        return None

    if key in mapping:
        return mapping[key]

    key_norm = _normalize_text(str(key))

    for stored_key, value in mapping.items():
        if (
            _normalize_text(str(stored_key))
            == key_norm
        ):
            return value

    return None


def _pest_allowed_for_crop(
    crop: str,
    pest_name: str,
) -> bool:

    allowed: list[str] = []

    crop_entry = _lookup_mapping(
        _crop_pests,
        crop,
    )

    if isinstance(crop_entry, list):
        allowed = [
            str(item)
            for item in crop_entry
        ]

    target = _normalize_text(pest_name)

    return any(
        _normalize_text(item) == target
        for item in allowed
    )


def _resolve_pest_name(
    class_id: int,
    model_names: dict[Any, Any],
) -> str:

    name = ""

    if class_id in model_names:
        name = str(
            model_names[class_id]
        ).strip()

    elif str(class_id) in model_names:
        name = str(
            model_names[str(class_id)]
        ).strip()

    generic = (
        not name
        or _normalize_text(name).startswith("class")
    )

    if not generic:
        return name

    if class_id in _pest_class_names:
        return _pest_class_names[class_id]

    return f"class_{class_id}"


# ============================================================================
# CROP-SPECIFIC MODEL REGISTRY
# ============================================================================

CROP_MODEL_REGISTRY: dict[
    str,
    dict[str, Any],
] = {
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


def _normalize_crop_name(
    name: Optional[str],
) -> Optional[str]:

    if not name or not isinstance(name, str):
        return None

    cleaned = _normalize_text(
        name.strip()
    )

    if cleaned in CROP_ALIASES:
        return CROP_ALIASES[cleaned]

    for alias, canonical in CROP_ALIASES.items():
        if alias in cleaned:
            return canonical

    return cleaned


# ============================================================================
# CROP MODEL LOADING
# ============================================================================

def _get_crop_model(
    crop_key: str,
) -> tuple[nn.Module, dict[int, str], str]:
    """Load one crop model transiently.

    The returned model MUST be released by the caller.
    """

    if crop_key not in CROP_MODEL_REGISTRY:
        raise KeyError(
            f"No crop-specific model registered "
            f"for {crop_key!r}"
        )

    info = CROP_MODEL_REGISTRY[crop_key]

    model_path = os.path.join(
        BASE_DIR,
        "ai",
        "models",
        info["model_file"],
    )

    class_path = os.path.join(
        BASE_DIR,
        "ai",
        "models",
        info["class_json"],
    )

    if (
        not os.path.isfile(model_path)
        or not os.path.isfile(class_path)
    ):
        raise FileNotFoundError(
            f"Missing crop model files: "
            f"{model_path} or {class_path}"
        )

    class_names = _index_map(
        _load_json(class_path),
        class_path,
    )

    num_classes = len(class_names)

    checkpoint = None
    model = None

    try:
        try:
            checkpoint = torch.load(
                model_path,
                map_location=_torch_device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location=_torch_device,
            )

        model = _build_mobilenetv3_small(
            num_classes
        )

        model.load_state_dict(
            _extract_state_dict(checkpoint),
            strict=True,
        )

        model.to(_torch_device)
        model.eval()

    except Exception as exc:
        if model is not None:
            del model

        raise RuntimeError(
            "Failed to load crop-specific model "
            f"for {crop_key!r}: {exc}"
        ) from exc

    finally:
        if checkpoint is not None:
            del checkpoint

        _release_memory()

    canonical_name = info["canonical_name"]

    logger.info(
        "Loaded crop-specific model: %s "
        "(%d classes, transient)",
        canonical_name,
        num_classes,
    )

    return (
        model,
        class_names,
        canonical_name,
    )


def _parse_crop_disease(
    raw_label: str,
    canonical_crop: str,
) -> str:

    if "___" in raw_label:
        _, disease_part = raw_label.split(
            "___",
            1,
        )

        return (
            disease_part
            .replace("_", " ")
            .strip()
        )

    norm_label = _normalize_text(raw_label)
    norm_crop = _normalize_text(canonical_crop)

    if norm_label.startswith(norm_crop):
        remainder = raw_label[
            len(canonical_crop):
        ].lstrip(" _-")

        if remainder:
            return (
                remainder
                .replace("_", " ")
                .strip()
            )

    return (
        raw_label
        .replace("_", " ")
        .strip()
    )


# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def _recommended_action(
    raw_label: str,
    crop: str,
    disease: str,
    pest_name: str | None,
) -> str:

    disease_actions = _actions.get(
        "disease_actions",
        _actions,
    )

    pest_actions = _actions.get(
        "pest_actions",
        {},
    )

    if not isinstance(
        disease_actions,
        dict,
    ):
        disease_actions = {}

    if not isinstance(
        pest_actions,
        dict,
    ):
        pest_actions = {}

    action = _lookup_mapping(
        disease_actions,
        raw_label,
    )

    if (
        not isinstance(action, str)
        or not action.strip()
    ):
        action = _lookup_mapping(
            disease_actions,
            f"{crop}___{disease.replace(' ', '_')}",
        )

    if (
        not isinstance(action, str)
        or not action.strip()
    ):
        action = _lookup_mapping(
            disease_actions,
            f"{crop} {disease}",
        )

    if (
        not isinstance(action, str)
        or not action.strip()
    ):
        action = _lookup_mapping(
            disease_actions,
            disease,
        )

    is_healthy = (
        "healthy"
        in _normalize_text(disease)
    )

    if (
        not isinstance(action, str)
        or not action.strip()
    ):
        if is_healthy:
            action = (
                "Plant appears healthy. "
                "Continue regular monitoring "
                "and proper watering."
            )
        else:
            action = (
                "No specific disease action is listed. "
                "Continue monitoring and consult "
                "local agricultural guidance."
            )

    if pest_name:
        pest_action = _lookup_mapping(
            pest_actions,
            pest_name,
        )

        if (
            isinstance(pest_action, str)
            and pest_action.strip()
        ):
            action = (
                f"{action} "
                f"{pest_action.strip()}"
            )

    return action.strip()


def _coerce_recommendation(
    raw: Any,
    fallback_severity: str = "Low",
) -> dict[str, str]:

    payload = dict(
        _DEFAULT_DISEASE_RECOMMENDATION
    )

    payload["severity_level"] = (
        fallback_severity
    )

    if isinstance(raw, dict):
        for key in RECOMMENDATION_KEYS:
            value = raw.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):
                payload[key] = value.strip()

    return payload


def _lookup_recommendations(
    raw_label: str,
    disease: str,
    pest_name: str | None,
) -> dict[str, Any]:

    disease_table = _remedies.get(
        "diseases",
        {},
    )

    pest_table = _remedies.get(
        "pests",
        {},
    )

    if not isinstance(
        disease_table,
        dict,
    ):
        disease_table = {}

    if not isinstance(
        pest_table,
        dict,
    ):
        pest_table = {}

    disease_raw = _lookup_mapping(
        disease_table,
        raw_label,
    )

    if not disease_raw:
        disease_raw = _lookup_mapping(
            disease_table,
            disease,
        )

    is_healthy = (
        "healthy"
        in _normalize_text(disease)
    )

    disease_rec = _coerce_recommendation(
        disease_raw,
        fallback_severity=(
            "Low"
            if is_healthy
            else "Medium"
        ),
    )

    pest_rec = None

    if pest_name:
        pest_raw = _lookup_mapping(
            pest_table,
            pest_name,
        )

        pest_rec = _coerce_recommendation(
            pest_raw,
            fallback_severity="Medium",
        )

    return {
        "disease": disease_rec,
        "pest": pest_rec,
    }


# ============================================================================
# WARNINGS
# ============================================================================

LOW_CONFIDENCE_THRESHOLD = 0.40


def _build_warnings(
    selected_crop: Optional[str],
    filtered_crop: str,
    filtered_confidence: float,
    global_crop: str,
    global_confidence: float,
) -> Optional[str]:
    """Build diagnostic warnings.

    Cross-crop mismatch is only meaningful when a separate global
    28-class prediction exists.

    Crop-specific models intentionally do NOT run the 28-class model
    just to generate a warning, because doing so would defeat the
    memory-saving architecture and load an unnecessary model.

    Low-confidence warning remains applicable to every model.
    """

    messages: List[str] = []

    has_global_prediction = bool(
        global_crop
        and _normalize_text(global_crop)
        not in {
            "",
            "unknown",
        }
    )

    if (
        selected_crop
        and isinstance(selected_crop, str)
        and selected_crop.strip()
        and has_global_prediction
        and _normalize_text(global_crop)
        != _normalize_text(filtered_crop)
    ):
        messages.append(
            "Visual features closely resemble "
            f"{global_crop}, but classification was "
            f"restricted to {filtered_crop} based on "
            "your selection."
        )

    if (
        filtered_confidence
        < LOW_CONFIDENCE_THRESHOLD
        and not messages
    ):
        messages.append(
            "Low classification confidence "
            f"({filtered_confidence:.2%}). "
            "Diagnosis may be uncertain."
        )

    return (
        " | ".join(messages)
        if messages
        else None
    )


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_models() -> None:
    """Validate files and load lightweight JSON configuration only.

    IMPORTANT:
    This function NEVER loads ML model weights.
    """

    global _disease_class_names
    global _pest_class_names
    global _actions
    global _crop_pests
    global _remedies
    global _loaded

    if _loaded:
        return

    with _INFERENCE_LOCK:

        if _loaded:
            return

        _validate_required_files()

        _disease_class_names = _index_map(
            _load_json(DISEASE_CLASSES_PATH),
            DISEASE_CLASSES_PATH,
        )

        if (
            len(_disease_class_names)
            != NUM_DISEASE_CLASSES
        ):
            raise ValueError(
                f"{DISEASE_CLASSES_PATH} must contain "
                f"{NUM_DISEASE_CLASSES} class names, "
                f"found {len(_disease_class_names)}."
            )

        _pest_class_names = _index_map(
            _load_json(PEST_CLASSES_PATH),
            PEST_CLASSES_PATH,
        )

        actions_raw = _load_json(
            ACTIONS_PATH
        )

        if not isinstance(
            actions_raw,
            dict,
        ):
            raise ValueError(
                f"{ACTIONS_PATH} must be a JSON object."
            )

        _actions = {
            str(key):
                (
                    _string_key_map(value)
                    if isinstance(value, dict)
                    else value
                )
            for key, value
            in actions_raw.items()
        }

        crop_pests_raw = _load_json(
            CROP_PESTS_PATH
        )

        if not isinstance(
            crop_pests_raw,
            dict,
        ):
            raise ValueError(
                f"{CROP_PESTS_PATH} "
                "must be a JSON object."
            )

        _crop_pests = {}

        for key, value in crop_pests_raw.items():

            if not isinstance(
                value,
                list,
            ):
                raise ValueError(
                    f"{CROP_PESTS_PATH} value for "
                    f"{key!r} must be a list "
                    "of pest names."
                )

            _crop_pests[str(key)] = [
                str(item)
                for item in value
            ]

        remedies_raw = _load_json(
            REMEDIES_PATH
        )

        if not isinstance(
            remedies_raw,
            dict,
        ):
            raise ValueError(
                f"{REMEDIES_PATH} "
                "must be a JSON object."
            )

        _remedies = {
            str(section):
                (
                    _string_key_map(value)
                    if isinstance(value, dict)
                    else value
                )
            for section, value
            in remedies_raw.items()
        }

        _loaded = True

        logger.info(
            "Lightweight inference configuration loaded. "
            "ML weights remain transient."
        )


# ============================================================================
# ORIGINAL 28-CLASS MODEL
# ============================================================================

def _ensure_disease_model() -> nn.Module:
    """Load a fresh original 28-class disease model."""

    checkpoint = None
    model = None

    try:

        try:
            checkpoint = torch.load(
                DISEASE_MODEL_PATH,
                map_location=_torch_device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                DISEASE_MODEL_PATH,
                map_location=_torch_device,
            )

        model = _build_mobilenetv3_small(
            NUM_DISEASE_CLASSES
        )

        model.load_state_dict(
            _extract_state_dict(checkpoint),
            strict=True,
        )

        model.to(_torch_device)
        model.eval()

    except Exception as exc:

        if model is not None:
            del model

        raise RuntimeError(
            "Failed to load original 28-class "
            f"disease model: {exc}"
        ) from exc

    finally:

        if checkpoint is not None:
            del checkpoint

        _release_memory()

    logger.info(
        "Loaded original 28-class disease model "
        "[transient]"
    )

    return model


# ============================================================================
# YOLO PEST MODEL
# ============================================================================

def _ensure_pest_model() -> Any:
    """Load a fresh YOLO pest model."""

    from ultralytics import YOLO

    try:
        model = YOLO(
            PEST_MODEL_PATH
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load YOLO pest model: "
            f"{exc}"
        ) from exc

    logger.info(
        "Loaded YOLO pest model [transient]"
    )

    return model


# ============================================================================
# DISEASE CLASSIFICATION
# ============================================================================

def _classify_disease(
    image: Image.Image,
    selected_crop: Optional[str] = None,
) -> Tuple[
    str,
    str,
    str,
    float,
    str,
    float,
    str,
]:
    """Classify disease.

    Supported crop:
        crop-specific model

    Unsupported/no crop:
        original 28-class model

    Returns:
        (
            crop,
            disease,
            raw_label,
            disease_confidence,
            global_crop,
            global_confidence,
            model_used,
        )
    """

    input_tensor = (
        disease_transform(image)
        .unsqueeze(0)
        .to(_torch_device)
    )

    norm_crop = _normalize_crop_name(
        selected_crop
    )

    # ------------------------------------------------------------------------
    # CROP-SPECIFIC MODEL
    # ------------------------------------------------------------------------

    if (
        norm_crop
        and norm_crop in CROP_MODEL_REGISTRY
    ):

        crop_model = None
        output = None
        probabilities = None

        try:

            (
                crop_model,
                class_names,
                canonical_crop,
            ) = _get_crop_model(
                norm_crop
            )

            with torch.no_grad():

                output = crop_model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )[0]

            num_classes = len(
                class_names
            )

            top_idx = int(
                torch.argmax(
                    probabilities[
                        :num_classes
                    ]
                ).item()
            )

            raw_label = class_names.get(
                top_idx,
                f"unknown_{top_idx}",
            )

            disease_confidence = float(
                probabilities[
                    top_idx
                ].cpu().item()
            )

            disease = _parse_crop_disease(
                raw_label,
                canonical_crop,
            )

            model_used = (
                f"{norm_crop}_mobilenetv3"
            )

            logger.info(
                "Using crop-specific disease "
                "model: %s",
                model_used,
            )

            # IMPORTANT:
            # No separate 28-class global prediction is
            # performed here. This keeps the selected crop
            # path memory-efficient and ensures only the
            # selected crop model is used.
            return (
                canonical_crop,
                disease,
                raw_label,
                disease_confidence,
                "",
                0.0,
                model_used,
            )

        finally:

            if output is not None:
                del output

            if probabilities is not None:
                del probabilities

            if crop_model is not None:
                del crop_model

            del input_tensor

            _release_memory()

    # ------------------------------------------------------------------------
    # ORIGINAL 28-CLASS FALLBACK
    # ------------------------------------------------------------------------

    disease_model = None
    output = None
    probabilities = None

    try:

        disease_model = (
            _ensure_disease_model()
        )

        with torch.no_grad():

            output = disease_model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1,
            )[0]

        num_classes = len(
            _disease_class_names
        )

        if num_classes == 0:
            raise ValueError(
                "_disease_class_names is empty."
            )

        # Global prediction.
        global_idx = int(
            torch.argmax(
                probabilities[
                    :num_classes
                ]
            ).item()
        )

        global_label = (
            _disease_class_names.get(
                global_idx,
                f"unknown_{global_idx}",
            )
        )

        global_crop, _ = (
            _parse_disease_label(
                global_label
            )
        )

        global_confidence = float(
            probabilities[
                global_idx
            ].cpu().item()
        )

        top_idx: Optional[int] = None

        # Crop filtering for the original
        # 28-class fallback.
        if (
            selected_crop
            and isinstance(
                selected_crop,
                str,
            )
            and selected_crop.strip()
        ):

            crop_hint = (
                selected_crop
                .strip()
                .lower()
            )

            matching_indices = [
                idx
                for idx, name
                in _disease_class_names.items()
                if (
                    crop_hint
                    in name.lower()
                    and idx
                    < len(probabilities)
                )
            ]

            if matching_indices:

                masked_probs = (
                    torch.zeros_like(
                        probabilities
                    )
                )

                for idx in matching_indices:
                    masked_probs[idx] = (
                        probabilities[idx]
                    )

                top_idx = int(
                    torch.argmax(
                        masked_probs
                    ).item()
                )

                del masked_probs

        if (
            top_idx is None
            or top_idx >= num_classes
        ):
            top_idx = global_idx

        disease_confidence = float(
            probabilities[
                top_idx
            ].cpu().item()
        )

        raw_label = (
            _disease_class_names.get(
                top_idx,
                f"unknown_{top_idx}",
            )
        )

        crop, disease = (
            _parse_disease_label(
                raw_label
            )
        )

        model_used = (
            "plant_disease_mobilenetv3_28"
        )

        logger.info(
            "Using original 28-class "
            "disease model"
        )

        return (
            crop,
            disease,
            raw_label,
            disease_confidence,
            global_crop,
            global_confidence,
            model_used,
        )

    finally:

        if output is not None:
            del output

        if probabilities is not None:
            del probabilities

        if disease_model is not None:
            del disease_model

        del input_tensor

        _release_memory()


# ============================================================================
# PEST DETECTION
# ============================================================================

def _detect_pests(
    image_path: str,
    crop: str,
) -> tuple[
    str | None,
    float,
    list[dict[str, Any]],
]:
    """Run YOLO pest detection.

    YOLO is loaded only after disease classification has
    completely released its model.
    """

    pest_model = None
    results = None

    try:

        pest_model = (
            _ensure_pest_model()
        )

        results = pest_model.predict(
         source=image_path,
         verbose=False,
         device=device,
         imgsz=256,
         stream=True,
         max_det=10,
         )

        raw_pest_detections: list[
            dict[str, Any]
        ] = []

        for result in results:

            names = result.names or {}

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                xyxy = [
                    float(value)
                    for value
                    in box.xyxy[0].tolist()
                ]

                pest_name = (
                    _resolve_pest_name(
                        class_id,
                        names,
                    )
                )

                raw_pest_detections.append(
                    {
                        "class_id": class_id,
                        "name": pest_name,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                        "bbox_xyxy": xyxy,
                    }
                )

        raw_pest_detections.sort(
            key=lambda item:
                item["confidence"],
            reverse=True,
        )

        relevant = [
            item
            for item
            in raw_pest_detections
            if (
                _pest_allowed_for_crop(
                    crop,
                    item["name"],
                )
                and item["confidence"]
                >= 0.30
            )
        ]

        if not relevant:
            return (
                None,
                0.0,
                raw_pest_detections,
            )

        top = relevant[0]

        return (
            str(top["name"]),
            float(top["confidence"]),
            raw_pest_detections,
        )

    finally:

        if results is not None:
            del results

        if pest_model is not None:
            del pest_model

        _release_memory()


# ============================================================================
# SINGLE IMAGE ANALYSIS
# ============================================================================

def analyze_crop_image(
    image_path: str,
    selected_crop: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete disease + pest analysis."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Only lightweight configuration is loaded.
    load_models()

    try:

        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

    except OSError as exc:

        raise ValueError(
            f"Could not read image: {image_path}"
        ) from exc

    crop = None
    disease = None
    raw_label = None
    disease_confidence = 0.0
    global_crop = ""
    global_confidence = 0.0
    model_used = None
    pest_detected = None
    pest_confidence = 0.0
    raw_pest_detections = []

    # Only one inference request at a time.
    #
    # This is especially important on the 512 MB Render
    # instance because concurrent requests could otherwise
    # load multiple copies of MobileNet/YOLO.
    with _INFERENCE_LOCK:

        try:

            (
                crop,
                disease,
                raw_label,
                disease_confidence,
                global_crop,
                global_confidence,
                model_used,
            ) = _classify_disease(
                image,
                selected_crop=selected_crop,
            )

            # Disease model has already been released
            # before YOLO is loaded.
            (
                pest_detected,
                pest_confidence,
                raw_pest_detections,
            ) = _detect_pests(
                image_path,
                crop,
            )

        finally:

            del image

            _release_memory()

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

        "disease_confidence": round(
            max(
                0.0,
                float(disease_confidence),
            ),
            4,
        ),

        "pest_detected": pest_detected,

        "pest_confidence": round(
            pest_confidence,
            4,
        ),

        "recommended_action": (
            _recommended_action(
                raw_label,
                crop,
                disease,
                pest_detected,
            )
        ),

        "raw_pest_detections": (
            raw_pest_detections
        ),

        "recommendations": (
            _lookup_recommendations(
                raw_label,
                disease,
                pest_detected,
            )
        ),

        "warning": warning,

        "model_used": model_used,
    }


# ============================================================================
# MULTI-IMAGE ANALYSIS
# ============================================================================

def analyze_crop_images(
    image_paths: list[str],
    selected_crop: Optional[str] = None,
) -> list[dict[str, Any]]:

    # Each image is processed completely before the
    # next image starts.
    #
    # analyze_crop_image() itself handles the inference
    # lock and releases every model before returning.

    return [
        analyze_crop_image(
            path,
            selected_crop=selected_crop,
        )
        for path in image_paths
    ]


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":

    import sys

    logging.basicConfig(
        level=logging.INFO
    )

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python inference.py <image_path>"
        )

    print(
        json.dumps(
            analyze_crop_image(
                sys.argv[1]
            ),
            indent=2,
        )
    )