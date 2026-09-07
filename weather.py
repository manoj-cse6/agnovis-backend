"""
Weather service using Open-Meteo API (free, no API key required).

Environment variables:
  WEATHER_API_URL  - override the base API URL (default: Open-Meteo)
  WEATHER_API_KEY  - optional API key if using a paid provider

Fallback: if the API is unavailable, returns clearly-marked fallback data.
Cache: responses are cached for 1 hour per lat/lon coordinate pair.
"""

import os
import time
import hashlib
import logging
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# --- Configuration ---
WEATHER_API_URL = os.environ.get(
    "WEATHER_API_URL",
    "https://api.open-meteo.com/v1/forecast"
)
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")  # Optional for paid providers

# --- Simple in-memory cache: key -> (timestamp, data) ---
_weather_cache: Dict[str, tuple] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_key(latitude: float, longitude: float) -> str:
    """Round coordinates to 2 decimal places to improve cache hit rate."""
    lat_r = round(latitude, 2)
    lon_r = round(longitude, 2)
    return hashlib.md5(f"{lat_r}:{lon_r}".encode()).hexdigest()


def _get_cached(key: str) -> Optional[Dict[str, Any]]:
    entry = _weather_cache.get(key)
    if entry:
        ts, data = entry
        if time.time() - ts < CACHE_TTL_SECONDS:
            return data
    return None


def _set_cache(key: str, data: Dict[str, Any]) -> None:
    _weather_cache[key] = (time.time(), data)


def _fallback_weather() -> Dict[str, Any]:
    """Return clearly-marked fallback data when the API is unavailable."""
    return {
        "source": "fallback/unavailable",
        "note": "Live weather data is currently unavailable. Showing fallback values.",
        "forecast": []
    }


def get_7day_forecast(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Fetch a 7-day weather forecast from Open-Meteo (or configured provider).
    Returns parsed forecast data or fallback if unavailable.
    """
    key = _cache_key(latitude, longitude)
    cached = _get_cached(key)
    if cached:
        logger.info(f"Weather cache hit for {latitude},{longitude}")
        return cached

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "windspeed_10m_max",
            "weathercode",
        ],
        "timezone": "auto",
        "forecast_days": 7,
    }

    # If a provider key is configured, include it
    if WEATHER_API_KEY:
        params["apikey"] = WEATHER_API_KEY

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            raw = response.json()
    except Exception as exc:
        logger.warning(f"Weather API unavailable: {exc}")
        return _fallback_weather()

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    forecast = []
    for i, date in enumerate(dates):
        forecast.append({
            "date": date,
            "temperature_max_c": daily.get("temperature_2m_max", [None])[i],
            "temperature_min_c": daily.get("temperature_2m_min", [None])[i],
            "rainfall_mm": daily.get("precipitation_sum", [None])[i],
            "precipitation_probability_pct": daily.get("precipitation_probability_max", [None])[i],
            "windspeed_kmh": daily.get("windspeed_10m_max", [None])[i],
            "weather_code": daily.get("weathercode", [None])[i],
            "condition": _wmo_code_to_description(daily.get("weathercode", [None])[i]),
        })

    result = {
        "source": "live",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": raw.get("timezone", "unknown"),
        "forecast": forecast,
    }
    _set_cache(key, result)
    return result


def _wmo_code_to_description(code: Optional[int]) -> str:
    """Convert WMO weather code to human-readable description."""
    if code is None:
        return "Unknown"
    wmo_map = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return wmo_map.get(code, f"Code {code}")


def assess_risk(
    crop: str,
    disease: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute a risk assessment combining disease/crop type with real weather
    forecast when coordinates are available, or rule-based fallback otherwise.
    """
    # Fetch live weather if coordinates provided
    if latitude is not None and longitude is not None:
        weather_data = get_7day_forecast(latitude, longitude)
    else:
        weather_data = None

    disease_lower = disease.lower()
    is_fallback = weather_data is None or weather_data.get("source") != "live"

    # Extract today's forecast values if available
    today_temp = None
    today_humidity_proxy = None  # Open-Meteo free tier doesn't have hourly humidity in daily
    today_rainfall = None
    today_precip_prob = None
    today_condition = "Unknown"

    if weather_data and weather_data.get("forecast"):
        today = weather_data["forecast"][0]
        today_temp = today.get("temperature_max_c")
        today_rainfall = today.get("rainfall_mm", 0)
        today_precip_prob = today.get("precipitation_probability_pct", 0)
        today_condition = today.get("condition", "Unknown")

    # --- Risk classification rules ---
    risk_level = "Low"
    risk_reasons = []
    risk_advisory_parts = []

    # Disease-based risk
    if "healthy" in disease_lower:
        risk_level = "Low"
        risk_reasons.append("Crop appears healthy — no disease detected.")
        risk_advisory_parts.append("Continue regular monitoring.")
    elif "blight" in disease_lower or "mildew" in disease_lower:
        risk_level = "High"
        risk_reasons.append(f"{disease} detected — favored by high humidity and moderate temperatures.")
        risk_advisory_parts.append("Apply appropriate fungicides immediately. Ensure good air circulation.")
    elif "rot" in disease_lower or "scab" in disease_lower:
        risk_level = "Medium"
        risk_reasons.append(f"{disease} detected — excess moisture can worsen damage.")
        risk_advisory_parts.append("Monitor moisture levels and avoid over-watering.")
    elif "rust" in disease_lower:
        risk_level = "Medium"
        risk_reasons.append(f"{disease} detected — favored by warm, humid conditions.")
        risk_advisory_parts.append("Remove infected leaves and apply protective sprays.")
    else:
        risk_reasons.append(f"{disease} detected. Monitor the crop closely.")
        risk_advisory_parts.append("Follow recommended disease management practices.")

    # Weather-based risk escalation (only if live weather available)
    if not is_fallback:
        if today_precip_prob is not None and today_precip_prob >= 70:
            if risk_level != "High":
                risk_level = "High"
            risk_reasons.append(
                f"High precipitation probability ({today_precip_prob}%) forecast for today "
                f"— increases spread risk for fungal/bacterial diseases."
            )
            risk_advisory_parts.append("Apply preventive fungal treatment before rain arrives.")
        elif today_precip_prob is not None and today_precip_prob >= 40:
            if risk_level == "Low":
                risk_level = "Medium"
            risk_reasons.append(
                f"Moderate precipitation probability ({today_precip_prob}%) — wet conditions may worsen disease."
            )

        if today_temp is not None and today_temp > 30:
            risk_reasons.append(
                f"High temperature ({today_temp}°C) — may accelerate pest activity."
            )

    # Build weather summary for response
    if weather_data and not is_fallback:
        weather_summary = {
            "source": "live",
            "today_temperature_max_c": today_temp,
            "today_rainfall_mm": today_rainfall,
            "today_precipitation_probability_pct": today_precip_prob,
            "today_condition": today_condition,
            "7day_forecast_available": True,
        }
    else:
        weather_summary = {
            "source": "fallback/unavailable",
            "note": "Live weather data unavailable or no coordinates provided.",
            "7day_forecast_available": False,
        }

    return {
        "risk_level": risk_level,
        "risk_reason": " | ".join(risk_reasons),
        "risk_advisory": " ".join(risk_advisory_parts),
        "weather": weather_summary,
    }
