from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import IEM_GEOJSON_URL, NOMINATIM_URL


USER_AGENT = "station-pipeline/1.0 (educational use)"


@dataclass
class Location:
    city: str
    state: str
    latitude: float
    longitude: float
    display_name: str


def _json_get(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_city_state(city: str, state: str) -> Location:
    params = urlencode(
        {
            "city": city,
            "state": state,
            "country": "United States",
            "format": "jsonv2",
            "limit": 1,
        }
    )
    payload = _json_get(f"{NOMINATIM_URL}?{params}")
    if not payload:
        raise ValueError(f"Could not geocode city/state: {city}, {state}")

    result = payload[0]
    return Location(
        city=city,
        state=state.upper(),
        latitude=float(result["lat"]),
        longitude=float(result["lon"]),
        display_name=result.get("display_name", f"{city}, {state}"),
    )


def network_for_state(state: str) -> str:
    return f"{state.upper()}_ASOS"


def _extract_station_id(properties: dict[str, Any]) -> str | None:
    for key in ("sid", "station", "id", "icao", "sname"):
        value = properties.get(key)
        if value:
            return str(value)
    return None


def fetch_network_stations(network: str) -> list[dict[str, Any]]:
    url = IEM_GEOJSON_URL.format(network=network)
    payload = _json_get(url)
    stations: list[dict[str, Any]] = []

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [])
        if len(coords) < 2:
            continue

        station_id = _extract_station_id(properties)
        if not station_id:
            continue

        stations.append(
            {
                "station": station_id,
                "name": properties.get("sname", station_id),
                "network": network,
                "longitude": float(coords[0]),
                "latitude": float(coords[1]),
            }
        )
    if not stations:
        raise ValueError(f"No stations found for network '{network}'")
    return stations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def nearest_station(city: str, state: str, network: str | None = None) -> dict[str, Any]:
    location = geocode_city_state(city, state)
    selected_network = network or network_for_state(location.state)
    stations = fetch_network_stations(selected_network)

    best_station = None
    best_distance = float("inf")
    for station in stations:
        distance_km = haversine_km(
            location.latitude,
            location.longitude,
            station["latitude"],
            station["longitude"],
        )
        if distance_km < best_distance:
            best_distance = distance_km
            best_station = station

    if best_station is None:
        raise ValueError(f"No nearby station found in network '{selected_network}'")

    return {
        "requested_city": city,
        "requested_state": location.state,
        "resolved_location": {
            "display_name": location.display_name,
            "latitude": location.latitude,
            "longitude": location.longitude,
        },
        "network": selected_network,
        "station": best_station["station"],
        "station_name": best_station["name"],
        "station_latitude": best_station["latitude"],
        "station_longitude": best_station["longitude"],
        "distance_km": round(best_distance, 2),
    }

