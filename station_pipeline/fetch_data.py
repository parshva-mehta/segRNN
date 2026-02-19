from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import (
    ARTIFACTS_DIR,
    CONTINUOUS_FEATURES,
    DEFAULT_START_DAYS_BACK,
    IEM_ASOS_REQUEST_URL,
    RAW_DATA_DIR,
)
from station_lookup import nearest_station


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _default_date_range(days_back: int) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days_back)
    return start, end


def _ensure_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_iem_url(station: str, start_date: date, end_date: date) -> str:
    params: list[tuple[str, str]] = [
        ("station", station),
        ("year1", str(start_date.year)),
        ("month1", str(start_date.month)),
        ("day1", str(start_date.day)),
        ("year2", str(end_date.year)),
        ("month2", str(end_date.month)),
        ("day2", str(end_date.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "yes"),
        ("report_type", "1"),
        ("report_type", "2"),
    ]
    for feature in CONTINUOUS_FEATURES:
        params.append(("data", feature))
    return f"{IEM_ASOS_REQUEST_URL}?{urlencode(params)}"


def _download_csv(url: str) -> str:
    request = Request(url, headers={"User-Agent": "station-pipeline/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _validate_csv_payload(payload: str) -> None:
    rows = list(csv.reader(payload.splitlines()))
    if len(rows) < 2:
        raise ValueError("Downloaded payload is empty or missing data rows.")

    headers = set(rows[0])
    required_cols = set(CONTINUOUS_FEATURES + ["station", "valid"])
    missing = sorted(required_cols - headers)
    if missing:
        raise ValueError(f"Downloaded CSV missing expected columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch weather data from IEM ASOS based on city/state."
    )
    parser.add_argument("--city", required=True, help="City name (example: Rochester)")
    parser.add_argument("--state", required=True, help="US state code (example: NY)")
    parser.add_argument(
        "--network",
        default=None,
        help="IEM network override (default: <STATE>_ASOS)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date in YYYY-MM-DD format (default: today - 365 days).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_START_DAYS_BACK,
        help="If --start is not set, fetch this many days back from today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs()

    if args.start and args.end:
        start_date = _parse_date(args.start)
        end_date = _parse_date(args.end)
    elif args.start and not args.end:
        start_date = _parse_date(args.start)
        end_date = date.today()
    elif not args.start and args.end:
        end_date = _parse_date(args.end)
        start_date = end_date - timedelta(days=args.days_back)
    else:
        start_date, end_date = _default_date_range(args.days_back)

    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    station_info = nearest_station(args.city, args.state, args.network)
    station = station_info["station"]
    url = _build_iem_url(station, start_date, end_date)

    payload = _download_csv(url)
    _validate_csv_payload(payload)

    filename = f"{station}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    raw_path = RAW_DATA_DIR / filename
    raw_path.write_text(payload, encoding="utf-8")

    metadata = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "station_lookup": station_info,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "request_url": url,
        "raw_csv_path": str(raw_path),
    }

    metadata_path = ARTIFACTS_DIR / "fetch_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Resolved station: {station} ({station_info['station_name']})")
    print(f"Distance from requested location: {station_info['distance_km']} km")
    print(f"Saved raw data: {raw_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

