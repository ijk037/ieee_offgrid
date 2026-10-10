"""
Spatial Analysis & Spatial Intelligence Modules for CivicPulse AI.
Person B Implementation:
1. group_by_geography: Groups alerts by ward/area_id and category with graceful handling of unknown areas.
2. cross_category_analysis: Scans for multiple anomaly categories occurring in the same area & time window.
3. civic_ripple_engine: Time-lagged proximity analysis identifying cascading civic hypotheses.
"""

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas import AlertResponse


# ---------------------------------------------------------------------------
# Robust Extraction Helpers
# ---------------------------------------------------------------------------

def extract_area_id(alert: AlertResponse) -> str:
    """
    Safely extract the geographic ward / area_id from an AlertResponse.
    Gracefully handles None, empty strings, and missing attributes by returning 'UNKNOWN'.
    """
    # 1. Direct attribute on AlertResponse
    if hasattr(alert, "area_id") and alert.area_id:
        val = str(alert.area_id).strip()
        if val and val.upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return val

    # 2. Inside alert.metrics
    metrics = getattr(alert, "metrics", {}) or {}
    for key in ("area_id", "ward", "zone", "region", "district"):
        val = metrics.get(key)
        if val is not None and str(val).strip() and str(val).upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return str(val).strip()

    # 3. From title (e.g., "... - Area Zone-4")
    title = getattr(alert, "title", "") or ""
    match = re.search(r"-\s*Area\s+([A-Za-z0-9_\-]+)", title, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val and val.upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return val

    # 4. From alert_id (e.g., "ALT-ZONE-4-1234ABCD")
    alert_id = getattr(alert, "alert_id", "") or ""
    match = re.search(r"^ALT-([A-Za-z0-9_\-]+)-[A-F0-9]{8}$", alert_id, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val and val.upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return val

    # 5. From description (e.g., "in Area 'Zone-4'")
    desc = getattr(alert, "description", "") or ""
    match = re.search(r"in\s+Area\s+'([^']+)'", desc, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val and val.upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return val

    return "UNKNOWN"


def extract_category(alert: AlertResponse) -> str:
    """
    Safely extract civic category from an AlertResponse.
    Falls back gracefully to 'General' if not found.
    """
    # 1. Direct attribute
    if hasattr(alert, "category") and alert.category:
        val = str(alert.category).strip()
        if val and val.upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return val

    # 2. Inside alert.metrics
    metrics = getattr(alert, "metrics", {}) or {}
    for key in ("category", "issue_type", "incident_type", "type"):
        val = metrics.get(key)
        if val is not None and str(val).strip() and str(val).upper() not in ("NONE", "UNKNOWN", "NULL", "NAN", ""):
            return str(val).strip()

    # 3. From title (e.g., "[HIGH ALERT] Surge in Water Supply Outage - Area Zone-4")
    title = getattr(alert, "title", "") or ""
    match = re.search(r"Surge in\s+(.+?)\s*-\s*Area", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"Spike in\s+(.+?)\s*-\s*Area", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 4. From description (e.g., "surge of 'Water Supply Outage' incidents")
    desc = getattr(alert, "description", "") or ""
    match = re.search(r"surge of\s+'([^']+)'", desc, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"surge in\s+'([^']+)'", desc, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return "General"


def extract_time_window(alert: AlertResponse) -> str:
    """
    Safely extract or infer time window string from an AlertResponse.
    """
    # 1. Direct attribute
    if hasattr(alert, "time_window") and alert.time_window:
        return str(alert.time_window).strip()

    # 2. Inside metrics
    metrics = getattr(alert, "metrics", {}) or {}
    for key in ("time_window", "window", "interval"):
        val = metrics.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    # 3. From description
    desc = getattr(alert, "description", "") or ""
    match = re.search(r"during (?:time )?window\s+'([^']+)'", desc, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 4. Default to timestamp hour window
    ts = extract_timestamp(alert)
    return ts.strftime("%Y-%m-%d %H:00")


def extract_timestamp(alert: AlertResponse) -> datetime:
    """
    Safely extract and normalize timezone-aware UTC datetime from an AlertResponse.
    """
    ts = getattr(alert, "timestamp", None)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            pass
    return datetime.now(timezone.utc)


def extract_coordinates(alert: AlertResponse) -> Tuple[Optional[float], Optional[float]]:
    """
    Safely extract (latitude, longitude) coordinates from an AlertResponse.
    Returns (None, None) gracefully if missing, non-numeric, or sparse.
    """
    lat, lon = None, None

    # Check direct attributes
    if hasattr(alert, "latitude") and alert.latitude is not None:
        try:
            lat = float(alert.latitude)
        except (ValueError, TypeError):
            pass

    if hasattr(alert, "longitude") and alert.longitude is not None:
        try:
            lon = float(alert.longitude)
        except (ValueError, TypeError):
            pass

    # Check metrics dict safely
    raw_metrics = getattr(alert, "metrics", {}) or {}
    if hasattr(raw_metrics, "model_dump"):
        metrics = raw_metrics.model_dump()
    elif isinstance(raw_metrics, dict):
        metrics = raw_metrics
    else:
        metrics = {}
    if lat is None:
        for k in ("latitude", "lat"):
            if k in metrics and metrics[k] is not None:
                try:
                    lat = float(metrics[k])
                    break
                except (ValueError, TypeError):
                    pass

    if lon is None:
        for k in ("longitude", "lon", "lng"):
            if k in metrics and metrics[k] is not None:
                try:
                    lon = float(metrics[k])
                    break
                except (ValueError, TypeError):
                    pass

    # Check nested statistical_evidence
    evidence = metrics.get("statistical_evidence", {}) or {}
    if isinstance(evidence, dict):
        if lat is None:
            for k in ("latitude", "lat"):
                if k in evidence and evidence[k] is not None:
                    try:
                        lat = float(evidence[k])
                        break
                    except (ValueError, TypeError):
                        pass
        if lon is None:
            for k in ("longitude", "lon", "lng"):
                if k in evidence and evidence[k] is not None:
                    try:
                        lon = float(evidence[k])
                        break
                    except (ValueError, TypeError):
                        pass

    # Validate coordinate boundaries
    if lat is not None and not (-90.0 <= lat <= 90.0):
        lat = None
    if lon is not None and not (-180.0 <= lon <= 180.0):
        lon = None

    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two geographic points in kilometers.
    """
    R = 6371.0  # Earth's mean radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ---------------------------------------------------------------------------
# 1. Geographic & Category Grouping
# ---------------------------------------------------------------------------

class GeographyGroupDict(dict):
    """
    Specialized dictionary mapping geography/area_id -> List[AlertResponse].
    Provides robust, flexible access:
    - `groups["Ward-4"]` -> all alerts in Ward-4
    - `groups["Ward-4:Water Outage"]` -> alerts in Ward-4 with that category
    - `groups[("Ward-4", "Water Outage")]` -> tuple-based category lookup
    - Property `by_category`: dictionary grouped by category
    - Property `by_area_and_category`: dictionary with compound keys 'area:category'
    """

    def __missing__(self, key: Any) -> List[AlertResponse]:
        # Handle string compound keys e.g. "Ward-4:Water Outage" or "Ward-4 - Water Outage"
        if isinstance(key, str):
            for sep in (":", " - ", " / ", "|"):
                if sep in key:
                    parts = key.split(sep, 1)
                    area_part, cat_part = parts[0].strip(), parts[1].strip()
                    if area_part in self:
                        return [
                            a for a in self[area_part]
                            if extract_category(a).lower() == cat_part.lower()
                        ]
        # Handle tuple keys e.g. ("Ward-4", "Water Outage")
        if isinstance(key, tuple) and len(key) == 2:
            area_part, cat_part = str(key[0]).strip(), str(key[1]).strip()
            if area_part in self:
                return [
                    a for a in self[area_part]
                    if extract_category(a).lower() == cat_part.lower()
                ]
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for sep in (":", " - ", " / ", "|"):
                if sep in key:
                    parts = key.split(sep, 1)
                    area_part, cat_part = parts[0].strip(), parts[1].strip()
                    if super().__contains__(area_part):
                        return any(
                            extract_category(a).lower() == cat_part.lower()
                            for a in self[area_part]
                        )
        if isinstance(key, tuple) and len(key) == 2:
            area_part, cat_part = str(key[0]).strip(), str(key[1]).strip()
            if super().__contains__(area_part):
                return any(
                    extract_category(a).lower() == cat_part.lower()
                    for a in self[area_part]
                )
        return False

    @property
    def by_category(self) -> Dict[str, List[AlertResponse]]:
        """Group all contained alerts by category."""
        res: Dict[str, List[AlertResponse]] = {}
        for alert_list in self.values():
            for alert in alert_list:
                cat = extract_category(alert)
                res.setdefault(cat, []).append(alert)
        return res

    @property
    def by_area_and_category(self) -> Dict[str, List[AlertResponse]]:
        """Return a dictionary keyed by compound 'area_id:category' strings."""
        res: Dict[str, List[AlertResponse]] = {}
        for area_id, alert_list in self.items():
            for alert in alert_list:
                cat = extract_category(alert)
                compound_key = f"{area_id}:{cat}"
                res.setdefault(compound_key, []).append(alert)
        return res


def group_by_geography(
    alerts: List[AlertResponse],
    compound_keys: bool = False,
) -> Dict[str, List[AlertResponse]]:
    """
    Groups alerts by ward/area_id and category, handling missing or unknown
    area identifiers gracefully.

    - Alerts with missing, null, or empty area IDs are grouped under 'UNKNOWN'.
    - If `compound_keys` is True, returns a dict with 'area_id:category' keys.
    - By default, returns a GeographyGroupDict keyed by area_id that also supports
      compound and category lookups.

    Args:
        alerts: List of AlertResponse instances.
        compound_keys: If True, key dictionary directly by 'area_id:category'.

    Returns:
        Dict[str, List[AlertResponse]] grouping alerts geographically and by category.
    """
    if compound_keys:
        compound_result: Dict[str, List[AlertResponse]] = {}
        for alert in alerts:
            area = extract_area_id(alert)
            cat = extract_category(alert)
            key = f"{area}:{cat}"
            compound_result.setdefault(key, []).append(alert)
        return compound_result

    group_dict = GeographyGroupDict()
    for alert in alerts:
        area = extract_area_id(alert)
        group_dict.setdefault(area, []).append(alert)

    # Sort alerts within each geographic group by category and timestamp
    for area in group_dict:
        group_dict[area].sort(
            key=lambda a: (extract_category(a), extract_timestamp(a))
        )

    return group_dict


# ---------------------------------------------------------------------------
# 2. Cross-Category Analysis (Compounding Civic Crises)
# ---------------------------------------------------------------------------

def _format_compounding_narrative(
    area: str,
    tw: str,
    unique_categories: List[str],
    category_map: Dict[str, List[AlertResponse]],
    compound_severity: str,
    bucket_alerts: List[AlertResponse],
) -> Dict[str, str]:
    """
    Dynamically generates customized, localized titles, problem descriptions,
    localized neighborhood impacts, and recommended municipal actions
    for compounding civic crises, eliminating repetitive boilerplate strings.
    """
    # 1. Resolve localized ward title from alert metadata
    ward_title = None
    for a in bucket_alerts:
        raw_metrics = getattr(a, "metrics", {}) or {}
        if hasattr(raw_metrics, "model_dump"):
            m = raw_metrics.model_dump()
        elif isinstance(raw_metrics, dict):
            m = raw_metrics
        else:
            m = {}
        ev = m.get("statistical_evidence", {}) or {}
        if isinstance(ev, dict) and ev.get("ward_title"):
            ward_title = str(ev["ward_title"]).strip()
            break

    if ward_title and ward_title.lower() != str(area).lower():
        clean_num = str(area).replace("ward_", "Ward ").replace("ward", "Ward ")
        area_display = f"{ward_title} ({clean_num})"
    elif str(area).lower().startswith("ward_"):
        area_display = f"Ward {str(area).replace('ward_', '')}"
    else:
        area_display = str(area)

    # 2. Compute specific category counts and breakdown
    cat_details = []
    for cat in unique_categories:
        count = sum(
            int(
                (getattr(a, "metrics", {}) or {}).get("observed_count", 1)
                if isinstance(getattr(a, "metrics", {}), dict)
                else getattr(getattr(a, "metrics", {}), "observed_count", 1)
            )
            for a in category_map[cat]
        )
        cat_details.append(f"{cat} ({count} incident{'s' if count != 1 else ''})")

    breakdown_text = ", ".join(cat_details)

    # 3. Analyze specific interaction pairs
    cats_lower = [c.lower() for c in unique_categories]
    has_roads = any("road" in c or "pothole" in c or "infrastructure" in c for c in cats_lower)
    has_water = any("water" in c or "drainage" in c or "flood" in c for c in cats_lower)
    has_waste = any("sanitation" in c or "waste" in c or "garbage" in c or "clean" in c for c in cats_lower)
    has_light = any("light" in c or "electric" in c or "power" in c for c in cats_lower)
    has_traffic = any("traffic" in c or "signal" in c or "transit" in c for c in cats_lower)
    has_park = any("park" in c or "green" in c or "tree" in c or "animal" in c for c in cats_lower)
    has_health = any("health" in c or "disease" in c or "contamination" in c or "pollution" in c for c in cats_lower)

    if has_roads and has_water:
        title = f"Water Logging & Road Surface Deterioration in {area_display}"
        desc = (
            f"Severe water leakage and stormwater drainage overflow are actively compounding road sub-base deterioration in {area_display}. "
            f"Submerged potholes and road cave-ins are directly bottlenecking commuter transit, with {breakdown_text} recorded during window '{tw}'."
        )
        impact = f"Stagnant water runoff undermines asphalt foundations across {area_display}, doubling physical street deterioration and concealing road craters beneath standing pools."
        action = "Dispatch Water Supply & Sewerage (BWSSB) drainage pumps immediately to de-water carriageways before deploying Road Infrastructure emergency cold-patch asphalt teams."

    elif has_roads and has_traffic:
        title = f"Road Surface Cave-ins & Severe Traffic Bottlenecks in {area_display}"
        desc = (
            f"Major road surface defects and pothole clusters in {area_display} are severely throttling commuter lanes, "
            f"triggering spillover traffic congestion ({breakdown_text}) during window '{tw}'."
        )
        impact = f"Vehicles diverting onto narrow residential feeder lanes in {area_display} causes localized gridlock and acute safety hazards for pedestrians and schoolchildren."
        action = "Deploy municipal traffic marshals for intersection detour management while Road Maintenance emergency units install barricades and rapid asphalt patches."

    elif has_water and (has_waste or has_health):
        title = f"Drainage Blockage & Public Sanitation Hazard in {area_display}"
        desc = (
            f"Blocked stormwater drains and uncollected solid waste are compounding in {area_display}. "
            f"Accumulated garbage obstructs drainage outflows, threatening localized street sewage backflows ({breakdown_text}) during window '{tw}'."
        )
        impact = f"Overflowing sewage and stagnant runoff near residential doorsteps in {area_display} generates severe vector-borne disease risks, odor complaints, and environmental contamination."
        action = "Deploy mechanical drain jetting and suction tankers alongside priority Solid Waste clearing tippers to open clogged neighborhood culverts."

    elif has_waste and has_roads:
        title = f"Road Access Barriers & Garbage Collection Backlog in {area_display}"
        desc = (
            f"Severe road surface degradation in {area_display} is blocking heavy municipal collection vehicles, "
            f"triggering uncollected garbage pileups alongside road damage complaints ({breakdown_text}) during window '{tw}'."
        )
        impact = f"Sanitation compactor trucks cannot navigate damaged lane corridors, leaving uncollected solid waste mounds along primary thoroughfares in {area_display}."
        action = "Reroute solid waste collection trucks along alternative feeder lanes and prioritize road leveling on key waste transport routes."

    elif has_light and (has_traffic or has_roads):
        title = f"Corridor Blackout & Traffic Signal Malfunction in {area_display}"
        desc = (
            f"Electrical grid instability has triggered concurrent street lighting outages and traffic signal failures along transit corridors in {area_display} ({breakdown_text}) during window '{tw}'."
        )
        impact = f"Darkened transit corridors and unmonitored intersections in {area_display} significantly elevate evening road collision risks and pedestrian safety vulnerabilities."
        action = "Alert Electricity Supply Corporation (BESCOM) and Municipal Electrical Engineering for immediate transformer feeder inspection and signal battery backup restoration."

    elif has_park and (has_health or has_waste):
        title = f"Park Grounds Neglect & Open Waste Accumulation in {area_display}"
        desc = (
            f"Overgrown public park spaces, broken municipal enclosures, and illegal waste dumping are coinciding in {area_display} ({breakdown_text}) during window '{tw}'."
        )
        impact = f"Degraded public open spaces and open littering compromise community recreational access and provoke neighborhood environmental sanitation complaints."
        action = "Mobilize Horticulture Department cleanup crews alongside sanitation marshals for park perimeter fencing repair and thorough debris clearing."

    else:
        cat_1 = unique_categories[0]
        cat_2 = unique_categories[1]
        title = f"Concurrent Civic Infrastructure Stress: {cat_1} & {cat_2} in {area_display}"
        desc = (
            f"Concurrent municipal service disruptions observed in {area_display}: {breakdown_text} during window '{tw}'. "
            f"When distinct civic domains fail simultaneously in the same ward, municipal field response times double due to shared regional logistics."
        )
        impact = f"Simultaneous disruptions across {len(unique_categories)} distinct service domains place acute compounding strain on neighborhood operations in {area_display}."
        action = f"Convene joint multi-departmental field inspection in {area_display} with department heads of {cat_1} and {cat_2} for synchronized remediation."

    return {
        "title": title,
        "description": desc,
        "localized_impact": impact,
        "recommended_action": action,
        "area_display": area_display,
        "breakdown_text": breakdown_text,
    }


def cross_category_analysis(alerts: List[AlertResponse]) -> List[Dict[str, Any]]:
    """
    Scans for multiple anomaly categories occurring in the same geographic area
    during the same time window to detect compounding civic crises.

    A compounding crisis is detected when 2 or more distinct anomaly categories
    co-occur within the same geographic area during the same time window.

    Args:
        alerts: List of AlertResponse instances.

    Returns:
        List of compounding civic crisis records (Dict[str, Any]).
    """
    # 1. Bucket alerts by (area_id, time_window)
    time_geo_buckets: Dict[Tuple[str, str], List[AlertResponse]] = {}

    for alert in alerts:
        area = extract_area_id(alert)
        tw = extract_time_window(alert)
        time_geo_buckets.setdefault((area, tw), []).append(alert)

    crises: List[Dict[str, Any]] = []

    # 2. Identify compounding crises (multiple distinct categories)
    for (area, tw), bucket_alerts in time_geo_buckets.items():
        # Collect distinct categories
        category_map: Dict[str, List[AlertResponse]] = {}
        for a in bucket_alerts:
            cat = extract_category(a)
            category_map.setdefault(cat, []).append(a)

        unique_categories = sorted(list(category_map.keys()))

        # Compounding crises require at least 2 distinct anomaly categories
        if len(unique_categories) >= 2:
            alert_ids = [a.alert_id for a in bucket_alerts]
            severities = [a.severity.upper() for a in bucket_alerts]
            has_high = "HIGH" in severities

            # Compute compound severity level
            if has_high and len(unique_categories) >= 3:
                compound_severity = "CRITICAL"
            elif has_high or len(unique_categories) >= 2:
                compound_severity = "HIGH"
            else:
                compound_severity = "MEDIUM"

            clean_area = str(area).replace(" ", "-").upper()
            crisis_id = f"CRISIS-{clean_area}-{uuid.uuid4().hex[:6].upper()}"

            narrative = _format_compounding_narrative(
                area=area,
                tw=tw,
                unique_categories=unique_categories,
                category_map=category_map,
                compound_severity=compound_severity,
                bucket_alerts=bucket_alerts,
            )

            crisis_record: Dict[str, Any] = {
                "crisis_id": crisis_id,
                "area_id": area,
                "area_display": narrative["area_display"],
                "time_window": tw,
                "categories": unique_categories,
                "category_count": len(unique_categories),
                "alert_count": len(bucket_alerts),
                "alert_ids": alert_ids,
                "max_severity": "HIGH" if has_high else "MEDIUM",
                "compound_severity": compound_severity,
                "title": narrative["title"],
                "description": narrative["description"],
                "localized_impact": narrative["localized_impact"],
                "recommended_action": narrative["recommended_action"],
                "breakdown_text": narrative["breakdown_text"],
                "category_breakdown": {
                    cat: len(alts) for cat, alts in category_map.items()
                },
                "alerts": bucket_alerts,
            }
            crises.append(crisis_record)

    # Sort crises: highest category count first, then highest severity
    crises.sort(
        key=lambda c: (
            1 if c["max_severity"] == "HIGH" else 0,
            c["category_count"],
            c["alert_count"],
        ),
        reverse=True,
    )

    return crises


# ---------------------------------------------------------------------------
# 3. Civic Ripple Engine (Time-Lagged Proximity Cascading Analysis)
# ---------------------------------------------------------------------------

# Known municipal cascading archetypes (antecedent -> plausible cascade)
KNOWN_CASCADE_ARCHETYPES = {
    "water": ["health", "sanitation", "contamination", "drainage", "disease"],
    "infrastructure": ["health", "sanitation", "traffic", "safety", "water"],
    "power": ["traffic", "signal", "water", "transit", "safety"],
    "road": ["traffic", "transit", "accident", "emergency"],
    "waste": ["health", "sanitation", "pest", "odor", "disease"],
    "flooding": ["road", "traffic", "pothole", "contamination", "power"],
}


def _is_known_archetype(antecedent_cat: str, subsequent_cat: str) -> bool:
    """Check if the pair matches recognized domain cascading archetypes."""
    a_lower = antecedent_cat.lower()
    s_lower = subsequent_cat.lower()
    for trigger, cascades in KNOWN_CASCADE_ARCHETYPES.items():
        if trigger in a_lower:
            if any(c in s_lower for c in cascades):
                return True
    return False


def _format_plain_english_ripple(
    alert_a: AlertResponse,
    alert_b: AlertResponse,
    cat_a: str,
    cat_b: str,
    area_a: str,
    area_b: str,
    lag_hours: float,
    dist_km: Optional[float],
) -> Dict[str, str]:
    """
    Translates raw IDs, time intervals, and technical jargon into clear, plain-English
    narratives, notices, and recommended field actions tailored for non-technical municipal workers.
    """
    # 1. Ward title resolution
    def _get_ward_title(al: AlertResponse) -> Optional[str]:
        m = getattr(al, "metrics", {}) or {}
        if hasattr(m, "model_dump"):
            m = m.model_dump()
        ev = m.get("statistical_evidence", {}) if isinstance(m, dict) else getattr(m, "statistical_evidence", {})
        if isinstance(ev, dict) and ev.get("ward_title"):
            return str(ev["ward_title"]).strip()
        return None

    wt_a = _get_ward_title(alert_a)
    wt_b = _get_ward_title(alert_b)

    if area_a == area_b:
        if wt_a and wt_a.lower() != str(area_a).lower():
            clean_num = str(area_a).replace("ward_", "Ward ").replace("ward", "Ward ")
            loc_str = f"{wt_a} ({clean_num})"
        elif str(area_a).lower().startswith("ward_"):
            loc_str = f"Ward {str(area_a).replace('ward_', '')}"
        else:
            loc_str = f"Area {area_a}"
    else:
        name_a = f"{wt_a} ({area_a})" if wt_a else area_a
        name_b = f"{wt_b} ({area_b})" if wt_b else area_b
        loc_str = f"{name_a} and neighboring {name_b}"

    # 2. Plain-English time lag
    if lag_hours <= 1.5:
        lag_phrase = "within 1–2 hours"
        lag_badge = f"{int(round(lag_hours))}h Lag (Immediate)"
    elif lag_hours <= 12.0:
        lag_phrase = f"within {int(round(lag_hours))} hours"
        lag_badge = f"{int(round(lag_hours))}h Lag (Same Day)"
    elif 18.0 <= lag_hours <= 30.0:
        lag_phrase = "24 hours later (1 day later)"
        lag_badge = "24h Lag (1 Day)"
    elif 36.0 <= lag_hours <= 54.0:
        lag_phrase = "48 hours later (2 days later)"
        lag_badge = "48h Lag (2 Days)"
    else:
        days = int(round(lag_hours / 24.0))
        lag_phrase = f"{days} days later"
        lag_badge = f"{int(round(lag_hours))}h Lag ({days} Days)"

    # 3. Proximity plain-English
    if dist_km is not None and dist_km > 0:
        prox_text = f"Within {dist_km:.1f} km radius in {loc_str}"
    else:
        prox_text = f"Same ward location in {loc_str}"

    # 4. Domain-specific cause-and-effect notice for municipal employees
    c_a = cat_a.lower()
    c_b = cat_b.lower()

    if ("road" in c_a or "pothole" in c_a) and ("waste" in c_b or "sanitation" in c_b):
        title = f"Notice: Pothole & road complaints in {loc_str} often lead to garbage pileups {lag_phrase}"
        notice = (
            f"Field data shows road surface deterioration and pothole clusters in {loc_str} "
            f"are frequently followed by uncollected garbage complaints {lag_phrase}. "
            f"Damaged roads prevent municipal waste compactor trucks from accessing neighborhood lanes, causing missed collection rounds."
        )
        action = "Dispatch road patching crews immediately to clear route access and alert Solid Waste Management to deploy smaller tippers to bypass road obstructions."

    elif ("road" in c_a or "pothole" in c_a) and ("traffic" in c_b or "safety" in c_b):
        title = f"Notice: Road surface cave-ins in {loc_str} trigger severe traffic bottlenecks {lag_phrase}"
        notice = (
            f"Unaddressed road craters and asphalt degradation in {loc_str} reliably precipitate major traffic congestion {lag_phrase} "
            f"as vehicles abruptly swerve or decelerate to avoid road damage."
        )
        action = "Deploy emergency road asphalt patching and coordinate with municipal traffic wardens for temporary lane diversion."

    elif ("water" in c_a or "drainage" in c_a) and ("health" in c_b or "sanitation" in c_b):
        title = f"Notice: Water pipeline leaks in {loc_str} often precede sanitation & contamination complaints {lag_phrase}"
        notice = (
            f"Water pipeline bursts and drainage backflow in {loc_str} frequently precipitate sanitation overflow complaints {lag_phrase}. "
            f"Water main pressure loss and pooling water compromise local drainage networks and create stagnant pools."
        )
        action = "Dispatch Water & Sewerage (BWSSB) technicians to isolate pipeline leaks and deploy drain desilting crews for preventive flushing."

    elif ("water" in c_a or "drainage" in c_a) and ("road" in c_b or "infrastructure" in c_b):
        title = f"Notice: Water pipe leaks in {loc_str} frequently precede asphalt cave-ins {lag_phrase}"
        notice = (
            f"Subterranean water leakage and blocked stormwater runoff in {loc_str} are followed by road cave-ins {lag_phrase}. "
            f"Water seepage weakens underlying roadway soil sub-bases, causing sudden asphalt collapse under heavy vehicular load."
        )
        action = "Inspect underground pipeline joints and test road foundations for sub-surface voids before cave-ins expand."

    elif ("power" in c_a or "electric" in c_a) and ("light" in c_b or "signal" in c_b):
        title = f"Notice: Power supply drops in {loc_str} precede street lighting & traffic signal blackouts {lag_phrase}"
        notice = (
            f"Electrical grid voltage drops in {loc_str} frequently cascade into street lighting and traffic signal failures {lag_phrase}."
        )
        action = "Notify electricity distribution utility (BESCOM) to check substation feeder lines and ensure emergency signal battery backups are functional."

    else:
        title = f"Notice: '{cat_a}' spikes in {loc_str} often precede '{cat_b}' complaints {lag_phrase}"
        notice = (
            f"An initial civic surge of '{cat_a}' in {loc_str} was followed by an uptick in '{cat_b}' incidents {lag_phrase}. "
            f"Historical municipal tracking indicates early infrastructure strain in {cat_a} precipitates secondary service breakdowns."
        )
        action = f"Alert {cat_b} department supervisors to conduct early field inspection in {loc_str} before community complaints multiply."

    return {
        "plain_title": title,
        "plain_notice": notice,
        "plain_action": action,
        "plain_proximity": prox_text,
        "plain_lag": lag_badge,
        "loc_display": loc_str,
    }


def civic_ripple_engine(
    alerts: List[AlertResponse],
    min_lag_hours: float = 0.5,
    max_lag_hours: float = 48.0,
    max_distance_km: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    A time-lagged proximity analysis function that scans for cascading civic patterns
    (e.g., a water infrastructure complaint in an area preceding a health/sanitation
    complaint within a 24-48 hour window).

    IMPORTANT: Findings are labeled EXPLICITLY as hypotheses for investigation
    rather than definitive correlations.

    Coordinate Safety:
    - Gracefully handles missing latitude/longitude or sparse coordinates.
    - If coordinates are absent or sparse, safely falls back to administrative ward / area_id
      co-location without crashing.

    Args:
        alerts: List of AlertResponse instances.
        min_lag_hours: Minimum time lag in hours between antecedent and subsequent alerts (default 0.5h).
        max_lag_hours: Maximum time lag in hours to look for cascading effects (default 48.0h).
        max_distance_km: Maximum physical distance in km when coordinates are available (default 5.0 km).

    Returns:
        List of investigative hypothesis records (Dict[str, Any]).
    """
    hypotheses: List[Dict[str, Any]] = []

    # Sort alerts chronologically
    sorted_alerts = sorted(alerts, key=extract_timestamp)
    n = len(sorted_alerts)

    for i in range(n):
        alert_a = sorted_alerts[i]
        ts_a = extract_timestamp(alert_a)
        area_a = extract_area_id(alert_a)
        cat_a = extract_category(alert_a)
        lat_a, lon_a = extract_coordinates(alert_a)

        for j in range(i + 1, n):
            alert_b = sorted_alerts[j]
            ts_b = extract_timestamp(alert_b)
            area_b = extract_area_id(alert_b)
            cat_b = extract_category(alert_b)
            lat_b, lon_b = extract_coordinates(alert_b)

            # 1. Temporal Analysis: Calculate time lag in hours
            lag_hours = (ts_b - ts_a).total_seconds() / 3600.0

            if lag_hours < min_lag_hours:
                continue

            if lag_hours > max_lag_hours:
                break

            # Cascades occur across different issue categories
            if cat_a.lower() == cat_b.lower():
                continue

            # 2. Spatial Proximity Analysis (Safe Coordinate & Ward Handling)
            has_coords_a = lat_a is not None and lon_a is not None
            has_coords_b = lat_b is not None and lon_b is not None

            is_proximate = False
            proximity_basis = "NONE"
            dist_km: Optional[float] = None

            if has_coords_a and has_coords_b:
                try:
                    dist_km = haversine_distance(lat_a, lon_a, lat_b, lon_b)  # type: ignore[arg-type]
                    if dist_km <= max_distance_km:
                        is_proximate = True
                        proximity_basis = f"COORDINATE_PROXIMITY ({dist_km:.2f} km)"
                except Exception:
                    dist_km = None

            if not is_proximate:
                if area_a != "UNKNOWN" and area_b != "UNKNOWN" and area_a == area_b:
                    is_proximate = True
                    proximity_basis = f"WARD_CO_LOCATION (Area '{area_a}', sparse coordinates handled safely)"

            if not is_proximate:
                continue

            # 3. Formulate Cascading Ripple Hypothesis
            is_archetype = _is_known_archetype(cat_a, cat_b)
            hypothesis_id = f"HYP-RIPPLE-{uuid.uuid4().hex[:8].upper()}"

            clean_area = area_a if area_a == area_b else f"{area_a} / {area_b}"

            plain_ripple = _format_plain_english_ripple(
                alert_a=alert_a,
                alert_b=alert_b,
                cat_a=cat_a,
                cat_b=cat_b,
                area_a=area_a,
                area_b=area_b,
                lag_hours=lag_hours,
                dist_km=dist_km,
            )

            hypothesis_title = (
                f"[INVESTIGATIVE HYPOTHESIS] Potential Cascade: '{cat_a}' preceding '{cat_b}' "
                f"in {clean_area} ({lag_hours:.1f}h lag)"
            )

            hypothesis_desc = (
                f"INVESTIGATIVE HYPOTHESIS FOR FIELD INSPECTION: "
                f"An initial civic spike of '{cat_a}' (Alert: {alert_a.alert_id}) in Area '{area_a}' "
                f"preceded a subsequent surge in '{cat_b}' (Alert: {alert_b.alert_id}) in Area '{area_b}' "
                f"by {lag_hours:.1f} hours (within the {min_lag_hours:.0f}-{max_lag_hours:.0f}h ripple window). "
                f"Spatial linkage: {proximity_basis}. "
                f"Operators should investigate whether early failure in '{cat_a}' contributed to or accelerated "
                f"the subsequent '{cat_b}' incident."
            )

            record: Dict[str, Any] = {
                "hypothesis_id": hypothesis_id,
                # Mandatory explicit hypothesis labeling:
                "is_hypothesis": True,
                "correlation_status": "UNVERIFIED_HYPOTHESIS",
                "nature_of_finding": "HYPOTHESIS_FOR_INVESTIGATION",
                "disclaimer": (
                    "DISCLAIMER - HYPOTHESIS FOR INVESTIGATION ONLY: This cascading pattern is an investigative hypothesis "
                    "derived from temporal-spatial proximity. It does NOT establish definitive causal correlation "
                    "and requires field verification by municipal operations."
                ),
                "hypothesis_title": hypothesis_title,
                "hypothesis_description": hypothesis_desc,
                "plain_english_title": plain_ripple["plain_title"],
                "plain_english_notice": plain_ripple["plain_notice"],
                "plain_english_action": plain_ripple["plain_action"],
                "plain_english_proximity": plain_ripple["plain_proximity"],
                "plain_english_lag": plain_ripple["plain_lag"],
                "location_display": plain_ripple["loc_display"],
                "area_id": clean_area,
                "antecedent_alert_id": alert_a.alert_id,
                "subsequent_alert_id": alert_b.alert_id,
                "antecedent_category": cat_a,
                "subsequent_category": cat_b,
                "antecedent_timestamp": ts_a.isoformat(),
                "subsequent_timestamp": ts_b.isoformat(),
                "time_lag_hours": round(lag_hours, 2),
                "spatial_distance_km": round(dist_km, 2) if dist_km is not None else None,
                "proximity_basis": proximity_basis,
                "is_known_archetype": is_archetype,
                "recommended_action": plain_ripple["plain_action"],
                "antecedent_alert": alert_a,
                "subsequent_alert": alert_b,
            }
            hypotheses.append(record)

    # Sort hypotheses: known archetypes first, then by earliest lag
    hypotheses.sort(
        key=lambda h: (
            1 if h["is_known_archetype"] else 0,
            -h["time_lag_hours"],
        ),
        reverse=True,
    )

    return hypotheses


# ---------------------------------------------------------------------------
# Standalone Mock Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta

    print("=" * 75)
    print("Running CivicPulse AI Spatial Analysis - Local Verification Suite")
    print("=" * 75)

    base_time = datetime(2026, 10, 8, 10, 0, 0, tzinfo=timezone.utc)

    # Mock Alerts spanning geography, concurrent windows, and cascading ripples
    mock_alerts = [
        # Alert 1: Water Infrastructure spike in Zone-4 at T-0 (with coordinates)
        AlertResponse(
            alert_id="ALT-ZONE-4-WATER-001",
            severity="HIGH",
            title="[HIGH ALERT] Surge in Water Infrastructure - Area Zone-4",
            description="Civic spike detected: An anomalous surge of 'Water Infrastructure' in Area 'Zone-4'.",
            metrics={"observed_count": 65, "expected_count": 15.0, "time_window": "2026-10-08 10:00-11:00"},
            timestamp=base_time,
            area_id="Zone-4",
            category="Water Infrastructure",
            time_window="2026-10-08 10:00-11:00",
            latitude=28.6139,
            longitude=77.2090,
        ),
        # Alert 2: Traffic Signal Malfunction in Zone-4 at SAME time window (compounding crisis)
        AlertResponse(
            alert_id="ALT-ZONE-4-TRAFFIC-002",
            severity="MEDIUM",
            title="[MEDIUM ALERT] Surge in Traffic Signal - Area Zone-4",
            description="Civic spike detected: Traffic Signal issues in Area 'Zone-4'.",
            metrics={"observed_count": 28, "expected_count": 14.0, "time_window": "2026-10-08 10:00-11:00"},
            timestamp=base_time,
            area_id="Zone-4",
            category="Traffic Signal",
            time_window="2026-10-08 10:00-11:00",
            latitude=28.6145,
            longitude=77.2095,
        ),
        # Alert 3: Health & Sanitation spike in Zone-4 at T + 28 hours (cascading ripple, MISSING coordinates)
        AlertResponse(
            alert_id="ALT-ZONE-4-HEALTH-003",
            severity="HIGH",
            title="[HIGH ALERT] Surge in Health & Sanitation - Area Zone-4",
            description="Civic spike detected: Health & Sanitation complaints in Area 'Zone-4'.",
            metrics={"observed_count": 42, "expected_count": 10.0, "time_window": "2026-10-09 14:00-15:00"},
            timestamp=base_time + timedelta(hours=28),
            area_id="Zone-4",
            category="Health & Sanitation",
            time_window="2026-10-09 14:00-15:00",
            latitude=None,   # Sparse/missing coordinates handled safely
            longitude=None,
        ),
        # Alert 4: Road Flooding in Sector-9 at T + 5 hours (coordinates provided)
        AlertResponse(
            alert_id="ALT-SECTOR-9-FLOOD-004",
            severity="HIGH",
            title="[HIGH ALERT] Surge in Road Flooding - Area Sector-9",
            description="Civic spike detected: Road Flooding in Area 'Sector-9'.",
            metrics={"observed_count": 50, "expected_count": 12.0, "time_window": "2026-10-08 15:00-16:00"},
            timestamp=base_time + timedelta(hours=5),
            area_id="Sector-9",
            category="Road Flooding",
            time_window="2026-10-08 15:00-16:00",
            latitude=28.5355,
            longitude=77.3910,
        ),
        # Alert 5: Pothole Hazards in Sector-9 at T + 35 hours (ripple from flooding, close coordinates)
        AlertResponse(
            alert_id="ALT-SECTOR-9-POTHOLE-005",
            severity="MEDIUM",
            title="[MEDIUM ALERT] Surge in Pothole Hazards - Area Sector-9",
            description="Civic spike detected: Pothole Hazards in Area 'Sector-9'.",
            metrics={"observed_count": 30, "expected_count": 15.0, "time_window": "2026-10-09 21:00-22:00"},
            timestamp=base_time + timedelta(hours=35),
            area_id="Sector-9",
            category="Pothole Hazards",
            time_window="2026-10-09 21:00-22:00",
            latitude=28.5380,
            longitude=77.3930,
        ),
        # Alert 6: Missing / Unknown area identifier alert (graceful handling)
        AlertResponse(
            alert_id="ALT-UNKNOWN-006",
            severity="MEDIUM",
            title="[MEDIUM ALERT] Surge in Noise Complaint - Area UNKNOWN",
            description="Civic spike detected in unspecified location.",
            metrics={"observed_count": 18, "expected_count": 10.0},
            timestamp=base_time,
            area_id=None,    # Missing area_id
            category="Noise Complaint",
            time_window="2026-10-08 10:00-11:00",
            latitude=None,
            longitude=None,
        ),
    ]

    # -----------------------------------------------------------------------
    # Test 1: group_by_geography
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Testing group_by_geography...")
    geo_groups = group_by_geography(mock_alerts)

    assert "Zone-4" in geo_groups, "Zone-4 should be a geographic group"
    assert "Sector-9" in geo_groups, "Sector-9 should be a geographic group"
    assert "UNKNOWN" in geo_groups, "Missing area_id should be grouped under UNKNOWN"
    assert len(geo_groups["Zone-4"]) == 3, f"Zone-4 should have 3 alerts, got {len(geo_groups['Zone-4'])}"
    assert len(geo_groups["UNKNOWN"]) == 1, "UNKNOWN should have 1 alert"

    # Compound lookup test
    water_alerts = geo_groups["Zone-4:Water Infrastructure"]
    assert len(water_alerts) == 1, f"Expected 1 water alert, got {len(water_alerts)}"
    print(f"[PASS] Geographic grouping succeeded: {list(geo_groups.keys())}")
    print(f"       Gracefully mapped missing area to: 'UNKNOWN' ({len(geo_groups['UNKNOWN'])} alert)")
    print(f"       Compound lookup 'Zone-4:Water Infrastructure' -> {len(water_alerts)} alert")

    # -----------------------------------------------------------------------
    # Test 2: cross_category_analysis
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Testing cross_category_analysis (compounding crises)...")
    crises = cross_category_analysis(mock_alerts)

    assert len(crises) >= 1, "Should detect at least 1 compounding crisis in Zone-4"
    zone_4_crisis = next((c for c in crises if c["area_id"] == "Zone-4"), None)
    assert zone_4_crisis is not None, "Zone-4 compounding crisis not found"
    assert zone_4_crisis["category_count"] == 2, f"Expected 2 categories, got {zone_4_crisis['category_count']}"
    assert "Water Infrastructure" in zone_4_crisis["categories"]
    assert "Traffic Signal" in zone_4_crisis["categories"]
    assert zone_4_crisis["max_severity"] == "HIGH"
    print(f"[PASS] Detected {len(crises)} compounding crisis cluster(s):")
    for c in crises:
        print(f"       * {c['title']} ({', '.join(c['categories'])})")

    # -----------------------------------------------------------------------
    # Test 3: civic_ripple_engine
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Testing civic_ripple_engine (cascading hypotheses & sparse coords)...")
    ripples = civic_ripple_engine(mock_alerts, min_lag_hours=1.0, max_lag_hours=48.0)

    assert len(ripples) >= 1, "Should detect at least 1 ripple hypothesis"
    # Find the Water -> Health ripple
    water_health_hyp = next(
        (r for r in ripples if r["antecedent_category"] == "Water Infrastructure" and r["subsequent_category"] == "Health & Sanitation"),
        None,
    )
    assert water_health_hyp is not None, "Water -> Health cascading hypothesis not detected"
    assert water_health_hyp["is_hypothesis"] is True, "Must be explicitly labeled as hypothesis"
    assert water_health_hyp["correlation_status"] == "UNVERIFIED_HYPOTHESIS"
    assert "DISCLAIMER" in water_health_hyp["disclaimer"].upper()
    assert 27.9 <= water_health_hyp["time_lag_hours"] <= 28.1, f"Expected ~28h lag, got {water_health_hyp['time_lag_hours']}"
    assert "WARD_CO_LOCATION" in water_health_hyp["proximity_basis"], "Should safely handle missing coordinates via ward co-location"

    # Find the Flooding -> Pothole ripple with coordinates
    flood_pothole_hyp = next(
        (r for r in ripples if r["antecedent_category"] == "Road Flooding" and r["subsequent_category"] == "Pothole Hazards"),
        None,
    )
    assert flood_pothole_hyp is not None, "Flooding -> Pothole cascading hypothesis not detected"
    assert flood_pothole_hyp["spatial_distance_km"] is not None
    assert flood_pothole_hyp["spatial_distance_km"] < 1.0, "Coordinates should be within 1 km"

    print(f"[PASS] Detected {len(ripples)} cascading ripple hypotheses:")
    for r in ripples:
        print(f"       * {r['hypothesis_title']}")
        print(f"         Proximity: {r['proximity_basis']}")
        print(f"         Lag: {r['time_lag_hours']} hrs | Explicit Hypothesis: {r['is_hypothesis']}")
        print(f"         Disclaimer: {r['disclaimer'][:80]}...\n")

    print("=" * 75)
    print("All Spatial Intelligence modules successfully verified locally!")
    print("=" * 75)
