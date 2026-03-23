from __future__ import annotations

import re

from .text_parsing import _normalize_text_token


_STREET_TOKEN_RE = re.compile(
    r"(stra(?:\u00dfe|sse)|str\.|weg|allee|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt)",
    re.IGNORECASE,
)


def _extract_project_city(project_address: str) -> str | None:
    if not project_address:
        return None
    match = re.search(r"\b\d{5}\s+([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-/ ]+)", project_address or "")
    if match:
        return match.group(1).split(",")[0].strip()
    return None


def _address_is_specific(address: str) -> bool:
    if not address:
        return False
    has_number = bool(re.search(r"\b\d+[A-Za-z]?\b", address))
    return has_number and bool(_STREET_TOKEN_RE.search(address))


def _classify_address_confidence(address: str) -> str:
    if not address:
        return "city"
    if _address_is_specific(address):
        return "address"
    has_postcode = bool(re.search(r"\b\d{5}\b", address))
    has_street_token = bool(_STREET_TOKEN_RE.search(address))
    if has_street_token:
        return "street"
    if has_postcode:
        return "postcode"
    return "city"


def _extract_road_codes(text: str) -> list[str]:
    if not text:
        return []
    codes: list[str] = []
    for match in re.finditer(r"\b([ABLKS])\s*(\d{1,4}[a-z]?)\b", text, re.IGNORECASE):
        letter = match.group(1).upper()
        digits = match.group(2)
        code = f"{letter}{digits}"
        if len(digits) >= 2 and digits[0] == "0":
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        context = text[line_start:line_end].lower()
        if any(tok in context for tok in ("din a", "blatt a", "format a", "paper a", "size a")):
            continue
        if re.search(r"\(\s*" + re.escape(letter) + r"\s*" + re.escape(digits) + r"\s*\)", text[max(0, match.start() - 3):match.end() + 3]):
            continue
        codes.append(code)
    return list(dict.fromkeys(codes))


def _extract_station_text(text: str) -> str | None:
    if not text:
        return None
    match = re.search(
        r"(Stat\.?\s*[0-9+., ]+\s*(?:bis|-)\s*[0-9+., ]+|km\s*=\s*[0-9+., ]+)",
        text,
        re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else None


def _clean_city_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" ,")
    value = re.split(r"\b(?:Schild|Stat\.?|km|Datum|Ma(?:\u00dfs|ss)stab|Projekt|Unterlage)\b", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip(" ,")


def _extract_postal_address_candidates(text: str) -> list[dict]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits: list[dict] = []
    seen: set[str] = set()
    street_pat = re.compile(
        r"([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-/ ]{2,60}?"
        r"(?:stra(?:\u00dfe|sse)|str\.|weg|allee|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt))"
        r"\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    )
    city_pat = re.compile(r"\b(\d{5})\s+([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-/ ]{2,50})")
    office_ctx_pat = re.compile(
        r"(regionalniederlassung|niederlassung|auftraggeber|landesbetrieb|verwaltung|b\u00fcro|buero|anschrift|ruhr)",
        re.IGNORECASE,
    )

    for idx, line in enumerate(lines):
        street_match = street_pat.search(line)
        if not street_match:
            continue
        city_line = line
        city_match = city_pat.search(city_line)
        if not city_match and idx + 1 < len(lines):
            city_line = lines[idx + 1]
            city_match = city_pat.search(city_line)
        if not city_match:
            continue

        street = re.sub(r"\s+", " ", street_match.group(1)).strip(" ,")
        house_no = street_match.group(2).strip()
        postcode = city_match.group(1)
        city = re.sub(r"\s+", " ", city_match.group(2)).strip(" ,")
        query = f"{street} {house_no}, {postcode} {city}"
        context = " ".join(lines[max(0, idx - 2):min(len(lines), idx + 3)])
        context_lower = context.lower()
        office_markers = (
            "telefon", "telefax", "fax", "e-mail", "email", "www.",
            "engineering", "gmbh", " ag ", " mbh", "hauptsitz", "zentrale",
        )
        role = "office" if office_ctx_pat.search(context) or any(marker in context_lower for marker in office_markers) else "site"
        key = _normalize_text_token(query)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "query": query,
                "street": street,
                "house_number": house_no,
                "postcode": postcode,
                "city": city,
                "role": role,
                "context": context,
                "source": "postal_address",
            }
        )
    return hits


def extract_structured_location_hints(
    text: str,
    vision: dict | None = None,
    titleblock_meta: dict | None = None,
) -> dict:
    hints = {
        "site_street": None,
        "site_house_number": None,
        "site_city": None,
        "site_postcode": None,
        "road_codes": [],
        "station_text": None,
        "parcel_refs": [],
        "landmarks": [],
        "site_addresses": [],
        "office_addresses": [],
        "client_address": None,
    }
    text = text or ""
    titleblock_meta = titleblock_meta or {}

    for addr in _extract_postal_address_candidates(text):
        if addr["role"] == "office":
            hints["office_addresses"].append(addr)
            if hints["client_address"] is None:
                hints["client_address"] = addr["query"]
        else:
            hints["site_addresses"].append(addr)
            if hints["site_street"] is None:
                hints["site_street"] = addr["street"]
                hints["site_house_number"] = addr["house_number"]
                hints["site_city"] = addr["city"]
                hints["site_postcode"] = addr["postcode"]

    street_city_pat = re.compile(
        r"([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-/]+(?:\s+[A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-/]+){0,4}\s+"
        r"(?:stra(?:\u00dfe|sse)|allee|weg|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt))"
        r"\s+(?:in|bei|nahe)\s+([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+(?:\s+[A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+)?)",
        re.IGNORECASE,
    )
    road_city_pat = re.compile(
        r"\b([ABLKS])\s*(\d{1,4}[a-z]?)\b[^\n]{0,100}?\s+(?:in|bei|nahe)\s+"
        r"([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+(?:\s+[A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\-]+)?)",
        re.IGNORECASE,
    )
    standort_pat = re.compile(r"Standort\s*:?\s*([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\- ]{2,40})", re.IGNORECASE)
    landmark_pat = re.compile(r"[Ss]child\s+['\"]([^'\"]{3,80})['\"]")
    parcel_pat = re.compile(
        r"(?:Gemarkung\s+([A-Z\u00c4\u00d6\u00dc][A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\- ]+))?[^\n]{0,40}?"
        r"Flur\s+(\d+)[^\n]{0,40}?(?:Flurst(?:\u00fcck|ueck))\s+([\d/]+)",
        re.IGNORECASE,
    )

    for match in street_city_pat.finditer(text):
        street = re.sub(r"\s+", " ", match.group(1)).strip(" ,")
        city = _clean_city_name(match.group(2))
        if hints["site_street"] is None:
            hints["site_street"] = street
        if hints["site_city"] is None:
            hints["site_city"] = city

    for match in road_city_pat.finditer(text):
        road = f"{match.group(1).upper()}{match.group(2)}"
        city = _clean_city_name(match.group(3))
        if road not in hints["road_codes"]:
            hints["road_codes"].append(road)
        if hints["site_city"] is None:
            hints["site_city"] = city

    if not hints["road_codes"]:
        hints["road_codes"] = _extract_road_codes(text)

    site_match = standort_pat.search(text)
    if site_match and not hints["site_city"]:
        hints["site_city"] = _clean_city_name(site_match.group(1))

    for match in landmark_pat.finditer(text):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        if name and name not in hints["landmarks"]:
            hints["landmarks"].append(name)

    for match in parcel_pat.finditer(text):
        gemarkung = re.sub(r"\s+", " ", (match.group(1) or "")).strip(" ,")
        flur = match.group(2).strip()
        flurstueck = match.group(3).strip()
        parcel = f"{gemarkung + ', ' if gemarkung else ''}Flur {flur}, Flurstueck {flurstueck}"
        if parcel not in hints["parcel_refs"]:
            hints["parcel_refs"].append(parcel)

    hints["station_text"] = _extract_station_text(text)

    if titleblock_meta:
        if titleblock_meta.get("project_site_street") and not hints["site_street"]:
            hints["site_street"] = str(titleblock_meta["project_site_street"]).strip()
        if titleblock_meta.get("project_site_house_number") and not hints["site_house_number"]:
            hints["site_house_number"] = str(titleblock_meta["project_site_house_number"]).strip()
        if titleblock_meta.get("project_site_city") and not hints["site_city"]:
            hints["site_city"] = str(titleblock_meta["project_site_city"]).strip()
        if titleblock_meta.get("project_site_postcode") and not hints["site_postcode"]:
            hints["site_postcode"] = str(titleblock_meta["project_site_postcode"]).strip()
        if titleblock_meta.get("project_road_code"):
            road_code = re.sub(r"\s+", "", str(titleblock_meta["project_road_code"]).upper())
            if road_code and road_code not in hints["road_codes"]:
                hints["road_codes"].insert(0, road_code)
        if titleblock_meta.get("station_text") and not hints["station_text"]:
            hints["station_text"] = str(titleblock_meta["station_text"]).strip()
        if titleblock_meta.get("landmark_name"):
            landmark = str(titleblock_meta["landmark_name"]).strip()
            if landmark and landmark not in hints["landmarks"]:
                hints["landmarks"].insert(0, landmark)
        if titleblock_meta.get("client_address") and not hints["client_address"]:
            hints["client_address"] = str(titleblock_meta["client_address"]).strip()

    if vision and not hints["site_city"]:
        location_name = str(vision.get("location_name") or "").strip()
        if location_name:
            if "," in location_name:
                hints["site_city"] = location_name.split(",")[-1].strip()
                if not hints["site_street"]:
                    hints["site_street"] = location_name.split(",")[0].strip()
            else:
                hints["site_city"] = location_name

    return hints
