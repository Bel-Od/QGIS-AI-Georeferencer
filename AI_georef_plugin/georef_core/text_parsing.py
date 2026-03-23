from __future__ import annotations

import re
import unicodedata


UTM_EASTING_RE = re.compile(r"\b(3\d{5}|4\d{5}|5\d{5}|6\d{5}|7\d{5}|8\d{5})\b")
UTM_NORTHING_RE = re.compile(r"\b(5[0-9]\d{5}|60\d{5}|61\d{5}|62\d{5}|63\d{5}|64\d{5})\b")
COORD_PAIR_RE = re.compile(
    r"(?:\b(?:E|R|x|Ost(?:wert)?)\b)[:\s]{0,6}(\d[\d\s.]{4,10})"
    r"[^\n]{0,40}?"
    r"(?:\b(?:N|H|y|Nord(?:wert)?)\b)[:\s]{0,6}(\d[\d\s.]{5,11})",
    re.IGNORECASE,
)


def _clean_text_for_match(text: str) -> str:
    return (text or "").replace("\u00e4", "a").replace("\u00f6", "o").replace("\u00fc", "u").replace("\u00df", "ss")


def _parse_scale_value(raw: str) -> int | None:
    cleaned = re.sub(r"[^\d]", "", raw or "")
    if not cleaned:
        return None
    try:
        scale_val = int(cleaned)
    except ValueError:
        return None
    if 100 <= scale_val <= 100_000:
        return scale_val
    return None


def _extract_scale_candidates(text: str) -> list[tuple[int, float, str]]:
    candidates: list[tuple[int, float, str]] = []
    if not text:
        return candidates
    lines = text.splitlines()
    pattern = re.compile(r"(?<!\d)1\s*:\s*([0-9][0-9.\s]{1,12})(?!\d)")
    for idx, line in enumerate(lines):
        for match in pattern.finditer(line):
            scale_val = _parse_scale_value(match.group(1))
            if scale_val is None:
                continue
            score = 1.0
            line_norm = _clean_text_for_match(line.lower())
            prev_norm = _clean_text_for_match(lines[idx - 1].lower()) if idx > 0 else ""
            next_norm = _clean_text_for_match(lines[idx + 1].lower()) if idx + 1 < len(lines) else ""
            context = "inline"
            if any(tok in line_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 2.5
                context = "titleblock"
            elif any(tok in prev_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 2.0
                context = "titleblock-prev"
            elif any(tok in next_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 1.5
                context = "titleblock-next"
            if any(tok in line_norm for tok in ("schnitt", "detail", "schema", "schematisch", "querschnitt", "langsschnitt", "l\u00e4ngsschnitt")):
                score -= 1.5
                context = f"{context}+detail"
            if scale_val < 200:
                score -= 0.5
            candidates.append((scale_val, score, context))
    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return candidates


def _extract_best_scale(text: str) -> int | None:
    candidates = _extract_scale_candidates(text)
    if candidates:
        return candidates[0][0]
    return None


def _normalize_text_token(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("\u00df", "ss")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _merge_text_sources(primary: str, secondary: str) -> str:
    primary = (primary or "").strip()
    secondary = (secondary or "").strip()
    if not primary:
        return secondary
    if not secondary:
        return primary

    merged: list[str] = []
    seen_norm: set[str] = set()

    def _add_lines(text: str, *, fuzzy_against: set[str] | None = None) -> None:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            norm = _normalize_text_token(line)
            if len(norm) < 3 or norm in seen_norm:
                continue
            if fuzzy_against:
                tokens = set(norm.split())
                if tokens and any(len(tokens & set(existing.split())) / max(len(tokens), 1) >= 0.8 for existing in fuzzy_against):
                    continue
            merged.append(line)
            seen_norm.add(norm)

    _add_lines(primary)
    _add_lines(secondary, fuzzy_against=set(seen_norm))
    return "\n".join(merged)


def parse_coordinates(text: str, *, logger=None) -> dict:
    result = {"eastings": [], "northings": [], "pairs": [], "scale": None, "crs_hints": []}
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]

    for idx, line in enumerate(lines):
        search_lines = [line]
        if idx + 1 < len(lines) and len(line) < 80:
            search_lines.append(lines[idx + 1])
        for candidate_line in search_lines:
            for match in COORD_PAIR_RE.finditer(candidate_line):
                try:
                    easting = float(re.sub(r"[\s.]", "", match.group(1)))
                    northing = float(re.sub(r"[\s.]", "", match.group(2)))
                except ValueError:
                    continue
                if 280_000 <= easting <= 850_000 and 5_200_000 <= northing <= 6_100_000:
                    result["pairs"].append((easting, northing))

    result["eastings"] = [int(v) for v in UTM_EASTING_RE.findall(text or "")]
    result["northings"] = [int(v) for v in UTM_NORTHING_RE.findall(text or "")]

    scale_candidates = _extract_scale_candidates(text or "")
    if scale_candidates:
        result["scale"] = scale_candidates[0][0]

    for crs_kw in ["UTM", "ETRS", "Gauss", "Kr\u00fcger", "DHDN", "WGS", "25832", "31467"]:
        if crs_kw.lower() in (text or "").lower():
            result["crs_hints"].append(crs_kw)

    if logger is not None:
        logger(
            f"[ok] Parsed coords - eastings: {result['eastings'][:5]}, "
            f"northings: {result['northings'][:5]}, pairs: {result['pairs'][:3]}, "
            f"scale: {result['scale']}, CRS hints: {result['crs_hints']}"
        )
    return result

