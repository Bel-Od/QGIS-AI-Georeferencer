from __future__ import annotations

from pathlib import Path

from .models import CandidateEvidence, GeoCandidate, StructuredHints, VisionResult
from .ranker import rerank_candidates
from .runtime import load_auto_georeference


_RADIUS_BY_TIER = {
    "address": 250.0,
    "street": 800.0,
    "feature": 2_000.0,
    "parcel": 500.0,
    "coordinates": 200.0,
    "city": 10_000.0,
    "fallback": 12_000.0,
}

_BASE_RANK_BY_SOURCE = {
    "library":          1.50,  # operator-verified ground truth — always ranked first
    "ocr_coordinates":  1.00,
    "parcel_ref":       0.95,
    "site_address":     0.93,
    "project_address":  0.90,
    "street_city":      0.82,
    "road_city":        0.78,
    "landmark_city":    0.68,
    "city_name":        0.45,
}

_TIER_BONUS = {
    "coordinates": 0.25,
    "parcel": 0.20,
    "address": 0.18,
    "street": 0.10,
    "feature": 0.06,
    "city": 0.00,
    "fallback": -0.05,
}


def build_candidates(
    *,
    src_path: Path,
    ocr_text: str,
    hints: StructuredHints,
    vision: VisionResult,
    parsed: dict,
    include_project_address: bool = True,
    library_path: Path | None = None,
) -> list[GeoCandidate]:
    ag = load_auto_georeference()

    candidates: list[GeoCandidate] = []
    seen: set[tuple[str, int, int]] = set()

    # ------------------------------------------------------------------
    # Library candidate — operator-verified ground truth for this plan.
    # Injected FIRST so it always wins heuristic ranking.  Conflicts are
    # intentionally skipped (the position was manually confirmed).
    # ------------------------------------------------------------------
    if library_path is not None:
        from .library import find_library_match, load_library
        _lib = load_library(library_path)
        _lib_entry = find_library_match(src_path.stem, _lib, plan_path=src_path)
        if _lib_entry is not None:
            _lib_label = (
                f"Library verified: {_lib_entry.plan_stem}"
                f" ({_lib_entry.source}, {_lib_entry.reviewed_at[:10]})"
            )
            _lib_evidence = [
                CandidateEvidence(
                    source="library",
                    text=f"Verified by {_lib_entry.reviewer or 'operator'} on {_lib_entry.reviewed_at[:10]}",
                    weight=2.0,
                )
            ]
            _lib_key = ("library", int(round(_lib_entry.center_easting)), int(round(_lib_entry.center_northing)))
            if _lib_key not in seen:
                seen.add(_lib_key)
                _lib_rank = _score_candidate(
                    source="library",
                    confidence_tier="coordinates",
                    evidence=_lib_evidence,
                    metadata={},
                    conflicts=[],  # never penalise verified entries
                )
                candidates.append(
                    GeoCandidate(
                        candidate_id="library-verified",
                        source="library",
                        confidence_tier="coordinates",
                        center_easting=float(_lib_entry.center_easting),
                        center_northing=float(_lib_entry.center_northing),
                        search_radius_m=150.0,  # tightest possible — we trust this position
                        label=_lib_label,
                        rank_score=_lib_rank,
                        evidence=_lib_evidence,
                        metadata={
                            "library_source":      _lib_entry.source,
                            "library_confidence":  _lib_entry.confidence,
                            "library_reviewed_at": _lib_entry.reviewed_at,
                            "library_delta_m":     _lib_entry.delta_from_pipeline_m,
                        },
                        conflicts=[],
                    )
                )

    project_address = ag.load_project_address() if include_project_address else None
    _extract_project_city = getattr(ag, "_extract_project_city", None)
    project_city = _extract_project_city(project_address) if (project_address and callable(_extract_project_city)) else None
    project_address_specific = bool(project_address and ag._address_is_specific(project_address))
    # If Vision AI flagged the title block address as an office/firm address,
    # don't use its city as an expected plan location — the office may be in a
    # completely different city from the actual construction site.
    _tb_is_office = str(vision.title_block.get("office") or "").strip().lower() in ("yes", "true", "1") if vision.title_block else False
    titleblock_city = ("" if _tb_is_office else str(vision.title_block.get("project_site_city") or "").strip()) if vision.title_block else ""
    overview_location = str(vision.overview.get("location_name") or "").strip() if vision.overview else ""
    trusted_project_city = project_city
    if project_city and not project_address_specific:
        _proj_norm = _norm_city(project_city)
        _site_norm = _norm_city(hints.site_city)
        _tb_norm = _norm_city(titleblock_city)
        if (_site_norm and _proj_norm != _site_norm) or (_tb_norm and _proj_norm != _tb_norm):
            trusted_project_city = None

    def add_candidate(
        candidate_id: str,
        source: str,
        confidence_tier: str,
        easting: float | None,
        northing: float | None,
        label: str,
        evidence: list[CandidateEvidence],
        metadata: dict | None = None,
    ) -> None:
        if easting is None or northing is None:
            return
        key = (source, int(round(easting)), int(round(northing)))
        if key in seen:
            return
        seen.add(key)
        metadata = metadata or {}
        conflicts = _infer_candidate_conflicts(
            source=source,
            metadata=metadata,
            label=label,
            hints=hints,
            project_city=trusted_project_city,
            titleblock_city=titleblock_city,
            overview_location=overview_location,
        )
        rank_score = _score_candidate(
            source=source,
            confidence_tier=confidence_tier,
            evidence=evidence,
            metadata=metadata,
            conflicts=conflicts,
        )
        candidates.append(
            GeoCandidate(
                candidate_id=candidate_id,
                source=source,
                confidence_tier=confidence_tier,
                center_easting=float(easting),
                center_northing=float(northing),
                search_radius_m=_RADIUS_BY_TIER.get(confidence_tier, _RADIUS_BY_TIER["fallback"]),
                label=label,
                rank_score=rank_score,
                evidence=evidence,
                metadata=metadata,
                conflicts=conflicts,
            )
        )

    if include_project_address:
        if project_address:
            gc = ag.geocode_address_to_utm32(project_address)
            if gc:
                tier = "address" if ag._address_is_specific(project_address) else "city"
                add_candidate(
                    "project-address",
                    "project_address",
                    tier,
                    gc[0],
                    gc[1],
                    project_address,
                    [CandidateEvidence(source="project_address", text=project_address, weight=1.0)],
                    {"query": project_address},
                )

    for idx, addr in enumerate(hints.site_addresses):
        query = addr.get("query")
        if not query:
            continue
        gc = ag.geocode_address_to_utm32(query)
        if gc:
            add_candidate(
                f"site-address-{idx}",
                "site_address",
                "address",
                gc[0],
                gc[1],
                query,
                [CandidateEvidence(source="ocr_address", text=query, weight=1.0, details=addr)],
                {"query": query},
            )

    if hints.site_street and hints.site_city:
        street_query = f"{hints.site_street}, {hints.site_city}"
        gc = ag.geocode_address_to_utm32(street_query)
        if gc:
            add_candidate(
                "street-city",
                "street_city",
                "street",
                gc[0],
                gc[1],
                street_query,
                [CandidateEvidence(source="structured_hints", text=street_query, weight=0.9)],
                {"street": hints.site_street, "city": hints.site_city},
            )

    for idx, road_code in enumerate(hints.road_codes):
        if not hints.site_city:
            continue
        road_query = f"{road_code}, {hints.site_city}"
        gc = ag.geocode_address_to_utm32(road_query)
        if gc:
            add_candidate(
                f"road-city-{idx}",
                "road_city",
                "street",
                gc[0],
                gc[1],
                road_query,
                [CandidateEvidence(source="road_code", text=road_code, weight=0.85)],
                {"road_code": road_code, "city": hints.site_city},
            )

    for idx, landmark in enumerate(hints.landmarks):
        if not hints.site_city:
            continue
        query = f"{landmark}, {hints.site_city}"
        gc = ag.geocode_address_to_utm32(query)
        if gc:
            add_candidate(
                f"landmark-{idx}",
                "landmark_city",
                "feature",
                gc[0],
                gc[1],
                query,
                [CandidateEvidence(source="landmark", text=landmark, weight=0.7)],
                {"landmark": landmark, "city": hints.site_city},
            )

    for idx, parcel_ref in enumerate(hints.parcel_refs):
        resolved = ag._lookup_nrw_parcel_centroid(
            parcel_ref,
            city=hints.site_city,
        )
        if resolved:
            easting, northing, details = resolved
            add_candidate(
                f"parcel-{idx}",
                "parcel_ref",
                "parcel",
                easting,
                northing,
                parcel_ref,
                [CandidateEvidence(source="parcel_ref", text=parcel_ref, weight=0.95)],
                details,
            )

    if parsed.get("pairs"):
        pair = parsed["pairs"][0]
        add_candidate(
            "ocr-coordinates",
            "ocr_coordinates",
            "coordinates",
            pair[0],
            pair[1],
            "OCR coordinate anchor",
            [CandidateEvidence(source="ocr_coordinates", text=str(pair), weight=1.0)],
            {"pair": pair},
        )

    if hints.site_city and not any(item.source == "project_address" for item in candidates):
        city_query = f"{hints.site_city}"
        gc = ag.geocode_address_to_utm32(city_query)
        if gc:
            add_candidate(
                "city-fallback",
                "city_name",
                "city",
                gc[0],
                gc[1],
                city_query,
                [CandidateEvidence(source="site_city", text=hints.site_city, weight=0.5)],
                {"city": hints.site_city},
            )

    if not candidates:
        auto_seed = ag.derive_auto_seed(vision.overview, ocr_text or "", src_path)
        if auto_seed:
            add_candidate(
                "legacy-auto-seed",
                auto_seed.get("_source", "legacy_auto_seed"),
                str(auto_seed.get("_seed_confidence") or "fallback"),
                auto_seed.get("center_easting"),
                auto_seed.get("center_northing"),
                auto_seed.get("_address") or "Legacy auto seed",
                [CandidateEvidence(source="legacy", text=auto_seed.get("_address"), weight=0.4)],
                dict(auto_seed),
            )

    candidates.sort(key=lambda item: (item.rank_score, -item.search_radius_m), reverse=True)
    return candidates


def select_top_candidates(
    candidates: list[GeoCandidate],
    limit: int = 3,
    *,
    hints: StructuredHints | None = None,
    vision: VisionResult | None = None,
    model_path: Path | None = None,
) -> list[GeoCandidate]:
    if limit <= 0:
        return []
    ranked = rerank_candidates(candidates, hints=hints, vision=vision, model_path=model_path)
    return ranked[:limit]


def _score_candidate(
    *,
    source: str,
    confidence_tier: str,
    evidence: list[CandidateEvidence],
    metadata: dict,
    conflicts: list[str],
) -> float:
    base = _BASE_RANK_BY_SOURCE.get(source, 0.30)
    if source == "project_address":
        if confidence_tier == "city":
            base -= 0.18
        elif confidence_tier == "address":
            base += 0.03
    tier_bonus = _TIER_BONUS.get(confidence_tier, 0.0)
    evidence_weight = sum(max(float(item.weight), 0.0) for item in evidence) * 0.03
    signal_bonus = 0.0
    if metadata.get("road_code"):
        signal_bonus += 0.02
    if metadata.get("city"):
        signal_bonus += 0.02
    if metadata.get("query"):
        signal_bonus += 0.01
    conflict_penalty = 0.18 * len(conflicts)
    if any(item.startswith("city_conflict") for item in conflicts):
        conflict_penalty += 0.22
    if any(item.startswith("office_like") for item in conflicts):
        conflict_penalty += 0.20
    return round(base + tier_bonus + evidence_weight + signal_bonus - conflict_penalty, 4)


def _infer_candidate_conflicts(
    *,
    source: str,
    metadata: dict,
    label: str,
    hints: StructuredHints,
    project_city: str | None,
    titleblock_city: str | None,
    overview_location: str | None,
) -> list[str]:
    # Operator-verified library entries are trusted — never penalised.
    if source == "library":
        return []
    conflicts: list[str] = []
    candidate_city = str(metadata.get("city") or "").strip()
    query = str(metadata.get("query") or "").strip().lower()
    if not candidate_city:
        parts = [part.strip() for part in label.split(",") if part.strip()]
        if len(parts) >= 2:
            candidate_city = parts[-2] if parts[-1].upper() == "NRW" else parts[-1]
    cand_norm = _norm_city(candidate_city)
    # overview_city from Vision AI is unreliable when it conflicts with the
    # project_city (Vision often reads office letterhead addresses or misreads
    # place names). Only include it as an expected city when it agrees with at
    # least one other source (project or title block) or when project_city is empty.
    _ov_city_norm = _norm_city(_extract_city_from_location(overview_location))
    _proj_city_norm = _norm_city(project_city)
    _tb_city_norm = _norm_city(titleblock_city)
    _ov_city_trusted = (
        not _ov_city_norm                           # empty → no constraint
        or not _proj_city_norm                      # no project address → trust vision
        or _ov_city_norm == _proj_city_norm         # agrees with project city
        or _ov_city_norm == _tb_city_norm           # agrees with title block city
    )
    expected_cities = {
        "project_city": _proj_city_norm,
        "titleblock_city": _tb_city_norm,
        "site_city": _norm_city(hints.site_city),
    }
    if _ov_city_trusted:
        expected_cities["overview_city"] = _ov_city_norm
    for name, expected in expected_cities.items():
        if cand_norm and expected and cand_norm != expected:
            conflicts.append(f"city_conflict:{name}")
    if source in {"site_address", "street_city"} and any(tok in query for tok in ("telefon", "tel.", "email", "e-mail", "www.", "engineering", "gmbh", "ag", "d-")):
        conflicts.append("office_like_context")
    if hints.client_address and query and query in str(hints.client_address).lower():
        conflicts.append("matches_client_address")
    return conflicts


def _norm_city(value: str | None) -> str:
    import re as _re
    raw = " ".join(str(value or "").strip().lower().split())
    # Strip leading postal/zip codes (e.g. "59174 kamen" → "kamen", "D-44575 castrop-rauxel" → "castrop-rauxel")
    raw = _re.sub(r'^[d-]?\d{4,5}\s+', '', raw).strip()
    return raw


def _extract_city_from_location(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts[-1] if parts else raw
