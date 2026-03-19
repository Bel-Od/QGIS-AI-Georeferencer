from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .models import GeoCandidate, StructuredHints, VisionResult
from .ranker import LinearRanker, extract_candidate_features


@dataclass(slots=True)
class RankingExample:
    case_id: str
    candidate_id: str
    label: int
    features: dict[str, float]


def load_ranking_examples(dataset_path: Path) -> list[RankingExample]:
    examples: list[RankingExample] = []
    try:
        content = dataset_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return examples
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        selected_candidate_id = row.get("selected_candidate_id")
        candidates = list(row.get("candidates") or [])
        hints = StructuredHints.from_dict(dict(row.get("structured_hints") or {}))
        vision_raw = dict(row.get("vision") or {})
        vision = VisionResult(
            overview=vision_raw.get("overview") if isinstance(vision_raw.get("overview"), dict) else vision_raw,
            title_block=vision_raw.get("title_block") if isinstance(vision_raw.get("title_block"), dict) else {},
        )
        case_id = str(row.get("case_path") or row.get("input_path") or f"case-{len(examples)}")
        review = dict(row.get("review") or {})
        status = review.get("status")
        # accepted: pipeline was correct — selected candidate is the positive.
        # corrected: pipeline was wrong — find which candidate was closest to the
        #            human-verified corrected position (target_center) and use THAT
        #            as the positive label.  Using selected_candidate_id here would
        #            train the model to reproduce the pipeline's mistake.
        # rejected:  all candidates were wrong — all negative.
        if status not in {"accepted", "corrected", "rejected"}:
            continue
        is_rejected = status == "rejected"

        # Determine the true positive candidate id for this case.
        target_candidate_id = selected_candidate_id
        if status == "corrected":
            target_center = row.get("target_center")
            if target_center and target_center.get("easting") is not None and target_center.get("northing") is not None:
                t_e = float(target_center["easting"])
                t_n = float(target_center["northing"])
                best_id: str | None = None
                best_dist2 = math.inf
                for cd in candidates:
                    c_e = cd.get("center_easting")
                    c_n = cd.get("center_northing")
                    if c_e is not None and c_n is not None:
                        d2 = (float(c_e) - t_e) ** 2 + (float(c_n) - t_n) ** 2
                        if d2 < best_dist2:
                            best_dist2 = d2
                            best_id = str(cd.get("candidate_id") or "")
                if best_id is not None:
                    target_candidate_id = best_id

        for candidate_data in candidates:
            from .models import CandidateEvidence
            evidence = [
                CandidateEvidence(
                    source=str(ev.get("source") or ""),
                    text=ev.get("text"),
                    weight=float(ev.get("weight") or 0.0),
                )
                for ev in (candidate_data.get("evidence") or [])
            ]
            candidate = GeoCandidate(
                candidate_id=str(candidate_data.get("candidate_id") or ""),
                source=str(candidate_data.get("source") or ""),
                confidence_tier=str(candidate_data.get("confidence_tier") or "fallback"),
                center_easting=candidate_data.get("center_easting"),
                center_northing=candidate_data.get("center_northing"),
                search_radius_m=float(candidate_data.get("search_radius_m") or 0.0),
                label=str(candidate_data.get("label") or ""),
                rank_score=float(candidate_data.get("rank_score") or 0.0),
                evidence=evidence,
                metadata=dict(candidate_data.get("metadata") or {}),
                conflicts=list(candidate_data.get("conflicts") or []),
            )
            examples.append(
                RankingExample(
                    case_id=case_id,
                    candidate_id=candidate.candidate_id,
                    label=0 if is_rejected else (1 if candidate.candidate_id == target_candidate_id else 0),
                    features=extract_candidate_features(candidate, hints=hints, vision=vision),
                )
            )
    return examples


def train_linear_ranker(
    examples: list[RankingExample],
    *,
    epochs: int = 40,
    learning_rate: float = 0.08,
    seed: int = 42,
) -> LinearRanker:
    random.seed(seed)
    weights: dict[str, float] = {}
    bias = 0.0
    data = list(examples)
    for _ in range(max(1, epochs)):
        random.shuffle(data)
        for example in data:
            score = bias
            for key, value in example.features.items():
                score += weights.get(key, 0.0) * float(value)
            pred = 1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0)))
            error = float(example.label) - pred
            bias += learning_rate * error
            for key, value in example.features.items():
                weights[key] = weights.get(key, 0.0) + learning_rate * error * float(value)
    return LinearRanker(bias=bias, weights=weights)


def evaluate_ranker(examples: list[RankingExample], ranker: LinearRanker | None = None) -> dict[str, float]:
    if not examples:
        return {"cases": 0.0, "top1_accuracy": 0.0}
    grouped: dict[str, list[RankingExample]] = {}
    for item in examples:
        grouped.setdefault(item.case_id, []).append(item)
    correct = 0
    for items in grouped.values():
        if ranker is None:
            scored = sorted(items, key=lambda item: item.features.get("heuristic_rank", 0.0), reverse=True)
        else:
            scored = sorted(items, key=lambda item: ranker.score_features(item.features), reverse=True)
        predicted = scored[0].candidate_id if scored else None
        target = next((item.candidate_id for item in items if item.label == 1), None)
        if predicted is not None and predicted == target:
            correct += 1
    return {
        "cases": float(len(grouped)),
        "top1_accuracy": float(correct) / float(len(grouped)),
    }
