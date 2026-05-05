from __future__ import annotations

import json
from typing import Any, Dict, List

from .model import (
    AttachmentPoint,
    ExperimentConfig,
    TargetDefinition,
    TargetPoint,
)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ExperimentConfig(**data)


def save_config(path: str, config: ExperimentConfig) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def load_target_definition(path: str) -> TargetDefinition:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    contour_points = [
        TargetPoint(id=pt["id"], x=pt["x"], y=pt["y"], normal=pt.get("normal"))
        for pt in data.get("contour_points", [])
    ]
    dense_points = [
        TargetPoint(id=pt["id"], x=pt["x"], y=pt["y"], normal=pt.get("normal"))
        for pt in data.get("dense_points", [])
    ]
    attachment_points = [
        AttachmentPoint(
            id=ap["id"],
            point_id=ap["point_id"],
            x=ap["x"],
            y=ap["y"],
            normal=ap.get("normal"),
            label=ap.get("label"),
        )
        for ap in data.get("attachment_points", [])
    ]

    return TargetDefinition(
        name=data.get("name", "target"),
        contour_points=contour_points,
        dense_points=dense_points,
        attachment_points=attachment_points,
    )


def save_target_definition(path: str, target: TargetDefinition) -> None:
    payload = {
        "name": target.name,
        "contour_points": [
            {"id": pt.id, "x": pt.x, "y": pt.y, "normal": pt.normal}
            for pt in target.contour_points
        ],
        "dense_points": [
            {"id": pt.id, "x": pt.x, "y": pt.y, "normal": pt.normal}
            for pt in target.dense_points
        ],
        "attachment_points": [
            {
                "id": ap.id,
                "point_id": ap.point_id,
                "x": ap.x,
                "y": ap.y,
                "normal": ap.normal,
                "label": ap.label,
            }
            for ap in target.attachment_points
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_target_definition(target: TargetDefinition) -> List[str]:
    errors: List[str] = []
    if len(target.contour_points) < 3:
        errors.append("contour_points must contain at least 3 points.")
    dense_ids = {pt.id for pt in target.dense_points}
    if len(dense_ids) != len(target.dense_points):
        errors.append("dense_points contain duplicate ids.")
    attachment_ids = {ap.id for ap in target.attachment_points}
    if len(attachment_ids) != len(target.attachment_points):
        errors.append("attachment_points contain duplicate ids.")
    for ap in target.attachment_points:
        if ap.point_id not in dense_ids:
            errors.append(f"attachment_point {ap.id} references missing point_id {ap.point_id}.")
    return errors


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

