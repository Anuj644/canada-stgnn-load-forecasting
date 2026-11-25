"""Compare experiment settings using their persisted JSON representation."""
import json


def validate_frozen_plan(prior, proposed):
    normalized = json.loads(json.dumps(proposed))
    if prior != normalized:
        differences = [key for key in sorted(set(prior) | set(normalized)) if prior.get(key) != normalized.get(key)]
        raise ValueError(f"Frozen experiment settings differ: {', '.join(differences)}")
    return normalized
