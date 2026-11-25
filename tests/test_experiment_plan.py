import json

import pytest

from src.evaluation.experiment_plan import validate_frozen_plan


def test_json_roundtrip_of_split_tuples_is_a_valid_resume():
    proposed = {"splits": {"train": ("2017", "2019")}, "seeds": [42, 123], "epochs": 200}
    prior = json.loads(json.dumps(proposed))
    assert validate_frozen_plan(prior, proposed) == prior


def test_actual_training_setting_changes_are_rejected():
    with pytest.raises(ValueError, match="epochs"):
        validate_frozen_plan({"epochs": 200, "seeds": [42, 123]}, {"epochs": 100, "seeds": [42, 123]})
    with pytest.raises(ValueError, match="features"):
        validate_frozen_plan({"features": ["a", "b"]}, {"features": ["b", "a"]})
