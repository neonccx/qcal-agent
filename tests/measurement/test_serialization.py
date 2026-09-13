import json

import numpy as np

from qcal.models import ExperimentResult


def test_complex_raw_data_round_trip_shape(tmp_path):
    result = ExperimentResult("test", "Q0", {}, {"s21": np.array([1 + 2j, 3 + 4j])})
    path = result.save_json(tmp_path / "result.json")
    payload = json.loads(path.read_text())
    assert payload["raw_data"]["s21"]["real"] == [1.0, 3.0]
    assert payload["raw_data"]["s21"]["imag"] == [2.0, 4.0]
