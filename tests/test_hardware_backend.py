import copy

import pytest

from qmagent.contracts import DEFAULT_STATE
from qmagent.hardware_backend import HardwareApprovalError, RegisteredHardwareBackend
from qmagent.physical_backend import PhysicalSimulator


def raw_s21(request):
    return PhysicalSimulator(7, noise_scale=0).acquire(
        request["tool"], request["state"], request["scan"]
    )


def test_registered_hardware_backend_requires_action_time_approval():
    backend = RegisteredHardwareBackend({"sq.s21": raw_s21}, lambda request: False)
    with pytest.raises(HardwareApprovalError):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})


def test_registered_hardware_backend_returns_measured_fit_and_audit_trace():
    requests = []
    backend = RegisteredHardwareBackend(
        {"sq.s21": raw_s21}, lambda request: requests.append(request) or True
    )
    observation = backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert observation["synthetic"] is False
    assert observation["quality"]["reliable"] is True
    assert observation["tool_trace"][0]["operation"] == "acquire"
    assert requests[0]["sequence"] == 1


def test_hardware_failure_invokes_safe_shutdown():
    shutdowns = []

    def fail(_request):
        raise RuntimeError("instrument timeout")

    backend = RegisteredHardwareBackend(
        {"sq.s21": fail}, lambda request: True, lambda: shutdowns.append(True)
    )
    with pytest.raises(RuntimeError, match="instrument timeout"):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert shutdowns == [True]
