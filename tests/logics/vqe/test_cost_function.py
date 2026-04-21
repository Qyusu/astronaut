import pytest

from astronaut.logics.vqe.cost_function import cost_function


def test_cost_function() -> None:
    assert cost_function(1.0, 0.8) == pytest.approx(0.2)
    assert cost_function(0.5, 1.0) == pytest.approx(0.5)
    assert cost_function(-1.0, -1.2) == pytest.approx(0.2)
