import numpy as np
import pytest

from astronaut.logics.vqe.ansatz_scalability import (
    AnsatzLoader,
    AnsatzLoadingError,
    ModelFitter,
    ScalabilityAnalyzer,
)


def test_model_fitter_models() -> None:
    N = np.array([4, 8, 12])

    # Const model
    res = ModelFitter.const_model(N, 10.0)
    assert np.all(res == 10.0)

    # Poly model
    res = ModelFitter.poly_model(N, 1.0, 2.0)
    assert np.all(res == N**2)

    # Exp model
    res = ModelFitter.exp_model(N, 1.0, 0.1)
    assert np.all(res == np.exp(0.1 * N))

    # Log model
    res = ModelFitter.log_model(N, 1.0, 0.1)
    assert np.all(res == np.log(0.1 * N))


def test_model_fitter_fit_and_error() -> None:
    num_qubits = np.array([4, 8, 12, 16])
    targets = 2.0 * num_qubits**2  # poly

    rss, popt = ModelFitter.fit_and_error(
        ModelFitter.poly_model, (1.0, 1.0), num_qubits, targets
    )
    assert rss < 1e-5
    assert popt[0] == pytest.approx(2.0, rel=1e-2)
    assert popt[1] == pytest.approx(2.0, rel=1e-2)


def test_scalability_analyzer_detect_const() -> None:
    analyzer = ScalabilityAnalyzer()
    num_qubits = [4, 8, 12, 16]
    targets = [10, 10, 10, 10]
    result = analyzer.detect_scalability(num_qubits, targets)
    assert result == "O(1)"


def test_scalability_analyzer_detect_poly() -> None:
    analyzer = ScalabilityAnalyzer()
    num_qubits = [4, 8, 12, 16]
    targets = [16, 64, 144, 256]
    result = analyzer.detect_scalability(num_qubits, targets)
    assert "O(N^2.00)" in result


def test_ansatz_loader_fail() -> None:
    loader = AnsatzLoader("non.existent.module", "MyAnsatz")
    with pytest.raises(AnsatzLoadingError):
        loader.load_ansatz_class()
