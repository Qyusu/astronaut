import importlib
from collections import Counter
from typing import Callable, Iterable, cast

import numpy as np
import pennylane as qml
from loguru import logger
from qxmt.ansatze import BaseAnsatz
from qxmt.devices import PennyLaneDevice
from scipy.optimize import curve_fit

BASE_OPERATIONS = {"RX", "RY", "RZ", "CNOT"}


class AnsatzLoadingError(Exception):
    pass


class ScalabilityDetectionError(Exception):
    pass


class CircuitAnalysisError(Exception):
    pass


class AnsatzLoader:
    """Handles dynamic loading of ansatz classes."""

    def __init__(self, module_path: str, ansatz_name: str) -> None:
        self.module_path = module_path
        self.ansatz_name = ansatz_name

    def load_ansatz_class(self) -> Callable:
        """Load ansatz class from module path."""
        try:
            mod = importlib.import_module(self.module_path)
        except ModuleNotFoundError as e:
            logger.error(f"Module not found: {self.module_path}")
            raise AnsatzLoadingError(
                f"Module not found: {self.module_path}"
            ) from e

        try:
            ansatz_cls = getattr(mod, self.ansatz_name)
        except AttributeError as e:
            logger.error(
                f"Class '{self.ansatz_name}' not found in module '{self.module_path}'"
            )
            raise AnsatzLoadingError(
                f"Class '{self.ansatz_name}' not found in module '{self.module_path}'"
            ) from e

        return ansatz_cls


class CircuitAnalyzer:
    """Analyzes quantum circuit specifications."""

    @staticmethod
    def tape_in_basis(
        qnode: qml.QNode,
        init_params: np.ndarray,
        base_ops: Iterable[str] = ("RX", "RY", "RZ", "CNOT"),
        max_passes: int = 20,
    ) -> qml.tape.QuantumScript:
        """Convert quantum tape to base operations."""
        tape = qml.workflow.construct_tape(qnode)(init_params)

        base_ops = set(base_ops)

        def stop_at(op: qml.operation.Operation) -> bool:
            from pennylane.measurements import MeasurementProcess

            return (
                (op.name in base_ops)
                or (not op.has_decomposition)
                or isinstance(op, MeasurementProcess)
            )

        for _ in range(max_passes):
            nxt = tape.expand(stop_at=stop_at)
            if len(nxt.operations) == len(tape.operations):
                return nxt
            tape = nxt

        return tape

    def get_circuit_spec(
        self,
        device: qml.devices.Device,
        ansatz: BaseAnsatz,
        base_ops: Iterable[str],
    ) -> dict[str, int | dict[str, int]]:
        """Get circuit specifications including depth, gates count, and histogram."""
        try:

            def _circuit(params: np.ndarray) -> float:
                ansatz.circuit(params)
                return qml.expval(qml.PauliZ(0))

            qnode = qml.QNode(_circuit, device)
            tape = self.tape_in_basis(
                qnode, np.random.randn(ansatz.n_params), base_ops
            )
            ops = tape.operations

            return {
                "depth": tape.graph.get_depth(),
                "total_gates": len(ops),
                "hist": dict(Counter(op.name for op in ops)),
            }
        except Exception as e:
            logger.error(f"Failed to analyze circuit: {e}")
            raise CircuitAnalysisError(f"Failed to analyze circuit: {e}") from e


class ModelFitter:
    """Handles mathematical model fitting for scalability detection."""

    @staticmethod
    def const_model(N: np.ndarray, c: float) -> np.ndarray:
        """Constant model: f(N) = c"""
        return np.full_like(N, fill_value=c, dtype=float)

    @staticmethod
    def poly_model(N: np.ndarray, a: float, k: float) -> np.ndarray:
        """Polynomial model: f(N) = a * N^k"""
        return a * N**k

    @staticmethod
    def exp_model(N: np.ndarray, a: float, b: float) -> np.ndarray:
        """Exponential model: f(N) = a * exp(b * N)"""
        return a * np.exp(b * N)

    @staticmethod
    def log_model(N: np.ndarray, a: float, b: float) -> np.ndarray:
        """Logarithmic model: f(N) = a * log(b * N)"""
        with np.errstate(invalid="ignore"):
            return a * np.log(b * N)

    @staticmethod
    def fit_and_error(
        model: Callable,
        p0: tuple[float, ...],
        num_qubits: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """Fit model and return RSS and parameters."""
        try:
            popt, _ = curve_fit(model, num_qubits, targets, p0=p0, maxfev=5000)
            residuals = targets - model(num_qubits, *popt)
            rss = float(np.sum(residuals**2))
            return rss, popt
        except RuntimeError as e:
            logger.warning(f"Failed to fit model: {e}")
            return np.inf, None


class ScalabilityAnalyzer:
    """Analyzes scalability patterns in quantum circuits."""

    def __init__(self, model_fitter: ModelFitter | None = None) -> None:
        self.model_fitter = model_fitter or ModelFitter()

    def detect_scalability(
        self,
        num_qubits: np.ndarray | list[int],
        targets: np.ndarray | list[int],
    ) -> str:
        """Detect scalability order by fitting different models."""
        if len(num_qubits) != len(targets):
            raise ValueError("num_qubits and targets must have the same length")

        if len(num_qubits) < 2:
            logger.warning(
                "Cannot determine scalability with less than 2 data points"
            )
            return "unknown"

        num_qubits_arr = np.array(num_qubits)
        targets_arr = np.array(targets)

        try:
            # Fit different models
            rss_const, popt_const = self.model_fitter.fit_and_error(
                self.model_fitter.const_model,
                p0=(float(np.median(targets_arr)),),
                num_qubits=num_qubits_arr,
                targets=targets_arr,
            )

            rss_poly, popt_poly = self.model_fitter.fit_and_error(
                self.model_fitter.poly_model,
                p0=(1, 1),
                num_qubits=num_qubits_arr,
                targets=targets_arr,
            )

            rss_exp, popt_exp = self.model_fitter.fit_and_error(
                self.model_fitter.exp_model,
                p0=(1, 0.1),
                num_qubits=num_qubits_arr,
                targets=targets_arr,
            )

            rss_log, popt_log = self.model_fitter.fit_and_error(
                self.model_fitter.log_model,
                p0=(1, 1),
                num_qubits=num_qubits_arr,
                targets=targets_arr,
            )

            # Select the model with the smallest RSS
            best_type = min(
                [
                    ("const", rss_const, popt_const),
                    ("poly", rss_poly, popt_poly),
                    ("exp", rss_exp, popt_exp),
                    ("log", rss_log, popt_log),
                ],
                key=lambda x: x[1],
            )

            return self._format_scalability_result(best_type)

        except Exception as e:
            logger.error(f"Failed to detect scalability: {e}")
            raise ScalabilityDetectionError(
                f"Failed to detect scalability: {e}"
            ) from e

    def _format_scalability_result(self, best_type: tuple) -> str:
        """Format the scalability result based on best model."""
        model_name, _, popt = best_type

        if model_name == "const":
            return "O(1)" if popt is not None else "unknown constant order"
        elif model_name == "poly":
            return (
                f"O(N^{popt[1]:.2f})"
                if popt is not None
                else "unknown polynomial order"
            )
        elif model_name == "exp":
            return (
                f"O(exp({popt[1]:.2f}*N))"
                if popt is not None
                else "unknown exponential order"
            )
        else:  # log
            return (
                f"O({popt[1]:.2f}*log(N))"
                if popt is not None
                else "unknown log order"
            )


class AnsatzScalabilityDetector:
    """Main detector class that orchestrates scalability analysis."""

    def __init__(self, module_path: str, ansatz_name: str) -> None:
        self.module_path = module_path
        self.ansatz_name = ansatz_name

        self.ansatz_loader = AnsatzLoader(module_path, ansatz_name)
        self.circuit_analyzer = CircuitAnalyzer()
        self.scalability_analyzer = ScalabilityAnalyzer()

        self.ansatz_cls = self._load_ansatz_cls()
        self.qubits_list, self.params_list, self.gates_list = (
            self._get_scalability_list()
        )

    def detect_n_params_order(self) -> str:
        """Detect scalability order for number of parameters."""
        if not self.qubits_list or not self.params_list:
            logger.warning(
                "Failed to calculate number of parameters scalability: no data"
            )
            return "unknown"

        try:
            n_params_scalability = self.scalability_analyzer.detect_scalability(
                self.qubits_list, self.params_list
            )
            logger.info(
                f"Number of parameters scalability: {n_params_scalability}"
            )
            return n_params_scalability
        except ScalabilityDetectionError as e:
            logger.error(f"Failed to detect parameters scalability: {e}")
            return "unknown"

    def get_n_gates_order(self) -> str:
        """Detect scalability order for number of gates."""
        if not self.qubits_list or not self.gates_list:
            logger.warning(
                "Failed to calculate number of gates scalability: no data"
            )
            return "unknown"

        try:
            n_gates_scalability = self.scalability_analyzer.detect_scalability(
                self.qubits_list, self.gates_list
            )
            logger.info(f"Number of gates scalability: {n_gates_scalability}")
            return n_gates_scalability
        except ScalabilityDetectionError as e:
            logger.error(f"Failed to detect gates scalability: {e}")
            return "unknown"

    def _load_ansatz_cls(self) -> Callable | None:
        try:
            return self.ansatz_loader.load_ansatz_class()
        except AnsatzLoadingError:
            return None

    def _get_scalability_list(
        self, min_qubits: int = 4, max_qubits: int = 128, step: int = 2
    ) -> tuple[list[int], list[int], list[int]]:
        if self.ansatz_cls is None:
            logger.warning("Ansatz class not loaded, returning empty lists")
            return [], [], []

        qubits_list: list[int] = []
        params_list: list[int] = []
        gates_list: list[int] = []

        for n_qubits in range(min_qubits, max_qubits, step):
            try:
                device = PennyLaneDevice(
                    platform="pennylane",
                    device_name="default.qubit",
                    backend_name=None,
                    n_qubits=n_qubits,
                    shots=None,
                )
                hamiltonian = DummyHamiltonian(n_qubits)
                ansatz = self.ansatz_cls(device, hamiltonian)
                n_params = ansatz.n_params

                circuit_spec = self.circuit_analyzer.get_circuit_spec(
                    device.get_device(),
                    ansatz,
                    BASE_OPERATIONS,
                )
                n_gates = cast(int, circuit_spec["total_gates"])

                qubits_list.append(n_qubits)
                params_list.append(n_params)
                gates_list.append(n_gates)

            except (CircuitAnalysisError, Exception) as e:
                logger.error(
                    f"Failed to calculate ansatz scalability for {n_qubits} qubits: {e}"
                )
                break

        return qubits_list, params_list, gates_list


class DummyHamiltonian:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.n_orbitals = int(n_qubits / 2)
        n_occ_spatial = max(1, min(self.n_orbitals - 1, self.n_orbitals // 2))
        self.n_electrons = 2 * n_occ_spatial

    def get_n_qubits(self) -> int:
        return self.n_qubits

    def get_electrons(self) -> int:
        return self.n_electrons

    def get_molecular_orbitals(self) -> int:
        return self.n_orbitals

    def get_spin_orbitals(self) -> int:
        return self.n_qubits

    def get_active_electrons(self) -> int:
        return self.n_electrons

    def get_active_orbitals(self) -> int:
        return self.n_orbitals

    def get_active_spin_orbitals(self) -> int:
        return self.n_qubits
