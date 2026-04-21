import json
from pathlib import Path
from typing import Any

from astronaut.logics.common.validation import (
    PennyLaneDocValidator,
    SyntaxValidator,
    ValidationResult,
    is_same_code,
)


class TestValidationResult:
    def test_add_error(self) -> None:
        result = ValidationResult(is_valid=True, errors=[])
        result.add_error("Error 1")
        assert not result.is_valid
        assert result.errors == ["Error 1"]

    def test_merge(self) -> None:
        result1 = ValidationResult(is_valid=True, errors=[], cost=1.0)
        result2 = ValidationResult(is_valid=False, errors=["Error 2"], cost=2.0)
        result1.merge(result2)
        assert not result1.is_valid
        assert result1.errors == ["Error 2"]
        assert result1.cost == 3.0


class TestSyntaxValidator:
    def test_validate_by_py_compile_valid(self) -> None:
        validator = SyntaxValidator()
        source_code = "print('hello')"
        result = validator.validate_by_py_compile(source_code)
        assert result.is_valid
        assert result.errors == []

    def test_validate_by_py_compile_invalid(self) -> None:
        validator = SyntaxValidator()
        source_code = "print('hello'"
        result = validator.validate_by_py_compile(source_code)
        assert not result.is_valid
        assert any("py_compile" in e for e in result.errors)

    def test_validate_by_ast_valid(self) -> None:
        validator = SyntaxValidator()
        source_code = "a = 1 + 2"
        result = validator.validate_by_ast(source_code)
        assert result.is_valid
        assert result.errors == []

    def test_validate_by_ast_invalid(self) -> None:
        validator = SyntaxValidator()
        source_code = "a = 1 +"
        result = validator.validate_by_ast(source_code)
        assert not result.is_valid
        assert any("ast" in e for e in result.errors)

    def test_validate_all(self) -> None:
        validator = SyntaxValidator()
        source_code = "x = 10"
        result = validator.validate(source_code)
        assert result.is_valid
        assert result.errors == []


class TestCodeValidator:
    def test_validate_all(self, mocker: Any, tmp_path: Path) -> None:
        import astronaut.constants as exp_const

        ref_path = tmp_path / "pennylane_reference.json"
        ref_path.write_text(json.dumps({"qml.RX": {"args": []}}))
        mocker.patch.object(
            exp_const, "PENNYLANE_REFERENCE_PATH", str(ref_path)
        )

        from astronaut.logics.common.validation import CodeValidator

        validator = CodeValidator(
            code="import pennylane as ql\nql.RX(0.1, wires=0)"
        )

        is_valid, errors, cost = validator.validate_all()

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        assert isinstance(cost, float)


class TestPennyLaneDocValidator:
    def _write_reference(self, tmp_path: Path) -> str:
        reference = {
            "qml.RX": {
                "args": [
                    {
                        "name": "theta",
                        "type": "float",
                        "description": "Rotation angle",
                        "required": True,
                    },
                    {
                        "name": "wires",
                        "type": "int | Sequence[int]",
                        "description": "Target wires",
                        "required": True,
                    },
                ]
            },
            "qml.AmplitudeEmbedding": {
                "args": [
                    {
                        "name": "features",
                        "type": "Sequence[float]",
                        "description": "Features to embed",
                        "required": True,
                    },
                    {
                        "name": "wires",
                        "type": "Sequence[int]",
                        "description": "Target wires",
                        "required": True,
                    },
                    {
                        "name": "normalize",
                        "type": "bool",
                        "description": "Whether to normalize",
                        "required": False,
                    },
                ]
            },
            "qml.qchem.nuclear_attraction": {
                "args": [
                    {
                        "name": "i",
                        "type": "int",
                        "description": "Index of the first electron",
                        "required": True,
                    },
                    {
                        "name": "j",
                        "type": "int",
                        "description": "Index of the second electron",
                        "required": True,
                    },
                ]
            },
        }

        ref_path = tmp_path / "pennylane_reference.json"
        ref_path.write_text(json.dumps(reference))
        return str(ref_path)

    def test_pennylane_doc_validator_valid(self, tmp_path: Path) -> None:
        ref_path = self._write_reference(tmp_path)
        validator = PennyLaneDocValidator(reference_path=ref_path)

        source_code = (
            "def circuit():\n"
            "    qml.RX(0.1, wires=0)\n"
            "    qml.AmplitudeEmbedding(features=[0.1,0.2,0.3,0.4], wires=[0,1], normalize=True)\n"
        )

        result = validator.validate(source_code)
        assert result.is_valid
        assert result.errors == []

        source_code_qchem = (
            "def circuit():\n"
            "    qml.RX(0.1, wires=0)\n"
            "    qml.qchem.nuclear_attraction(i=0, j=1)\n"
        )

        result = validator.validate(source_code_qchem)
        assert result.is_valid
        assert result.errors == []

    def test_pennylane_doc_validator_skip(self, tmp_path: Path) -> None:
        ref_path = self._write_reference(tmp_path)
        validator = PennyLaneDocValidator(reference_path=ref_path)

        source_code = "def circuit():\n    qml.numpy.zeros(10)\n"

        result = validator.validate(source_code)
        assert result.is_valid
        assert result.errors == []

    def test_pennylane_doc_validator_method_not_found(
        self, tmp_path: Path
    ) -> None:
        ref_path = self._write_reference(tmp_path)
        validator = PennyLaneDocValidator(reference_path=ref_path)

        source_code = "def circuit():\n    qml.UnknownOp(1.23, wires=0)\n"

        result = validator.validate(source_code)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Method 'qml.UnknownOp'" in result.errors[0]
        assert "Not found in PennyLane Library" in result.errors[0]

    def test_pennylane_doc_validator_missing_required_arg(
        self, tmp_path: Path
    ) -> None:
        ref_path = self._write_reference(tmp_path)
        validator = PennyLaneDocValidator(reference_path=ref_path)

        # Missing required 'wires'
        source_code = "def circuit():\n    qml.RX(0.1)\n"

        result = validator.validate(source_code)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Method 'qml.RX'" in result.errors[0]
        assert "Missing required argument 'wires'" in result.errors[0]

    def test_pennylane_doc_validator_unexpected_arg(
        self, tmp_path: Path
    ) -> None:
        ref_path = self._write_reference(tmp_path)
        validator = PennyLaneDocValidator(reference_path=ref_path)

        # Unexpected kwarg 'bogus'
        source_code = "def circuit():\n    qml.RX(0.1, wires=0, bogus=1)\n"

        result = validator.validate(source_code)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Method 'qml.RX'" in result.errors[0]
        assert "Unexpected argument 'bogus'" in result.errors[0]


def test_is_same_code() -> None:
    original_code = (
        "def circuit():\n"
        "    qml.RX(0.1, wires=0)\n"
        "    qml.AmplitudeEmbedding(features=[0.1,0.2,0.3,0.4], wires=[0,1], normalize=True)\n"
    )
    same_code = (
        "def circuit():\n"
        "    qml.RX(0.1, wires=0)\n"
        "    qml.AmplitudeEmbedding(features=[0.1,0.2,0.3,0.4], wires=[0,1], normalize=True)\n"
    )
    different_code = (
        "def circuit():\n"
        "    qml.RX(0.1, wires=0)\n"
        "    qml.RZ(0.1, wires=1)\n"
        "    qml.AmplitudeEmbedding(features=[0.1,0.2,0.3,0.4], wires=[0,1], normalize=True)\n"
    )
    assert is_same_code(original_code, same_code)
    assert not is_same_code(original_code, different_code)


def test_dry_run(mocker: Any, tmp_path: Path) -> None:
    from astronaut.logics.common.validation import dry_run

    mock_experiment = mocker.MagicMock()
    mock_experiment.name = "test_exp"
    config = {"param": 1}
    config_path = str(tmp_path / "config.yaml")

    error, warnings = dry_run(mock_experiment, config, config_path)
    assert error == ""
    assert mock_experiment.run.called

    mock_experiment.run.side_effect = Exception("Dry run failed")
    error, warnings = dry_run(mock_experiment, config, config_path)
    assert "Dry run failed" in error
