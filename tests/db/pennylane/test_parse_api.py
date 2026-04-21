import json
from pathlib import Path
from types import ModuleType

import pytest
from pytest_mock import MockerFixture

from astronaut.db.pennylane.parse_api import (
    parse_all_modules,
    parse_api,
    walk_modules,
)


class DummyClass:
    """Dummy docstring."""

    def method(self) -> None:
        pass


@pytest.fixture
def mock_package() -> ModuleType:
    module = ModuleType("my_pkg")  # Changed from test_package to avoid filter
    module.__path__ = ["/path/to/package"]  # type: ignore
    return module


def test_walk_modules(mocker: MockerFixture, mock_package: ModuleType) -> None:
    # return_value is a list of (importer, modname, ispkg)
    mocker.patch(
        "pkgutil.walk_packages",
        return_value=[
            (None, "my_pkg.mod1", False),
            (None, "my_pkg.test_mod", False),
            (None, "my_pkg.mod2", False),
        ],
    )

    modules = list(walk_modules(mock_package))

    # "my_pkg.test_mod" should be excluded because it contains "test_"
    assert "my_pkg.mod1" in modules
    assert "my_pkg.mod2" in modules
    assert "my_pkg.test_mod" not in modules
    assert len(modules) == 2


def test_parse_api(mocker: MockerFixture, mock_package: ModuleType) -> None:
    # Mock walk_modules to return one module
    mocker.patch(
        "astronaut.db.pennylane.parse_api.walk_modules",
        return_value=["my_pkg.mod1"],
    )

    # Mock import_module
    mock_submodule = ModuleType("my_pkg.mod1")
    mocker.patch("importlib.import_module", return_value=mock_submodule)

    # Attach DummyClass to submodule
    mock_submodule.DummyClass = DummyClass

    # Attach DummyClass to root package (required by parse_api logic: hasattr(module, attr_name))
    setattr(mock_package, "DummyClass", DummyClass)

    result = parse_api(mock_package, "my_pkg")

    assert "my_pkg.DummyClass" in result
    entry = result["my_pkg.DummyClass"]
    assert "Dummy docstring" in entry["docstring"]


def test_parse_all_modules(mocker: MockerFixture, tmp_path: Path) -> None:
    # Mock parse_api
    mocker.patch(
        "astronaut.db.pennylane.parse_api.parse_api",
        return_value={"mod.Class": {"docstring": "doc"}},
    )

    output_file = tmp_path / "output.json"
    mock_module = ModuleType("my_pkg")

    parse_all_modules([(mock_module, "my_pkg")], output_file)

    assert output_file.exists()
    with open(output_file, "r") as f:
        data = json.load(f)
        assert "mod.Class" in data
        assert data["mod.Class"]["docstring"] == "doc"
