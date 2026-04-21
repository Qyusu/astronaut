import importlib
import importlib.util
import inspect
import json
import pkgutil
from pathlib import Path
from types import ModuleType

from loguru import logger


def walk_modules(package):
    prefix = package.__name__ + "."
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, prefix):
        # ignore test files and temporary files
        if any(
            excluded in mod_name
            for excluded in [
                ".tests.",
                ".test_",
                "test_",
                "_test",
                ".conftest",
                "conftest",
                ".benchmark",
                "benchmark",
            ]
        ):
            continue
        yield mod_name


def parse_api(
    module: ModuleType, module_name: str
) -> dict[str, dict[str, str]]:
    result = {}
    for mod_name in walk_modules(module):
        try:
            sub_module = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr_name, obj in inspect.getmembers(
            sub_module,
            predicate=lambda o: inspect.isclass(o) or inspect.isfunction(o),
        ):
            if not hasattr(module, attr_name):
                continue

            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = None

            try:
                source = inspect.getsource(obj)
            except (OSError, TypeError):
                source = None

            result[f"{module_name}.{attr_name}"] = {
                "signature": sig,
                "docstring": inspect.getdoc(obj) or "",
                "source": source,
            }

    logger.info(f'Parsed {len(result)} entries from "{module_name}"')

    return result


def parse_all_modules(
    modules: list[tuple[ModuleType, str]], output_json_path: Path
) -> None:
    data = {}
    for module, module_name in modules:
        module_data = parse_api(module, module_name)
        data.update(module_data)

    with open(output_json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

    logger.info(
        f'Total {len(data)} entries are written to "{output_json_path}"'
    )
