from pathlib import Path

from click.testing import CliRunner
from pytest_mock import MockerFixture

from astronaut.db.pennylane.setup_local_file import main


def test_main(mocker: MockerFixture, tmp_path: Path) -> None:
    mock_parse = mocker.patch(
        "astronaut.db.pennylane.setup_local_file.parse_all_modules"
    )
    mock_format = mocker.patch(
        "astronaut.db.pennylane.setup_local_file.format_docs_by_llm"
    )

    raw_path = tmp_path / "raw.json"
    formatted_path = tmp_path / "formatted.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--save_raw_json_path",
            str(raw_path),
            "--save_formatted_json_path",
            str(formatted_path),
        ],
    )

    assert result.exit_code == 0
    mock_parse.assert_called_once()
    mock_format.assert_called_once()

    args_parse = mock_parse.call_args
    assert str(args_parse[0][1]) == str(raw_path)

    args_format = mock_format.call_args
    assert str(args_format[0][0]) == str(raw_path)
    assert str(args_format[0][1]) == str(formatted_path)
