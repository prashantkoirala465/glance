import pandas as pd
import pytest

from glance.cli import main


@pytest.fixture
def messy_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Amount": [1.0, 2.0, None, 4.0],
            "Category": ["a", "b", "b", None],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_main_generates_report_with_default_output_dir(messy_csv, capsys):
    exit_code = main([str(messy_csv)])

    out = capsys.readouterr().out
    expected_dir = messy_csv.parent / "data_glance"

    assert exit_code == 0
    assert (expected_dir / "report.md").exists()
    assert "Rows: 4, Columns: 2" in out
    assert str(expected_dir / "report.md") in out


def test_main_honors_output_flag(messy_csv, tmp_path):
    output_dir = tmp_path / "custom_out"

    exit_code = main([str(messy_csv), "-o", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "report.md").exists()


def test_main_returns_error_for_missing_file(tmp_path, capsys):
    exit_code = main([str(tmp_path / "does_not_exist.csv")])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "no such file" in err


def test_main_returns_error_for_unsupported_extension(tmp_path, capsys):
    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}")

    exit_code = main([str(bad_file)])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "don't know how to read" in err


def test_main_narrate_generates_summary_via_local_ollama(messy_csv, tmp_path, mock_ollama_url):
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(messy_csv),
            "-o",
            str(output_dir),
            "--narrate",
            "llama3.2",
            "--ollama-host",
            mock_ollama_url,
        ]
    )

    report_text = (output_dir / "report.md").read_text()

    assert exit_code == 0
    assert "## Summary" in report_text


def test_main_narrate_falls_back_when_ollama_unreachable(
    messy_csv, tmp_path, capsys, unreachable_ollama_url
):
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(messy_csv),
            "-o",
            str(output_dir),
            "--narrate",
            "llama3.2",
            "--ollama-host",
            unreachable_ollama_url,
        ]
    )

    err = capsys.readouterr().err
    report_text = (output_dir / "report.md").read_text()

    assert exit_code == 0
    assert "couldn't reach Ollama" in err
    assert "## Summary" not in report_text


def test_main_rejects_unknown_missing_strategy(messy_csv, capsys):
    with pytest.raises(SystemExit):
        main([str(messy_csv), "--missing-strategy", "bogus"])
