import pandas as pd
import pytest

from glance.loading import UnsupportedFormatError, load


def test_load_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age\nAda,30\nGrace,85\n")

    df = load(csv_path)

    assert list(df.columns) == ["name", "age"]
    assert len(df) == 2
    assert df.loc[0, "name"] == "Ada"
    assert df.loc[1, "age"] == 85


def test_load_tsv(tmp_path):
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("name\tage\nAda\t30\n")

    df = load(tsv_path)

    assert list(df.columns) == ["name", "age"]
    assert len(df) == 1


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "does-not-exist.csv")


def test_load_directory_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")

    with pytest.raises(UnsupportedFormatError, match="don't know how to read"):
        load(path)


def test_load_excel(tmp_path):
    xlsx_path = tmp_path / "data.xlsx"
    pd.DataFrame({"name": ["Ada"], "age": [30]}).to_excel(xlsx_path, index=False)

    df = load(xlsx_path)

    assert list(df.columns) == ["name", "age"]
    assert df.loc[0, "name"] == "Ada"
