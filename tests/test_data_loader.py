import os
import io
import pytest
import pandas as pd
from data_loader import load_dataset, DataLoaderError

def test_load_valid_csv():
    csv_data = "id,name,value\n1,alpha,10.5\n2,beta,20.0\n"
    df, meta = load_dataset(csv_data.encode("utf-8"), filename="test.csv")
    assert len(df) == 2
    assert meta["rows"] == 2
    assert meta["columns"] == 3
    assert "name" in meta["column_names"]

def test_load_valid_excel(tmp_path):
    df_orig = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    excel_path = tmp_path / "test.xlsx"
    df_orig.to_excel(excel_path, index=False)

    df_loaded, meta = load_dataset(str(excel_path))
    assert len(df_loaded) == 3
    assert meta["columns"] == 2
    assert meta["filename"] == "test.xlsx"

def test_unsupported_file_extension():
    with pytest.raises(DataLoaderError) as exc_info:
        load_dataset(b"some content", filename="malicious.exe")
    assert "Unsupported file format" in str(exc_info.value)

def test_empty_file():
    with pytest.raises(DataLoaderError) as exc_info:
        load_dataset(b"", filename="empty.csv")
    assert "empty" in str(exc_info.value).lower()

def test_file_size_limit():
    large_payload = b"a" * (2 * 1024 * 1024)  # 2MB
    with pytest.raises(DataLoaderError) as exc_info:
        load_dataset(large_payload, filename="big.csv", config_overrides={"MAX_FILE_SIZE_MB": 1})
    assert "exceeds maximum allowed limit" in str(exc_info.value)

def test_row_limit():
    # 5 rows but limit is 3
    csv_content = "id\n1\n2\n3\n4\n5\n"
    with pytest.raises(DataLoaderError) as exc_info:
        load_dataset(csv_content.encode("utf-8"), filename="rows.csv", config_overrides={"MAX_ROWS_LIMIT": 3})
    assert "exceeding the limit" in str(exc_info.value)
