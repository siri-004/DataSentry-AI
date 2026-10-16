import os
import io
from typing import Union, Tuple, Dict, Any, Optional
import pandas as pd
from config import Config

class DataLoaderError(Exception):
    """Custom exception raised when file loading or validation fails."""
    pass

def load_dataset(
    file_source: Union[str, bytes, io.BytesIO],
    filename: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads and validates a dataset from a file path, raw bytes, or BytesIO object.

    Args:
        file_source: File path (str), raw bytes, or BytesIO.
        filename: Optional name of the file (required if file_source is bytes or BytesIO).
        config_overrides: Optional dictionary of configuration overrides.

    Returns:
        Tuple of (pd.DataFrame, dataset_metadata_dict).

    Raises:
        DataLoaderError: If file validation or parsing fails.
    """
    cfg = config_overrides or {}
    max_size_mb = cfg.get("MAX_FILE_SIZE_MB", Config.MAX_FILE_SIZE_MB)
    max_rows = cfg.get("MAX_ROWS_LIMIT", Config.MAX_ROWS_LIMIT)

    # Determine filename
    actual_filename = ""
    if isinstance(file_source, str):
        actual_filename = os.path.basename(file_source)
    elif filename:
        actual_filename = filename
    else:
        actual_filename = "uploaded_dataset.csv"

    # Validate file extension
    ext = os.path.splitext(actual_filename)[1].lower()
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    if ext not in allowed_extensions:
        raise DataLoaderError(
            f"Unsupported file format '{ext}'. Supported formats are: {', '.join(sorted(allowed_extensions))}"
        )

    # Check file size if bytes or local file
    size_bytes = 0
    buffer: Union[io.BytesIO, str]
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise DataLoaderError(f"File not found: {file_source}")
        size_bytes = os.path.getsize(file_source)
        buffer = file_source
    elif isinstance(file_source, bytes):
        size_bytes = len(file_source)
        buffer = io.BytesIO(file_source)
    elif hasattr(file_source, "read"):
        file_source.seek(0, io.SEEK_END)
        size_bytes = file_source.tell()
        file_source.seek(0)
        buffer = file_source
    else:
        raise DataLoaderError("Invalid file source provided.")

    if size_bytes == 0:
        raise DataLoaderError("The provided file is empty (0 bytes).")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DataLoaderError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    # Load dataframe safely
    try:
        if ext == ".csv":
            # Attempt UTF-8 first, fallback to latin-1 / cp1252
            if isinstance(buffer, (io.BytesIO, io.StringIO)):
                raw_data = buffer.getvalue() if isinstance(buffer, io.BytesIO) else buffer.read().encode()
                try:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding="latin-1")
            else:
                try:
                    df = pd.read_csv(buffer, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(buffer, encoding="latin-1")
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(buffer, engine="openpyxl" if ext == ".xlsx" else None)
        else:
            raise DataLoaderError(f"Unsupported extension: {ext}")
    except Exception as e:
        if isinstance(e, DataLoaderError):
            raise e
        raise DataLoaderError(f"Failed to parse file '{actual_filename}': {str(e)}")

    if df.empty:
        raise DataLoaderError("The dataset contains 0 rows or is completely empty.")

    # Check row limit
    if len(df) > max_rows:
        raise DataLoaderError(
            f"Dataset has {len(df):,} rows, exceeding the limit of {max_rows:,} rows."
        )

    # Clean column names by trimming leading/trailing spaces
    df.columns = [str(col).strip() for col in df.columns]

    metadata = {
        "filename": actual_filename,
        "file_size_kb": round(size_bytes / 1024, 2),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    return df, metadata
