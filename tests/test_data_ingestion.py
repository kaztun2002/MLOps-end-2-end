from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.mlops_end_2_end.data.data_ingestion.data_ingestion import CSVDataIngestion


def test_ingest_reads_csv_and_saves_to_raw(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    csv_path = source_dir / "sales.csv"
    expected = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "sales": [100, 120]})
    expected.to_csv(csv_path, index=False)

    ingestion = CSVDataIngestion(project_root=tmp_path)
    result = ingestion.ingest(csv_path)

    assert result.equals(expected)
    saved_path = tmp_path / "data" / "raw" / "sales.csv"
    assert saved_path.exists()
    assert pd.read_csv(saved_path).equals(expected)


def test_ingest_raises_for_missing_csv(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    ingestion = CSVDataIngestion(project_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        ingestion.ingest(missing_path)
