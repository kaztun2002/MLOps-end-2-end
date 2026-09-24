"""Data ingestion utilities for loading CSV source files into the raw data folder."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils import get_logger


class CSVDataIngestion:
    """Load CSV files from a source location and persist them under data/raw.

    The class validates the input path, reads the CSV into a pandas DataFrame, 
    creates the destination directory when needed, and stores the file in the project 
    raw-data folder.
    """

    def __init__(self, project_root: Union[str, Path] | None = None) -> None:
        """Initialize the ingestion component.

        Args:
            project_root: Root directory of the project. Defaults to the current
                working directory if not supplied.
        """
        self.logger = get_logger("data_ingestion")
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.raw_data_dir = self.project_root / "data" / "raw"

    def load_data(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Read a CSV file and save a copy in the project's raw-data directory.

        Args:
            csv_path: Path to the input CSV file.

        Returns:
            The pandas DataFrame loaded from that CSV.

        Raises:
            FileNotFoundError: If the source CSV does not exist.
            ValueError: If the path is not a CSV file.
        """
        source_path = Path(csv_path)
        self.logger.info("Starting CSV ingestion for %s", source_path)

        if not source_path.exists():
            self.logger.error("CSV source file not found: %s", source_path)
            raise FileNotFoundError(f"CSV file not found: {source_path}")

        if source_path.suffix.lower() != ".csv":
            self.logger.error("Expected a CSV file but received: %s", source_path)
            raise ValueError(f"Expected a CSV file but received: {source_path}")

        self.logger.info("Reading CSV into pandas DataFrame")
        df = pd.read_csv(source_path, sep=",", encoding="utf-8", header=0)

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        destination_path = self.raw_data_dir / source_path.name

        self.logger.info("Saving ingested data to %s", destination_path)
        df.to_csv(destination_path, index=False)

        self.logger.info("CSV ingestion completed successfully for %s", source_path)
        return df
