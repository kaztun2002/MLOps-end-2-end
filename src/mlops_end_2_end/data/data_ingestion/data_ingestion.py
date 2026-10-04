"""Data ingestion utilities for loading CSV source files into the raw data folder."""

from __future__ import annotations

from pathlib import Path
from sys import path
from typing import Union
from urllib.parse import urlparse

import pandas as pd

from mlops_end_2_end.utils import get_logger


class DataIngestion:
    """Load files from a source location and persist them under data/raw.

    The class validates the input path, reads the file into a pandas DataFrame, 
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
        self.raw_data_dir = Path(f"{self.project_root}/data/raw")

    def is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        parsed_url = urlparse(path)
        return parsed_url.scheme in ("http", "https")


    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Read a file and save a copy in the project's raw-data directory.

        Args:
            data_path: Path to the input data file.

        Returns:
            The pandas DataFrame loaded from that data file.

        Raises:
            FileNotFoundError: If the source CSV does not exist.
        """
        self.logger.info("Starting CSV ingestion for %s", data_path)

        if self.is_url(str(data_path)):
            self.logger.info("The provided path is a URL: %s", data_path)
            source_path = data_path
            filename = Path(data_path.path).name
            self.logger.info("Downloading CSV from URL: %s", data_path)
        else:
            self.logger.info("The provided path is a local file: %s", data_path)           
            source_path = Path(data_path)
            filename = source_path.name
        
        self.logger.info("Reading CSV into pandas DataFrame")
        df = pd.read_csv(source_path, sep=",", encoding="utf-8", header=0)

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        destination_path = self.raw_data_dir / filename

        self.logger.info("Saving ingested data to %s", destination_path)
        df.to_csv(destination_path, index=False)

        self.logger.info("CSV ingestion completed successfully for %s", source_path)
        return df
