"""Utility helpers for the forecasting project."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Create and return a configured logger for the project.

    Args:
        name: Logger name.
        log_dir: Optional directory where logs should be written. If omitted,
            logs are written to the project-level ``logs`` folder.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        project_root = Path(__file__).resolve().parents[1]
        log_path = Path(log_dir) if log_dir is not None else project_root / "logs"
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
