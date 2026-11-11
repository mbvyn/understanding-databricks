"""Shared helpers: command-line arguments, Unity Catalog paths, volumes, and logging."""

import argparse
import logging
import sys

from databricks.sdk.runtime import dbutils, spark

PIPELINE_VOLUMES = ("landing", "checkpoints")


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser


def volume_path(catalog: str, schema: str, volume: str) -> str:
    return f"/Volumes/{catalog}/{schema}/{volume}"


def ensure_volumes(catalog: str, schema: str) -> None:
    """Create the schema and the volumes the pipeline needs. Safe to run repeatedly."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    for volume in PIPELINE_VOLUMES:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")


def list_batches(path: str) -> list[str]:
    """Sorted batch names (batch=0001, batch=0002, ...) found under `path`."""
    try:
        entries = dbutils.fs.ls(path)
    except Exception:  # noqa: BLE001 - dbutils raises a generic error when the path doesn't exist yet
        return []
    names = (entry.name.rstrip("/") for entry in entries)
    return sorted(name for name in names if name.startswith("batch="))


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to stdout, which is where serverless job output shows up."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger
