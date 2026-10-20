from __future__ import annotations
import csv
from pathlib import Path


def read_epoch_csv(path):
    """Read rows containing SourceID, TargetLabel, Telescope, Band, MJD."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["MJD"] = float(row["MJD"])
            rows.append(row)
    return rows


def group_epochs_by_source(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["SourceID"], []).append(row)
    return grouped
