from __future__ import annotations

import csv
import os
from typing import Dict, List


class NicheLogger:
    """Compatibility logger; now stores stage + subpopulation rows."""

    def __init__(self, output_dir: str | None):
        self.output_dir = output_dir
        self.stage_rows: List[Dict] = []
        self.subpop_rows: List[Dict] = []

    def log_stage(self, row: dict) -> None:
        self.stage_rows.append(row)

    def log_subpops(self, rows: list[dict]) -> None:
        if rows:
            self.subpop_rows.extend(rows)

    # Backward-compatible alias
    def log_niches(self, rows: list[dict]) -> None:
        self.log_subpops(rows)

    def flush(self, stage_csv_path: str, subpop_csv_path: str) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_csv(stage_csv_path, self.stage_rows)
        self._write_csv(subpop_csv_path, self.subpop_rows)

    @staticmethod
    def _write_csv(path: str, rows: List[Dict]) -> None:
        if len(rows) == 0:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
