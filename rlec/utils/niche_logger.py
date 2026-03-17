from __future__ import annotations

import csv
import os
from typing import Dict, List


class NicheLogger:
    def __init__(self, output_dir: str | None):
        self.output_dir = output_dir
        self.stage_rows: List[Dict] = []
        self.niche_rows: List[Dict] = []

    def log_stage(self, row: dict) -> None:
        self.stage_rows.append(row)

    def log_niches(self, rows: list[dict]) -> None:
        if rows:
            self.niche_rows.extend(rows)

    def flush(self, stage_csv_path: str, niche_csv_path: str) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_csv(stage_csv_path, self.stage_rows)
        self._write_csv(niche_csv_path, self.niche_rows)

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

