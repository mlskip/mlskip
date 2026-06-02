from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from nnv_tools.metadata_paths import REPO_ROOT, setup_path


@dataclass(frozen=True)
class DatabaseSetup:
    database: str
    description: str
    source_format: str
    csv_source: str | None
    scale_factor: int
    training_row_count: int
    tables: list[str]
    duckdb_file: Path
    data_dir: Path

    def training_block_count(self, block_size: int) -> int:
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if self.training_row_count < 1:
            raise ValueError("training_row_count must be at least 1.")
        return math.ceil(self.training_row_count / block_size)


def load_database_setup(database: str) -> DatabaseSetup:
    path = setup_path(database)
    raw = json.loads(path.read_text())
    return DatabaseSetup(
        database=raw["database"],
        description=raw["description"],
        source_format=raw["source_format"],
        csv_source=raw.get("csv_source"),
        scale_factor=int(raw["scale_factor"]),
        training_row_count=int(raw["training_row_count"]),
        tables=list(raw["tables"]),
        duckdb_file=REPO_ROOT / raw["duckdb_file"],
        data_dir=REPO_ROOT / raw["data_dir"],
    )
