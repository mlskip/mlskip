from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from nnv_tools.database_setup import DatabaseSetup, load_database_setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dataset CSVs and load them into a DuckDB cache."
    )
    parser.add_argument(
        "--database",
        default="tpch",
        help="Database identifier, for example 'tpch'.",
    )
    parser.add_argument(
        "--force-csv",
        action="store_true",
        help="Regenerate CSV files even if they already exist.",
    )
    parser.add_argument(
        "--force-duckdb",
        action="store_true",
        help="Rebuild the DuckDB cache even if it already exists.",
    )
    return parser


def preprocess_database(
    setup: DatabaseSetup,
    *,
    force_csv: bool = False,
    force_duckdb: bool = False,
) -> None:
    setup.data_dir.mkdir(parents=True, exist_ok=True)
    setup.duckdb_file.parent.mkdir(parents=True, exist_ok=True)

    if setup.csv_source == "duckdb_dbgen":
        _ensure_duckdb_generated_csvs(setup, generator="dbgen", force=force_csv)
    elif setup.csv_source == "duckdb_dsdgen":
        _ensure_duckdb_generated_csvs(setup, generator="dsdgen", force=force_csv)
    else:
        _ensure_csvs_exist(setup)

    _load_csvs_into_duckdb(setup, force=force_duckdb)


def _ensure_duckdb_generated_csvs(
    setup: DatabaseSetup,
    *,
    generator: str,
    force: bool,
) -> None:
    if not force and all(_csv_path(setup, table).exists() for table in setup.tables):
        print(f"[setup] Reusing existing CSV files in {setup.data_dir}")
        return

    print(
        f"[setup] Generating {setup.database} CSV files with DuckDB {generator} "
        f"in {setup.data_dir}"
    )
    with duckdb.connect(":memory:") as con:
        con.execute(f"CALL {generator}(sf={setup.scale_factor})")
        for table in setup.tables:
            csv_path = _csv_path(setup, table)
            con.execute(
                f"""
                COPY {table}
                TO '{csv_path.as_posix()}'
                (FORMAT CSV, HEADER, DELIMITER ',')
                """
            )
            print(f"[setup] Wrote {csv_path}")


def _ensure_csvs_exist(setup: DatabaseSetup) -> None:
    missing = [
        _csv_path(setup, table)
        for table in setup.tables
        if not _csv_path(setup, table).exists()
    ]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing CSV files for database '{setup.database}': {missing_str}"
        )
    print(f"[setup] Found CSV files in {setup.data_dir}")


def _load_csvs_into_duckdb(
    setup: DatabaseSetup,
    *,
    force: bool,
) -> None:
    if force and setup.duckdb_file.exists():
        setup.duckdb_file.unlink()

    print(f"[setup] Loading DuckDB cache at {setup.duckdb_file}")
    with duckdb.connect(str(setup.duckdb_file)) as con:
        for table in setup.tables:
            csv_path = _csv_path(setup, table)
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT
                    source_table.*,
                    CAST(ROW_NUMBER() OVER () - 1 AS BIGINT) AS row_id
                FROM read_csv_auto('{csv_path.as_posix()}', header=true)
                AS source_table
                """
            )
            print(f"[setup] Loaded table {table} with row_id from {csv_path.name}")
            for legacy_table in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE ?",
                [f"{table}_blocks_%"],
            ).fetchall():
                con.execute(f"DROP TABLE {legacy_table[0]}")
                print(f"[setup] Dropped legacy blocked table {legacy_table[0]}")


def _csv_path(setup: DatabaseSetup, table: str) -> Path:
    return setup.data_dir / f"{table}.csv"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup = load_database_setup(args.database)
    print(f"[setup] Using setup metadata for database '{args.database}'")
    preprocess_database(
        setup,
        force_csv=args.force_csv,
        force_duckdb=args.force_duckdb,
    )


if __name__ == "__main__":
    main()
