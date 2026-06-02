from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from nnv_tools.database_setup import load_database_setup
from nnv_tools.dataset_duckdb import fetch_function_dataset
from nnv_tools.function_catalog import get_function_specs
from nnv_tools.metadata_paths import functions_dir, models_dir
from nnv_tools.model_runtime import ModelKind
from nnv_tools.modeling import (
    DEFAULT_MODEL_KIND,
    load_existing_training_artifacts,
    train_and_export_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or reuse models defined in metadata/functions/<database>/."
    )
    parser.add_argument("--database", default="tpch")
    parser.add_argument("--functions-path", type=Path)
    parser.add_argument("--function", action="append", dest="functions")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument(
        "--model-kind",
        choices=["shallow", "deep"],
        default=DEFAULT_MODEL_KIND,
        help="Model architecture family to train and reuse.",
    )
    parser.add_argument("--force-retrain", action="store_true")
    return parser


def run_pipeline(args: argparse.Namespace) -> list[dict]:
    database_setup = load_database_setup(args.database)
    db_path = args.db_path if args.db_path is not None else database_setup.duckdb_file
    training_rows = database_setup.training_row_count
    model_kind: ModelKind = args.model_kind
    root = args.models_root if args.models_root is not None else models_dir(args.database) / model_kind
    functions_path = args.functions_path
    specs = get_function_specs(args.database, args.functions, functions_path)
    summaries: list[dict] = []
    print(
        f"[train] Running {len(specs)} function(s) for database '{args.database}' "
        f"from {functions_path or functions_dir(args.database)}"
    )
    print(f"[train] Model kind: {model_kind}")
    print(f"[train] DuckDB cache: {db_path}")
    print(f"[train] Training on the first {training_rows} row(s)")

    for spec in specs:
        print(f"[train] Starting function '{spec.name}'")
        artifact_dir = root / spec.task_type / spec.table / spec.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / f"{spec.name}.summary.json"
        training = None
        if not args.force_retrain:
            if _can_reuse_training_summary(summary_path, training_rows, args.epochs, model_kind):
                training = load_existing_training_artifacts(spec, artifact_dir)
            elif summary_path.exists():
                print(
                    f"[train] Existing summary at {summary_path} was created with a different "
                    f"training-row, epoch, or model-kind configuration; retraining '{spec.name}'."
                )

        if training is None:
            dataframe = fetch_function_dataset(
                spec,
                sample_size=min(args.sample_size, training_rows),
                db_path=db_path,
            )
            training = train_and_export_classifier(
                dataframe,
                spec,
                artifact_dir,
                epochs=args.epochs,
                model_kind=model_kind,
            )

        summary = {
            "database": args.database,
            "model_kind": model_kind,
            "function_name": spec.name,
            "description": spec.description,
            "table": spec.table,
            "task_type": spec.task_type,
            "db_path": str(db_path),
            "training_rows": training_rows,
            "model_path": str(training.model_path),
            "metadata_path": str(training.metadata_path),
            "model_info": training.model_info,
            "metrics": training.metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
        summaries.append(summary)
        print(f"[train] Finished function '{spec.name}', summary at {summary_path}")

    results_path = root / "training_results.json"
    payload = {
        "database": args.database,
        "model_kind": model_kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "training_rows": training_rows,
        "results": summaries,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"[train] Wrote aggregate training results to {results_path}")

    return summaries


def _can_reuse_training_summary(
    summary_path: Path,
    training_rows: int,
    epochs: int,
    model_kind: ModelKind,
) -> bool:
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    metrics = summary.get("metrics", {})
    model_info = summary.get("model_info", {})
    summary_model_kind = summary.get("model_kind", model_info.get("model_kind"))
    return (
        summary.get("training_rows") == training_rows
        and metrics.get("epochs") == epochs
        and summary_model_kind == model_kind
        and isinstance(model_info.get("num_parameters"), int)
        and _contains_only_finite_numbers(metrics)
    )


def _contains_only_finite_numbers(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_contains_only_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_contains_only_finite_numbers(item) for item in value)
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summaries = run_pipeline(args)

    for summary in summaries:
        badge, note = _score_badge(summary)
        print(f"Function: {summary['function_name']}")
        print(f"  Score: {badge} {note}")
        print(f"  Model kind: {summary['model_kind']}")
        print(f"  Model: {summary['model_path']}")
        if summary.get("model_info"):
            print(f"  Parameters: {summary['model_info']['num_parameters']}")
        if summary["task_type"] == "regressor":
            print(f"  Test RMSE: {summary['metrics']['test_rmse']:.4f}")
            print(
                f"  Normalized RMSE: {summary['metrics']['test_normalized_rmse']:.4f} "
                f"(vs test std)"
            )
        else:
            print(f"  Test accuracy: {summary['metrics']['test_accuracy']:.4f}")
        print(f"  Summary: {summary['summary_path']}")

    _print_training_summary(summaries)


def _score_badge(summary: dict) -> tuple[str, str]:
    metrics = summary["metrics"]
    if summary["task_type"] == "classifier":
        accuracy = float(metrics["test_accuracy"])
        if accuracy >= 0.95:
            return "🌟", "excellent classifier fit"
        if accuracy >= 0.80:
            return "✅", "strong classifier fit"
        if accuracy >= 0.60:
            return "🙂", "usable classifier fit"
        if accuracy >= 0.40:
            return "😐", "weak classifier fit"
        return "⚠️", "poor classifier fit"

    normalized_rmse = float(metrics.get("test_normalized_rmse", metrics["test_rmse"]))
    if normalized_rmse <= 0.10:
        return "🌟", "excellent regressor fit"
    if normalized_rmse <= 0.25:
        return "✅", "strong regressor fit"
    if normalized_rmse <= 0.50:
        return "🙂", "usable regressor fit"
    if normalized_rmse <= 1.00:
        return "😐", "weak regressor fit"
    return "⚠️", "poor regressor fit"


def _print_training_summary(summaries: list[dict]) -> None:
    classifier_summaries = [s for s in summaries if s["task_type"] == "classifier"]
    regressor_summaries = [s for s in summaries if s["task_type"] == "regressor"]

    if classifier_summaries:
        print("Classifier summary:")
        for summary in sorted(
            classifier_summaries,
            key=lambda item: float(item["metrics"]["test_accuracy"]),
            reverse=True,
        ):
            badge, note = _score_badge(summary)
            print(
                f"  {badge} [{summary['table']}] {summary['function_name']}: "
                f"accuracy={summary['metrics']['test_accuracy']:.4f} ({note})"
            )

    if regressor_summaries:
        print("Regressor summary:")
        for summary in sorted(
            regressor_summaries,
            key=lambda item: float(item["metrics"]["test_rmse"]),
        ):
            badge, note = _score_badge(summary)
            print(
                f"  {badge} [{summary['table']}] {summary['function_name']}: "
                f"rmse={summary['metrics']['test_rmse']:.4f}, "
                f"nrmse={summary['metrics']['test_normalized_rmse']:.4f} ({note})"
            )


if __name__ == "__main__":
    main()
