from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from nnv_tools.function_catalog import FunctionSpec
from nnv_tools.model_runtime import ModelKind


DEFAULT_HIDDEN_WIDTH = 32
DEFAULT_MODEL_KIND: ModelKind = "shallow"


class TabularModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int = DEFAULT_HIDDEN_WIDTH,
        model_kind: ModelKind = DEFAULT_MODEL_KIND,
    ) -> None:
        super().__init__()
        self.model_kind = model_kind

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
        ]
        if model_kind == "deep":
            layers.extend(
                [
                    nn.Linear(hidden_width, hidden_width),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(hidden_width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    metadata_path: Path
    metrics: dict
    feature_names: list[str]
    model_info: dict


def train_and_export_classifier(
    dataframe: pd.DataFrame,
    spec: FunctionSpec,
    artifact_dir: str | Path,
    *,
    epochs: int = 120,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    model_kind: ModelKind = DEFAULT_MODEL_KIND,
    seed: int = 123,
) -> TrainingArtifacts:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    print(f"[train] Writing artifacts to {artifact_path}")

    feature_names = [feature.name for feature in spec.features]
    target_name = spec.target_name
    print(
        f"[train] Training '{spec.name}' ({spec.task_type}) as a {model_kind} model "
        f"to predict '{target_name}' from {feature_names}"
    )

    train_frame, test_frame = _split_frame(dataframe, seed=seed)
    train_x = torch.tensor(train_frame[feature_names].to_numpy(), dtype=torch.float32)
    test_x = torch.tensor(test_frame[feature_names].to_numpy(), dtype=torch.float32)

    torch.manual_seed(seed)
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0).clamp_min(1e-6)
    normalized_train_x = (train_x - feature_mean) / feature_std
    normalized_test_x = (test_x - feature_mean) / feature_std

    if spec.task_type == "regressor":
        train_y = torch.tensor(
            train_frame[target_name].to_numpy().reshape(-1, 1),
            dtype=torch.float32,
        )
        test_y = torch.tensor(
            test_frame[target_name].to_numpy().reshape(-1, 1),
            dtype=torch.float32,
        )
        target_mean = train_y.mean(dim=0)
        target_std = train_y.std(dim=0).clamp_min(1e-6)
        normalized_train_y = (train_y - target_mean) / target_std
        output_dim = 1
        criterion: nn.Module = nn.MSELoss()
    elif spec.task_type == "classifier":
        train_y = torch.tensor(train_frame[target_name].to_numpy(), dtype=torch.long)
        test_y = torch.tensor(test_frame[target_name].to_numpy(), dtype=torch.long)
        target_mean = None
        target_std = None
        normalized_train_y = train_y
        output_dim = spec.num_classes or int(train_y.max().item()) + 1
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task_type: {spec.task_type}")

    model = TabularModel(
        len(feature_names),
        output_dim=output_dim,
        hidden_width=hidden_width,
        model_kind=model_kind,
    )
    model_info = _build_model_info(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(
        f"[train] Starting training for {epochs} epoch(s), "
        f"train_rows={len(train_frame)}, test_rows={len(test_frame)}"
    )

    log_every = max(1, epochs // 5)
    for epoch_index in range(epochs):
        permutation = torch.randperm(normalized_train_x.shape[0])
        for batch_start in range(0, normalized_train_x.shape[0], batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            batch_x = normalized_train_x[batch_indices]
            batch_y = normalized_train_y[batch_indices]

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch_index + 1) % log_every == 0 or epoch_index == 0:
            print(f"[train] Epoch {epoch_index + 1}/{epochs} loss={loss.item():.6f}")

    model.eval()
    if spec.task_type == "regressor":
        train_predictions = model(normalized_train_x) * target_std + target_mean
        test_predictions = model(normalized_test_x) * target_std + target_mean
        train_errors = train_predictions - train_y
        test_errors = test_predictions - test_y
        train_rmse = float(torch.sqrt((train_errors**2).mean()).item())
        test_rmse = float(torch.sqrt((test_errors**2).mean()).item())
        test_target_std = float(test_y.std().clamp_min(1e-6).item())
        test_target_range = float((test_y.max() - test_y.min()).clamp_min(1e-6).item())
        baseline_test_predictions = torch.full_like(test_y, fill_value=float(train_y.mean().item()))
        baseline_test_errors = baseline_test_predictions - test_y
        baseline_test_rmse = float(torch.sqrt((baseline_test_errors**2).mean()).item())
        metrics = {
            "task_type": "regressor",
            "train_mae": float(train_errors.abs().mean().item()),
            "test_mae": float(test_errors.abs().mean().item()),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_target_std": test_target_std,
            "test_target_range": test_target_range,
            "test_normalized_rmse": test_rmse / test_target_std,
            "test_range_normalized_rmse": test_rmse / test_target_range,
            "baseline_test_rmse": baseline_test_rmse,
            "baseline_rmse_ratio": test_rmse / max(baseline_test_rmse, 1e-6),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_rows": int(train_frame.shape[0]),
            "test_rows": int(test_frame.shape[0]),
        }
        export_model = _fold_input_normalization(model, feature_mean, feature_std)
        export_model = _fold_output_denormalization(export_model, target_mean, target_std)
    else:
        train_logits = model(normalized_train_x)
        test_logits = model(normalized_test_x)
        train_predictions = train_logits.argmax(dim=1)
        test_predictions = test_logits.argmax(dim=1)
        metrics = {
            "task_type": "classifier",
            "train_accuracy": float((train_predictions == train_y).float().mean().item()),
            "test_accuracy": float((test_predictions == test_y).float().mean().item()),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_rows": int(train_frame.shape[0]),
            "test_rows": int(test_frame.shape[0]),
            "num_classes": output_dim,
        }
        export_model = _fold_input_normalization(model, feature_mean, feature_std)

    model_path = artifact_path / f"{spec.name}.onnx"
    metadata_path = artifact_path / f"{spec.name}.metadata.json"
    print(f"[train] Exporting ONNX model to {model_path}")
    _export_onnx(export_model, len(feature_names), model_path)
    _write_metadata(metadata_path, spec, metrics, dataframe, feature_names, model_info)
    if spec.task_type == "regressor":
        print(
            f"[train] Finished '{spec.name}': "
            f"test_rmse={metrics['test_rmse']:.4f}, test_mae={metrics['test_mae']:.4f}"
        )
    else:
        print(
            f"[train] Finished '{spec.name}': "
            f"test_accuracy={metrics['test_accuracy']:.4f}"
        )
    print(f"[train] Wrote metadata to {metadata_path}")

    return TrainingArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
        feature_names=feature_names,
        model_info=model_info,
    )


def load_existing_training_artifacts(
    spec: FunctionSpec,
    artifact_dir: str | Path,
) -> TrainingArtifacts | None:
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / f"{spec.name}.onnx"
    metadata_path = artifact_path / f"{spec.name}.metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    print(f"[train] Reusing existing model at {model_path}")
    return TrainingArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metadata["metrics"],
        feature_names=[feature.name for feature in spec.features],
        model_info=metadata.get("model_info", {}),
    )


def _build_model_info(model: TabularModel) -> dict:
    hidden_layer_widths = [
        int(layer.out_features)
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ][:-1]
    return {
        "model_kind": model.model_kind,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "num_hidden_layers": len(hidden_layer_widths),
        "hidden_layer_widths": hidden_layer_widths,
    }


def _split_frame(dataframe: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cutoff = max(1, int(len(shuffled) * 0.8))
    return shuffled.iloc[:cutoff].copy(), shuffled.iloc[cutoff:].copy()


ONNX_OPSET_VERSION = 18


def _export_onnx(model: nn.Module, input_dim: int, output_path: Path) -> None:
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def _fold_input_normalization(
    model: TabularModel,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> TabularModel:
    export_model = copy.deepcopy(model)
    first_layer = export_model.network[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError("Expected the first layer to be linear.")

    with torch.no_grad():
        scaled_weight = first_layer.weight / feature_std.unsqueeze(0)
        shifted_bias = first_layer.bias - scaled_weight @ feature_mean
        first_layer.weight.copy_(scaled_weight)
        first_layer.bias.copy_(shifted_bias)

    return export_model


def _fold_output_denormalization(
    model: TabularModel,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> TabularModel:
    export_model = copy.deepcopy(model)
    last_layer = export_model.network[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Expected the last layer to be linear.")

    scale = float(target_std.squeeze().item())
    shift = float(target_mean.squeeze().item())
    with torch.no_grad():
        last_layer.weight.copy_(last_layer.weight * scale)
        last_layer.bias.copy_(last_layer.bias * scale + shift)

    return export_model


def _write_metadata(
    output_path: Path,
    spec: FunctionSpec,
    metrics: dict,
    dataframe: pd.DataFrame,
    feature_names: list[str],
    model_info: dict,
) -> None:
    payload = {
        "function_name": spec.name,
        "description": spec.description,
        "task_type": spec.task_type,
        "num_classes": spec.num_classes,
        "target_expression": spec.target_expression,
        "features": [
            {"name": feature.name, "expression": feature.expression}
            for feature in spec.features
        ],
        "metrics": metrics,
        "model_info": model_info,
        "observed_feature_ranges": {
            name: {
                "min": float(dataframe[name].min()),
                "max": float(dataframe[name].max()),
            }
            for name in feature_names
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
