from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2pytorch import ConvertModel

from nnv_tools.function_catalog import FunctionSpec
from nnv_tools.metadata_paths import models_dir


ModelKind = Literal["shallow", "deep"]
_PYTORCH_DEVICE = torch.device("cpu")


def model_artifact_dir(
    database: str,
    model_kind: ModelKind,
    task_type: str,
    table: str,
    model_name: str,
) -> Path:
    return models_dir(database) / model_kind / task_type / table / model_name


def model_metadata_path(
    database: str,
    model_kind: ModelKind,
    task_type: str,
    table: str,
    model_name: str,
) -> Path:
    return (
        model_artifact_dir(database, model_kind, task_type, table, model_name)
        / f"{model_name}.metadata.json"
    )


def model_onnx_path(
    database: str,
    model_kind: ModelKind,
    task_type: str,
    table: str,
    model_name: str,
) -> Path:
    return (
        model_artifact_dir(database, model_kind, task_type, table, model_name)
        / f"{model_name}.onnx"
    )


def load_model_metadata(
    database: str,
    model_kind: ModelKind,
    task_type: str,
    table: str,
    model_name: str,
) -> dict:
    return json.loads(
        model_metadata_path(database, model_kind, task_type, table, model_name).read_text()
    )


@lru_cache(maxsize=32)
def _session_for_path(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=32)
def _pytorch_model_for_path(model_path: str) -> torch.nn.Module:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Using experimental implementation that allows 'batch_size > 1'.*",
            category=UserWarning,
        )
        model = ConvertModel(onnx.load(model_path), experimental=True)
    model.to(_PYTORCH_DEVICE)
    model.eval()
    return model


def predict_row(spec: FunctionSpec, model_path: str | Path, values: list[float]) -> float | int:
    session = _session_for_path(str(model_path))
    output = session.run(None, {"input": np.array([values], dtype=np.float32)})[0][0]
    if spec.task_type == "classifier":
        return int(np.argmax(output))
    return float(output[0] if np.ndim(output) > 0 else output)


def predict_array(spec: FunctionSpec, model_path: str | Path, values: np.ndarray) -> np.ndarray:
    """Predict many rows, falling back to row-wise ONNX calls for fixed-batch exports."""
    inputs = np.asarray(values, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={inputs.shape}.")

    session = _session_for_path(str(model_path))
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    try:
        output = session.run(None, {input_name: inputs})[0]
    except Exception:
        output = np.asarray(
            [
                session.run(None, {input_name: row.reshape(1, -1)})[0][0]
                for row in inputs
            ]
        )

    if spec.task_type == "classifier":
        return np.argmax(output, axis=1).astype(np.int64)
    return output.reshape(-1).astype(np.float64)


def predict_array_pytorch(
    spec: FunctionSpec,
    model_path: str | Path,
    values: np.ndarray,
) -> np.ndarray:
    inputs = np.asarray(values, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={inputs.shape}.")

    model = _pytorch_model_for_path(str(model_path))
    with torch.no_grad():
        output = model(torch.from_numpy(inputs).to(_PYTORCH_DEVICE)).detach().cpu().numpy()

    if spec.task_type == "classifier":
        return np.argmax(output, axis=1).astype(np.int64)
    return output.reshape(-1).astype(np.float64)
