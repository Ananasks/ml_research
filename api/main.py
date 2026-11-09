"""Local HTTP API for the React UI; ML work stays in the shared ml_lab engine."""
from __future__ import annotations

import json
import uuid
import base64
import io
import re
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import Future
from threading import RLock
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, Response

from ml_lab.core import DEFAULTS, run_experiments, generate_candidates, eligible_rows, split_data
from ml_lab.data import prepare_frame, read_dataset
from ml_lab.anomaly_detection import MODEL_DETECTORS
from ml_lab.result_series import result_series, winning_validation
from ml_lab.datasets import combine_datasets, dataset_record
from ml_lab.experiments import ExperimentStore, html_report, clean
from ml_lab.model_registry import MODELS, MODEL_REGISTRY, OPTIONAL_STATUS, available_models, catalog_rows, documentation, online_catalog_status
from ml_lab.time_index import detect_time_columns, parse_datetime_values, normalize_time_index

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / ".local" / "uploads"
WORKFLOW_DIR = ROOT / ".local" / "workflows"
STORE = ExperimentStore(ROOT / ".local" / "experiments.sqlite3")
from ml_lab.run_store import RunStore
RUNS = RunStore(ROOT / ".local" / "runs")
RUN_CONFIGS: dict[str, dict[str, Any]] = {}
app = FastAPI(title="ML Research API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str
    target: str
    features: list[str] = Field(min_length=1)
    task: str = "Регрессия"
    model: str = "Ridge"
    params: dict[str, Any] = Field(default_factory=dict)
    preparation: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "manual"
    count: int = Field(default=5, ge=1, le=100)
    search_space: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


def get_dataset(name: str) -> pd.DataFrame:
    if Path(name).name != name or not name.lower().endswith((".csv", ".tsv", ".xlsx")):
        raise HTTPException(404, "Dataset не найден.")
    path = DATA_DIR / name
    if not path.is_file():
        path = UPLOAD_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Dataset не найден.")
    try:
        settings_path = path.with_suffix(path.suffix + ".json")
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
        return read_dataset(path.read_bytes(), name, **settings.get("settings", {}))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


def table_records(frame: pd.DataFrame, limit: int = 100) -> list[dict[str, Any]]:
    payload = frame.head(limit).to_json(orient="records", date_format="iso")
    return json.loads(payload)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ML Research", "backend": "python"}


@app.get("/api/datasets")
def datasets():
    result = []
    for path in sorted([*DATA_DIR.glob("*"), *UPLOAD_DIR.glob("*")]):
        if path.suffix.lower() not in {".csv", ".tsv", ".xlsx"}:
            continue
        try:
            settings_path = path.with_suffix(path.suffix + ".json")
            metadata = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
            frame = read_dataset(path.read_bytes(), path.name, **metadata.get("settings", {}))
            result.append({
                "id": path.name,
                "name": path.stem.replace("_", " ").title(),
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "columnNames": frame.columns.tolist(),
                "timeColumns": detect_time_columns(frame),
                "numericColumns": [c for c in frame.select_dtypes(include="number").columns if str(c).strip().lower() not in ("index","level_0","unnamed: 0")],
                "readSettings": metadata.get("settings", {}),
            })
        except Exception:
            continue
    return result


class UploadRequest(BaseModel):
    name: str
    content: str
    separator: str | None = None
    encoding: str = "utf-8-sig"
    decimal: str = "."
    sheet: str | int = 0


class InspectWorkbookRequest(BaseModel):
    name: str
    content: str


class CombineDatasetsRequest(BaseModel):
    datasets: list[str] = Field(min_length=2)
    primary: str
    method: str = "datetime"
    dayfirst: bool | None = None
    drop_invalid_time: bool = True


@app.post("/api/datasets/upload")
def upload_dataset(request: UploadRequest):
    filename = Path(request.name).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".tsv", ".xlsx"} or not filename:
        raise HTTPException(400, "Поддерживаются файлы CSV, TSV и XLSX.")
    safe_stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem, flags=re.UNICODE).strip("._") or "dataset"
    try:
        raw = base64.b64decode(request.content, validate=True)
        settings = {"separator": request.separator, "encoding": request.encoding,
                    "decimal": request.decimal, "sheet": request.sheet}
        settings = {key: value for key, value in settings.items() if key != "separator" or value is not None}
        frame = read_dataset(raw, filename, **settings)
    except Exception as exc:
        raise HTTPException(400, f"Не удалось прочитать файл: {exc}") from exc
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    digest = uuid.uuid5(uuid.NAMESPACE_OID, raw.hex()).hex[:10]
    stored_name = f"{safe_stem}-{digest}{suffix}"
    (UPLOAD_DIR / stored_name).write_bytes(raw)
    (UPLOAD_DIR / f"{stored_name}.json").write_text(json.dumps({"settings": settings}, ensure_ascii=False), encoding="utf-8")
    return {
        "id": stored_name,
        "name": Path(filename).stem,
        "file": stored_name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "columnNames": frame.columns.tolist(),
        "timeColumns": detect_time_columns(frame),
        "numericColumns": [c for c in frame.select_dtypes(include="number").columns if str(c).strip().lower() not in ("index","level_0","unnamed: 0")],
        "readSettings": settings,
    }


@app.post("/api/datasets/inspect-workbook")
def inspect_workbook(request: InspectWorkbookRequest):
    filename = Path(request.name).name
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Для выбора листа нужен файл XLSX.")
    try:
        raw = base64.b64decode(request.content, validate=True)
        return {"sheets": pd.ExcelFile(io.BytesIO(raw)).sheet_names}
    except Exception as exc:
        raise HTTPException(400, f"Не удалось прочитать книгу Excel: {exc}") from exc


@app.post("/api/datasets/combine")
def combine_dataset_request(request: CombineDatasetsRequest):
    if request.method not in {"datetime", "rows"}:
        raise HTTPException(400, "Выберите объединение по времени или по строкам.")
    if request.primary not in request.datasets:
        raise HTTPException(400, "Основной набор должен входить в список объединения.")
    records = []
    for name in request.datasets:
        path = DATA_DIR / Path(name).name
        if not path.is_file():
            path = UPLOAD_DIR / Path(name).name
        if not path.is_file() or path.name != name:
            raise HTTPException(404, f"Набор «{name}» не найден.")
        try:
            settings_path = path.with_suffix(path.suffix + ".json")
            metadata = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
            records.append(dataset_record(path.read_bytes(), path.name, metadata.get("settings", {})))
        except Exception as exc:
            raise HTTPException(400, f"Не удалось открыть «{name}»: {exc}") from exc
    records.sort(key=lambda record: record["name"] != request.primary)
    try:
        frame, diagnostics = combine_datasets(records, request.method, request.dayfirst, request.drop_invalid_time)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        digest = uuid.uuid4().hex[:10]
        stored_name = f"combined-{digest}.csv"
        frame.to_csv(UPLOAD_DIR / stored_name, index=False)
        (UPLOAD_DIR / f"{stored_name}.json").write_text(json.dumps({"settings": {}}, ensure_ascii=False), encoding="utf-8")
        return {"dataset": {"id": stored_name, "name": f"Сборка · {len(records)} наборов", "file": stored_name,
                "rows": len(frame), "columns": len(frame.columns), "columnNames": frame.columns.tolist(),
                "timeColumns": detect_time_columns(frame), "numericColumns": [c for c in frame.select_dtypes(include="number").columns if str(c).strip().lower() not in ("index","level_0","unnamed: 0")]}, "diagnostics": diagnostics}
    except Exception as exc:
        raise HTTPException(422, f"Не удалось объединить наборы: {exc}") from exc


class WorkflowRunRequest(BaseModel):
    nodes: list[dict[str, Any]] = Field(min_length=2)
    edges: list[dict[str, Any]] = Field(min_length=1)


class WorkflowSaveRequest(BaseModel):
    id: str | None = None
    name: str = "Новое исследование"
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def workflow_record(payload: dict[str, Any]) -> dict[str, Any]:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    identifier = payload.get("id") or uuid.uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise HTTPException(400, "Некорректный ID исследования.")
    record = {
        "id": identifier,
        "name": str(payload.get("name") or "Новое исследование")[:100],
        "updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "nodes": payload.get("nodes", []),
        "edges": payload.get("edges", []),
    }
    path = WORKFLOW_DIR / f"{identifier}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return record


@app.get("/api/workflows")
def list_workflows():
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(WORKFLOW_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append({key: record[key] for key in ("id", "name", "updated")})
        except (OSError, ValueError, KeyError):
            continue
    return records


@app.get("/api/workflows/{workflow_id}")
def get_workflow(workflow_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", workflow_id):
        raise HTTPException(404, "Схема исследования не найдена.")
    path = WORKFLOW_DIR / f"{workflow_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Схема исследования не найдена.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/workflows")
def save_workflow(request: WorkflowSaveRequest):
    return workflow_record(request.model_dump())


def _preview(frame: pd.DataFrame, limit: int = 100) -> list[dict[str, Any]]:
    return table_records(frame, limit)


def _profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for column in frame.columns:
        series = frame[column]
        columns.append({
            "column": str(column), "type": str(series.dtype), "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        })
    numeric = frame.select_dtypes(include="number")
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)), "memoryBytes": int(frame.memory_usage(deep=True).sum()),
        "missingTotal": int(frame.isna().sum().sum()), "columnProfile": columns,
        "statistics": table_records(numeric.describe().reset_index().rename(columns={"index": "statistic"})) if not numeric.empty else [],
        "correlation": table_records(numeric.corr().reset_index().rename(columns={"index": "feature"})) if len(numeric.columns) > 1 else [],
    }


@app.post("/api/workflows/run")
def run_workflow(request: WorkflowRunRequest):
    nodes = request.nodes
    edges = request.edges
    by_id = {str(node.get("id")): node for node in nodes}
    if len(by_id) != len(nodes) or "" in by_id:
        raise HTTPException(422, "Каждый узел должен иметь уникальный ID.")
    allowed = {"source", "profile", "preprocess", "selection", "split", "model", "evaluation", "explain", "export"}
    for node in nodes:
        kind = node.get("data", {}).get("kind")
        if kind not in allowed:
            raise HTTPException(422, f"Инструмент «{kind}» пока не реализован.")
    incoming: dict[str, list[str]] = {identifier: [] for identifier in by_id}
    adjacency: dict[str, list[str]] = {identifier: [] for identifier in by_id}
    for edge in edges:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in by_id or target not in by_id:
            raise HTTPException(422, "Связь ссылается на отсутствующий узел.")
        incoming[target].append(source)
        adjacency[source].append(target)
    if any(len(sources) > 1 for sources in incoming.values()):
        raise HTTPException(422, "В первом сценарии каждый инструмент принимает данные только из одного входа.")
    source_nodes = [node for node in nodes if node.get("data", {}).get("kind") == "source"]
    if len(source_nodes) != 1:
        raise HTTPException(422, "В схеме должен быть ровно один источник данных.")
    roots = [identifier for identifier, sources in incoming.items() if not sources]
    if roots != [str(source_nodes[0]["id"])]:
        raise HTTPException(422, "Все инструменты должны начинаться от одного источника данных.")
    order: list[str] = []
    queue = roots[:]
    while queue:
        identifier = queue.pop(0)
        order.append(identifier)
        for target in adjacency[identifier]:
            incoming[target].remove(identifier)
            if not incoming[target]:
                queue.append(target)
    if len(order) != len(nodes):
        raise HTTPException(422, "В схеме есть цикл или отключённые узлы.")
    data: dict[str, Any] = {}
    node_results: dict[str, dict[str, Any]] = {}
    preprocess_options = {"scaler": "standard", "imputer_strategy": "median"}
    training_result = None
    training_id = None
    workflow_selection: dict[str, Any] = {}
    workflow_split: dict[str, Any] = {}
    for identifier in order:
        node = by_id[identifier]
        kind = node.get("data", {}).get("kind")
        config = node.get("data", {}).get("config") or {}
        started = time.perf_counter()
        predecessor = data.get(next((edge["source"] for edge in edges if edge["target"] == identifier), ""))
        if kind == "source":
            frame = get_dataset(str(config.get("dataset", "")))
            data[identifier] = frame
            node_results[identifier] = {"status": "complete", "rows": len(frame), "columns": len(frame.columns), "preview": _preview(frame), "summary": {"dataset": config.get("dataset"), "rows": len(frame), "columns": len(frame.columns)}}
        elif kind == "profile":
            if not isinstance(predecessor, pd.DataFrame):
                raise HTTPException(422, f"Узел {identifier}: профиль ожидает таблицу.")
            profile = _profile(predecessor)
            data[identifier] = predecessor
            node_results[identifier] = {"status": "complete", "rows": len(predecessor), "columns": len(predecessor.columns), "preview": _preview(predecessor), "profile": profile, "summary": {"rows": profile["rows"], "columns": profile["columns"], "missing": profile["missingTotal"]}}
        elif kind == "selection":
            if not isinstance(predecessor, pd.DataFrame):
                raise HTTPException(422, f"Узел {identifier}: выбор цели ожидает таблицу.")
            workflow_selection = {"target": str(config.get("target", "")), "features": list(config.get("features") or [])}
            data[identifier] = predecessor
            node_results[identifier] = {"status": "complete", "rows": len(predecessor), "columns": len(predecessor.columns), "preview": _preview(predecessor), "summary": {"target": workflow_selection["target"], "features": workflow_selection["features"]}}
        elif kind == "split":
            if not isinstance(predecessor, pd.DataFrame):
                raise HTTPException(422, f"Узел {identifier}: настройка разбиения ожидает таблицу.")
            workflow_split = dict(config.get("training") or {})
            data[identifier] = predecessor
            node_results[identifier] = {"status": "complete", "rows": len(predecessor), "columns": len(predecessor.columns), "preview": _preview(predecessor), "summary": {"split": workflow_split.get("split", "random"), "test_size": workflow_split.get("test_size", .2), "validation_size": workflow_split.get("validation_size", .2)}}
        elif kind == "preprocess":
            if not isinstance(predecessor, pd.DataFrame):
                raise HTTPException(422, f"Узел {identifier}: предобработка ожидает таблицу.")
            imputer = config.get("imputer", "median")
            scaler = config.get("scaler", "standard")
            if imputer not in {"median", "mean", "most_frequent"} or scaler not in {"standard", "robust", "minmax", "none"}:
                raise HTTPException(422, f"Узел {identifier}: неизвестный способ предобработки.")
            preprocess_options.update(imputer_strategy=imputer, scaler=scaler)
            prep_config = {"calendar": True, **(config.get("preparation") or {})}
            if workflow_selection.get("target"):
                prep_config.setdefault("target", workflow_selection["target"])
            prep_config.pop("generate_index", None)
            if prep_config.get("target"):
                prep_config["target"] = str(prep_config["target"])
            try:
                prepared_frame = predecessor.copy() if predecessor.attrs.get("workflow_prepared") else prepare_frame(predecessor, **prep_config)
            except Exception as exc:
                raise HTTPException(422, f"Узел {identifier}: ошибка подготовки данных: {exc}") from exc
            prepared_frame.attrs["workflow_prepared"] = True
            data[identifier] = prepared_frame
            node_results[identifier] = {"status": "complete", "rows": len(prepared_frame), "columns": len(prepared_frame.columns), "preview": _preview(prepared_frame), "profile": _profile(prepared_frame), "inputPreview": _preview(predecessor), "summary": {"imputer": imputer, "scaler": scaler, "fit": "Модельный preprocessing обучается только на train внутри ML pipeline.", "featuresAdded": len(prepared_frame.columns)-len(predecessor.columns), "invalidTimeRows": prep_config.get("drop_invalid_time", False)}}
        elif kind == "model":
            if not isinstance(predecessor, pd.DataFrame):
                raise HTTPException(422, f"Узел {identifier}: обучение ожидает таблицу.")
            task = str(config.get("task", "Регрессия"))
            target = workflow_selection.get("target") or str(config.get("target", ""))
            features = workflow_selection.get("features") or list(config.get("features") or [])
            model_name = str(config.get("model", "Random Forest"))
            model_params = config.get("params") or {}
            if not target or not features:
                raise HTTPException(422, f"Узел {identifier}: выберите целевую колонку и хотя бы один признак.")
            prepared = predecessor if predecessor.attrs.get("workflow_prepared") else prepare_frame(predecessor, target=target, calendar=True)
            training_options = workflow_split | dict(config.get("training") or {})
            if training_options.get("forecast_end") and len(str(training_options["forecast_end"])) == 10:
                training_options["forecast_end"] = (pd.Timestamp(training_options["forecast_end"]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).isoformat()
            params = DEFAULTS | preprocess_options | training_options | model_params
            if predecessor.attrs.get("time") is None and not detect_time_columns(predecessor) and not training_options.get("start_datetime"):
                params["split"] = "random"
            strategy = config.get("strategy", "manual")
            count = max(1, min(100, int(config.get("count", 5))))
            model_schema = MODEL_REGISTRY[model_name]["parameters"]
            fixed_params = {key:value for key,value in model_params.items() if key in model_schema}
            search_space = config.get("searchSpace") or {key:value for key,value in model_schema.items() if value.get("type") != "layers"}
            if strategy == "manual":
                candidates_to_run = [{"name": model_name, "params": fixed_params}]
                search = None
            elif strategy == "range":
                candidates_to_run = generate_candidates(task, [model_name], count, params, {model_name: {**search_space, "__fixed__": fixed_params}})
                search = None
            elif strategy == "optuna":
                candidates_to_run = []
                search = {"count": count, "families": [model_name], "spaces": {model_name: {**search_space, "__fixed__": fixed_params}}}
            else:
                raise ValueError("Неизвестная стратегия обучения.")
            try:
                training_result = run_experiments(prepared, features, target, task, candidates_to_run, params, search=search)
            except Exception as exc:
                raise HTTPException(422, f"Узел {identifier}: обучение завершилось ошибкой: {exc}") from exc
            if "error" in training_result:
                failure = training_result.get("leaderboard", [{}])[0].get("Статус", training_result["error"])
                raise HTTPException(422, f"Узел {identifier}: {failure}")
            eligible_count = len(eligible_rows(prepared, features, target, task))
            used_count = sum(int(training_result["sizes"].get(key, 0)) for key in ("train", "validation", "test"))
            training_result["size_details"] = {"rawRows": len(predecessor), "preparedRows": len(prepared), "eligibleRows": eligible_count,
                "usedRows": used_count, "availableOutside": max(0, eligible_count-used_count), "notEligible": max(0, len(prepared)-eligible_count)}
            training_id = uuid.uuid4().hex
            training_result.update(config={"dataset": by_id[str(source_nodes[0]["id"])].get("data", {}).get("config", {}).get("dataset"), "model": model_name, "task": task, "features": features, "target": target, "params": params}, source="workflow")
            workflow_config = {"dataset": by_id[str(source_nodes[0]["id"])].get("data", {}).get("config", {}).get("dataset"), "model": model_name, "task": task, "params": params, "preparation": next((by_id[item].get("data", {}).get("config", {}).get("preparation", {}) for item in reversed(order[:order.index(identifier)]) if by_id[item].get("data", {}).get("kind") == "preprocess"), {}), "strategy": strategy, "count": count, "features": features, "target": target, "training": training_options, "searchSpace": config.get("searchSpace") or {}, "nodes": nodes, "edges": edges}
            RUN_CONFIGS[training_id] = workflow_config
            training_result["config"] = workflow_config
            RUNS[training_id] = training_result
            record = STORE.save(training_result, workflow_config)
            transformed = training_result["pipeline"].named_steps["preprocess"].transform(training_result["explain_train"])
            try:
                transformed_names = training_result["pipeline"].named_steps["preprocess"].get_feature_names_out().tolist()
            except (AttributeError, ValueError):
                transformed_names = [f"feature_{index + 1}" for index in range(transformed.shape[1])]
            transformed_frame = pd.DataFrame(transformed, columns=transformed_names)
            for upstream in reversed(order[:order.index(identifier)]):
                if by_id[upstream].get("data", {}).get("kind") == "preprocess":
                    before_frame = predecessor[features].head(40).reset_index(drop=True)
                    node_results[upstream]["inputPreview"] = _preview(before_frame)
                    node_results[upstream]["outputPreview"] = _preview(transformed_frame)
                    node_results[upstream]["transformedFeatures"] = transformed_names
                    break
            data[identifier] = training_result
            node_results[identifier] = {"status": "complete", "rows": len(predecessor), "columns": len(features), "preview": _preview(predecessor[features + [target]]), "metrics": training_result["metrics"], "validation": training_result["validation"], "sizes": training_result["sizes"], "leaderboard": training_result["leaderboard"], "summary": {"model": model_name, "task": task, "target": target, "features": features, "recordId": record["id"]}}
        elif kind == "explain":
            if not isinstance(predecessor, dict) or "pipeline" not in predecessor:
                raise HTTPException(422, f"Узел {identifier}: объяснение ожидает обученную модель.")
            method = str(config.get("method", "Permutation Importance"))
            try:
                from ml_lab.explain import permutation_scores, shap_scores, integrated_gradients
                if method == "Permutation Importance": payload = permutation_scores(predecessor)
                elif method == "SHAP": payload = shap_scores(predecessor)
                elif method == "Integrated Gradients": payload = integrated_gradients(predecessor)
                else: raise ValueError("Неизвестный метод объяснения.")
            except Exception as exc:
                raise HTTPException(422, f"Узел {identifier}: ошибка объяснения модели: {exc}") from exc
            predecessor.setdefault("explanations", {})[method] = payload
            data[identifier] = predecessor
            rows = payload.get("summary", payload) if isinstance(payload, dict) else payload
            node_results[identifier] = {"status": "complete", "rows": len(rows) if isinstance(rows, list) else 0, "columns": 0, "preview": _json_safe(rows[:100]) if isinstance(rows, list) else [], "summary": {"message": f"Рассчитано: {method}", "method": method, "runId": training_id}}
        elif kind == "export":
            if not isinstance(predecessor, dict) or "metrics" not in predecessor or not training_id:
                raise HTTPException(422, f"Узел {identifier}: экспорт ожидает обученную модель.")
            data[identifier] = predecessor
            node_results[identifier] = {"status": "complete", "rows": len(predecessor.get("predictions", [])), "columns": len(predecessor.get("predictions", pd.DataFrame()).columns), "summary": {"runId": training_id, "message": "Артефакты эксперимента готовы к скачиванию."}}
        elif kind == "evaluation":
            if not isinstance(predecessor, dict) or "metrics" not in predecessor:
                raise HTTPException(422, f"Узел {identifier}: оценка ожидает обученную модель.")
            data[identifier] = predecessor
            prediction_frame = predecessor["predictions"]
            node_results[identifier] = {"status": "complete", "rows": len(prediction_frame), "columns": len(prediction_frame.columns), "preview": _preview(prediction_frame), "metrics": predecessor["metrics"], "sizes": predecessor["sizes"], "summary": {"testRows": len(prediction_frame), "metrics": predecessor["metrics"]}}
        node_results[identifier]["seconds"] = round(time.perf_counter() - started, 3)
    if training_result is None or training_id is None:
        raise HTTPException(422, "Добавьте и подключите блок обучения модели и блок оценки.")
    predictions = training_result["predictions"]
    return {
        "runId": training_id,
        "nodeResults": node_results,
        "experiment": {
            "id": training_id, "model": training_result["name"], "task": training_result["task"],
            "target": training_result["target"], "features": training_result["features"],
            "metrics": training_result["metrics"], "trainMetrics": training_result["train_metrics"], "validation": training_result.get("validation", {}),
            "sizes": training_result["sizes"], "leaderboard": training_result["leaderboard"],
            "sizeDetails": training_result.get("size_details", {}),
            "predictions": _preview(predictions, 1000), "training": _preview(training_result["train_predictions"], 1000),
            "validationPredictions": _preview(training_result["comparison"], 1000),
            "config": workflow_config, "series": result_series(training_result),
        },
    }


@app.get("/api/datasets/{name}/preview")
def dataset_preview(name: str, limit: int = 100):
    frame = get_dataset(name)
    numeric = [c for c in frame.select_dtypes(include="number").columns if str(c).strip().lower() not in ("index","level_0","unnamed: 0")]
    series = []
    time_columns = detect_time_columns(frame)
    times = parse_datetime_values(frame[time_columns].astype(str).agg(' '.join, axis=1)) if time_columns else None
    time_values = [None if pd.isna(value) else pd.Timestamp(value).isoformat() for value in times] if times is not None else []
    if numeric:
        column = numeric[-1]
        points = pd.DataFrame({'x': times if times is not None else range(len(frame)), 'y': frame[column]})
        if times is not None:
            points = points.loc[points.x.notna()].sort_values('x', kind='stable')
        step = max(1, (len(points)+2999)//3000)
        series = table_records(points.iloc[::step], len(points))
    return {
        "columns": [{"field": str(c), "headerName": str(c)} for c in frame.columns],
        "rows": table_records(frame, max(1, min(limit, 5000))),
        "totalRows": len(frame), "times": time_values[:max(1,min(limit,5000))],
        "chart": {"name": str(numeric[-1]) if numeric else "", "points": series},
    }


@app.get("/api/datasets/{name}/profile")
def dataset_profile(name: str):
    frame = get_dataset(name)
    profile = _profile(frame)
    profile["duplicates"] = int(frame.duplicated().sum())
    profile["emptyColumns"] = [str(column) for column in frame.columns if frame[column].isna().all()]
    profile["numericColumns"] = [c for c in frame.select_dtypes(include="number").columns if str(c).strip().lower() not in ("index","level_0","unnamed: 0")]
    profile["categoricalColumns"] = [str(column) for column in frame.columns if column not in frame.select_dtypes(include="number").columns]
    return profile


@app.get("/api/datasets/{name}/download")
def download_dataset(name: str):
    if Path(name).name != name:
        raise HTTPException(404, "Dataset не найден.")
    path = DATA_DIR / name
    if not path.is_file():
        path = UPLOAD_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Dataset не найден.")
    return FileResponse(path, filename=path.name)


class PreparationRequest(BaseModel):
    target: str
    task: str = "Регрессия"
    preparation: dict[str, Any] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


_PREPARED_MODELS = OrderedDict()
_PREPARING_MODELS = {}
_PREPARED_MODELS_LOCK = RLock()


def prepare_dataset(raw, target, task, options):
    if target not in raw.columns:
        raise ValueError("Целевая колонка отсутствует в наборе данных.")
    if task not in MODELS:
        raise ValueError("Неизвестный тип задачи.")
    preparation = {"calendar": True, **options, "target": target}
    generated = preparation.pop("generate_index", None)
    if generated is False:
        preparation.pop("start_datetime", None)
        preparation.pop("index_frequency", None)
    # Saved drafts may still reference the row index removed during datetime normalization.
    if detect_time_columns(raw) or isinstance(raw.index,pd.DatetimeIndex) or preparation.get('start_datetime'):
        service=lambda name:str(name).strip().lower() in ('index','level_0','unnamed: 0')
        for stage in ('repair','anomaly_stage','restoration'):
            if preparation.get(stage):
                preparation[stage]={**preparation[stage],'columns':[c for c in preparation[stage].get('columns',[]) if not service(c)]}
        if preparation.get('feature_columns') is not None:
            preparation['feature_columns']=[c for c in preparation['feature_columns'] if not service(c)]
    if task == "Классификация":
        repair = preparation.get("repair")
        if repair and target in repair.get("columns", []):
            raise ValueError("Категории целевой классификации нельзя восстанавливать. Исключите Y из колонок восстановления.")
    if task == "Классификация":
        for stage in ("anomaly_stage", "restoration"):
            if target in (preparation.get(stage) or {}).get("columns", []):
                raise ValueError("Категории целевой классификации нельзя восстанавливать.")
    repair = preparation.get("repair") or {}
    if preparation.get("restoration") or preparation.get("anomaly_stage") or repair.get("fill_method") == "model" or repair.get("anomaly_method") in MODEL_DETECTORS:
        digest = hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()).hexdigest()
        key = (digest, tuple(raw.columns), tuple(map(str,raw.dtypes)), json.dumps(preparation,sort_keys=True,default=str))
        with _PREPARED_MODELS_LOCK:
            if key in _PREPARED_MODELS:
                _PREPARED_MODELS.move_to_end(key)
                return _PREPARED_MODELS[key].copy(deep=True), preparation
            future=_PREPARING_MODELS.get(key)
            owner=future is None
            if owner:
                future=Future()
                _PREPARING_MODELS[key]=future
        if not owner:
            return future.result().copy(deep=True),preparation
        try:
            prepared=prepare_frame(raw,**preparation)
            with _PREPARED_MODELS_LOCK:
                _PREPARED_MODELS[key]=prepared
                if len(_PREPARED_MODELS)>2: _PREPARED_MODELS.popitem(last=False)
                future.set_result(prepared)
            return prepared.copy(deep=True),preparation
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with _PREPARED_MODELS_LOCK: _PREPARING_MODELS.pop(key,None)
    return prepare_frame(raw, **preparation), preparation


def preparation_params(prepared, features, target, task, options):
    params = DEFAULTS | options
    if prepared.attrs.get("time") is None:
        params["split"] = "random"
    if params.get("forecast_start"):
        end = params.get("forecast_end") or params["forecast_start"]
        if len(str(end)) == 10:
            end = (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).isoformat()
        params["forecast_end"] = end
        if params.get("size_mode") == "rows":
            work = eligible_rows(prepared, features, target, task)
            times = prepared.attrs.get("time")
            if times is None:
                raise ValueError("Для периода теста нужна дата.")
            target_times = times.shift(-int(prepared.attrs.get("horizon", 0))).iloc[work.index]
            params["test_rows"] = int(((target_times >= pd.Timestamp(params["forecast_start"])) &
                                        (target_times <= pd.Timestamp(end))).sum())
    return params


@app.post("/api/datasets/{name}/prepare")
def preview_preparation(name: str, request: PreparationRequest):
    raw = get_dataset(name)
    try:
        frame, _ = prepare_dataset(raw, request.target, request.task, request.preparation)
        feature_names = [str(c) for c in frame.columns if c != request.target]
        time_keys = ("time_columns", "dayfirst", "strict_dates", "drop_invalid_time", "start_datetime", "index_frequency")
        source_options = {key: value for key, value in request.preparation.items() if key in time_keys}
        if request.preparation.get("generate_index") is False:
            source_options.pop("start_datetime", None)
            source_options.pop("index_frequency", None)
        source, source_time, _ = normalize_time_index(raw, target=request.target, **source_options)
        source_numeric = source.select_dtypes(include="number")
        source_numeric = source_numeric.copy()
        source_numeric.insert(0, "__source_x__", source_time if source_time is not None else range(len(source)))
        source_step = max(1, (len(source) + 2999) // 3000)
        source_sample = source_numeric.iloc[::source_step]

        display = frame.reset_index() if isinstance(frame.index, pd.DatetimeIndex) else frame
        # Charts cover the entire period; include all anomalous points as well as regular samples.
        step = max(1, (len(frame) + 2999) // 3000)
        positions = set(range(0, len(frame), step))
        if len(frame):
            positions.add(len(frame) - 1)
        masks = frame.attrs.get("repair_masks", {})
        for mask in masks.values():
            positions.update(i for i, value in enumerate(mask["anomaly"]) if value)
        positions = sorted(positions)
        series = {}
        for column, before in frame.attrs.get("repair_originals", {}).items():
            series[column] = table_records(pd.DataFrame({
                "x": frame.index, "before": before.to_numpy(),
                "after": frame.attrs["repair_values"][column].to_numpy(),
                "anomaly": masks[column]["anomaly"],
                "score": masks[column].get("anomaly_score", [None]*len(frame)),
            }).iloc[positions], len(positions))
        split = None
        split_error = None
        if request.features:
            try:
                params = preparation_params(frame, request.features, request.target, request.task, request.params)
                eligible = len(eligible_rows(frame, request.features, request.target, request.task))
                _, _, indices = split_data(frame, request.features, request.target, request.task, params)
                sizes = dict(zip(("train", "validation", "test"), map(len, indices)))
                used = sum(sizes.values())
                split = {**sizes, "eligible": eligible, "unused": len(frame)-used,
                         "outside": eligible-used, "notEligible": len(frame)-eligible}
            except (ValueError, KeyError) as exc:
                split_error = str(exc)
        times=frame.attrs.get("time")
        recommendations={"lags":[1,2,3],"rolling_windows":[6,12],"calendar_features":[],"notes":[]}
        if times is not None and len(times)>1:
            gaps=pd.Series(times).drop_duplicates().sort_values().diff().dropna()
            if len(gaps):
                step=gaps.mode().iloc[0]
                recommendations["step"]=str(step)
                daily=max(1,round(pd.Timedelta(days=1)/step))
                recommendations["notes"].append(f"Наиболее частый шаг: {step}. Суточный цикл — примерно {daily} строк.")
                recommendations["calendar_features"]=["calendar_dayofweek","calendar_year_sin","calendar_year_cos"]
                if step<pd.Timedelta(days=1):
                    recommendations["calendar_features"] += ["calendar_hour_sin","calendar_hour_cos"]
                if daily>=2 and daily<len(frame)//3:
                    recommendations["lags"].append(daily)
                    recommendations["rolling_windows"].append(daily)
        recommendations["lags"]=[n for n in recommendations["lags"] if n<max(2,len(frame)//3)]
        recommendations["rolling_windows"]=[n for n in recommendations["rolling_windows"] if n<max(3,len(frame)//3)]
        recommendations["notes"].append(f"Доступно {len(frame)} строк и {len(source_numeric.columns)-1} числовых колонок. Окна рекомендованы с запасом истории; полезность проверьте на валидации.")
        return {"recommendations":recommendations,"sourceRows": table_records(source_sample, len(source_sample)), "sourceSampled": source_step > 1, "features": feature_names, "columns": [{"field": str(c)} for c in display.columns],
                "rows": table_records(display, 1000), "totalRows": len(frame),
                "profile": _profile(frame), "repairRecords": frame.attrs.get("repair_records", []),
                "repairSeries": series, "sampled": len(positions) < len(frame),
                "diagnostics": {k: frame.attrs.get(k, 0) for k in
                    ("invalid_time_rows", "grid_added_rows", "duplicates_removed", "time_source", "time_step")},
                "split": split, "splitError": split_error}
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/datasets/{name}/prepared.csv")
def export_preparation(name: str, request: PreparationRequest):
    try:
        frame, _ = prepare_dataset(get_dataset(name), request.target, request.task, request.preparation)
        return Response(frame.to_csv(index=isinstance(frame.index, pd.DatetimeIndex)).encode("utf-8-sig"),
                        media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="prepared_data.csv"'})
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/models")
def models(task: str = "Регрессия"):
    if task not in MODELS:
        raise HTTPException(400, "Неизвестный тип задачи.")
    rows = catalog_rows(task)
    for row in rows:
        if row["Модель"] in OPTIONAL_STATUS:
            row["Статус"] = OPTIONAL_STATUS[row["Модель"]]
    return {
        "task": task,
        "models": available_models(task),
        "catalog": rows,
        "schemas": {
            name: MODEL_REGISTRY[name]["parameters"]
            for name in MODELS[task]
        },
    }


@app.get("/api/models/details")
def model_details(name: str, task: str = "Регрессия"):
    if name not in MODEL_REGISTRY or task not in MODELS or name not in MODELS[task]:
        raise HTTPException(404, "Описание модели недоступно для этой задачи.")
    from ml_lab.model_theory import theory, architecture_dot
    import inspect
    from importlib.metadata import PackageNotFoundError, version
    spec = MODEL_REGISTRY[name]
    estimator_class = spec.get("classes", {}).get(task)
    library_description = ""
    signature = ""
    installed_version = "не установлена"
    if estimator_class is not None:
        signature = str(inspect.signature(estimator_class))
        library_description = " ".join((inspect.getdoc(estimator_class) or "").split())[:5000]
    try:
        installed_version = version(spec.get("package", "scikit-learn"))
    except PackageNotFoundError:
        pass
    return {
        "name": name, "task": task, "group": spec.get("group"), "package": spec.get("package"),
        "description": spec.get("description", ""), "parameters": spec.get("parameters", {}),
        "documentation": documentation(name, task), "theory": theory(name, task),
        "architecture": architecture_dot(name, task, {}), "signature": signature,
        "className": getattr(estimator_class, "__name__", name),
        "libraryDescription": library_description, "installedVersion": installed_version,
    }


@app.get("/api/models/online")
def online_models():
    return online_catalog_status(include_extensions=True)


@app.get("/api/experiments/{run_id}/model-view")
def experiment_model_view(run_id: str):
    """Return a compact, inspectable view of the fitted pipeline."""
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полный результат запуска не сохранён. Повторите расчёт по сохранённой конфигурации, чтобы восстановить графики и объяснимость.")
    pipeline = result["pipeline"]
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps.get("preprocess")
    try:
        transformed = list(preprocessor.get_feature_names_out()) if preprocessor is not None else list(result["features"])
    except Exception:
        transformed = list(result["features"])
    values = None
    kind = None
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "Важность признака"
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_
        kind = "Коэффициент модели"
    effects = []
    if values is not None:
        import numpy as np
        array = np.asarray(values)
        if array.ndim > 1:
            array = np.mean(np.abs(array), axis=0)
            kind = "Средний модуль коэффициента"
        for name, value in zip(transformed, array.reshape(-1)):
            if np.isfinite(value):
                effects.append({"feature": str(name), "value": float(value)})
        effects.sort(key=lambda row: abs(row["value"]), reverse=True)
    params = {}
    for key, value in estimator.get_params(deep=False).items():
        try:
            params[key] = clean(value)
        except Exception:
            params[key] = str(value)
    return {
        "runId": run_id, "model": result["name"], "task": result["task"],
        "target": result["target"], "features": list(result["features"]),
        "transformedFeatures": transformed, "estimator": estimator.__class__.__name__,
        "parameters": params, "effects": effects, "effectLabel": kind,
        "metrics": clean(result["metrics"]), "trainMetrics": clean(result.get("train_metrics", {})),
        "sizes": clean(result.get("sizes", {})),
        "stages": [
            {"id": "input", "title": "Данные", "detail": f"{len(result['features'])} признаков · {result['target']} как цель"},
            {"id": "prep", "title": "Подготовка", "detail": f"{len(transformed)} признаков после кодирования и преобразования"},
            {"id": "model", "title": result["name"], "detail": estimator.__class__.__name__},
            {"id": "output", "title": "Оценка", "detail": f"{result['sizes'].get('test', 0)} наблюдений в test"},
        ],
    }


class CandidateRequest(BaseModel):
    task: str
    models: list[str] = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=100)
    params: dict[str, Any] = Field(default_factory=dict)
    spaces: dict[str, dict[str, Any]] = Field(default_factory=dict)


@app.post("/api/candidates")
def candidates(request: CandidateRequest):
    if request.task not in MODELS or any(name not in MODELS[request.task] for name in request.models):
        raise HTTPException(400, "В списке есть недоступная модель для выбранной задачи.")
    try:
        return generate_candidates(request.task, request.models, request.count, DEFAULTS | request.params, request.spaces)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/experiments")
def train(request: TrainRequest):
    raw = get_dataset(request.dataset)
    if request.task not in MODELS:
        raise HTTPException(400, "Неизвестный тип задачи.")
    if request.model not in MODELS[request.task]:
        raise HTTPException(400, "Модель недоступна для выбранной задачи.")
    if request.target not in raw.columns:
        raise HTTPException(400, "Целевая колонка отсутствует в наборе данных.")
    try:
        prepared, preparation = prepare_dataset(raw, request.target, request.task, request.preparation)
        if any(feature not in prepared.columns for feature in request.features):
            raise ValueError("Один или несколько выбранных признаков отсутствуют после подготовки данных.")
        params = preparation_params(prepared, request.features, request.target, request.task, request.params)
        model_params = {key: value for key, value in request.params.items() if key in MODEL_REGISTRY[request.model]["parameters"]}
        if request.strategy == "manual":
            candidates_to_run = [{"name": request.model, "params": model_params}]
            search = None
        elif request.strategy == "range":
            space = request.search_space or {key: schema for key, schema in MODEL_REGISTRY[request.model]["parameters"].items() if schema.get("type") != "layers"}
            spaces = {request.model: {**space, "__fixed__": model_params}}
            if request.candidates:
                allowed = set(MODEL_REGISTRY[request.model]["parameters"])
                for candidate in request.candidates:
                    if candidate.get("name") != request.model or not isinstance(candidate.get("params"), dict):
                        raise ValueError("Варианты должны относиться к выбранной модели и содержать параметры.")
                    if set(candidate["params"]) - allowed:
                        raise ValueError("В варианте есть неизвестные параметры модели.")
                candidates_to_run = request.candidates
            else:
                candidates_to_run = generate_candidates(request.task, [request.model], request.count, params, spaces)
            search = None
        elif request.strategy == "optuna":
            space = request.search_space or {key: schema for key, schema in MODEL_REGISTRY[request.model]["parameters"].items() if schema.get("type") != "layers"}
            candidates_to_run = []
            search = {"count": request.count, "families": [request.model], "spaces": {request.model: {**space, "__fixed__": model_params}}}
        else:
            raise ValueError("Неизвестная стратегия обучения.")
        result = run_experiments(prepared, request.features, request.target, request.task, candidates_to_run, params, search=search)
        if "error" in result:
            details = result.get("leaderboard", [{}])[0].get("Статус", result["error"])
            raise ValueError(details)
        eligible_count = len(eligible_rows(prepared, request.features, request.target, request.task))
        used_count = sum(int(result["sizes"].get(key, 0)) for key in ("train", "validation", "test"))
        result["size_details"] = {"rawRows": len(raw), "preparedRows": len(prepared), "eligibleRows": eligible_count,
            "usedRows": used_count, "availableOutside": max(0, eligible_count-used_count), "notEligible": max(0, len(prepared)-eligible_count)}
        run_id = uuid.uuid4().hex
        config = {"dataset": request.dataset, "model": request.model, "task": request.task,
            "features": request.features, "target": request.target, "params": params,
            "preparation": preparation, "strategy": request.strategy, "count": request.count,
            "search_space": request.search_space, "candidates": request.candidates}
        result.update(config=config, source="react")
        RUNS[run_id] = result
        record = STORE.save(result, config)
        RUN_CONFIGS[run_id] = config
        return {
            "id": run_id, "record": record, "model": result["name"], "task": result["task"],
            "target": result["target"], "features": result["features"], "metrics": result["metrics"],
            "trainMetrics": result["train_metrics"], "validation": result.get("validation", {}), "sizes": result["sizes"],
            "leaderboard": result["leaderboard"], "predictions": table_records(result["predictions"], 1000),
            "training": table_records(result["train_predictions"], 1000), "validationPredictions": table_records(winning_validation(result), 1000),
            "config": config, "sizeDetails": result["size_details"], "series": result_series(result),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


def _json_safe(value: Any) -> Any:
    import numpy as np
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records", date_format="iso"))
    if isinstance(value, pd.Series):
        return json.loads(value.to_json(date_format="iso"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return clean(value)


@app.post("/api/experiments/{run_id}/explain")
def explain_experiment(run_id: str, request: dict[str, Any]):
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полный результат запуска не сохранён. Повторите расчёт по сохранённой конфигурации, чтобы восстановить графики и объяснимость.")
    method = str(request.get("method", ""))
    class_index = int(request.get("classIndex", 0))
    try:
        from ml_lab.explain import permutation_scores, shap_scores, integrated_gradients
        if method == "Permutation Importance": payload = permutation_scores(result)
        elif method == "SHAP": payload = shap_scores(result, class_index=class_index)
        elif method == "Integrated Gradients": payload = integrated_gradients(result)
        else: raise HTTPException(400, "Неизвестный метод объяснения.")
        key = f"{method}:{class_index}"
        result.setdefault("explanations", {})[key] = payload
        RUNS[run_id] = result
        return {"key": key, "method": method, "payload": _json_safe(payload)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/experiments/{run_id}/series")
def experiment_series(run_id: str):
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полные временные ряды этого запуска не сохранены. Повторите расчёт по сохранённой конфигурации.")
    return result_series(result)


@app.get("/api/experiments/{run_id}/diagnostics")
def experiment_diagnostics(run_id: str):
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полный результат запуска не сохранён. Повторите расчёт по сохранённой конфигурации, чтобы восстановить графики и объяснимость.")
    from ml_lab.diagnostics import (timeline_figure, daily_figure, regression_figures,
        classification_figures, probability_figures, comparison_figures, training_figure)
    from ml_lab.diagnostics import daily_metric_figures
    figures = {"timeline": timeline_figure(result), **daily_metric_figures(result), **daily_metric_figures(result,basis='prepared')}
    daily = daily_figure(result)
    if daily is not None: figures["daily"] = daily
    quality = regression_figures(result["predictions"]) if result["task"] == "Регрессия" else classification_figures(result)
    figures.update({f"quality:{key}": fig for key, fig in quality.items()})
    figures.update({f"comparison:{key}": fig for key, fig in comparison_figures(result).items()})
    training = training_figure(result)
    if training is not None: figures["training"] = training
    if result.get("test_probabilities") is not None:
        selected = result.get("display_probability_class", result["probability_classes"][0])
        figures.update({f"probability:{key}": fig for key, fig in probability_figures(result, selected).items()})
    return {key: json.loads(fig.to_json()) for key, fig in figures.items()}


@app.get("/api/experiments/{run_id}/export/{format_name}")
def export_experiment(run_id: str, format_name: str):
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полный результат запуска не сохранён. Повторите расчёт по сохранённой конфигурации, чтобы восстановить графики и объяснимость.")
    config = RUN_CONFIGS.get(run_id, result.get("config", {}))
    if format_name == "html":
        return Response(html_report(result, config), media_type="text/html; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}.html"'})
    if format_name == "zip":
        from ml_lab.result_export import results_bundle
        content = results_bundle(result, result.get("chart_styles", {}), include_model=True)
        return Response(content, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}.zip"'})
    if format_name == "model":
        import io, joblib
        buffer = io.BytesIO(); joblib.dump({"pipeline": result["pipeline"], "config": config}, buffer)
        return Response(buffer.getvalue(), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}.joblib"'})
    if format_name == "predictions":
        content = result["predictions"].to_csv(index=True).encode("utf-8-sig")
        return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}-predictions.csv"'})
    if format_name == "config":
        return Response(json.dumps(clean(config), ensure_ascii=False, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}.json"'})
    raise HTTPException(404, "Формат экспорта не поддерживается.")


@app.get("/api/experiments/{run_id}/report")
def experiment_report(run_id: str):
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(404, "Полный результат запуска не сохранён. Повторите расчёт по сохранённой конфигурации, чтобы восстановить графики и объяснимость.")
    from ml_lab.experiments import html_report
    report = html_report(result, RUN_CONFIGS.get(run_id, result.get("config", {})))
    return Response(
        content=report,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="experiment-{run_id[:8]}.html"'},
    )


@app.get("/api/journal/export")
def export_journal():
    return STORE.list()


@app.get("/api/journal")
def journal():
    return STORE.list()


@app.delete("/api/journal")
def clear_journal():
    return {"removed": STORE.clear()}


class RadioAnalysisRequest(BaseModel):
    column: str
    source: str = 'prepared'
    preparation: dict[str, Any] = Field(default_factory=dict)
    max_points: int = Field(default=4096,ge=128,le=8192)
    lags: int = Field(default=48,ge=1,le=200)
    m: int = Field(default=2,ge=1,le=4)
    ratio: float = Field(default=.2,ge=.01,le=1)
    method: str = 'welch'
    segment: int = Field(default=512,ge=16,le=8192)
    trend: str = 'constant'


@app.post('/api/datasets/{name}/radio-analysis')
def radio_analysis(name: str, request: RadioAnalysisRequest):
    from ml_lab.radio_analysis import analyze_channel
    raw=get_dataset(name)
    if request.source not in ('raw','prepared'): raise HTTPException(422,'Неизвестный источник анализа.')
    try:
        allowed=('time_columns','dayfirst','strict_dates','drop_invalid_time','start_datetime','index_frequency','generate_index')
        options={k:v for k,v in request.preparation.items() if k in allowed}
        options.update(calendar=False,drop_invalid_time=True)
        if request.source=='prepared':
            for key in ('frequency','duplicate_policy'):
                if key in request.preparation: options[key]=request.preparation[key]
            for stage in ('repair','anomaly_stage','restoration'):
                config=request.preparation.get(stage)
                if config and request.column in config.get('columns',[]):
                    options[stage]={**config,'columns':[request.column]}
        frame,_=prepare_dataset(raw,request.column,'Регрессия',options)
        result=analyze_channel(frame,request.column,request.max_points,request.lags,request.m,request.ratio,request.method,request.segment,request.trend)
        result['source']=request.source
        result['processingApplied']=any(options.get(key) for key in ('repair','anomaly_stage','restoration'))
        return result
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(422,str(exc)) from exc


# Radio simulation stays independent of dataset training.
from api.radio import router as radio_router
app.include_router(radio_router)
