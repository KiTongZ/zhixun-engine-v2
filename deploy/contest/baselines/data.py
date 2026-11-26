"""Read only explicitly supplied public training/validation data."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

MAX_BYTES = 128 * 1024 * 1024
FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}


def sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("nonfinite JSON number")


_DECODER = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=_reject_constant)


def json_value(raw):
    if isinstance(raw, str):
        return _DECODER.decode(raw)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=_reject_constant)


@lru_cache(maxsize=4096)
def _encoded_path(path):
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1024)
def _allowed_key(key):
    return re.sub(r"[^a-z0-9]", "", key.lower()) not in FORBIDDEN_INPUTS


@lru_cache(maxsize=128)
def _numeric_list_paths(path, size):
    return tuple(_encoded_path(path + (("index", index),)) for index in range(size))


ENGINE_PATHS = [
    *_numeric_list_paths((("key", "sensors"),), 21),
    *_numeric_list_paths((("key", "settings"),), 3),
]
_ENGINE_KEYS = frozenset(("sensors", "settings"))
_FEATURE_BATCH_ROWS = 2048


def _finite_numbers(values):
    return all((type(value) is float or type(value) is int) and math.isfinite(value)
               for value in values)


def flatten(inputs):
    if type(inputs) is dict and inputs.keys() == _ENGINE_KEYS:
        sensors, settings = inputs["sensors"], inputs["settings"]
        if type(sensors) is list and len(sensors) == 21 and type(settings) is list and len(settings) == 3:
            combined = sensors + settings
            if _finite_numbers(combined):
                return ENGINE_PATHS.copy(), list(map(float, combined))

    paths, values = [], []
    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or not _allowed_key(key):
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + (("key", key),), depth + 1)
        elif isinstance(value, list):
            if depth < 16 and 0 < len(value) <= 256 and all(
                type(child) in (int, float) and math.isfinite(child) for child in value
            ):
                paths.extend(_numeric_list_paths(path, len(value)))
                values.extend(map(float, value))
            else:
                for index, child in enumerate(value):
                    visit(child, path + (("index", index),), depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(_encoded_path(path))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")
    visit(inputs, (), 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def _hashed_lines(path, digest):
    total = 0
    with path.open("rb") as source:
        while block := source.read(min(1024 * 1024, MAX_BYTES + 1 - total)):
            # Decode through the next LF so a UTF-8 character or JSON row
            # cannot be split by an arbitrary read boundary.
            if not block.endswith(b"\n") and total + len(block) <= MAX_BYTES:
                block += source.readline(MAX_BYTES + 1 - total - len(block))
            total += len(block)
            if total > MAX_BYTES:
                raise ValueError("dataset must be a regular JSONL file no larger than 128 MiB")
            digest.update(block)
            yield from block.decode("utf-8").splitlines()


def read_dataset(path, task, classes=None, expected_paths=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("dataset must be a regular JSONL file no larger than 128 MiB")
    ids, features, targets = [], [], []
    feature_batches = []
    seen = set()
    paths = expected_paths
    digest = hashlib.sha256()
    for number, line in enumerate(_hashed_lines(path, digest), 1):
        if not line.strip():
            continue
        row = json_value(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError(f"line {number}: id, inputs and answer are required")
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"line {number}: sample IDs must be nonempty and unique")
        current_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = current_paths
        if current_paths != paths:
            raise ValueError(f"line {number}: feature schema changed")
        answer = row["answer"]
        if task == "classification":
            if not isinstance(answer, str) or not classes or answer not in classes:
                raise ValueError(f"line {number}: label is outside the configured class list")
            target = classes.index(answer)
        elif task == "regression":
            if isinstance(answer, bool) or not isinstance(answer, (float, int)) or not math.isfinite(answer) or answer < 0:
                raise ValueError(f"line {number}: RUL must be a finite nonnegative number")
            target = float(answer)
        else:
            raise ValueError("unsupported task")
        ids.append(identity); seen.add(identity); features.append(values); targets.append(target)
        if len(features) == _FEATURE_BATCH_ROWS:
            feature_batches.append(np.asarray(features, dtype=np.float64))
            features.clear()
    if not ids:
        raise ValueError("dataset is empty")
    if features:
        feature_batches.append(np.asarray(features, dtype=np.float64))
    matrix = feature_batches[0] if len(feature_batches) == 1 else np.concatenate(feature_batches, axis=0)
    return {"ids": ids, "x": matrix,
            "y": np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
            "paths": paths, "sha256": "sha256:" + digest.hexdigest()}
