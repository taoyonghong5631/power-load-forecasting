# -*- coding: utf-8 -*-
"""训练好的模型落盘/加载，避免每次打开界面都重训一遍。"""
from __future__ import annotations

import os
from typing import Optional

import joblib

from .config import RESULTS_DIR

MODEL_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def model_path(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return os.path.join(MODEL_DIR, safe + ".joblib")


def save_model(model, name: str) -> str:
    path = model_path(name)
    joblib.dump(model, path, compress=3)
    return path


def load_model(name: str):
    path = model_path(name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def has_model(name: str) -> bool:
    return os.path.exists(model_path(name))
