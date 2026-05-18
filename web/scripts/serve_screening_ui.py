"""
一键启动：真实 3D-IPTNet + 前端。

在 il_property_prediction 目录下执行:
  python scripts/serve_screening_ui.py

- 网页版（简洁排版 + 可选论文导出）：http://127.0.0.1:8765/
- App 简洁版（无论文导出）：http://127.0.0.1:8765/app

需 configs/default.yaml 与可用检查点（见 _default_config_and_ckpt）。
"""

from __future__ import annotations

import os

# Windows 上 Anaconda 与 PyTorch CPU  wheel 可能各带一份 OpenMP，避免 import 阶段直接退出
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_SCRIPT_DIR = Path(__file__).resolve().parent
_IL_ROOT = _SCRIPT_DIR.parent

if str(_IL_ROOT) not in sys.path:
    sys.path.insert(0, str(_IL_ROOT))

_TASK_ROOT = _IL_ROOT.parent.parent
_STATIC_INDEX = _IL_ROOT / "static" / "index.html"
_STATIC_APP = _IL_ROOT / "static" / "app.html"
_STATIC_ASSETS = _IL_ROOT / "static" / "assets"
_FALLBACK_INDEX = _TASK_ROOT / "IL_Screening_UI_Deliverable" / "index.html"
_FALLBACK_ASSETS = _TASK_ROOT / "IL_Screening_UI_Deliverable" / "assets"


def _index_html_path() -> Path:
    if _STATIC_INDEX.exists():
        return _STATIC_INDEX
    if _FALLBACK_INDEX.exists():
        return _FALLBACK_INDEX
    return _STATIC_INDEX


def _assets_dir() -> Path | None:
    if _STATIC_ASSETS.is_dir():
        return _STATIC_ASSETS
    if _FALLBACK_ASSETS.is_dir():
        return _FALLBACK_ASSETS
    return None


_assets_mount = _assets_dir()
_lock = threading.Lock()
_state: dict[str, Any] = {"bundle": None, "config_path": None, "checkpoint_path": None, "load_error": None}


def _default_config_and_ckpt() -> tuple[Path, Path]:
    cfg = _IL_ROOT / "configs" / "default.yaml"
    ckpt_primary = _IL_ROOT / "outputs" / "checkpoints" / "best_model.pt"
    ckpt_alt = _IL_ROOT / "outputs" / "checkpoints" / "finetune_viscosity_from_weak_seed42" / "best_model.pt"
    if ckpt_primary.exists():
        return cfg, ckpt_primary
    if ckpt_alt.exists():
        return cfg, ckpt_alt
    return cfg, ckpt_primary


def _ensure_model(cfg_opt: Path | None, ckpt_opt: Path | None) -> tuple[bool, str | None]:
    from ui_model_runtime import load_model_bundle  # noqa: PLC0415

    if _state["bundle"] is not None:
        return True, None
    d_cfg, d_ckpt = _default_config_and_ckpt()
    cfg = cfg_opt if cfg_opt is not None else d_cfg
    ckpt = ckpt_opt if ckpt_opt is not None else d_ckpt
    if not cfg.exists():
        return False, f"未找到配置文件：{cfg}"
    if not ckpt.exists():
        return False, f"未找到检查点：{ckpt}"
    try:
        bundle = load_model_bundle(cfg, ckpt)
    except Exception as exc:  # noqa: BLE001
        _state["load_error"] = str(exc)
        return False, "加载模型失败：" + str(exc)
    _state["bundle"] = bundle
    _state["config_path"] = str(cfg.resolve())
    _state["checkpoint_path"] = str(ckpt.resolve())
    _state["load_error"] = None
    return True, None


class OnePredictBody(BaseModel):
    IL_SMILES: str
    Temperature_K: float = 298.15
    Pressure_kPa: float | None = None
    case_label: str | None = None
    config_path: str | None = None
    checkpoint_path: str | None = None


class ModelLoadBody(BaseModel):
    config_path: str
    checkpoint_path: str


def _build_response(rows: list[dict], preds: np.ndarray, errors: list[str | None]) -> dict[str, Any]:
    from ui_model_runtime import PROPERTY_NAMES, PROPERTY_UNITS  # noqa: PLC0415

    results = []
    for i, row in enumerate(rows):
        sig = preds[i, PROPERTY_NAMES.index("ElectricalConductivity")]
        eta = preds[i, PROPERTY_NAMES.index("Viscosity")]
        ratio = float(sig / eta) if np.isfinite(sig) and np.isfinite(eta) and float(eta) > 1e-12 else None
        vals = {}
        for j, name in enumerate(PROPERTY_NAMES):
            v = preds[i, j]
            vals[name] = float(v) if np.isfinite(v) else None
        results.append(
            {
                "label": row.get("label") or f"第{i + 1}条",
                "IL_SMILES": row.get("IL_SMILES", ""),
                "Temperature_K": row.get("Temperature_K"),
                "Pressure_kPa": row.get("Pressure_kPa"),
                "graph_error": errors[i],
                "values": vals,
                "sigma_over_eta": ratio,
            }
        )
    return {
        "property_names": PROPERTY_NAMES,
        "property_units": PROPERTY_UNITS,
        "property_labels_zh": {
            "Density": "密度",
            "ElectricalConductivity": "电导率",
            "HeatCapacity": "热容",
            "SurfaceTension": "表面张力",
            "ThermalConductivity": "热导率",
            "Viscosity": "黏度",
        },
        "results": results,
    }


app = FastAPI(title="IL 六性质 UI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_df_lock = threading.Lock()
_il_table: Any = None
_il_table_error: str | None = None


def _get_il_table():
    """懒加载 clean_csv，供名称搜索。"""
    global _il_table, _il_table_error
    with _df_lock:
        if _il_table is not None:
            return _il_table, _il_table_error
        if _il_table_error is not None:
            return None, _il_table_error
        try:
            import pandas as pd  # noqa: PLC0415
            from src.utils.io import load_config, resolve_path  # noqa: PLC0415

            cfg = load_config(_IL_ROOT / "configs" / "default.yaml")
            base = Path(cfg["_base_dir"])
            csv_path = resolve_path(cfg["data"]["clean_csv"], base)
            if not csv_path.exists():
                _il_table_error = f"未找到数据表：{csv_path}"
                return None, _il_table_error
            _il_table = pd.read_csv(csv_path)
            return _il_table, None
        except Exception as exc:  # noqa: BLE001
            _il_table_error = str(exc)
            return None, _il_table_error


@app.get("/api/search")
def api_search(q: str = Query("", max_length=200), limit: int = Query(18, ge=1, le=50)):
    """按离子液体名称、阴阳离子名称等关键词在数据表中检索，返回 SMILES 供内部预测。"""
    df, err = _get_il_table()
    if err:
        raise HTTPException(status_code=503, detail=err)
    if df is None:
        raise HTTPException(status_code=503, detail="数据表未加载")
    raw = (q or "").strip().lower()
    terms = [t for t in raw.replace("，", " ").split() if t]
    if not terms:
        return {"results": []}
    cols = ["IL_Name", "Cation_FullName", "Cation_ShortName", "Anion_FullName", "Anion_ShortName", "IL_SMILES"]
    import pandas as pd  # noqa: PLC0415

    mask = pd.Series(True, index=df.index)
    for tok in terms:
        mcol = pd.Series(False, index=df.index)
        for c in cols:
            if c not in df.columns:
                continue
            mcol = mcol | df[c].astype(str).str.lower().str.contains(tok, regex=False, na=False)
        mask = mask & mcol
    sub = df.loc[mask].copy()
    if sub.empty:
        return {"results": []}
    use = [c for c in ["IL_Name", "Cation_ShortName", "Anion_ShortName", "IL_SMILES"] if c in sub.columns]
    sub = sub[use]
    if "IL_SMILES" in sub.columns:
        sub = sub.drop_duplicates(subset=["IL_SMILES"])
    sub = sub.head(limit)
    out = []
    for _, r in sub.iterrows():
        out.append(
            {
                "il_name": str(r.get("IL_Name", "")),
                "cation": str(r.get("Cation_ShortName", "")),
                "anion": str(r.get("Anion_ShortName", "")),
                "smiles": str(r.get("IL_SMILES", "")),
            }
        )
    return {"results": out}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap():
    cfg, ckpt = _default_config_and_ckpt()
    with _lock:
        loaded = _state["bundle"] is not None
        err = _state["load_error"]
    if loaded:
        msg = "模型已就绪。"
    elif err:
        msg = err
    elif not ckpt.exists():
        msg = f"未找到默认检查点：{ckpt}"
    else:
        msg = "首次预测时将自动加载模型。"
    return {
        "model_ready": loaded,
        "message": msg,
        "defaults": {"config_path": str(cfg.resolve()), "checkpoint_path": str(ckpt.resolve())},
        "last_error": err,
    }


@app.post("/api/model/load")
def model_load(body: ModelLoadBody):
    from ui_model_runtime import load_model_bundle  # noqa: PLC0415

    cfg, ckpt = Path(body.config_path), Path(body.checkpoint_path)
    try:
        bundle = load_model_bundle(cfg, ckpt)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _state["load_error"] = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _lock:
        _state["bundle"] = bundle
        _state["config_path"] = str(cfg.resolve())
        _state["checkpoint_path"] = str(ckpt.resolve())
        _state["load_error"] = None
    return {"ok": True, "config_path": _state["config_path"], "checkpoint_path": _state["checkpoint_path"]}


@app.post("/api/predict/one")
def predict_one(body: OnePredictBody):
    from ui_model_runtime import predict_batch_loaded  # noqa: PLC0415

    smi = body.IL_SMILES.strip()
    if not smi:
        raise HTTPException(status_code=400, detail="SMILES 不能为空")
    p = body.Pressure_kPa
    rows = [
        {
            "IL_SMILES": smi,
            "Temperature_K": float(body.Temperature_K),
            "Pressure_kPa": float(p) if p is not None else float("nan"),
            "label": "预测",
        }
    ]
    cfg_opt = Path(body.config_path) if body.config_path and body.config_path.strip() else None
    ckpt_opt = Path(body.checkpoint_path) if body.checkpoint_path and body.checkpoint_path.strip() else None
    with _lock:
        ok, err = _ensure_model(cfg_opt, ckpt_opt)
        if not ok:
            raise HTTPException(status_code=503, detail=err or "模型不可用")
        preds, errors = predict_batch_loaded(_state["bundle"], rows)
    out = _build_response(rows, preds, errors)
    from paper_output import build_latex_table, build_markdown_report  # noqa: PLC0415

    r0 = out["results"][0]
    if body.Pressure_kPa is None:
        p_view = 101.325
    else:
        try:
            p_view = float(body.Pressure_kPa)
            if p_view != p_view:
                p_view = 101.325
        except (TypeError, ValueError):
            p_view = 101.325
    case = (body.case_label or "").strip() or "—"
    out["paper_markdown"] = build_markdown_report(
        case_label=case,
        il_smiles=smi,
        temperature_k=float(body.Temperature_K),
        pressure_kpa=p_view,
        values=r0["values"],
        graph_note=r0.get("graph_error"),
        model_line="3D-IPTNet（本仓库默认检查点）",
    )
    out["paper_latex"] = build_latex_table(
        case_label=case,
        il_smiles=smi,
        temperature_k=float(body.Temperature_K),
        pressure_kpa=p_view,
        values=r0["values"],
    )
    return JSONResponse(content=out)


@app.get("/")
def index_page():
    path = _index_html_path()
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="缺少 static/index.html。请确认文件存在："
            + str(_STATIC_INDEX)
            + "（或备用："
            + str(_FALLBACK_INDEX)
            + "）",
        )
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/app")
def app_shell_page():
    """无「论文用稿」区块的精简界面（仍调用同一套 /api）。"""
    if not _STATIC_APP.exists():
        raise HTTPException(
            status_code=500,
            detail="缺少 static/app.html：" + str(_STATIC_APP),
        )
    return FileResponse(_STATIC_APP, media_type="text/html; charset=utf-8")


if _assets_mount is not None:
    app.mount("/assets", StaticFiles(directory=str(_assets_mount)), name="assets")


@app.on_event("startup")
def startup():
    ecfg = os.environ.get("IL_SCREENING_CONFIG", "").strip()
    eckpt = os.environ.get("IL_SCREENING_CKPT", "").strip()
    with _lock:
        if ecfg and eckpt:
            try:
                from ui_model_runtime import load_model_bundle  # noqa: PLC0415

                _state["bundle"] = load_model_bundle(Path(ecfg), Path(eckpt))
                _state["config_path"] = str(Path(ecfg).resolve())
                _state["checkpoint_path"] = str(Path(eckpt).resolve())
                _state["load_error"] = None
            except Exception as exc:  # noqa: BLE001
                _state["load_error"] = str(exc)
        if _state["bundle"] is None and not os.environ.get("IL_SCREENING_SKIP_AUTOLOAD"):
            cfg, ckpt = _default_config_and_ckpt()
            if cfg.exists() and ckpt.exists():
                try:
                    _ensure_model(None, None)
                except Exception as exc:  # noqa: BLE001
                    _state["load_error"] = (
                        _state.get("load_error") or ""
                    ) + f"；自动加载模型失败（页面仍可用）：{exc}"


if __name__ == "__main__":
    import uvicorn

    os.chdir(_IL_ROOT)
    port = int(os.environ.get("IL_SCREENING_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
