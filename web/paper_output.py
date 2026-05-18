"""论文风格输出：中英性质名、符号、单位与 LaTeX 表格片段（无数据库依赖）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# 与 ui_model_runtime.PROPERTY_UNITS 保持一致（本模块避免 import torch 依赖）
_PROPERTY_UNITS: dict[str, str] = {
    "Density": "kg·m⁻³",
    "ElectricalConductivity": "S·m⁻¹",
    "HeatCapacity": "J·mol⁻¹·K⁻¹",
    "SurfaceTension": "mN·m⁻¹",
    "ThermalConductivity": "W·m⁻¹·K⁻¹",
    "Viscosity": "Pa·s",
}

# (英文名, 中文名, LaTeX 符号不含 $, 单位展示)
PROP_SPEC: list[tuple[str, str, str, str]] = [
    ("Density", "密度", r"\rho", _PROPERTY_UNITS["Density"]),
    ("ElectricalConductivity", "电导率", r"\sigma", _PROPERTY_UNITS["ElectricalConductivity"]),
    ("HeatCapacity", "热容", r"C_{p}", _PROPERTY_UNITS["HeatCapacity"]),
    ("SurfaceTension", "表面张力", r"\gamma", _PROPERTY_UNITS["SurfaceTension"]),
    ("ThermalConductivity", "热导率", r"\lambda", _PROPERTY_UNITS["ThermalConductivity"]),
    ("Viscosity", "黏度", r"\eta", _PROPERTY_UNITS["Viscosity"]),
]


def _fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not (v == v):  # NaN
        return "—"
    a = abs(v)
    if a >= 1000:
        return f"{v:.2f}"
    if a >= 10:
        return f"{v:.3f}"
    if a >= 1:
        return f"{v:.4f}"
    if a >= 0.01:
        return f"{v:.5f}"
    return f"{v:.3e}"


def build_markdown_report(
    *,
    case_label: str,
    il_smiles: str,
    temperature_k: float,
    pressure_kpa: float | None,
    values: dict[str, Any],
    graph_note: str | None,
    model_line: str,
) -> str:
    """供界面或附录粘贴的 Markdown 报告。"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    p_disp = f"{pressure_kpa:g}" if pressure_kpa is not None and pressure_kpa == pressure_kpa else "—"
    lines = [
        "### 预测条件（Conditions）",
        "",
        f"- **案例编号 / Case ID**：{case_label or '—'}",
        f"- **温度 / Temperature**：{temperature_k:g} K",
        f"- **压力 / Pressure**：{p_disp} kPa",
        f"- **离子液体 SMILES**：`{il_smiles.strip() or '—'}`",
        f"- **模型 / Model**：{model_line}",
        f"- **生成时间 / Generated**：{ts}",
        "",
    ]
    if graph_note:
        lines.extend(["> **构图说明 / Graph note**：" + graph_note, ""])
    lines.extend(
        [
            "### 性质预测表（Predicted properties）",
            "",
            "| Property (EN) | 性质（中文） | Symbol | Value | Unit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for en, zh, sym, unit in PROP_SPEC:
        val = values.get(en)
        lines.append(f"| {en} | {zh} | ${sym}$ | {_fmt_num(val)} | {unit} |")
    lines.append("")
    lines.append(
        "*注：数值由 3D-IPTNet 前向推断得到；若用于论文请说明网络结构、训练数据范围与不确定度来源。*"
    )
    return "\n".join(lines)


def build_latex_table(
    *,
    case_label: str,
    il_smiles: str,
    temperature_k: float,
    pressure_kpa: float | None,
    values: dict[str, Any],
    caption: str | None = None,
) -> str:
    """可粘贴进 manuscript 的 `tabular` 片段（需自行放入 table 环境并引入 amsmath 等）。"""
    cap = caption or "Predicted thermophysical properties (3D-IPTNet)"
    p_tex = f"{pressure_kpa:g}" if pressure_kpa is not None and pressure_kpa == pressure_kpa else "---"
    case_tex = (case_label or "---").replace("_", "\\_").replace("%", "\\%").replace("&", r"\&")
    rows_tex = []
    for en, zh, sym, unit in PROP_SPEC:
        val = _fmt_num(values.get(en))
        unit_esc = unit.replace("·", r"$\cdot$")
        rows_tex.append(f"{en} & {zh} & ${sym}$ & {val} & {unit_esc} \\\\")
    body = "\n".join(rows_tex)
    smi_comment = il_smiles.strip().replace("%", "\\%")
    return "\n".join(
        [
            r"% --- paste inside \begin{table}...\end{table} ---",
            r"% IL_SMILES (verbatim or SI): " + smi_comment,
            r"\centering",
            f"\\caption{{{cap}. Case ID: \\texttt{{{case_tex}}}; $T={temperature_k:g}\\ \\mathrm{{K}}$; $P={p_tex}$ kPa.}}",
            r"\begin{tabular}{llccc}",
            r"\hline",
            r"Property & (中文) & Symbol & Value & Unit \\",
            r"\hline",
            body,
            r"\hline",
            r"\end{tabular}",
        ]
    )
