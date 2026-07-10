from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ..utils.io import resolve_path
from .candidate_generator import generate_candidates
from .constraints import apply_task_scoring, get_task
from .fragment_library import load_fragment_library
from .pareto import assign_pareto_ranks
from .predictor import DesignPredictor


def _safe_run_name(task_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task_name}_{stamp}"


def _report_markdown(
    task_name: str,
    generated_count: int,
    valid_count: int,
    passing_count: int,
    rejected_count: int,
    prediction_failure_count: int,
    top: pd.DataFrame,
) -> str:
    lines = [
        f"# MIPGraph Design Agent Report",
        "",
        f"- task: `{task_name}`",
        f"- generated candidates: {generated_count}",
        f"- valid generated candidates: {valid_count}",
        f"- candidates passing constraints: {passing_count}",
        f"- rejected fragment combinations: {rejected_count}",
        f"- prediction failures: {prediction_failure_count}",
        "",
        "## Top Candidates",
        "",
    ]
    props = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]
    for _, row in top.iterrows():
        lines.append(f"### {row['candidate_id']}")
        lines.append("")
        lines.append(f"- SMILES: `{row['IL_SMILES']}`")
        lines.append(f"- Pareto rank: {int(row['pareto_rank'])}")
        lines.append(f"- scalar score: {float(row['objective_score']):.4f}")
        lines.append(f"- reason: {row.get('selection_reason', '')}")
        for prop in props:
            pred = row.get(f"{prop}_pred")
            if pd.isna(pred):
                continue
            std = row.get(f"{prop}_pred_std")
            if pd.notna(std):
                lines.append(f"- {prop}: {float(pred):.6g} +/- {float(std):.3g}")
            else:
                lines.append(f"- {prop}: {float(pred):.6g}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("Scores and constraints are local screening heuristics over MIPGraph predictions. Physics-inspired terms are regularizers/structured priors, not exact thermodynamic equations.")
    return "\n".join(lines)


def run_design_agent(
    config: dict,
    checkpoint: str | Path,
    task_name: str,
    cations_path: str | Path,
    anions_path: str | Path,
    temperature: float,
    pressure: float,
    top_k: int = 50,
    output_dir: str | Path = "outputs/design_agent",
    run_name: str | None = None,
    max_candidates: int | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    task = get_task(task_name)
    base = config.get("_base_dir")
    cations, anions = load_fragment_library(cations_path, anions_path)
    candidates, rejected = generate_candidates(cations, anions, max_candidates=max_candidates)
    predictor = DesignPredictor(config, checkpoint, device=device)
    predictions, prediction_failures = predictor.predict(candidates, temperature, pressure, batch_size=batch_size)
    if predictions.empty:
        raise RuntimeError("No valid candidates could be predicted.")
    scored = apply_task_scoring(predictions, task_name)
    rank_source = scored[scored["passes_constraints"]].copy()
    if rank_source.empty:
        rank_source = scored.copy()
    ranked = assign_pareto_ranks(rank_source, task["objectives"])
    scored = scored.merge(ranked[["candidate_id", "pareto_rank"]], on="candidate_id", how="left")
    scored["pareto_rank"] = scored["pareto_rank"].fillna(9999).astype(int)
    scored = scored.sort_values(["pareto_rank", "objective_score"], ascending=[True, False]).reset_index(drop=True)
    pareto_front = scored[scored["pareto_rank"] == 0].copy()
    top = scored.head(top_k).copy()

    run_name = run_name or _safe_run_name(task_name)
    out_dir = resolve_path(output_dir, base) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "candidates.csv", index=False)
    pareto_front.to_csv(out_dir / "pareto_front.csv", index=False)
    config_used = {
        "task": task_name,
        "checkpoint": str(checkpoint),
        "temperature": temperature,
        "pressure": pressure,
        "top_k": top_k,
        "cations_path": str(cations_path),
        "anions_path": str(anions_path),
        "max_candidates": max_candidates,
        "batch_size": batch_size,
        "task_definition": task,
    }
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_used, f, sort_keys=False)
    report = _report_markdown(
        task_name,
        generated_count=len(candidates) + len(rejected),
        valid_count=len(candidates),
        passing_count=int(scored["passes_constraints"].sum()),
        rejected_count=len(rejected),
        prediction_failure_count=len(prediction_failures),
        top=top,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "candidates": len(candidates),
        "rejected": len(rejected),
        "prediction_failures": len(prediction_failures),
        "passing_constraints": int(scored["passes_constraints"].sum()),
        "pareto_front": len(pareto_front),
    }
