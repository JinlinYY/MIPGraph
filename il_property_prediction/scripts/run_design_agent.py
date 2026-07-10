from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.agent.design_agent import run_design_agent
from src.agent.constraints import TASKS
from src.utils.io import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MIPGraph design agent.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="electrolyte_low_viscosity_high_conductivity")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--pressure", type=float, default=101.325)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs/design_agent")
    parser.add_argument("--cations", default=str(REPO_ROOT / "data" / "design_fragments" / "cations.csv"))
    parser.add_argument("--anions", default=str(REPO_ROOT / "data" / "design_fragments" / "anions.csv"))
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_design_agent(
        cfg,
        checkpoint=resolve_path(args.checkpoint, cfg["_base_dir"]),
        task_name=args.task,
        cations_path=resolve_path(args.cations, cfg["_base_dir"]),
        anions_path=resolve_path(args.anions, cfg["_base_dir"]),
        temperature=args.temperature,
        pressure=args.pressure,
        top_k=args.top_k,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
