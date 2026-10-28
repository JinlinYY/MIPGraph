# Baseline models

The manuscript compares MIPGraph with eight baselines: RF, XGBoost, LightGBM,
ChemBERTa with an MLP prediction head, MPNNConcat, GCN, GAT and GraphSAGE.

Descriptor-based baselines combine molecular features and thermodynamic-condition
inputs. The molecular-language baseline uses frozen SMILES representations with
property-specific regressors. Graph baselines use the cached molecular graphs
and condition inputs. Model-specific configurations define the exact features,
architecture, optimization and checkpoint-selection settings.

The selected study uses four fixed data partitions and training seeds 0, 42 and
123. The tree-boosting configurations use LR=0.01. The selected graph baselines
use LR=0.00005, batch size 256, at most 100 epochs, and validation-based early
stopping. Keep each published configuration together with its corresponding
result; other script options represent separate experiments.

Targets are natural-log-transformed and standardized using original experimental
training observations. Original validation/test labels are evaluated separately
from copied and interpolated training labels. NMAE divides each property's
log-space MAE by its training-fitted scale, followed by an equal-weight mean
over the six properties.

Inspect available command options from the repository root:

```bash
python experiments/baseline_comparison/scripts/run_baseline_comparison.py --help
python experiments/baseline_comparison/scripts/run_split_baseline_comparison.py --help
```

`il_property_prediction/scripts/run_split_baselines.py` supplies the shared
descriptor and graph implementations. The manuscript and SI provide the
reported comparisons; numerical performance-figure sources are linked from
`experiments/result_analysis/figure_source_map.csv`.
