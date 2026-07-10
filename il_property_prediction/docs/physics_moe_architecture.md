# MIPGraph Architecture

## Architecture

```text
cation -> shared Uni-Mol2-84M -> CLS + atom representations --\
                                                                  chemistry-biased
anion  -> shared Uni-Mol2-84M -> CLS + atom representations --/   cross-ion attention
                                                                          |
                                                        CLS bilinear + atom interaction
                                                                          |
                                                              pair representation
                                                                  +-> T/P FiLM
                                                                  |
                     packing / cohesion / transport / thermal experts
                                                                  |
                    condition-aware property-specific top-k router
                                                                  |
                         six independent property heads -> log targets
```

The four experts follow the mechanism-oriented latent factors in Eqs. (9)-(10) of the
MIPGraph manuscript. The original static property gate is replaced by a sample- and
condition-dependent router:

```text
z[p] = sum(m) gate[p, m](ion_pair, T, P) * expert[m](ion_pair)
```

The router starts from the manuscript's qualitative property-mechanism priors and selects
the top two experts by default. The six heads do not consume predictions from other heads.
Each head receives only the condition-modulated pair representation, its routed physical
latent, and temperature/pressure basis terms.

## Uni-Mol2 backbone

The encoder is the official `UniMolV2Model` from `unimol_tools`, initialized with the
`dptech/Uni-Mol2` 84M checkpoint. The 12-layer backbone produces a 768-dimensional CLS
representation. A single shared backbone encodes deduplicated cations and anions in one
forward pass; separate trainable projections preserve their ionic roles.

The ion interaction path combines two complementary terms. The CLS path uses a bilinear
interaction over whole-ion representations. The atom path applies bidirectional cross-ion
attention between Uni-Mol2 atomic representations. Attention logits receive trainable
biases for formal-charge complementarity, hydrogen-bond donor-acceptor compatibility,
aromatic contacts, and charge strength. No cross-ion distance is used because the dataset
does not provide reliable liquid-phase ion-pair geometries.

For the Python 3.9 environment used by this project, install the compatible package
without replacing the project's NumPy and pandas versions:

```powershell
E:\anaconda\envs\ggnn39\python.exe -m pip install --no-deps unimol_tools==0.1.4.post1
E:\anaconda\envs\ggnn39\python.exe -m pip install addict huggingface_hub
```

The default configuration freezes Uni-Mol2 and trains the projections, interaction,
Physics-MoE and property heads. Set `unimol2_unfreeze_last_n_layers: 2` for a later
low-learning-rate backbone fine-tuning stage.

## Feature cache

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts\build_unimol2_ion_cache.py `
  --graph-cache data\processed\graph_cache_fg.pt `
  --output data\processed\unimol2_ion_features.pt
```

## Training

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts\train_mipgraphnet.py `
  --config configs/default.yaml `
  --run-name mipgraph_seed42 `
  --monitor-space log `
  --skip-test-evaluation
```

Select checkpoints only with validation metrics. Evaluate the fixed test split after model
and hyperparameter selection is complete.
