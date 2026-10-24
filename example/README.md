# win95pts linear Gaussian example

## Source and model

The source DAG and variable mapping are in `win95pts/`:

- `win95pts_graph.csv`: the original 76-node DAG.
- `win95pts_node_mapping.csv`: V identifiers and original variable names.

The target is V63. Three nodes are renamed as latent variables:
V23 to L1, V3 to L2, and V29 to L3. The underlying DAG is otherwise unchanged.
Latent variables participate in simulation but are excluded
from the sample tables, which contain 73 observed columns.

Each environment has 5,000 samples from the linear Gaussian model
`X_j = sum_i(w_ij * X_i) + noise_j`, with independent standard normal noise.
Weights are sampled from [-2,-0.5] or [0.5,2] and saved in `weight_matrix.csv`.
The fixed seed is 20261024. Sampling uses seed + environment_index * 1009, with
indices starting at 1.

| Environment | Hard-intervention targets |
|---|---|
| 1 | V34, V56, V67 |
| 2 | V4, V51, V67 |
| 3 | V34, V51, V56 |

Hard interventions remove all incoming edges to each intervention target and
replace its equation with independent standard normal noise. Matrices use rows
as sources and columns as destinations. These are simulated continuous samples,
not the original discrete win95pts observations.

## Running

From the package root:

```bash
python MILCS.py
```

## Recorded target-only results

Default target V63, alpha 0.01, current local R1-R3 and global R1:

| Mode | Target metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Fisher_Z | Adjacency | 1.0000 | 1.0000 | 1.0000 |
| Fisher_Z | Mark | 1.0000 | 1.0000 | 1.0000 |

Only edges incident to V63 are scored, including both endpoints. The current
scoring convention accepts circle or tail at a causal source endpoint when direct
causation and latent confounding coexist. Perfect scores therefore do not imply
that every endpoint is fully oriented. These are results for one fixed example,
not averages across seeds. Runtime and traversal order can vary between runs.

## Files

- `config.json`: fixed settings.
- `metadata.json`: intervention sets, latent renaming, and observed labels.
- `truth_matrix.csv`: pre-intervention truth DAG including latent nodes.
- `weight_matrix.csv`: linear SEM weights.
- `data_1.csv` through `data_3.csv`: 5,000 rows and 73 columns per environment.
- `result_fisher_z.txt`: actual full-run output.
- `win95pts/`: original DAG and variable mapping.
