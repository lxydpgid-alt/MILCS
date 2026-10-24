# win95pts linear Gaussian example

## Source and model

The unchanged source DAG and variable mapping are in `win95pts/`:

- `win95pts_graph.csv`: the original 76-node DAG.
- `win95pts_node_mapping.csv`: V identifiers and original variable names.

`generate.py` uses the project's existing generator with the fixed settings in
`config.json`. The target is V63. Three nodes are renamed as latent variables:
V23 to L1, V3 to L2, and V29 to L3. The underlying DAG is otherwise unchanged.
Latent variables participate in simulation and Oracle queries but are excluded
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

## Running and regeneration

From the package root:

```bash
python MILCS.py
python MILCS.py --ci-type D_sep
python example/generate.py
```

Regeneration overwrites the generated data, DAGs, weights, and metadata. The source
DAG files and config are preserved. Requires the dependencies in requirements.txt.

## Recorded target-only results

Default target V63, alpha 0.01, current local R1-R3 and global R1:

| Mode | Target metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Fisher_Z | Adjacency | 1.0000 | 1.0000 | 1.0000 |
| Fisher_Z | Mark | 1.0000 | 1.0000 | 1.0000 |
| D_sep | Adjacency | 1.0000 | 1.0000 | 1.0000 |
| D_sep | Mark | 0.8000 | 0.8000 | 0.8000 |

Only edges incident to V63 are scored, including both endpoints. The current
scoring convention accepts circle or tail at a causal source endpoint when direct
causation and latent confounding coexist. Perfect scores therefore do not imply
that every endpoint is fully oriented. These are results for one fixed example,
not averages across seeds. Oracle uses true intervention DAGs rather than samples.
Runtime and Oracle traversal order can vary between runs.

## Files

- `config.json`: fixed generation settings.
- `metadata.json`: intervention sets, latent renaming, and observed labels.
- `generate.py`: reproducible generation entry point.
- `truth_matrix.csv`: pre-intervention truth DAG including latent nodes.
- `weight_matrix.csv`: linear SEM weights.
- `data_1.csv` through `data_3.csv`: 5,000 rows and 73 columns per environment.
- `data_1_matrix01.csv` through `data_3_matrix01.csv`: intervention DAGs for Oracle.
- `result_fisher_z.txt` and `result_oracle.txt`: actual full-run output.
- `win95pts/`: original DAG and variable mapping.
