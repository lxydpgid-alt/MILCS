# MILCS

**Multi-Intervention Local Causal Structure Learning**

MILCS is a Python toolkit for learning the causal structure around a target
variable from multiple interventional environments in the presence of latent
variables. It provides local PAG learning, multi-environment graph merging,
induced-edge pruning, global orientation, continuous data simulation, and
target-specific evaluation.

A complete runnable example is available in [example.ipynb](example.ipynb).
The bundled [win95pts example](example/README.md) includes data, intervention DAGs,
fixed generation settings, and recorded results.

## Main Components

| Component | Location | Purpose |
|---|---|---|
| `load_data` | `MILCS.py` | Load sample tables or Oracle DAGs for each environment. |
| `learn_local` / `collect_local_pags` | `MILCS.py` | Learn local PAGs and accumulate separation sets. |
| `merge_pags` / `prune_edges` | `MILCS.py` | Merge target edges and remove induced edges. |
| `process_target` / `update_waitlist` | `MILCS.py` | Run one learning round and update the processing queue. |
| `PMB_Learner` | `utils/PMB_learn.py` | Learn local skeletons and apply local R1-R3. |
| `G_orient` | `Graph/G_orient.py` | Apply global R1 using support from any one environment. |
| `CausalLearnCITest` | `utils/CI_test.py` | Run conditional independence tests with query caching. |
| `build_design` / `generate_data_from_design` | `utils/data_gen_manual_continuous.py` | Build intervention designs and simulate data. |
| `build_eval_matrices` / `evaluate_local` | `utils/evaluate.py` | Evaluate adjacency and endpoint marks around the original target. |
| `print_g` | `utils/print_g.py` | Print learned edges using their graph representation. |

## Typical Workflow

1. Load a DAG and define latent variables and intervention environments, or load
   existing sample tables.
2. Generate continuous observations when using the bundled simulation workflow.
3. Learn a local PAG for the current target in each environment.
4. Merge local target edges, prune induced edges, and apply global R1.
5. Expand the processing queue until a stopping condition is reached.
6. Evaluate only edges incident to the original target, including both endpoints.

Conflicting definite endpoint marks become circles during graph merging. Global
orientation can subsequently refine circles when its rule conditions are met.

## Environment

Tested with **Python 3.10.18** and the following packages:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| networkx | 3.4.2 |
| causal-learn | 0.1.4.3 |
| igraph | 1.0.0 |
| joblib | 1.5.2 |

Install the tested dependencies:

```bash
python -m pip install -r requirements.txt
```

Standard-library modules do not require separate installation. To use the optional
notebook, install Jupyter in the same environment and select that Python kernel:

```bash
python -m pip install notebook
python -m notebook example.ipynb
```

## Quick Start

Run the bundled continuous-data example from the repository root:

```bash
python MILCS.py
```

Run the corresponding Oracle experiment:

```bash
python MILCS.py --ci-type D_sep
```

The default target is `V63` and the significance level is `0.01`. Example paths
are resolved relative to `MILCS.py`, independently of the working directory.
Oracle mode uses true intervention DAGs for d-separation instead of sample-based
CI tests; it still runs the learning and target-evaluation pipeline.

### Custom data

```bash
python MILCS.py \
  --target V63 \
  --ci-type Fisher_Z \
  --alpha 0.01 \
  --intervention-dir /path/to/environments \
  --truth-matrix /path/to/truth_matrix.csv
```

Sample tables must be named `data_1.csv`, `data_2.csv`, etc., with variable names
in the header and no saved row index. Oracle matrices must be named
`data_1_matrix01.csv`, `data_2_matrix01.csv`, etc., with matching row and column
labels. A matrix entry of 1 represents an edge from its row to its column.

| Argument | Default | Description |
|---|---|---|
| `--target` | `V63` | Original target to learn and evaluate. |
| `--alpha` | `0.01` | CI significance level. |
| `--ci-type` | `Fisher_Z` | Fisher_Z, KCI, RCIT, FastKCI, Poly_Z, G_sq, or D_sep. |
| `--intervention-dir` | `example/` | Environment samples or Oracle DAGs. |
| `--truth-matrix` | `example/truth_matrix.csv` | Original DAG used to construct evaluation truth. |
| `--max-processed-nodes` | Automatic | Processing guard; for graphs larger than 50 nodes, the default is floor(20% of truth nodes). |
| `--waitlist-policy` | `all_local` | all_local, circle_circle, or at_least_one_circle. |

Latent variables use the names `L1` through `L100`. They are retained in Oracle
DAGs and excluded from continuous observation tables.

## win95pts Example

The source win95pts DAG and variable mapping are preserved in
[example/win95pts](example/win95pts). The example generates **linear Gaussian**
samples from that DAG; it does not contain the original discrete observations.

- 76 nodes: 73 observed variables and 3 latent variables.
- 3 intervention environments, with **5,000 samples per environment**.
- 3 hard-intervention targets in each environment.
- Fixed generation seed: `20261024`.
- Original target: `V63`.

Regenerate the bundled data:

```bash
python example/generate.py
```

The actual results for the fixed configuration are:

| Mode | Target metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Fisher_Z | Adjacency | 1.0000 | 1.0000 | 1.0000 |
| Fisher_Z | Mark | 1.0000 | 1.0000 | 1.0000 |
| D_sep | Adjacency | 1.0000 | 1.0000 | 1.0000 |
| D_sep | Mark | 0.8000 | 0.8000 | 0.8000 |

These are single-configuration demonstration results, not averages across seeds.
See the full [continuous output](example/result_fisher_z.txt),
[Oracle output](example/result_oracle.txt), and [example documentation](example/README.md).

## Target-Specific Evaluation

Let `T` be the original target. Adjacency evaluation compares whether each other
observed variable is connected to `T`. Mark evaluation compares the circle, tail,
and arrowhead at **both endpoints** of every edge incident to `T`. The evaluated
neighbors are the union of predicted and true neighbors, so missing and extra
target edges are included. Edges elsewhere in the graph are not scored.

For both adjacency and Mark evaluation:

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{F1}=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
$$

At the mark level, an incorrect endpoint contributes one FP and one FN. An extra
endpoint contributes an FP, and a missing endpoint contributes an FN. If both
predicted and true target neighborhoods are empty, all metrics are 1; if only
one is empty, all metrics are 0. Other zero denominators yield 0.

The truth conversion represents direct causal edges and directly shared latent
parents. If direct causation and latent confounding coexist, the current scoring
convention accepts either a circle or a tail at the causal source endpoint.
Perfect Mark scores therefore do not require every endpoint to be fully oriented.
This conversion does not implement a general latent projection for arbitrary
hidden paths. Oracle mode does not substitute a fully identified true PAG for
the learned output.

## Repository Structure

```text
MILCS.py
README.md
requirements.txt
example.ipynb
Graph/
    mixgraph.py
    sepset.py
    G_orient.py
utils/
    PMB_learn.py
    CI_test.py
    mb.py
    evaluate.py
    print_g.py
    data_gen_manual_continuous.py
example/
    README.md
    config.json
    metadata.json
    generate.py
    truth_matrix.csv
    weight_matrix.csv
    data_1.csv ... data_3.csv
    data_1_matrix01.csv ... data_3_matrix01.csv
    result_fisher_z.txt
    result_oracle.txt
    win95pts/
```

## Questions and Feedback

Please open a GitHub issue to report a reproducible problem or suggest an improvement.
Include the command, Python and dependency versions, and relevant output.
