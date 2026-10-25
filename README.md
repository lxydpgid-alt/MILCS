# MILCS

MILCS (Multi-Intervention Local Causal Structure Learning) is a Python toolkit for learning the causal structure around a target variable from multiple interventional environments in the presence of latent variables. It provides local PAG learning, multi-environment graph merging, induced-edge pruning, global orientation, and target-specific evaluation. A complete end-to-end example is available in `example.ipynb`.

## 😀 Main Components

- `learn_local` / `collect_local_pags`: learn local PAGs and accumulate separation sets in `MILCS.py`.
- `merge_pags` / `prune_edges`: merge target edges and remove induced edges in `MILCS.py`.
- `PMB_Learner`: learns local skeletons and applies local R1-R3 in `utils/PMB_learn.py`.
- `G_orient`: applies global R1 using support from any one environment in `Graph/G_orient.py`.
- `CausalLearnCITest`: runs conditional independence tests with query caching in `utils/CI_test.py`.

## 🗒️ Typical Workflow

A typical workflow loads the sample tables for each environment together with the associated configuration metadata. Starting from the target variable, MILCS iteratively processes relevant variables. For each anchor variable, it learns a local PAG from each environment using Markov blanket discovery and conditional independence (CI) tests, then extracts the local edges and V-structures relevant to that anchor. The resulting local structures are integrated into a global graph by merging adjacencies and endpoint marks and removing spurious adjacencies identified through cross-environment information. The algorithm then applies the global orientation rule (Rule 1) using the recorded separating sets. Newly discovered variables in the Markov blankets are added to the processing queue, and the recursive search continues until the local structure around the target is determined or the queue becomes empty.Finally, evaluation is restricted to edges incident to the original target variable.

## 🛠️ Environment

Python version in the current environment:

- Python 3.10.18

Third-party packages used by the codebase and their versions in the current environment:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| causal-learn | 0.1.4.3 |
| joblib | 1.5.2 |

Note: standard-library modules such as `os`, `argparse`, `json`, `time`, `re`, `logging`, `typing`, `itertools`, `collections`, `dataclasses`, `enum`, and `warnings` are also used, but they are included with Python and do not require separate installation.

Install the tested dependencies:

```bash
python -m pip install -r requirements.txt
```

## 🚀 Evaluation

Let `T` be the original target. Adjacency evaluation compares whether each other observed variable is connected to `T`. Mark evaluation compares the circle, tail, and arrowhead at **both endpoints** of every edge incident to `T`. The evaluated neighbors are the union of predicted and true neighbors, so missing and extra target edges are included. Edges elsewhere in the graph are not scored.

For both adjacency and Mark evaluation:

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{F1}=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
$$

At the mark level, an incorrect endpoint contributes one FP and one FN. If both predicted and true target neighborhoods are empty, all metrics are 1; if only one is empty, all metrics are 0. Other zero denominators yield 0.


## 🤝 Contributing & Contact

If you have questions or suggestions, feel free to open an issue or pull request.
