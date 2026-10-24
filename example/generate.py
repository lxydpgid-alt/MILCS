"""Generate reproducible linear Gaussian intervention data from win95pts."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.data_gen_manual_continuous import build_design, generate_data_from_design


def main():
    folder = Path(__file__).resolve().parent
    config = json.loads((folder / 'config.json').read_text())
    design = build_design(
        L=config['latent_count'], Is=config['environment_count'],
        I=config['intervention_size'], seed=config['seed'],
        dag_matrix_path=str(folder / 'win95pts' / 'win95pts_graph.csv'),
        forbid_latent_nodes=[config['target']],
        prefer_intervention_near=[config['target']],
    )
    design['truth_dag_df'].to_csv(folder / 'truth_matrix.csv')
    for i, matrix in enumerate(design['env_matrices'][:-1], start=1):
        matrix.to_csv(folder / f'data_{i}_matrix01.csv')
    labels = list(design['truth_dag_df'].columns)
    pd.DataFrame(design['W_base'], index=labels, columns=labels).to_csv(
        folder / 'weight_matrix.csv')
    generate_data_from_design(
        design, n_samples=config['samples_per_environment'], noise_scale=1.0,
        seed=config['seed'], intervention_dir=str(folder), data_model='linear',
    )
    metadata = dict(config)
    original_labels = pd.read_csv(folder / 'win95pts' / 'win95pts_graph.csv', index_col=0).columns
    metadata.update({
        'source_graph': 'win95pts/win95pts_graph.csv',
        'model': 'linear Gaussian SEM; independent N(0,1) noise; hard interventions',
        'observed_labels': design['observed_labels'],
        'latent_renaming': {old: new for old, new in zip(original_labels, labels) if old != new},
        'interventions': design['interventions'][:-1],
        'sample_seed_rule': 'seed + environment_index * 1009 (index starts at 1)',
        'matrix_convention': 'row source, column destination',
    })
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
