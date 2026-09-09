# GNN-Based BERT for Understanding Context from Music

CSE425 project: hybrid BERT + GNN models for music tagging, structure graphs, multimodal fusion, and caption–audio retrieval.

## Setup

```bash
pip install -r requirements.txt
```

Expected data layout (not committed):

- `data/raw/fma/fma_small/` + `data/raw/fma/fma_metadata/`
- `data/raw/musiccaps/musiccaps-public.csv` + `data/raw/musiccaps/audio/`
- `data/raw/deam/MEMD_audio/` + `data/raw/deam/annotations/`

## Preprocess

```bash
python -m src.audio_features --config config.yaml --write-splits
python -m src.audio_features --config config.yaml --dataset fma_small
python -m src.audio_features --config config.yaml --dataset deam
```

## Train

```bash
python -m src.train --task 1 --config config.yaml
python -m src.train --task 2 --config config.yaml
python -m src.train --task 3 --config config.yaml
python -m src.train --task 4 --config config.yaml
python -m src.postprocess --config config.yaml
```

## Notebooks

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/demo_context.ipynb
jupyter notebook notebooks/human_eval_task4.ipynb
```

## Outputs

| Path | Contents |
|------|----------|
| `results/metrics.json` | Metrics for all tasks |
| `results/checkpoints/` | Best model weights |
| `results/plots/` | Training and analysis plots |
| `results/case_studies/` | Task 3 chord-path figures |
| `results/retrieval_examples/` | Task 4 retrieval examples |
| `results/human_eval_task4.json` | Listener ratings for retrieval |
| `results/graph_samples/` | Example graph summaries |
| `data/splits/` | Train/val/test split files |

## Notes

- Primary audio corpus: FMA-small
- Audio: 22 050 Hz, 128 mel / 12 chroma, 5 s segments
- Graphs: chord-transition (Tasks 2–3) and segment graphs (Task 4 / samples)
- Task 3 uses DEAM valence/arousal as an auxiliary loss
- Task 4 uses InfoNCE with temperature `τ = 0.07`
