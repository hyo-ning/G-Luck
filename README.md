# Accurate Graph Classification via Contextualized Structural Tokens

Anonymous code for **G-LUCK (Graph Classification via Contextualized Structural
Tokens)**

---

## Requirements

Use Python 3.10 or 3.11 with the dependencies specified in `requirements.txt`:

```text
torch==2.5.1
torch-geometric==2.6.1
numpy==1.26.4
networkx==3.4.2
scipy==1.15.3
scikit-learn==1.7.2
```

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Use a PyTorch build compatible with your CPU or CUDA environment.

---

## Code Structure

- `main.py`: Runs training, validation and evaluation for the selected folds.
- `gluck/model.py`: Implements cross-level message passing, additive token-to-context attention and layer-wise context readout.
- `gluck/tokenizer.py`: Extracts structural tokens and constructs the multi-level graph.
- `gluck/data.py`: Downloads TU datasets, encodes node features and generates stratified splits.
- `gluck/train.py`: Trains the model, keeps the validation-best weights in memory and evaluates accuracy.
- `gluck/constants.py`: Defines node and directed edge types.
- `data/raw/`: Stores dataset files downloaded on first use (created automatically).

---

## How to Run

Run the following commands from the repository root. Start training with:

```bash
python main.py
```

This runs MUTAG fold 0 with depth 5, dropout 0.3 and seed 10. Training runs for
up to 500 epochs with an early-stopping patience of 100. The device defaults to
`cuda:0` when CUDA is available, and `cpu` otherwise.

Run a specific dataset on GPU:

```bash
python main.py --dataset NCI109 --device cuda:0
```

Run a single fold on CPU:

```bash
python main.py --dataset MUTAG --folds 0 --device cpu
```

Run all ten folds:

```bash
python main.py --dataset MUTAG --folds 0 1 2 3 4 5 6 7 8 9
```

On first use, the selected dataset is downloaded automatically from the TU
dataset collection and stored under `data/raw/<dataset>/`. Later runs reuse
these files. The first download requires an internet connection.

Graph files and split files do not need to be uploaded. With the dependencies
installed, `main.py` and the `gluck/` folder are sufficient to run training.
The `data/` folder is created automatically for downloaded files.

### Main Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | One of the seven datasets below | `MUTAG` |
| `--device` | Device such as `cuda:0` or `cpu` | CUDA if available, otherwise CPU |
| `--folds` | Zero-based fold IDs from 0 to 9 | `0` |
| `--layers` | Depth including the input representation: 4, 5 or 6 | `5` |
| `--dropout` | Dropout on prediction logits: 0 or 0.3 | `0.3` |
| `--seed` | Model and training random seed | `10` |
| `--epochs` | Maximum training epochs | `500` |
| `--patience` | Epochs without improved validation accuracy before stopping | `100` |

---

## Supported Datasets

| Dataset | Domain | Input features |
|---------|--------|----------------|
| `IMDB-BINARY` | Social | One-hot node degrees |
| `PROTEINS` | Protein | One-hot node labels |
| `MUTAG` | Molecular | One-hot node labels |
| `PTC_MR` | Molecular | One-hot node labels |
| `NCI1` | Molecular | One-hot node labels |
| `NCI109` | Molecular | One-hot node labels |
| `Mutagenicity` | Molecular | One-hot node labels |

Data originate from the [TU graph dataset collection](https://chrsmrrs.github.io/datasets/).
The loader preserves the original TU graph ordering.
