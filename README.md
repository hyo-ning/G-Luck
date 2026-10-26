# Accurate Graph Classification via Contextualized Structural Tokens

Anonymous code for **G-LUCK (Graph Classification via Contextualized Structural
Tokens)**, accompanying the paper **Accurate Graph Classification via
Contextualized Structural Tokens**.

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

### Data Splits

The code generates stratified ten-fold splits using a fixed split seed of `0`.
Each outer fold reserves approximately 10% of graphs for testing. A stratified
inner split reserves approximately 10% of the remaining graphs for validation.
The rest are used for training. Splits are generated in memory and stay the same
across training seeds; no split files are read or written.

This follows the split-generation routine in the experiment code. The original
IMDB-BINARY, PROTEINS and NCI1 experiments used precomputed partitions, so their
generated partitions differ from the saved partitions used for the paper.

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

### Training Configuration

The model uses a hidden dimension of 64 and a two-layer MLP in each update.
Adam is configured with learning rate `0.001`, weight decay `0.0001` and batch
size `32`. The learning rate remains constant.

Token features are initialized as the mean of member-node features, and the
context is initialized as the mean of token features. Token-to-context messages
use additive attention, context-to-token messages broadcast directly, and the
GIN self-term uses fixed epsilon 0. Layer-wise context prediction logits are
summed. The depth convention includes the input representation: `--layers 5`
performs four message-passing updates and uses five prediction heads.

The tokenizer extracts cliques, rings, residual stars and residual chains,
with singleton tokens for uncovered nodes. The same extraction rules apply to
all seven datasets. The maximum ring length is 15 for NCI109 and 6 otherwise.

The first model state reaching the highest validation accuracy is selected;
accuracy ties do not reset patience. Test evaluation uses those weights after
training. Compare configurations using validation accuracy, and run all ten
folds to obtain a ten-fold mean.

### Console Results

Training progress shows loss, training accuracy, validation accuracy and the best
validation accuracy at the first epoch, every ten epochs and the final epoch.
At the end, a table reports the selected epoch, validation accuracy and test
accuracy for each fold. Multiple folds also report mean test accuracy and its
population standard deviation across the selected folds. All accuracies are
displayed as percentages.

The validation-best weights are kept in memory. Checkpoints, training histories
and result files are not written to disk.

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
