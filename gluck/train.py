import copy
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from .model import GLuck


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def batches(graphs, indices, device, generator=None):
    indices = list(indices)
    if generator is not None:
        indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
    for start in range(0, len(indices), 32):
        yield Batch.from_data_list([graphs[i] for i in indices[start:start + 32]]).to(device)


@torch.no_grad()
def evaluate(model, graphs, indices, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for batch in batches(graphs, indices, device):
        logits = model(batch)
        targets = batch.y.reshape(-1)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
    return {"loss": total_loss / len(indices), "accuracy": correct / len(indices)}


def train_fold(graphs, split, n_classes, args, fold):
    seed_everything(args.seed)
    model = GLuck(graphs[0].num_node_features, n_classes, args.layers, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    groups = dict(split["model_selection"][0], test=split["test"])
    best_accuracy, stale = -1.0, 0
    best_state, best_validation, best_epoch = None, None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for batch in batches(graphs, groups["train"], args.device, generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            targets = batch.y.reshape(-1)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.detach().argmax(-1) == targets).sum().item()
        validation = evaluate(model, graphs, groups["validation"], args.device)
        if not np.isfinite(validation["loss"]):
            raise FloatingPointError("Non-finite validation loss")
        # Strict improvement preserves the first checkpoint when accuracy ties.
        if validation["accuracy"] > best_accuracy:
            best_accuracy, stale = validation["accuracy"], 0
            best_state = copy.deepcopy(model.state_dict())
            best_validation, best_epoch = validation, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs or stale >= args.patience:
            print(
                f"Fold {fold} | Epoch {epoch}/{args.epochs} | "
                f"Loss {total_loss / len(groups['train']):.4f} | "
                f"Train {100 * correct / len(groups['train']):.2f}% | "
                f"Val {100 * validation['accuracy']:.2f}% | "
                f"Best val {100 * best_accuracy:.2f}%",
                flush=True,
            )
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    test = evaluate(model, graphs, groups["test"], args.device)
    result = {
        "fold": fold,
        "selected_epoch": best_epoch,
        "epochs_completed": epoch,
        "validation": best_validation,
        "test": test,
    }
    print(
        f"Fold {fold} finished | Best epoch {best_epoch} | "
        f"Validation {100 * best_validation['accuracy']:.2f}% | Test {100 * test['accuracy']:.2f}%",
        flush=True,
    )
    return result
