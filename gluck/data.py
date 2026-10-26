import os
from pathlib import Path
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data

from .tokenizer import build_multilevel_graph


DATASETS = ("IMDB-BINARY", "PROTEINS", "MUTAG", "PTC_MR", "NCI1", "NCI109", "Mutagenicity")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SPLIT_SEED = 0


def ensure_dataset(dataset, root):
    raw_root = Path(root) / "raw"
    directory = raw_root / dataset
    suffixes = ["A", "graph_indicator", "graph_labels"]
    if dataset != "IMDB-BINARY":
        suffixes.append("node_labels")
    names = [f"{dataset}_{suffix}.txt" for suffix in suffixes]
    if all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in names):
        return directory / dataset

    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset}.zip"
    print(f"Downloading {dataset} from {url}", flush=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=raw_root) as temporary:
            temporary = Path(temporary)
            archive_path = temporary / "dataset.zip"
            with urlopen(url, timeout=60) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
            # Only the files used by this model are extracted.
            with zipfile.ZipFile(archive_path) as archive:
                for name in names:
                    with archive.open(f"{dataset}/{name}") as source, (temporary / name).open("wb") as target:
                        shutil.copyfileobj(source, target)
            directory.mkdir(exist_ok=True)
            for name in names:
                os.replace(temporary / name, directory / name)
    except (URLError, TimeoutError, zipfile.BadZipFile, KeyError) as error:
        raise RuntimeError(
            f"Could not download {dataset} from {url}. "
            "Check your internet connection and run the command again."
        ) from error
    return directory / dataset


def generate_splits(labels):
    indices = np.arange(len(labels))
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=SPLIT_SEED)
    splits = []
    for trainval, test in outer.split(indices, labels):
        validation_size = max(1, int(round(0.1 * len(trainval))))
        inner_folds = max(2, min(10, len(trainval) // validation_size))
        inner = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=SPLIT_SEED
        )
        train, validation = next(inner.split(trainval, labels[trainval]))
        splits.append({
            "test": indices[test].tolist(),
            "model_selection": [{
                "train": trainval[train].tolist(),
                "validation": trainval[validation].tolist(),
            }],
        })
    return splits


def validate_splits(splits, n_graphs):
    if len(splits) != 10:
        raise ValueError("Expected 10 outer folds")
    all_test = []
    for split in splits:
        if len(split["model_selection"]) != 1:
            raise ValueError("Expected one fixed validation split per fold")
        groups = dict(split["model_selection"][0], test=split["test"])
        for indices in groups.values():
            if not indices or any(type(i) is not int or i < 0 or i >= n_graphs for i in indices):
                raise ValueError("Invalid split indices")
        joined = groups["train"] + groups["validation"] + groups["test"]
        if sorted(joined) != list(range(n_graphs)):
            raise ValueError("Train, validation and test must partition the graphs")
        all_test.extend(groups["test"])
    if sorted(all_test) != list(range(n_graphs)):
        raise ValueError("Test folds must cover each graph exactly once")


def load_raw_dataset(dataset, root=DATA_ROOT):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    prefix = ensure_dataset(dataset, root)

    def read(suffix, **kwargs):
        return np.loadtxt(f"{prefix}_{suffix}.txt", **kwargs)

    indicator = np.atleast_1d(read("graph_indicator", dtype=np.int64))
    ids, sizes = np.unique(indicator, return_counts=True)
    if not np.array_equal(ids, np.arange(1, len(sizes) + 1)) or np.any(np.diff(indicator) < 0):
        raise ValueError("Expected contiguous TU graph ordering")
    edges = np.atleast_2d(read("A", dtype=np.int64, delimiter=",")) - 1
    if np.any(indicator[edges[:, 0]] != indicator[edges[:, 1]]):
        raise ValueError("Found an edge between different graphs")
    adjacency = csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(indicator), len(indicator)),
    )
    values = np.asarray(adjacency.sum(1)) if dataset == "IMDB-BINARY" else read("node_labels", dtype=np.int64).reshape(-1, 1)
    features = OneHotEncoder(sparse_output=False).fit_transform(values)
    labels = LabelEncoder().fit_transform(np.atleast_1d(read("graph_labels", dtype=np.int64)))
    if len(labels) != len(sizes):
        raise ValueError("Graph-label count does not match graph indicators")
    splits = generate_splits(labels)
    validate_splits(splits, len(labels))

    graphs, offset = [], 0
    for i, size in enumerate(sizes):
        graph = nx.from_scipy_sparse_array(adjacency[offset:offset + size, offset:offset + size])
        graph.remove_edges_from(nx.selfloop_edges(graph))
        directed_edges = list(graph.to_directed().edges())
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous() if directed_edges else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(
            x=torch.tensor(features[offset:offset + size], dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor([labels[i]], dtype=torch.long),
            num_nodes=int(size),
        ))
        offset += size
    return graphs, splits, len(np.unique(labels))


def load_dataset(dataset):
    graphs, splits, n_classes = load_raw_dataset(dataset)
    return [build_multilevel_graph(g, dataset) for g in graphs], splits, n_classes
