import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONTEXT, TC, TOKEN


def segment_softmax(scores, index, size):
    maxima = scores.new_full((size,), -float("inf"))
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[index])
    totals = scores.new_zeros(size)
    totals.index_add_(0, index, weights)
    return weights / totals[index].clamp_min(1e-12)


class UpdateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class AdditiveAttentionScore(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.context_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.token_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, pair):
        context, token = pair.chunk(2, dim=-1)
        return self.score(torch.tanh(self.context_proj(context) + self.token_proj(token)))


class ContextualizedTokenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.update_mlp = UpdateMLP(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.sc_weight_mlp = AdditiveAttentionScore(input_dim, hidden_dim)

    def edge_weights(self, h, edge_index, edge_type):
        src, dst = edge_index
        weights = h.new_ones(edge_index.size(1), 1)
        token_context = edge_type == TC
        token_src, context_dst = src[token_context], dst[token_context]
        pair = torch.cat([h[context_dst], h[token_src]], dim=-1)
        scores = self.sc_weight_mlp(pair).view(-1)
        weights[token_context] = segment_softmax(scores, context_dst, h.size(0)).unsqueeze(-1)
        return weights

    def forward(self, h, edge_index, edge_type):
        src, dst = edge_index
        weights = self.edge_weights(h, edge_index, edge_type)
        token_context = edge_type == TC
        fixed = ~token_context
        indices = torch.stack([dst[fixed], src[fixed]])
        adjacency = torch.sparse_coo_tensor(
            indices, h.new_ones(indices.size(1)), size=(h.size(0), h.size(0))
        ).coalesce()
        pooled = torch.sparse.mm(adjacency, h)

        # Only token-to-context messages receive learned weights (Eq. 5).
        pooled.index_add_(0, dst[token_context], h[src[token_context]] * weights[token_context])
        return torch.relu(self.bn(self.update_mlp(pooled + h)))


class GLuck(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=5, dropout=0.3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers includes the input and must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = dropout
        # Experiment depth includes the input: D means D-1 message-passing steps.
        dims = [num_features] + [64] * (num_layers - 1)
        self.gnn_layers = nn.ModuleList(
            ContextualizedTokenLayer(dims[i], 64) for i in range(num_layers - 1)
        )
        self.context_jk_heads = nn.ModuleList(
            nn.Linear(dim, num_classes) for dim in dims
        )

    @staticmethod
    def context_readout(h, context_mask, context_batch, n_graphs):
        context_h = h[context_mask]
        output = h.new_zeros(n_graphs, h.size(-1))
        output.index_add_(0, context_batch, context_h)
        counts = h.new_zeros(n_graphs, 1)
        counts.index_add_(0, context_batch, h.new_ones(context_h.size(0), 1))
        return output / counts.clamp_min(1)

    def forward(self, data):
        token_mask = data.node_type == TOKEN
        context_mask = data.node_type == CONTEXT
        token_batch = data.batch[token_mask]
        context_batch = data.batch[context_mask]
        n_graphs = data.num_graphs
        h = data.x.clone()
        means = h.new_zeros(n_graphs, h.size(-1))
        counts = h.new_zeros(n_graphs, 1)
        means.index_add_(0, token_batch, data.x[token_mask])
        counts.index_add_(0, token_batch, h.new_ones(token_batch.size(0), 1))
        h[context_mask] = (means / counts.clamp_min(1))[context_batch]

        context_states = [self.context_readout(h, context_mask, context_batch, n_graphs)]
        for layer in self.gnn_layers:
            h = layer(h, data.edge_index, data.edge_type)
            context_states.append(self.context_readout(h, context_mask, context_batch, n_graphs))
        logits = [
            F.dropout(head(state), p=self.dropout, training=self.training)
            for head, state in zip(self.context_jk_heads, context_states)
        ]
        return torch.stack(logits).sum(0)
