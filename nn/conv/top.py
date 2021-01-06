from operator import ne, neg
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, BCEWithLogitsLoss
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparse_sum, mul
from torch_geometric.utils import add_remaining_self_loops, negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.nn.inits as tgi
from torch.nn import functional as F


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparse_sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class TOP(nn.Module):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]
    
    def __init__(self, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):

        super(TOP, self).__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        
        self.r_scaling_1, self.r_bias_1 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_2, self.r_bias_2 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_3, self.r_bias_3 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_4, self.r_bias_4 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_5, self.r_bias_5 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        
        self.cache = {
            "num_updated": 0,
            "edge_score": None,  # Use as sij for edge score.
            "edge_label": None,  # Use as label for sij for supervision.
            "new_edge": None,
        }

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

        for name, param in self.named_parameters():
            if name.startswith("r_scaling"):
                tgi.ones(param)
            elif name.startswith("r_bias"):
                tgi.zeros(param)

    def forward(self, x: Tensor, edge_index: Adj,
                        edge_weight: OptTensor = None):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(-2),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(-2),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # Super-GAT
        num_neg_samples = int(edge_index.size(1))

        neg_edge_index = negative_sampling(
                            edge_index=edge_index,
                            num_nodes=x.size(0),
                            num_neg_samples=num_neg_samples,
                        )

        edge_score, edge_label = self._get_edge_and_label_with_negatives(x, edge_index, neg_edge_index)

        new_edge_index, new_edge_weight = self._get_new_edge(edge_score, edge_index, neg_edge_index)

        self._update_cache("edge_score", edge_score)
        self._update_cache("edge_label", edge_label)
        self._update_cache("new_edge", new_edge_index)

        return new_edge_index, new_edge_weight

    def _update_cache(self, key, val):
        self.cache[key] = val
        self.cache["num_updated"] += 1

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
    
    def _get_edge_score(self, x_i, x_j) -> torch.Tensor:
        """
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        """

        edge_score = torch.einsum("ef,ef->e", x_i, x_j)

        edge_score = self.r_scaling_1 * F.elu(edge_score) + self.r_bias_1
        edge_score = self.r_scaling_2 * F.elu(edge_score) + self.r_bias_2
        edge_score = self.r_scaling_3 * F.elu(edge_score) + self.r_bias_3
        edge_score = self.r_scaling_4 * F.elu(edge_score) + self.r_bias_4
        edge_score = self.r_scaling_5 * F.elu(edge_score) + self.r_bias_5

        return edge_score
    
    def _get_edge_and_label_with_negatives(self, x, pos_edge_index, neg_edge_index):
        """
        :param pos_edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]]
        :return: [E + neg_E, 1]
        """

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]

        edge_score = self._get_edge_score(x_i, x_j)
        
        edge_label = torch.zeros_like(edge_score)
        edge_label[:pos_edge_index.size(1)] = 1

        return edge_score, edge_label

    def _get_new_edge(self, edge_score, pos_edge_index, neg_edge_index):

        # edge_mask = edge_score > 0

        # neg_edge_mask = edge_mask[pos_edge_index.size(1):]

        # neg_edge_index = neg_edge_index[:, neg_edge_mask]

        # new_edge_index = neg_edge_index
        # new_edge_weight = None

        neg_edge_score = edge_score[pos_edge_index.size(1):]

        sorted_neg_edge_score, indices = torch.sort(neg_edge_score, descending=True)
        print('sorted_score', sorted_neg_edge_score, sorted_neg_edge_score.shape)
        print('indices', indices, indices.shape)

        target_index = neg_edge_index.clone()
        sorted_index_j = target_index[0].scatter_(src=target_index[0], dim=-1, index=indices)
        print('sorted_index_j', sorted_index_j, sorted_index_j.shape)
        sorted_index_i = target_index[1].scatter_(src=target_index[1], dim=-1, index=indices)
        print('sorted_index_i', sorted_index_i, sorted_index_i.shape)
        sorted_target_index = torch.stack([sorted_index_j, sorted_index_i])
        print('sorted_target_index', sorted_target_index, sorted_target_index.shape)

        new_edge_index = sorted_target_index[:, :1000]
        new_edge_weight = None
        print('new_edge_index', new_edge_index, new_edge_index.shape)

        return new_edge_index, new_edge_weight

    @staticmethod
    def get_link_prediction_loss(model):

        loss_list = []
        cache_list = [(m, m.cache) for m in model.modules() if m.__class__.__name__ == TOP.__name__]

        device = next(model.parameters()).device
        criterion = BCEWithLogitsLoss()
        for i, (module, cache) in enumerate(cache_list):
            # Edge Score (X)
            score = cache["edge_score"]  # [E + neg_E]
            num_total_samples = score.size(0)

            # Edge Labels (Y)
            label = cache["edge_label"]  # [E + neg_E]

            permuted = torch.randperm(num_total_samples)
            permuted = permuted.to(device)
            # print('Link pred loss: label[permuted]', label[permuted], label[permuted].shape)
            # print(label[label>0], label[label>0].shape)
            loss = criterion(score[permuted], label[permuted])
            loss_list.append(loss)
            del permuted

        return sum(loss_list)