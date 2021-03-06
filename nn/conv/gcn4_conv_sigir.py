from operator import ne, neg
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter, BCEWithLogitsLoss
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparse_sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, negative_sampling, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.nn.inits as tgi
from torch.nn import functional as F

from ..inits import glorot, zeros


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


class GCN4ConvSIGIR(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j}
        \frac{1}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`i` to target
    node :obj:`j` (default: :obj:`1`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, alpha=10, beta=-3, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN4ConvSIGIR, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.alpha = alpha
        self.beta = beta
        self.r_scaling_1, self.r_bias_1 = Parameter(
            torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_2, self.r_bias_2 = Parameter(
            torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_3, self.r_bias_3 = Parameter(
            torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_4, self.r_bias_4 = Parameter(
            torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_5, self.r_bias_5 = Parameter(
            torch.Tensor(1)), Parameter(torch.Tensor(1))

        self.cache = {
            "num_updated": 0,
            "edge_score": None,  # Use as sij for edge score.
            "edge_label": None,  # Use as label for sij for supervision.
            "new_edge": None,
            "del_edge": None,
            "total_edge_index": None
        }

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # torch.nn.init.xavier_uniform_(self.a, gain=torch.nn.init.calculate_gain('relu'))

        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

        for name, param in self.named_parameters():
            if name.startswith("r_scaling"):
                tgi.ones(param)
            elif name.startswith("r_bias"):
                tgi.zeros(param)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""

        if edge_index.size(0) != 2:
            out = torch.matmul(x, self.weight)
            out = torch.matmul(edge_index, out)
            if self.bias is not None:
                out += self.bias
            return out

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        wh = torch.matmul(x, self.weight)

        out = self.propagate(edge_index, x=wh, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # PyGAT style
        # print(f'Edge index: {edge_index}, {len(edge_index[0])}')
        # print(f'Neg edge index: {neg_edge_index}, {len(neg_edge_index[0])}')
        # print('x', x, x.shape)
        # s = self._prepare_toptimize_input(out)
        # print('s', s, s.shape)
        # print('self.a', self.a)
        # s = torch.matmul(s, self.a).squeeze(2)
        # print('a * s', s, s.shape)
        # e_new = _prepare_toptimize_input(x)
        # print('e_new', e_new, e_new.shape)
        # print('edge_index', edge_index, edge_index.shape)
        # print('edge_weight', edge_weight, edge_weight.shape)
        # input()

        # Super-GAT
        num_neg_samples = int(edge_index.size(1))
        # num_neg_samples = 10000
        # print('num_neg_samples', num_neg_samples)

        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=wh.size(0),
            num_neg_samples=num_neg_samples,
        )
        # assert torch.any(torch.eq(edge_index, neg_edge_index))

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # edge_score, edge_label, new_edge, del_edge = self._get_new_edge(
        #     wh, edge_index, neg_edge_index)  # OUT
        edge_score, edge_label, new_edge, del_edge, total_edge_index = self._get_new_edge(
            wh, edge_index, neg_edge_index)  # OUT

        # print('x', x, x.shape)
        # print('denser_edge_index', denser_edge_index, denser_edge_index.shape)
        # print('new_edge_index', denser_edge_index, denser_edge_index.shape)
        if self.training:
            self._update_cache("edge_score", edge_score)
            self._update_cache("edge_label", edge_label)
            self._update_cache("new_edge", new_edge)
            # print('cache_new_edge',
            #       self.cache["new_edge"], self.cache["new_edge"].shape)
            # input()
            self._update_cache("del_edge", del_edge)
            self._update_cache("total_edge_index", total_edge_index)

        # # Super-GAT
        # num_neg_samples = int(edge_index.size(1))
        # # print('num_neg_samples', num_neg_samples)

        # neg_edge_index = negative_sampling(
        #     edge_index=edge_index,
        #     num_nodes=x.size(0),
        #     num_neg_samples=num_neg_samples,
        # )
        # # assert torch.any(torch.eq(edge_index, neg_edge_index))

        # # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # edge_score, edge_label, new_edge, del_edge = self._get_new_edge(
        #     out, edge_index, neg_edge_index) #OUT

        # # print(new_edge)
        # # print('x', x, x.shape)
        # # print('denser_edge_index', denser_edge_index, denser_edge_index.shape)
        # # print('new_edge_index', denser_edge_index, denser_edge_index.shape)

        # self._update_cache("edge_score", edge_score)
        # self._update_cache("edge_label", edge_label)
        # self._update_cache("new_edge", new_edge)
        # self._update_cache("del_edge", del_edge)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # edge_score = self._get_edge_score(x_i, x_j)
        # self._update_cache("edge_score", edge_score)
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

        # PyGAT style
        # def _prepare_toptimize_input(self, x):
        #     Wh = torch.clone(x)
        #     N = Wh.size()[0] # number of nodes
        #     Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        #     Wh_repeated_alternating = Wh.repeat(N, 1)
        #     all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        #     return all_combinations_matrix.view(N, N, 2 * 16) #self.out_features)

    def _update_cache(self, key, val):
        if key == "new_edge" and self.cache["new_edge"] == None and len(val[0]) != 0:
            self.cache["new_edge"] = val
        elif key == "new_edge" and self.cache["new_edge"] != None and len(val[0]) != 0:
            prev = self.cache["new_edge"]
            # print('prev', prev)
            # print('prev[0]', prev[0], prev[0].shape)
            # print('val[0]', val[0], val[0].shape)
            a = torch.cat((prev[0], val[0]), dim=-1).unsqueeze(0)
            # print('a', a)
            # print('prev[1]', prev[1], prev[1].shape)
            # print('val[1]', val[1], val[1].shape)
            b = torch.cat((prev[1], val[1]), dim=-1).unsqueeze(0)
            # print('b',b)
            ab = torch.cat((a, b), dim=0)
            # print('ab',ab)
            self.cache["new_edge"] = ab

        if key == "del_edge" and self.cache["del_edge"] == None and len(val[0]) != 0:
            self.cache["del_edge"] = val
        elif key == "del_edge" and self.cache["del_edge"] != None and len(val[0]) != 0:
            prev = self.cache["del_edge"]
            # print('prev', prev)
            # print('prev[0]', prev[0], prev[0].shape)
            # print('val[0]', val[0], val[0].shape)
            a = torch.cat((prev[0], val[0]), dim=-1).unsqueeze(0)
            # print('a', a)
            # print('prev[1]', prev[1], prev[1].shape)
            # print('val[1]', val[1], val[1].shape)
            b = torch.cat((prev[1], val[1]), dim=-1).unsqueeze(0)
            # print('b',b)
            ab = torch.cat((a, b), dim=0)
            # print('ab',ab)
            self.cache["del_edge"] = ab

        if key == "edge_score" or key == "edge_label" or key == "total_edge_index":
            self.cache[key] = val

        # self.cache[key] = val
        self.cache["num_updated"] += 1

    def _get_edge_score(self, x_i, x_j) -> torch.Tensor:
        """
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        """
        # print('torch.cat([x_i, x_j], dim=-1)', torch.cat([x_i, x_j], dim=-1), torch.cat([x_i, x_j], dim=-1).shape)
        # print('self.a', self.a, self.a.shape)

        # [E, F] * [1, F] -> [1, E]
        # edge_score = torch.einsum("ef,xf->e",
        #                 torch.cat([x_i, x_j], dim=-1), # 26517, 32
        #                 self.a) # 1, 32
        edge_score = torch.einsum("ef,ef->e", x_i, x_j)

        # edge_score = torch.matmul(torch.cat([x_i, x_j], dim=-1), self.a)
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        # edge_score = torch.sigmoid(s)

        edge_score = self.r_scaling_1 * F.elu(edge_score) + self.r_bias_1
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        edge_score = self.r_scaling_2 * F.elu(edge_score) + self.r_bias_2
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        edge_score = self.r_scaling_3 * F.elu(edge_score) + self.r_bias_3
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        edge_score = self.r_scaling_4 * F.elu(edge_score) + self.r_bias_4
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        edge_score = self.r_scaling_5 * F.elu(edge_score) + self.r_bias_5
        # print('edge_score', edge_score, edge_score.shape) # 26517, 1
        # edge_score = self.r_scaling_6 * F.elu(edge_score) + self.r_bias_6
        # edge_score = self.r_scaling_7 * F.elu(edge_score) + self.r_bias_7
        # edge_score = self.r_scaling_8 * F.elu(edge_score) + self.r_bias_8
        # edge_score = self.r_scaling_9 * F.elu(edge_score) + self.r_bias_9
        # edge_score = self.r_scaling_10 * F.elu(edge_score) + self.r_bias_10

        return edge_score  # , torch.max(edge_score)

    def _get_new_edge(self, x, edge_index, neg_edge_index):
        """
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]]
        :return: [E + neg_E, 1]
        """
        total_edge_index = torch.cat(
            [edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        # [E + neg_E]
        # total_edge_index_j, total_edge_index_i = total_edge_index
        total_edge_index_i, total_edge_index_j = total_edge_index
        # [E + neg_E, heads * F]
        x_i = torch.index_select(x, 0, total_edge_index_i)
        # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)

        # node_degree = degree(edge_index[0], num_nodes=2708) - 1
        # total_node_degree = node_degree + node_degree

        # print(edge_score, total_node_degree)

        # edge_score = edge_score * total_node_degree
        # print(edge_score)
        # print('edge_score', edge_score)

        edge_score = self._get_edge_score(x_i, x_j)

        edge_label = torch.zeros_like(edge_score)
        edge_label[:edge_index.size(1)] = 1

        edge_mask = edge_score > self.alpha
        edge_mask = edge_mask[edge_index.size(1):]
        new_edge = neg_edge_index[:, edge_mask]

        # torch.set_printoptions(edgeitems=50)
        # node_degree = degree(edge_index[0], num_nodes=2708) - 1
        # Cora 2708, Citeseer 3703
        # mask_by_degree = []
        # for i in range(len(total_edge_index[0])):
        #     if node_degree[total_edge_index[0][i]] > 1:
        #         mask_by_degree.append(1)
        #     else:
        #         mask_by_degree.append(0)
        # mask_by_degree = torch.Tensor(mask_by_degree).cuda()
        # edge_score = torch.mul(mask_by_degree, edge_score)

        # print('edge_score_before', edge_score, edge_score.shape)

        # edge_score_max = edge_score[edge_index.size(1):]
        # alpha = torch.max(edge_score_max)
        # edge_mask = edge_score_max >= alpha
        # new_edge = neg_edge_index[:, edge_mask]

        # alpha = max(max_edge_score/1.5, self.alpha) - 0.0001
        # print(alpha, max_edge_score/1.5, self.alpha)
        # input()
        # edge_mask = edge_score > alpha

        # print(new_edge, new_edge.shape)
        # input()

        # new_edge_0 = new_edge[0]
        # new_edge_1 = new_edge[1]

        # if new_edge.shape[1] != 0:
        #     for i in range(len(new_edge[0])):
        #         if total_node_degree[new_edge[0][i]] < 2:
        #             new_edge_0[i] = -1
        #             new_edge_1[i] = -1
        #     if -1 in new_edge[0]:
        #         new_edge_0 = new_edge_0[new_edge_0 != -1]
        #         new_edge_1 = new_edge_1[new_edge_1 != -1]
        #         new_edge = torch.stack((new_edge_0, new_edge_1), dim=0)

        edge_mask = edge_score < self.beta
        edge_mask = edge_mask[:edge_index.size(1)]
        del_edge = edge_index[:, edge_mask]

        return edge_score, edge_label, new_edge, del_edge, total_edge_index

    # def _get_new_edge(self, x, edge_index, neg_edge_index):
    #     """
    #     :param edge_index: [2, E]
    #     :param neg_edge_index: [2, neg_E]]
    #     :return: [E + neg_E, 1]
    #     """

    #     total_edge_index = torch.cat(
    #         [edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]
    #     # print('edge_index', edge_index, edge_index.shape)
    #     # print('neg_edge_index', neg_edge_index, neg_edge_index.shape)
    #     # print('total_edge_index', total_edge_index, total_edge_index.shape)

    #     # [E + neg_E]
    #     total_edge_index_j, total_edge_index_i = total_edge_index
    #     # [E + neg_E, heads * F]

    #     x_i = torch.index_select(x, 0, total_edge_index_i)

    #     # [E + neg_E, heads * F]
    #     x_j = torch.index_select(x, 0, total_edge_index_j)

    #     edge_score = self._get_edge_score(x_i, x_j)

    #     edge_label = torch.zeros_like(edge_score)
    #     edge_label[:edge_index.size(1)] = 1
    #     # print('edge_label', edge_label, edge_label.shape)

    #     ###### SORTING #######
    #     # Forcing maximum {x} edges per step
    #     max_num = 2
    #     neg_edge_score = edge_score[edge_index.size(1):]
    #     _, sorted_indices = torch.sort(neg_edge_score, descending=True)
    #     new_edge = neg_edge_index[:, sorted_indices[:max_num]]
    #     new_edge_score = neg_edge_score[sorted_indices[:max_num]]

    #     ###### THRESHOLD #######
    #     edge_mask = new_edge_score > self.alpha
    #     new_edge = new_edge[:, edge_mask]
    #     # if new_edge.size(1) != 0:
    #     #     print('1', neg_edge_index, neg_edge_index.shape)
    #     #     print('0', neg_edge_score, neg_edge_score.shape)
    #     #     print('1', sorted_indices, sorted_indices.shape)
    #     #     print('2', sorted_indices[:50], sorted_indices[:50].shape)
    #     #     print('3', new_edge, new_edge.shape)
    #     #     print('4', new_edge_score, new_edge_score.shape)
    #     #     print('5', edge_mask, edge_mask.shape)
    #     #     print('6', new_edge, new_edge.shape)
    #     #     input()

    #     edge_mask = edge_score < self.beta
    #     edge_mask = edge_mask[:edge_index.size(1)]
    #     del_edge = edge_index[:, edge_mask]

    #     return edge_score, edge_label, new_edge, del_edge

    @ staticmethod
    def loss(model):

        loss_list = []
        cache_list = [(m, m.cache) for m in model.modules()
                      if m.__class__.__name__ == GCN4ConvSIGIR.__name__]

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
            # print(score, score.shape, num_total_samples,permuted, permuted.shape)
            # input()
            # print('label[--permuted]', label[permuted], label[permuted].shape)
            # print(label[label>0], label[label>0].shape)

            loss = criterion(score[permuted], label[permuted])
            loss_list.append(loss)
            del permuted

        return sum(loss_list)

    # @staticmethod
    # def add_link_prediction_loss(task_loss, model):
    #     link_loss =  1.0 * GCN3Conv.get_link_prediction_loss(
    #         model=model
    #     )
    #     print('Link loss', link_loss)
    #     return task_loss + link_loss
