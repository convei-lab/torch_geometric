from operator import ne, neg
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter, BCEWithLogitsLoss
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparse_sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, negative_sampling
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
                 bias: bool = True, **kwargs):

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

        self.a = Parameter(torch.Tensor(1, 2*out_channels))
        self.r_scaling_1, self.r_bias_1 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_2, self.r_bias_2 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_3, self.r_bias_3 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_4, self.r_bias_4 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        self.r_scaling_5, self.r_bias_5 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_6, self.r_bias_6 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_7, self.r_bias_7 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_8, self.r_bias_8 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_9, self.r_bias_9 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_10, self.r_bias_10 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))

        self.cache = {
            "num_updated": 0,
            "edge_score": None,  # Use as sij for edge score.
            "edge_label": None,  # Use as label for sij for supervision.
            "new_edge": None,
        }

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # torch.nn.init.xavier_uniform_(self.a, gain=torch.nn.init.calculate_gain('relu'))
        
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

        glorot(self.a)
        for name, param in self.named_parameters():
            if name.startswith("r_scaling"):
                tgi.ones(param)
            elif name.startswith("r_bias"):
                tgi.zeros(param)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""

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

        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
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

        if self.training:
            # Super-GAT
            num_neg_samples = int(edge_index.size(1))
            # print('num_neg_samples', num_neg_samples)

            neg_edge_index = negative_sampling(
                                edge_index=edge_index,
                                num_nodes=x.size(0),
                                num_neg_samples=num_neg_samples,
                            )
            # assert torch.any(torch.eq(edge_index, neg_edge_index))

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            edge_score, edge_label, new_edge, del_edge = self._get_new_edge(x, edge_index, neg_edge_index)

            # print('x', x, x.shape)
            # print('denser_edge_index', denser_edge_index, denser_edge_index.shape)
            # print('new_edge_index', denser_edge_index, denser_edge_index.shape)
            
            self._update_cache("edge_score", edge_score)
            self._update_cache("edge_label", edge_label)
            self._update_cache("new_edge", new_edge)
            self._update_cache("del_edge", del_edge)

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

        '''epoch 마다 추가
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
            ab = torch.cat((a,b), dim=0)
            # print('ab',ab)
            self.cache["new_edge"] = ab

        if key == "edge_score" or key == "edge_label":
            self.cache[key] = val
        '''    
        self.cache[key] = val
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

        return edge_score
    
    def _get_new_edge(self, x, edge_index, neg_edge_index):
        """
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]]
        :return: [E + neg_E, 1]
        """

        total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]
        # print('edge_index', edge_index, edge_index.shape)
        # print('neg_edge_index', neg_edge_index, neg_edge_index.shape)
        # print('total_edge_index', total_edge_index, total_edge_index.shape)

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]

        edge_score = self._get_edge_score(x_i, x_j)
        
        edge_label = torch.zeros_like(edge_score)
        edge_label[:edge_index.size(1)] = 1
        # print('edge_label', edge_label, edge_label.shape)

        edge_mask = edge_score > 10
        edge_mask = edge_mask[edge_index.size(1):]
        new_edge = neg_edge_index[:, edge_mask]

        edge_mask = edge_score < -3
        edge_mask = edge_mask[:edge_index.size(1)]
        del_edge = edge_index[:, edge_mask]

        return edge_score, edge_label, new_edge, del_edge

    @staticmethod
    def get_link_prediction_loss(model):

        loss_list = []
        cache_list = [(m, m.cache) for m in model.modules() if m.__class__.__name__ == GCN4ConvSIGIR.__name__]

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
