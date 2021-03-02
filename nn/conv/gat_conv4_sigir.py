from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
from torch.nn import Parameter, BCEWithLogitsLoss
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, negative_sampling
import torch_geometric.nn.inits as tgi

from ..inits import glorot, zeros


class GAT4ConvSIGIR(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.6,
                 add_self_loops: bool = True, bias: bool = True, alpha=10, beta=-3, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GAT4ConvSIGIR, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.beta = beta
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
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
        # self.r_scaling_6, self.r_bias_6 = Parameter(
        #     torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_7, self.r_bias_7 = Parameter(
        #     torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_8, self.r_bias_8 = Parameter(
        #     torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_9, self.r_bias_9 = Parameter(
        #     torch.Tensor(1)), Parameter(torch.Tensor(1))
        # self.r_scaling_10, self.r_bias_10 = Parameter(
        #     torch.Tensor(1)), Parameter(torch.Tensor(1))

        self.cache = {
            "num_updated": 0,
            "edge_score": None,  # Use as sij for edge score.
            "edge_label": None,  # Use as label for sij for supervision.
            "new_edge": None,
        }

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        # glorot(self.weight)
        zeros(self.bias)

        for name, param in self.named_parameters():
            if name.startswith("r_scaling"):
                tgi.ones(param)
            elif name.startswith("r_bias"):
                tgi.zeros(param)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        # wh = torch.matmul(x, self.weight)
        # x2 = x.clone()
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            # wh = self.lin_l(x).view(-1, H, C).clone()
            # x_l = x_r = F.dropout(x, p=0.6, training=self.training)
            x_l = x_r = self.lin_l(x).view(-1, H, C)  # Wh 2708, 8, 8
            # assert (x_l == x_r).all()
            wh = x_l.clone()  # 2708, 8, 8
            # awhi 1, 8, 8 -> 2708, 8, 8 -> 2708, 8
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)  # awhj
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        # Super-GAT
        num_neg_samples = int(edge_index.size(1))
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=num_neg_samples,
        )
        # assert torch.any(torch.eq(edge_index, neg_edge_index))

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        edge_score, edge_label, new_edge, del_edge = self._get_new_edge(
            wh, edge_index, neg_edge_index)  # OUT

        # print('x', x, x.shape)
        # print('denser_edge_index', denser_edge_index, denser_edge_index.shape)
        # print('new_edge_index', denser_edge_index, denser_edge_index.shape)

        self._update_cache("edge_score", edge_score)
        self._update_cache("edge_label", edge_label)
        self._update_cache("new_edge", new_edge)
        self._update_cache("del_edge", del_edge)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

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
        # print('xi', x_i)
        # print('xj', x_j)
        # input()
        # print('torch.cat([x_i, x_j], dim=-1)', torch.cat([x_i, x_j], dim=-1), torch.cat([x_i, x_j], dim=-1).shape)
        # print('self.a', self.a, self.a.shape)

        # [E, F] * [1, F] -> [1, E]
        # edge_score = torch.einsum("ef,xf->e",
        #                 torch.cat([x_i, x_j], dim=-1), # 26517, 32
        #                 self.a) # 1, 32

        edge_score = torch.einsum("ehf,ehf->e", x_i, x_j)

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

        total_edge_index = torch.cat(
            [edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        # [E + neg_E]
        total_edge_index_j, total_edge_index_i = total_edge_index
        # [E + neg_E, heads * F]
        x_i = torch.index_select(x, 0, total_edge_index_i)
        # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)

        edge_score = self._get_edge_score(x_i, x_j)

        edge_label = torch.zeros_like(edge_score)
        edge_label[:edge_index.size(1)] = 1

        edge_mask = edge_score > self.alpha
        edge_mask = edge_mask[edge_index.size(1):]
        new_edge = neg_edge_index[:, edge_mask]

        edge_mask = edge_score < self.beta
        edge_mask = edge_mask[:edge_index.size(1)]
        del_edge = edge_index[:, edge_mask]

        return edge_score, edge_label, new_edge, del_edge

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
    #     max_num = 20
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

    @staticmethod
    def loss(model):

        loss_list = []
        cache_list = [(m, m.cache) for m in model.modules()
                      if m.__class__.__name__ == GAT4ConvSIGIR.__name__]

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
