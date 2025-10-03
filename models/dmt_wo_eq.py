import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from . import utils
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse
from .layers import *
from . import utils
# DiffSpectra models import
from models.specformer import SpecFormer
# disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0


def coord2dist(x, edge_index):
    # coordinates to distance
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    return radial

def remove_mean(pos, batch):
    mean_pos = scatter(pos, batch, dim=0, reduce="mean")  # shape = [B, 3]
    pos = pos - mean_pos[batch]
    return pos

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class LearnedSinusoidalposEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class TransLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(TransLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.proj.reset_parameters()

    def forward(
        self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    # @torch.compile(dynamic=True, disable=disable_compile)
    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(
            self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels)
        )
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class TransLayerOptimV2(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(TransLayerOptimV2, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_qkv = nn.Linear(in_channels, heads * out_channels * 3, bias=bias)
        self.lin_kv_e = nn.Linear(edge_dim, heads * out_channels * 2, bias=False)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.lin_kv_e.reset_parameters()
        self.proj.reset_parameters()

    def forward(
        self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        """"""
        x_feat = x

        qkv = self.lin_qkv(x_feat).view(-1, self.heads, 3, self.out_channels)
        query, key, value = qkv.unbind(dim=2)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr
        )

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        query_i: [N, heads, out_channels]
        key_j: [N, heads, out_channels]
        value_j: [N, heads, out_channels]
        """
        edge_qkv = self.lin_kv_e(edge_attr).view(-1, self.heads, 2, self.out_channels)
        edge_key_ij, edge_value_ij = edge_qkv.unbind(
            dim=2
        )  # shape [N, heads, out_channels]

        query_ij = query_i
        edge_key_ij = key_j + edge_key_ij
        edge_value_ij = value_j + edge_value_ij

        alpha_ij = (query_ij * edge_key_ij).sum(dim=-1) / math.sqrt(
            self.out_channels
        )  # shape [N * N, heads]
        alpha_ij = softmax(alpha_ij, index, ptr, size_i)
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        # node feature message
        msg = edge_value_ij * alpha_ij.view(
            -1, self.heads, 1
        )  # shape [N * N, heads, out_channels]
        return msg

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class TransLayerOptim(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(TransLayerOptim, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_qkv = nn.Linear(in_channels, heads * out_channels * 3, bias=bias)

        self.lin_edge = nn.Linear(edge_dim, heads * out_channels * 2, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.lin_edge.reset_parameters()
        self.proj.reset_parameters()

    def forward(
        self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels
        x_feat = x
        qkv = self.lin_qkv(x_feat).view(-1, H, 3, C)
        query, key, value = qkv.unbind(dim=2)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    # @torch.compile(dynamic=True, disable=disable_compile)
    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tuple[Tensor, Tensor]:

        edge_key, edge_value = (
            torch.tanh(self.lin_edge(edge_attr))
            .view(-1, self.heads, 2, self.out_channels)
            .unbind(dim=2)
        )

        alpha = (query_i * key_j * edge_key).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j * edge_value * alpha.view(-1, self.heads, 1)
        return msg

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""

    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class DMT_WO_EQ_Block(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(
        self,
        node_dim,
        edge_dim,
        time_dim,
        num_heads,
        cond_time=True,
        mlp_ratio=4,
        act=nn.GELU,
        dropout=0.0,
        pair_update=True,
        trans_ver="v1",
    ):
        super().__init__()
        self.dropout = dropout
        self.act = act()
        self.cond_time = cond_time
        self.pair_update = pair_update

        if not self.pair_update:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * 2),
                nn.GELU(),
                nn.Linear(edge_dim * 2, edge_dim),
                nn.LayerNorm(edge_dim),
            )

        # message passing layer
        if trans_ver == "v1":
            self.attn_mpnn = TransLayer(
                node_dim,
                node_dim // num_heads,
                num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
        else:
            self.attn_mpnn = TransLayerOptimV2(
                node_dim,
                node_dim // num_heads,
                num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)

        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim * 2, edge_dim)
            # Feed forward block -> edge.
            self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
            self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)

        # equivariant edge update layer
        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_dim, node_dim * 6)
            )
            # Normalization for MPNN
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(), nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=False, eps=1e-6
                )
                self.norm2_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=False, eps=1e-6
                )
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=True, eps=1e-6
                )
                self.norm2_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=True, eps=1e-6
                )

    def _ff_block_node(self, x):
        x = F.dropout(
            self.act(self.ff_linear1(x)), p=self.dropout, training=self.training
        )
        return F.dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x):
        x = F.dropout(
            self.act(self.ff_linear3(x)), p=self.dropout, training=self.training
        )
        return F.dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward(self, h, edge_attr, edge_index, node_time_emb=None, edge_time_emb=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
        """
        h_in_node = h
        h_in_edge = edge_attr

        if self.cond_time:
            ## prepare node features
            (
                node_shift_msa,
                node_scale_msa,
                node_gate_msa,
                node_shift_mlp,
                node_scale_mlp,
                node_gate_mlp,
            ) = self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)

            ## prepare edge features
            if self.pair_update:
                (
                    edge_shift_msa,
                    edge_scale_msa,
                    edge_gate_msa,
                    edge_shift_mlp,
                    edge_scale_mlp,
                    edge_gate_mlp,
                ) = self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(
                    self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa
                )
            else:
                edge_attr = self.edge_emb(edge_attr)

            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            h_node = self.attn_mpnn(h, edge_index, edge_attr)

            ## update edge features
            h_out = self.node_update(
                h_in_node,
                h_node,
                node_gate_msa,
                node_shift_mlp,
                node_scale_mlp,
                node_gate_mlp,
            )
            if self.pair_update:
                # h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
                h_edge = torch.cat(
                    [h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1
                )
                h_edge_out = self.edge_update(
                    h_in_edge,
                    h_edge,
                    edge_gate_msa,
                    edge_shift_mlp,
                    edge_scale_mlp,
                    edge_gate_mlp,
                )
            else:
                h_edge_out = h_in_edge
        else:
            ## prepare node features
            h = self.norm1_node(h)
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)
            else:
                edge_attr = self.edge_emb(edge_attr)

            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            h_node = self.attn_mpnn(h, edge_index, edge_attr)

            # update node features
            h_out = self.node_update(h_in_node, h_node)

            # update edge features
            if self.pair_update:
                # h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
                h_edge = torch.cat(
                    [h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1
                )
                h_edge_out = self.edge_update(h_in_edge, h_edge)
            else:
                h_edge_out = h_in_edge
        return h_out, h_edge_out

    # @torch.compile(dynamic=True, disable=disable_compile)
    def node_update(
        self,
        h_in_node,
        h_node,
        node_gate_msa=None,
        node_shift_mlp=None,
        node_scale_mlp=None,
        node_gate_mlp=None,
    ):
        h_node = (
            h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        )
        _h_node = (
            modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp)
            if self.cond_time
            else self.norm2_node(h_node)
        )
        h_out = (
            h_node + node_gate_mlp * self._ff_block_node(_h_node)
            if self.cond_time
            else h_node + self._ff_block_node(_h_node)
        )
        return h_out

    # @torch.compile(dynamic=True, disable=disable_compile)
    def edge_update(
        self,
        h_in_edge,
        h_edge,
        edge_gate_msa=None,
        edge_shift_mlp=None,
        edge_scale_mlp=None,
        edge_gate_mlp=None,
    ):
        h_edge = self.node2edge_lin(h_edge)
        h_edge = (
            h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        )
        _h_edge = (
            modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp)
            if self.cond_time
            else self.norm2_edge(h_edge)
        )
        h_edge_out = (
            h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge)
            if self.cond_time
            else h_edge + self._ff_block_edge(_h_edge)
        )
        return h_edge_out


class NodeEmbed(nn.Module):
    def __init__(
        self,
        in_node_features,
        hidden_size
    ):
        super().__init__()
        self.x_linear = nn.Linear(in_node_features, hidden_size * 2)
        self.pos_linear = nn.Linear(3, hidden_size * 2)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, x, pos):
        x = self.x_linear(x)
        pos = self.pos_linear(pos)
        return self.mlp(x + pos)


@utils.register_model(name='DMT_WO_EQ')
class DMT_WO_EQ(nn.Module):
    """Diffusion Graph Transformer with DMT blocks for DiffSpectra models."""

    def __init__(self, config):
        super().__init__()

        in_node_dim = config.data.atom_types + int(config.model.include_fc_charge)  # 5+1 (atom types + formal charge)
        hidden_dim = config.model.nf                                    # 256
        edge_hidden_dim = config.model.nf // 4                          # 256 // 4 = 64
        n_heads = config.model.n_heads                                  # 16
        dropout = config.model.dropout                                  # 0.1
        self.dist_gbf = dist_gbf = config.model.dist_gbf                # True
        gbf_name = config.model.gbf_name                                # 'CondGaussianLayer'
        self.edge_th = config.model.edge_quan_th                        # 0.0  edge quantization threshold
        n_extra_heads = config.model.n_extra_heads                      # 2  number of attention heads from adjacency matrix
        self.CoM = config.model.CoM                                     # True  keep output of each layer at CoM
        mlp_ratio = config.model.mlp_ratio                              # 2 FFN channel ratio
        self.spatial_cut_off = config.model.spatial_cut_off             # 2.0  distance threshold for spatial adjacency matrix
        self.trans_ver = config.model.trans_ver if hasattr(config.model, "trans_ver") else "v2"         # can directly modify TransLayerOptimV2 in config as needed               # True

        if dist_gbf:
            dist_dim = edge_hidden_dim
        else:
            dist_dim = 1
        in_edge_dim = config.model.edge_ch * 2 + dist_dim           # 2*2+64 = 68
        self.cond_time = cond_time = config.model.cond_time         # True
        self.n_layers = n_layers = config.model.n_layers            # 8
        self.pred_data = config.model.pred_data                     # True  predict data instead of noise
        time_dim = hidden_dim * 4                                   # 256 * 4 = 1024
        self.dist_dim = dist_dim                                    # 64

        # Initial encoding layer
        # self.node_emb = nn.Linear(in_node_dim * 2+3, hidden_dim)      # 15 -> 256
        self.node_emb = NodeEmbed(in_node_dim * 2, hidden_dim)
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)     # 68 -> 64

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

        # Output dimension for each layer
        cat_node_dim = (hidden_dim * 2) // n_layers
        cat_edge_dim = (edge_hidden_dim * 2) // n_layers

        # Create DMT modules
        for i in range(n_layers):
            self.add_module("dmt_block_%d" % i, DMT_WO_EQ_Block(
                hidden_dim, edge_hidden_dim, time_dim, n_heads, 
                cond_time=cond_time, mlp_ratio=mlp_ratio, 
                act=nn.GELU, dropout=dropout, 
                pair_update=True, trans_ver=self.trans_ver))
            self.add_module("node_%d" % i, nn.Linear(hidden_dim, cat_node_dim))
            self.add_module("edge_%d" % i, nn.Linear(edge_hidden_dim, cat_edge_dim))

        # Prediction module
        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim)
        )
        
        self.pos_pred_mlp = nn.Sequential(
                    nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim, bias=False),
                    # nn.SiLU(),
                    # nn.Linear(hidden_dim, hidden_dim // 2),
                    # nn.SiLU(),
                    # nn.Linear(hidden_dim // 2, 3)  # predict 3D coordinates
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 3, bias=False)  # predict 3D coordinates
        )

        self.edge_type_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, config.model.edge_ch - 1)
        )

        self.edge_exist_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, 1)
        )


        # Time encoding
        if cond_time:
            learned_dim = 16
            sinu_pos_emb = LearnedSinusoidalposEmb(learned_dim)
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(learned_dim + 1, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

        # Spectral condition encoder
        self.spectra_version = config.data.spectra_version
        self.cond_encoder = SpecFormer(patch_len=config.model.patch_len, 
                                        stride=config.model.stride, 
                                        output_dim=hidden_dim, 
                                        spectra_version=config.data.spectra_version)
        self.cond_lin = nn.Linear(hidden_dim, time_dim)

        # Load pretrained SpecFormer weights
        if hasattr(config.model, 'pretrained_specformer_path') and config.model.pretrained_specformer_path:
            print("Load pretrained SpecFormer")
            self.load_pretrained_specformer(config.model.pretrained_specformer_path)
        else:
            print("Train SpecFormer from scratch")

    def load_pretrained_specformer(self, ckpt_path):
        """Load pretrained SpecFormer weights into current model"""
        print(f"Loading pretrained SpecFormer from {ckpt_path}")
        
        # Load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        
        if 'state_dict' not in ckpt:
            print(f"Error: No state_dict key in checkpoint!")
            return
        
        state_dict = ckpt['state_dict']
        current_dict = self.cond_encoder.state_dict()
        matched_keys = 0
        
        # Find correct prefix
        prefix = None
        for possible_prefix in ['model.representation_spec_model', 'model.representation_model']:
            if any(k.startswith(possible_prefix) for k in state_dict.keys()):
                prefix = possible_prefix
                print(f"Found matching prefix: {prefix}")
                break
        
        if prefix is None:
            print("Unable to find matching model prefix")
            return
        
        # Map key names and exclude mismatched shapes
        for target_key in current_dict.keys():
            # Try direct matching
            source_key = f"{prefix}.{target_key}"
            
            
            if target_key == "out_norm.weight" or target_key == "out_norm.bias":
                source_key = f"model.representation_model.out_norm.{target_key.split('.')[-1]}"
            
            if source_key in state_dict and current_dict[target_key].shape == state_dict[source_key].shape:
                current_dict[target_key] = state_dict[source_key]
                matched_keys += 1
                print(f"Successfully matched: {target_key} <- {source_key}")
        
        # Load updated state dict
        if matched_keys > 0:
            self.cond_encoder.load_state_dict(current_dict, strict=False)
            print(f"\nTotal loaded {matched_keys}/{len(current_dict)} keys ({matched_keys/len(current_dict)*100:.1f}%)")
        else:
            print("\nWarning: No weights loaded successfully!")

    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        """
        Parameters
        ----------
        t: [B] time steps in [0, 1]
        xh: [B, N, ch1] atom feature (positions, types, formal charges)
        node_mask: [B, N, 1]
        edge_mask: [B*N*N, 1]
        context: [B, 1, spectrum_length] spectral data
        kwargs: 'edge_x' [B, N, N, ch2]

        Returns
        -------
        atom_features: [B, N, 3+atom_types]
        edge_features: [B, N, N, edge_types]
        """
        edge_x, cond_x, cond_edge_x = kwargs['edge_x'], kwargs['cond_x'], kwargs['cond_edge_x']
        bs, n_nodes, dims = xh.shape
        pos_init = xh[:, :, 0:3].clone()  # save original positions [B, N, 3]
        h_feat = xh[:, :, 3:].clone()  # node features [B, N, feat_dim]

        # alpha_t, sigma_t = kwargs['alpha_t'], kwargs['sigma_t']  # get noise parameters

        # Build adjacency matrix
        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)

        # Process conditional structure features
        if cond_x is None:
            cond_x = torch.zeros_like(xh)
            cond_edge_x = torch.zeros_like(edge_x)
            cond_adj_2d = torch.ones((edge_index.size(1), 1), device=edge_x.device)
        else:
            with torch.no_grad():
                cond_adj_2d = cond_edge_x[dense_index][:, 0:1].clone()
                cond_adj_2d[cond_adj_2d >= self.edge_th] = 1.
                cond_adj_2d[cond_adj_2d < self.edge_th] = 0.

        # concat self_cond node feature
        cond_pos = cond_x[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        cond_h = cond_x[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        node_inputs = torch.cat([
            h_feat.reshape(bs * n_nodes, -1),  # original node features
            cond_h,                            # conditional node features
            # pos_init.reshape(bs * n_nodes, -1) # position features
        ], dim=-1)
        # h = self.node_emb(node_inputs)  # node embedding [bs*n_nodes, hidden_dim]
        pos_init = pos_init.reshape(bs * n_nodes, -1)
        h = self.node_emb(node_inputs, pos_init)


        if context is not None:
            # Process spectral condition data
            context = self.cond_encoder(context)
            context = self.cond_lin(context)  # [B, time_dim]

        # Time encoding
        if self.cond_time:
            noise_level = kwargs['noise_level']
            time_emb = self.time_mlp(noise_level)  # [B, time_dim]
            if context is not None:
                time_emb = time_emb + context     # add spectral condition to time encoding
            node_time_emb = time_emb.unsqueeze(1).expand(-1, n_nodes, -1).reshape(bs * n_nodes, -1)
            edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
            edge_time_emb = time_emb[edge_batch_id]
        else:
            node_time_emb = None
            edge_time_emb = None

        # Get distance features
        distances, cond_adj_spatial = utils.coord2diff_adj(cond_pos, edge_index, self.spatial_cut_off)
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        else:
            if self.dist_gbf:
                distances = self.dist_layer(distances, edge_time_emb)
        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]

        # Concatenate edge features
        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat([cur_edge_attr, cond_edge_attr, distances], dim=-1)
        edge_attr = self.edge_emb(edge_attr)

        # Run DMT modules
        atom_hids = [h]
        edge_hids = [edge_attr]
        
        for i in range(self.n_layers):
            # Update node and edge features
            h, edge_attr = self._modules['dmt_block_%d' % i](
                h, edge_attr, edge_index, node_time_emb, edge_time_emb
            )
            
            # Collect features from each layer for final prediction
            atom_hids.append(self._modules['node_%d' % i](h))
            edge_hids.append(self._modules['edge_%d' % i](edge_attr))

        # Feature prediction
        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        edge_pred = torch.cat([self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)
        pos_pred = self.pos_pred_mlp(atom_hids).reshape(bs, n_nodes, 3) * node_mask


        # Convert sparse edge prediction to dense form
        edge_final = torch.zeros_like(edge_x).reshape(bs * n_nodes * n_nodes, -1)
        edge_final = utils.to_dense_edge_attr(edge_index, edge_pred, edge_final, bs, n_nodes)
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

        # Post-processing
        if self.pred_data:
            pos_pred = pos_pred * node_mask
        else:
            # pos_pred = (pos_pred - pos_init.reshape(bs, n_nodes, -1)) * node_mask
            pos_pred = pos_pred * node_mask

        if torch.any(torch.isnan(pos_pred)):
            print('Warning: detected nan, resetting model output to zero.')
            pos_pred = torch.zeros_like(pos_pred)

        pos_pred = pos_pred.reshape(bs, n_nodes, -1)
        pos_pred = utils.remove_mean_with_mask(pos_pred, node_mask)

        return torch.cat([pos_pred, atom_pred], dim=2), edge_final





def to_dense_edge_attr(edge_index, edge_attr, dense_edge, bs, n_nodes):
    edge_index_a = edge_index[0] % n_nodes
    edge_index_b = edge_index[1] % n_nodes
    batch_edge_index = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
    reshape_edge_pos = batch_edge_index * n_nodes * n_nodes + edge_index_a * n_nodes + edge_index_b
    reshape_edge_pos = reshape_edge_pos.long()
    dense_edge[reshape_edge_pos] = edge_attr
    return dense_edge.reshape(bs, n_nodes, n_nodes, -1)
