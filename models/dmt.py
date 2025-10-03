# DGT, data prediction model

from torch import nn
import torch
from . import utils
from .layers import *
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter
import functools
from .specformer import SpecFormer


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MultiCondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, extra_heads):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        update_heads = 1 + extra_heads
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, update_heads, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb, adj_extra):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        if time_emb is not None:
            shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
            inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        else:
            inv = self.ln(self.input_lin(h_input))
        inv = torch.tanh(self.coord_mlp(inv))

        # multi channel adjacency matrix
        adj_dense = torch.ones((adj_extra.size(0), 1), device=adj_extra.device)
        adjs = torch.cat([adj_dense, adj_extra], dim=-1)
        inv = (inv * adjs).mean(-1, keepdim=True)

        # aggregate position
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0, reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


class EquivariantMixBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer."""

    def __init__(self, node_dim, edge_dim, time_dim, num_extra_heads, num_heads, cond_time, dist_gbf, softmax_inf,
                 mlp_ratio=2, act=nn.SiLU(), dropout=0.0, gbf_name='GaussianLayer', trans_name='TransMixLayer'):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.cond_time = cond_time
        self.dist_gbf = dist_gbf
        if dist_gbf:
            dist_dim = edge_dim
        else:
            dist_dim = 1
        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
        self.node2edge_lin = nn.Linear(node_dim, edge_dim)

        # message passing layer
        self.attn_mpnn = eval(trans_name)(node_dim, node_dim // num_heads, num_extra_heads, num_heads,
                                          edge_dim=edge_dim, inf=softmax_inf)

        # Normalization for MPNN
        self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)
        self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        # equivariant edge update layer
        self.equi_update = MultiCondEquiUpdate(node_dim, edge_dim, dist_dim, time_dim, num_extra_heads)

        self.node_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim * 6)
        )
        self.edge_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, edge_dim * 6)
        )

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, pos, h, edge_attr, edge_index, node_mask, extra_heads, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        distance = utils.coord2dist(pos, edge_index)
        if self.dist_gbf:
            distance = self.dist_layer(distance, edge_time_emb)
        edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr, extra_heads)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
                 self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(h_node)) * node_mask

        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                 self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(h_edge) if self.cond_time else \
                     h_edge + self._ff_block_edge(h_edge)

        # apply equivariant coordinate update
        pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb, extra_heads)

        return h_out, h_edge_out, pos



@utils.register_model(name='DMT')
class DMT(nn.Module):
    """Conditional Diffusion Graph Transformer with self-conditioning."""

    def __init__(self, config):
        super().__init__()

        in_node_dim = config.data.atom_types + int(config.model.include_fc_charge)  # 5+1 (5 atom types + 1 formal charge bit)
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
        softmax_inf = config.model.softmax_inf                          # True
        cond_ch = config.model.cond_ch                                  # 1

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

        self.node_emb = nn.Linear(in_node_dim * 2, hidden_dim)      # 12 -> 256
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)     # 68 -> 64

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

        cat_node_dim = (hidden_dim * 2) // n_layers
        cat_edge_dim = (edge_hidden_dim * 2) // n_layers

        for i in range(n_layers):
            self.add_module("e_block_%d" % i, EquivariantMixBlock(hidden_dim, edge_hidden_dim, time_dim, n_extra_heads,
                            n_heads, cond_time, dist_gbf, softmax_inf, mlp_ratio=mlp_ratio, dropout=dropout,
                            gbf_name=gbf_name))
            self.add_module("node_%d" % i, nn.Linear(hidden_dim, cat_node_dim))
            self.add_module("edge_%d" % i, nn.Linear(edge_hidden_dim, cat_edge_dim))

        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim)
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

        if cond_time:
            learned_dim = 16
            sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(learned_dim + 1, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

        # Implement spectral encoder
        self.spectra_version = config.data.spectra_version
        self.cond_encoder = SpecFormer(patch_len=config.model.patch_len, stride=config.model.stride, output_dim=hidden_dim, spectra_version=config.data.spectra_version)
        self.cond_lin = nn.Linear(hidden_dim, time_dim)
        if hasattr(config.model, 'pretrained_specformer_path') and config.model.pretrained_specformer_path:
            print("Load pretrained SpecFormer")
            self.load_pretrained_specformer(config.model.pretrained_specformer_path)
        else:
            print("Train SpecFormer from scratch")
    def load_pretrained_specformer(self, ckpt_path):
        print(f"get pretrained specformer from {ckpt_path}")
        #load the pretrained SpecFormer model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'state_dict' not in ckpt:
            print(f"Warning: pretrained model does not contain 'state_dict' key. Loading the entire checkpoint.")
            return
        
        state_dict = ckpt['state_dict']
        current_dict=self.cond_encoder.state_dict()
        matched_keys=0
        
        prefix=None
        for possible_prefix in ['model.representation_spec_model', 'model.representation_model']:
            if any(key.startswith(possible_prefix) for key in state_dict.keys()):
                prefix=possible_prefix
                print(f"Using prefix: {prefix}")
                break
        if prefix is None:
            print("Warning: No matching prefix found in the state_dict.")
            return
        for target_key in current_dict.keys():
            source_key = f"{prefix}.{target_key}"
            if target_key == "out_norm.weight" or target_key == "out_norm.bias":
                source_key = f"model.representation_model.out_norm.{target_key.split('.')[-1]}"
                
            if source_key in state_dict and current_dict[target_key].shape == state_dict[source_key].shape:
                current_dict[target_key] = state_dict[source_key]
                matched_keys += 1
                print(f"Loaded {source_key} into {target_key}")
        if matched_keys > 0:
            self.cond_encoder.load_state_dict(current_dict,strict=False)
            print(f"Loaded {matched_keys} keys from the pretrained SpecFormer model.")
        else:
            print(f"{matched_keys} Warning: No matching keys found in the pretrained SpecFormer model.")
        

    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        """
        Parameters
        ----------
        t: [B] time steps in [0, 1]
        xh: [B, N, ch1] atom feature (positions, types, formal charges)
        node_mask: [B, N, 1]
        edge_mask: [B*N*N, 1]
        context:
        kwargs: 'edge_x' [B, N, N, ch2]

        Returns
        -------

        """
        edge_x, cond_x, cond_edge_x = kwargs['edge_x'], kwargs['cond_x'], kwargs['cond_edge_x']

        bs, n_nodes, dims = xh.shape
        pos_init = pos = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        h = xh[:, :, 3:].clone().reshape(bs * n_nodes, -1)

        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)

        # extra structural features
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
        h = torch.cat([h, cond_h], dim=-1)

        # Encode condition information  context.shape = [128, 1, 3501]
        if context is not None:
            context = self.cond_encoder(context) # [128, 1, 3501] -> [128, 256]
            context = self.cond_lin(context)  # [128, 1024]

        if self.cond_time:
            noise_level = kwargs['noise_level']
            time_emb = self.time_mlp(noise_level) + context  # [B, hid_dim*4]     # spectral condition embedding is added to time_emb
            node_time_emb = time_emb.unsqueeze(1).expand(-1, n_nodes, -1).reshape(bs * n_nodes, -1)
            edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
            edge_time_emb = time_emb[edge_batch_id]
        else:
            node_time_emb = None
            edge_time_emb = None

        # obtain distance from self_cond position
        distances, cond_adj_spatial = utils.coord2diff_adj(cond_pos, edge_index, self.spatial_cut_off)
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        else:
            if self.dist_gbf:
                distances = self.dist_layer(distances, edge_time_emb)
        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]

        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat([cur_edge_attr, cond_edge_attr, distances], dim=-1)  # [N_edge, ch]

        # add structural features
        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)

        # run the equivariant block
        atom_hids = [h]
        edge_hids = [edge_attr]
        for i in range(0, self.n_layers):
            h, edge_attr, pos = self._modules['e_block_%d' % i](pos, h, edge_attr, edge_index, node_mask.reshape(-1, 1),
                                                                extra_adj, node_time_emb, edge_time_emb)
            if self.CoM:
                pos = utils.remove_mean_with_mask(pos.reshape(bs, n_nodes, -1), node_mask).reshape(bs * n_nodes, -1)
            atom_hids.append(self._modules['node_%d' % i](h))
            edge_hids.append(self._modules['edge_%d' % i](edge_attr))

        # type prediction
        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        edge_pred = torch.cat([self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)  # [N_edge, ch]

        # convert sparse edge_pred to dense form
        edge_final = torch.zeros_like(edge_x).reshape(bs * n_nodes * n_nodes, -1)  # [B*N*N, ch]
        edge_final = utils.to_dense_edge_attr(edge_index, edge_pred, edge_final, bs, n_nodes)
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

        # post-processing
        if self.pred_data:
            pos = pos * node_mask.reshape(-1, 1)
        else:
            pos = (pos - pos_init) * node_mask.reshape(-1, 1)

        if torch.any(torch.isnan(pos)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            pos = torch.zeros_like(pos)

        pos = pos.reshape(bs, n_nodes, -1)
        pos = utils.remove_mean_with_mask(pos, node_mask)

        return torch.cat([pos, atom_pred], dim=2), edge_final