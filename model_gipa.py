import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

def get_act_by_str(name: str, negative_slope: float = 0):
    if name == "leaky_relu":
        res = nn.LeakyReLU(negative_slope, inplace=True)
    elif name == "tanh":
        res = nn.Tanh()
    elif name == "none":
        res = nn.Identity()
    elif name == "relu":
        res = nn.ReLU()
    else:
        res = nn.Softplus()
    return res


class GIPAWideConv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            out_feats,
            n_heads,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax"
    ):
        super(GIPAWideConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._norm = norm
        self._agg_layer_norm = batch_norm
        self._edge_agg_mode = edge_agg_mode
        self._edge_att_act = edge_att_act

        # project function
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)

        # propagation function
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False) if use_attn_dst else None
        self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False) if edge_feats > 0 else None
        self.edge_norm = nn.BatchNorm1d(edge_feats) if edge_feats > 0 else None
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.edge_drop = edge_drop

        #  aggregation function
        self.offset = nn.Parameter(torch.zeros(size=(1, n_heads, out_feats)))
        self.scale = nn.Parameter(torch.ones(size=(1, n_heads, out_feats)))
        self.agg_fc = nn.Linear(out_feats * n_heads, out_feats * n_heads)

        # apply function
        self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
        self.activation = activation
        print("Init %s" % str(self.__class__))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

    def agg_func(self, h):
        if self._agg_layer_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale * torch.rsqrt(var) + self.offset
        return self.agg_fc(h.view(-1, self._out_feats * self._n_heads)).view(-1, self._n_heads, self._out_feats)

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # project function: source node
            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)

            # propagation function: source node
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            # propagation function: dst node
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            # propagation function: edge
            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.edge_att_actv(e)

            # edge drop
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)

            # edge softmax
            if self._edge_agg_mode == "single_softmax":
                graph.edata["a"][eids] = edge_softmax(graph, e[eids], eids=eids, norm_by='dst')
            else:
                graph.edata["a"][eids] = e[eids]

            # graph normalize
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm_adjust"][eids].view(-1, 1, 1)
            if self._norm == "avg":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm"][eids].view(-1, 1, 1)

            # aggregation
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
            agg_msg = self.agg_func(graph.dstdata["feat_src_fc"])

            # apply part
            if self.dst_fc is not None:
                feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
                rst = agg_msg + feat_dst_fc  # apply = fc(concat([h_{k-1}, msg]))
            else:
                rst = agg_msg

            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst


class GIPADeepConv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_head,
            out_feats,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax",
            use_att_edge=True,
            use_prop_edge=False,
            edge_prop_size=20
    ):
        super(GIPADeepConv, self).__init__()
        self._norm = norm
        self._batch_norm = batch_norm
        self._edge_agg_mode = edge_agg_mode
        self._use_prop_edge = use_prop_edge
        self._edge_prop_size = edge_prop_size

        # optional fc
        self.prop_edge_fc = None
        self.attn_dst_fc = None
        self.attn_edge_fc = None
        self.attn_dst_fc_e = None
        self.attn_edge_fc_e = None

        # propagation src feature
        self.prop_src_fc = nn.Linear(node_feats, n_head, bias=False)

        # attn fc
        self.attn_src_fc = nn.Linear(node_feats, n_head, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(node_feats, n_head, bias=False)
        if edge_feats > 0 and use_att_edge:
            self.attn_edge_fc = nn.Linear(edge_feats, n_head, bias=False)

        # msg BN
        if batch_norm:
            self.agg_batch_norm = nn.BatchNorm1d(n_head)

        # agg function
        self.agg_fc = nn.Linear(n_head, out_feats)

        # apply function
        self.apply_dst_fc = nn.Linear(node_feats, out_feats)
        self.apply_fc = nn.Linear(out_feats, out_feats)

        if use_prop_edge and edge_prop_size > 0:
            self.prop_edge_fc = nn.Linear(edge_feats, edge_prop_size, bias=False)
            self.prop_src_fc_e = nn.Linear(node_feats, edge_prop_size)
            self.attn_src_fc_e = nn.Linear(node_feats, edge_prop_size, bias=False)
            if use_attn_dst:
                self.attn_dst_fc_e = nn.Linear(node_feats, edge_prop_size, bias=False)
            if edge_feats > 0 and use_att_edge:
                self.attn_edge_fc_e = nn.Linear(edge_feats, edge_prop_size, bias=False)
            if batch_norm:
                self.agg_batch_norm_e = nn.BatchNorm1d(edge_prop_size)
            self.agg_fc_e = nn.Linear(edge_prop_size, out_feats)

        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.activation = activation

        print("Init %s" % str(self.__class__))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.prop_src_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_dst_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        # nn.init.zeros_(self.attn_src_fc.bias)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_dst_fc.bias)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_edge_fc.bias)
        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

        if self._use_prop_edge and self._edge_prop_size > 0:
            nn.init.xavier_normal_(self.prop_src_fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.prop_edge_fc.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_src_fc_e.weight, gain=gain)
            if self.attn_dst_fc_e is not None:
                nn.init.xavier_normal_(self.attn_dst_fc_e.weight, gain=gain)
            if self.attn_edge_fc_e is not None:
                nn.init.xavier_normal_(self.attn_edge_fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.agg_fc_e.weight, gain=gain)
            nn.init.zeros_(self.agg_fc_e.bias)

    def agg_function(self, h, idx):
        out = h
        if self._batch_norm:
            out = self.agg_batch_norm(h) if idx == 0 else self.agg_batch_norm_e(h)

        return self.agg_fc(out) if idx == 0 else self.agg_fc_e(out)

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # propagation value prepare
            feat_src_fc = self.prop_src_fc(feat_src)
            graph.srcdata.update({"_feat_src_fc": feat_src_fc})

            # src node attention
            attn_src = self.attn_src_fc(feat_src)
            graph.srcdata.update({"_attn_src": attn_src})

            # dst node attention
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst)
                graph.dstdata.update({"_attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("_attn_src", "_attn_dst", "_attn_node"))
            else:
                graph.apply_edges(fn.copy_u("_attn_src", "_attn_node"))

            e = graph.edata["_attn_node"]
            if self.attn_edge_fc is not None:
                attn_edge = self.attn_edge_fc(feat_edge)
                graph.edata.update({"_attn_edge": attn_edge})
                e += graph.edata["_attn_edge"]
            e = self.edge_att_actv(e)

            if self._edge_agg_mode == "both_softmax":
                graph.edata["_a"] = torch.sqrt(edge_softmax(graph, e, norm_by='dst').clamp(min=1e-9)
                                               * edge_softmax(graph, e, norm_by='src').clamp(min=1e-9))
            elif self._edge_agg_mode == "single_softmax":
                graph.edata["_a"] = edge_softmax(graph, e, norm_by='dst')
            else:
                graph.edata["_a"] = e

            if self._norm == "adj":
                graph.edata["_a"] = graph.edata["_a"] * graph.edata["gcn_norm_adjust"].view(-1, 1)
            if self._norm == "avg":
                graph.edata["_a"] = (graph.edata["_a"] * graph.edata["gcn_norm"].view(-1, 1)) / 2

            graph.update_all(fn.u_mul_e("_feat_src_fc", "_a", "_m"), fn.sum("_m", "_feat_src_fc"))
            msg_sum = graph.dstdata["_feat_src_fc"]
            # print(msg_sum.size())
            # aggregation function
            rst = self.agg_function(msg_sum, 0)

            if self._use_prop_edge and self._edge_prop_size > 0:
                graph.edata["_v"] = self.prop_edge_fc(feat_edge)
                feat_src_fc_e = self.prop_src_fc_e(feat_src)
                graph.srcdata.update({"_feat_src_fc_e": feat_src_fc_e})
                graph.apply_edges(fn.u_add_e("_feat_src_fc_e", "_v", "_prop_edge"))

                # src node attention
                attn_src_e = self.attn_src_fc_e(feat_src)
                graph.srcdata.update({"_attn_src_e": attn_src_e})

                # dst node attention
                if self.attn_dst_fc is not None:
                    attn_dst_e = self.attn_dst_fc_e(feat_dst)
                    graph.dstdata.update({"_attn_dst_e": attn_dst_e})
                    graph.apply_edges(fn.u_add_v("_attn_src_e", "_attn_dst_e", "_attn_node_e"))
                else:
                    graph.apply_edges(fn.copy_u("_attn_src_e", "_attn_node_e"))

                e_e = graph.edata["_attn_node_e"]
                if self.attn_edge_fc is not None:
                    attn_edge_e = self.attn_edge_fc_e(feat_edge)
                    graph.edata.update({"_attn_edge_e": attn_edge_e})
                    e_e += graph.edata["_attn_edge_e"]
                e_e = self.edge_att_actv(e_e)

                if self._edge_agg_mode == "both_softmax":
                    graph.edata["_a_e"] = torch.sqrt(edge_softmax(graph, e_e, norm_by='dst').clamp(min=1e-9)
                                                     * edge_softmax(graph, e_e, norm_by='src').clamp(min=1e-9))
                elif self._edge_agg_mode == "single_softmax":
                    graph.edata["_a_e"] = edge_softmax(graph, e_e, norm_by='dst')
                else:
                    graph.edata["_a_e"] = e_e

                if self._norm == "adj":
                    graph.edata["_a_e"] = graph.edata["_a_e"] * graph.edata["gcn_norm_adjust"].view(-1, 1)
                if self._norm == "avg":
                    graph.edata["_a_e"] = (graph.edata["_a_e"] * graph.edata["gcn_norm"].view(-1, 1)) / 2

                graph.edata["_m_e"] = graph.edata["_a_e"] * graph.edata["_prop_edge"]
                graph.update_all(fn.copy_e("_m_e", "_m_copy_e"), fn.sum("_m_copy_e", "_feat_src_fc_e"))
                msg_sum_e = graph.dstdata["_feat_src_fc_e"]
                rst_e = self.agg_function(msg_sum_e, 1)
                rst += rst_e

            # apply function
            rst += self.apply_dst_fc(feat_dst)
            rst = self.leaky_relu(rst)
            rst = self.apply_fc(rst)
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst


class GipaWide(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_classes,
            n_layers,
            n_heads,
            n_hidden,
            edge_emb,
            activation,
            dropout,
            input_drop,
            edge_drop,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="none_softmax",
            use_node_sparse=False,
            input_norm=False,
            first_hidden=150,
            first_layer_act="relu",
            first_layer_drop=0.1,
            first_layer_norm=False,
            last_layer_drop=-1
    ):
        super().__init__()
        self.n_layers = n_layers
        self.has_edge = edge_emb > 0
        self._use_node_sparse = use_node_sparse

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_norm = nn.BatchNorm1d(node_feats) if input_norm else None
        self.first_layer_norm = nn.BatchNorm1d(first_hidden) if first_layer_norm else None

        self.node_encoder = nn.Linear(node_feats, first_hidden)

        if self.has_edge:
            self.edge_encoder = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else first_hidden
            out_hidden = n_hidden

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
            self.convs.append(
                GIPAWideConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    norm=norm,
                    batch_norm=batch_norm,
                    edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)
        self.first_layer_act = get_act_by_str(first_layer_act)
        self.input_drop = nn.Dropout(input_drop) if input_drop > 0 else None
        self.first_layer_drop = nn.Dropout(first_layer_drop)
        self.dropout = nn.Dropout(dropout)
        self.last_layer_drop = nn.Dropout(last_layer_drop if last_layer_drop > 0 else dropout)
        self.activation = activation
        print("The parameter are %s,%s,%s" % (batch_norm, edge_att_act, edge_agg_mode))
        print("Init %s" % str(self.__class__))

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["sparse"] if self._use_node_sparse else subgraphs[0].srcdata["feat"]

        if self.input_norm is not None:
            h = self.input_norm(h)
        if self.input_drop is not None:
            h = self.input_drop(h)
        h = self.first_layer_act(self.node_encoder(h))
        if self.first_layer_norm is not None:
            h = self.first_layer_norm(h)
        h = self.first_layer_drop(h)

        h_last = None

        for i in range(self.n_layers):

            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h) if i < self.n_layers - 1 else self.last_layer_drop(h)

        res = self.pred_linear(h)
        return res


class GipaDeep(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_classes,
            n_layers,
            n_hidden,
            n_head,
            edge_emb,
            activation,
            dropout,
            input_drop,
            edge_drop,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax",
            first_hidden=150,
            use_att_edge=True,
            use_prop_edge=False,
            edge_prop_size=20
    ):
        super().__init__()
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, first_hidden)

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else first_hidden
            out_hidden = n_hidden

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                GIPADeepConv(
                    in_hidden,
                    edge_emb,
                    n_head,
                    out_hidden,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    norm=norm,
                    batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode,
                    use_att_edge=use_att_edge,
                    use_prop_edge=use_prop_edge,
                    edge_prop_size=edge_prop_size
                )
            )
            self.norms.append(nn.BatchNorm1d(out_hidden))

        self.pred_linear = nn.Linear(n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        print("The parameter are %s,%s,%s" % (batch_norm, edge_att_act, edge_agg_mode))
        print("Init %s" % str(self.__class__))

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]

        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        res = self.pred_linear(h)
        return res


class GipaWideDeep(nn.Module):
    def __init__(self,
                node_feats,
                sparse_node_feats,
                edge_feats,
                n_classes,
                n_layers,
                n_deep_layers,
                n_heads,
                n_hidden,
                n_deep_hidden,
                edge_emb,
                activation,
                dropout,
                deep_drop_out,
                input_drop,
                deep_input_drop,
                edge_drop,
                use_attn_dst=True,
                norm="none",
                batch_norm=True,
                edge_att_act="leaky_relu",
                edge_agg_mode="none_softmax",
                use_node_sparse = False,
                input_norm = False,
                first_hidden = 150,
                first_layer_act = "relu",
                first_layer_drop = 0.1,
                first_layer_norm = False,
                last_layer_drop = -1,
                use_att_edge=True,
                use_prop_edge=False,
                edge_prop_size = 20):
        super(GipaWideDeep, self).__init__()
        self.n_layers = n_layers
        self.wide_part = GipaWide(sparse_node_feats,
                                   edge_feats,
                                   n_classes,
                                   n_layers,
                                   n_heads,
                                   n_hidden,
                                   edge_emb,
                                   activation,
                                   dropout,
                                   input_drop,
                                   edge_drop,
                                   use_attn_dst,
                                   norm,
                                   batch_norm,
                                   edge_att_act,
                                   edge_agg_mode,
                                   use_node_sparse,
                                   input_norm,
                                   first_hidden,
                                   first_layer_act,
                                   first_layer_drop,
                                   first_layer_norm,
                                   last_layer_drop)
        self.deep_part = GipaDeep(node_feats,
                                edge_feats,
                                n_classes,
                                n_deep_layers,
                                n_deep_hidden,
                                n_heads,
                                edge_emb,
                                activation,
                                deep_drop_out,
                                deep_input_drop,
                                edge_drop,
                                use_attn_dst,
                                norm,
                                batch_norm,
                                edge_att_act,
                                edge_agg_mode,
                                first_hidden,
                                use_att_edge,
                                use_prop_edge,
                                edge_prop_size)

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g
        res1 = self.wide_part(subgraphs)
        res2 = self.deep_part(subgraphs)
        return res1 + res2
