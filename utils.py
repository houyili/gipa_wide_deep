import random
import torch
import numpy as np
import dgl
import torch.nn.functional as F
from model_gipa import GipaWide, GipaDeep, GipaWideDeep2

def random_subgraph(num_clusters, graph, shuffle=True, save_e=[]):
    if shuffle:
        cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
    else:
        if not save_e:
            cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
            save_e.append(cluster_id)
        else:
            cluster_id = save_e[0]
    perm = np.arange(0, graph.num_nodes())
    batch_no = 0
    while batch_no < num_clusters:
        batch_nodes = perm[cluster_id == batch_no]
        batch_no += 1
        sub_g = graph.subgraph(batch_nodes)
        yield batch_nodes, sub_g

def count_model_parameters(model:torch.nn.Module):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    n_parameters = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return n_parameters

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def print_msg_and_write(out_msg, log_f):
    print(out_msg)
    log_f.write(out_msg)
    log_f.flush()

def get_model(args, n_node_feats, n_edge_feats, n_classes, n_node_sparse_feats):
    if args.model == "gipa_wide":
        model = GipaWide(
            n_node_sparse_feats if args.use_sparse_fea else n_node_feats,
            n_edge_feats,
            n_classes,
            n_layers = args.n_layers,
            n_heads = args.n_heads,
            n_hidden = args.n_hidden,
            edge_emb = args.edge_emb_size,
            activation = F.relu,
            dropout = args.dropout,
            input_drop = args.feature_drop,
            edge_drop = args.edge_drop,
            use_attn_dst = not args.no_attn_dst,
            norm = args.norm,
            batch_norm = not args.disable_fea_trans_norm,
            edge_att_act = args.edge_att_act,
            edge_agg_mode = args.edge_agg_mode,
            use_node_sparse = args.use_sparse_fea,
            first_hidden = args.first_hidden,
            input_norm = args.input_norm,
            first_layer_act = args.first_layer_act,
            first_layer_drop = args.input_drop,
            first_layer_norm = args.first_layer_norm,
            last_layer_drop = args.last_layer_drop
        )
    elif args.model == "gipa_deep":
        model = GipaDeep(
            n_node_feats,
            n_edge_feats,
            n_classes,
            n_layers = args.n_layers,
            n_hidden = args.n_hidden,
            n_head = args.n_heads,
            edge_emb = args.edge_emb_size,
            activation = F.relu,
            dropout = args.dropout,
            input_drop = args.input_drop,
            edge_drop = args.edge_drop,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax",
            first_hidden=150,
            use_att_edge=True,
            use_prop_edge=False,
            edge_prop_size=20
        )
    elif args.model == "gipa_wide_deep" or args.model == "gipa_deep_wide":
        model = GipaWideDeep2(
            n_node_feats,
            n_node_sparse_feats,
            n_edge_feats,
            n_classes,
            args.n_layers,
            args.n_deep_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            n_deep_hidden=args.n_deep_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            deep_drop_out=args.deep_drop_out,
            input_drop=args.feature_drop,
            deep_input_drop = args.deep_input_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act,
            edge_agg_mode=args.edge_agg_mode,
            use_node_sparse=args.use_sparse_fea,
            input_norm=args.input_norm,
            first_hidden=args.first_hidden,
            first_layer_act=args.first_layer_act,
            first_layer_drop=args.input_drop,
            first_layer_norm=args.first_layer_norm,
            last_layer_drop=args.last_layer_drop,
            use_att_edge=True,
            use_prop_edge=False,
            edge_prop_size=20
        )
    else:
        model = None
    return model