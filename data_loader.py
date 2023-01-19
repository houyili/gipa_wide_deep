import torch
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

interval_0_1 = [0.001, 1]
interval_3 = [0.001, 0.7, 1]
interval_4 = [0.001, 0.1, 0.2, 1]
interval_12 = [0.001, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1]
interval_15 = [0.001, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1]
interval = [interval_0_1, interval_4, interval_0_1, interval_4, interval_12, interval_12, interval_3, interval_15]

def trans_edge_fea_to_sparse(raw_edge_fea, graph, interval: list, is_log=False):
    edge_fea_list = []
    for i in range(8):
        print("Process edge feature == %d " % i)
        res = torch.reshape((raw_edge_fea[:, i] == 0.001).float(), [-1, 1])
        edge_fea_list.append(res)
        for j in range(1, len(interval[i])):
            small, big = float(interval[i][j - 1]), float(interval[i][j])
            print("process interval %0.3f < x <= %0.3f " % (small, big))
            cond = torch.logical_and((raw_edge_fea[:, i] > small), (raw_edge_fea[:, i] <= big))
            edge_fea_list.append(torch.reshape(cond.float(), [-1, 1]))
    sparse = torch.concat(edge_fea_list, dim=-1)
    print(sparse.size())
    graph.edata.update({"sparse": sparse})
    graph.update_all(fn.copy_e("sparse", "sparse_c"), fn.sum("sparse_c", "sparse_f" if is_log else "sparse"))
    if is_log:
        graph.apply_nodes(lambda nodes: {"sparse": torch.log2(nodes.data['sparse_f'] + 1)})
        del graph.ndata["sparse_f"]
    return sparse

def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt

def load_data(dataset, root_path):
    data = DglNodePropPredDataset(name=dataset, root=root_path)
    evaluator = Evaluator(name=dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels
    print(f"Nodes : {graph.number_of_nodes()}\n"
          f"Edges: {graph.number_of_edges()}\n"
          f"Train nodes: {len(train_idx)}\n"
          f"Val nodes: {len(val_idx)}\n"
          f"Test nodes: {len(test_idx)}")
    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph, labels, edge_agg_as_feat=True, user_adj=False, user_avg=False, sparse_encoder: str = None):
    if edge_agg_as_feat:
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    if sparse_encoder is not None and len(sparse_encoder) > 0:
        is_log = sparse_encoder.find("log") > -1
        edge_sparse = trans_edge_fea_to_sparse(graph.edata['feat'], graph, interval, is_log)

        if len(sparse_encoder) > 0 and sparse_encoder.find("edge_sparse") != -1:
            graph.edata.update({"feat": edge_sparse})
        del graph.edata["sparse"]

    if user_adj or user_avg:
        deg_sqrt, deg_isqrt = compute_norm(graph)
        if user_adj:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_sqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

        if user_avg:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_isqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.create_formats_()
    print(graph.ndata.keys())
    print(graph.edata.keys())
    return graph, labels