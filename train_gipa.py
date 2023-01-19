#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as func
import torch.optim as optim
from torch import nn

from data_loader import load_data, preprocess
from tools import count_model_parameters, print_msg_and_write, seed, get_model, random_subgraph

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = None
dataset = "ogbn-proteins"
act_set = ["leaky_relu", "tanh", "softplus", "none", "relu"]
n_node_feats, n_node_sparse_feats, n_edge_feats, n_classes = 0, 0, 8, 112


def train(args, graph, model, _labels, _train_idx, criterion, optimizer, _evaluator):
    model.train()
    train_pred = torch.zeros(graph.ndata["labels"].shape).to(device)
    loss_sum, total = 0, 0

    for batch_nodes, sub_graph in random_subgraph(args.train_partition_num, graph, shuffle=True):
        sub_graph = sub_graph.to(device)
        new_train_idx = torch.arange(0, len(batch_nodes), device=device)
        inner_train_mask = np.isin(batch_nodes[new_train_idx.cpu()], _train_idx.cpu())
        train_pred_idx = new_train_idx[inner_train_mask]
        pred = model(sub_graph)
        train_pred[batch_nodes] += pred
        loss = criterion(pred[train_pred_idx], sub_graph.ndata["labels"][train_pred_idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(batch_nodes)
        total += len(batch_nodes)
    train_score = _evaluator(train_pred[_train_idx], _labels[_train_idx])

    return loss_sum / total, train_score


@torch.no_grad()
def evaluate(args, graph, model, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    torch.cuda.empty_cache()
    model.eval()

    preds = torch.zeros(labels.shape).to(device)
    for _ in range(args.eval_times):
        for batch_nodes, subgraph in random_subgraph(args.eval_partition_num, graph, shuffle=False):
            subgraph = subgraph.to(device)
            pred = model(subgraph)
            preds[batch_nodes] += pred
    if args.eval_times > 1:
        preds /= args.eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()
    train_score = evaluator(preds[train_idx], labels[train_idx])
    val_score = evaluator(preds[val_idx], labels[val_idx])
    test_score = evaluator(preds[test_idx], labels[test_idx])

    return train_score, val_score, test_score, train_loss, val_loss, test_loss, preds


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, log_f):
    # generate model
    criterion = nn.BCEWithLogitsLoss()
    model = get_model(args, n_edge_feats, n_node_feats, n_classes, n_node_sparse_feats).to(device)

    def evaluator_wrapper(scores, real_scores):
        return evaluator.eval({"y_pred": scores, "y_true": real_scores})["rocauc"]

    title_msg = f"Number of node feature: {n_node_feats}\n" + f"Number of edge feature: {n_edge_feats}\n" + \
                f"Number of params: {count_model_parameters(model)}\n"
    print_msg_and_write(title_msg, log_f)

    # optimizer
    lr_scheduler = None
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.advanced_optimizer:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50,
                                                            verbose=True)

    total_time, eval_time, eval_num, val_score, best_val_score, final_test_score, best_step = 0, 0, 0, 0, 0, 0, 0
    final_pred = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        loss, t_score = train(args, graph, model, labels, train_idx, criterion, optimizer, evaluator_wrapper)
        total_time += time.time() - tic
        train_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, " \
                    f"this epoch time: {time.time() - tic:.2f}s Train loss/score: {loss:.4f}/{t_score:.4f}\n"
        print_msg_and_write(train_msg, log_f)

        if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            tic = time.time()
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, graph, model, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper)
            eval_num += 1
            eval_time += (time.time() - tic)

            if val_score > best_val_score:
                best_val_score, final_test_score, final_pred, best_step = val_score, test_score, pred, epoch

            if epoch % args.log_every == 0:
                out_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, " \
                          f"Average Train epoch time: {total_time / epoch:.2f}s, " \
                          f"Average Eval epoch time: {eval_time / eval_num:.2f}s\n" \
                          f"Loss: {loss:.4f} Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n" \
                          f"Train/Val/Test: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}\n" \
                          f"Best val/Final test score/Best Step: {best_val_score:.4f}/{final_test_score:.4f}/{best_step}\n"
                print_msg_and_write(out_msg, log_f)
        if args.advanced_optimizer:
            lr_scheduler.step(val_score)

    out_msg = "*" * 50 + f"\nBest val score: {best_val_score}, Final test score: {final_test_score}\n" + "*" * 50 + "\n"
    print_msg_and_write(out_msg, log_f)

    if args.save_pred:
        os.makedirs("../output", exist_ok=True)
        torch.save(func.sigmoid(final_pred), f"../output/{n_running}.pt")
    return best_val_score, final_test_score


def main():
    global device, n_node_feats, n_edge_feats, n_classes, n_node_sparse_feats

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="gipa_simple", choices=["gipa_wide", "gipa_deep",
                                                                                "gipa_deep_wide", "gipa_wide_deep"])
    argparser.add_argument("--root", type=str, default="/data/ogb/datasets/")
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--eval-times", type=int, default=1)
    argparser.add_argument("--advanced-optimizer", action="store_true")
    argparser.add_argument("--train-partition-num", type=int, default=10, help="number of partitions for training")
    argparser.add_argument("--eval-partition-num", type=int, default=3, help="number of partitions for evaluating")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--norm", type=str, default="none", choices=["none", "adj", "avg"])
    argparser.add_argument("--disable-fea-trans-norm", action="store_true", help="disable batch norm in fea trans part")
    argparser.add_argument("--edge-att-act", type=str, default="leaky_relu", choices=act_set)
    argparser.add_argument("--edge-agg-mode", type=str, default="none_softmax",
                           choices=["single_softmax", "none_softmax"])
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input layer drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--log-file-name", type=str, default="")
    argparser.add_argument("--first-hidden", type=int, default=150, help="first layer size")
    argparser.add_argument("--edge-emb-size", type=int, default=16)
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--use-sparse-fea", action="store_true")
    argparser.add_argument("--sparse-encoder", type=str, default="")
    argparser.add_argument("--input-norm", action="store_true")
    argparser.add_argument("--first-layer-norm", action="store_true")
    argparser.add_argument("--first-layer-act", type=str, default="relu", choices=act_set)
    argparser.add_argument("--feature-drop", type=float, default=0.0, help="raw feature drop rate")
    argparser.add_argument("--last-layer-drop", type=float, default=-1.0, help="last layer drop rate")

    argparser.add_argument("--n-deep-layers", type=int, default=6, help="number of deep layers, work only wide deep")
    argparser.add_argument("--n-deep-hidden", type=int, default=80,
                           help="number of deep hidden units, work only wide deep")
    argparser.add_argument("--deep-drop-out", type=float, default=0.25, help="dropout rate, work only wide deep")
    argparser.add_argument("--deep-input-drop", type=float, default=0.1,
                           help="input layer drop rate, work only wide deep")
    args = argparser.parse_args()
    print(args)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    print("Loading data......")
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, args.root)
    print("Preprocessing......")
    graph, labels = preprocess(graph, labels, user_adj=args.norm == "adj", user_avg=args.norm == "avg",
                               sparse_encoder=args.sparse_encoder if args.use_sparse_fea else None)
    n_node_feats = graph.ndata["feat"].shape[-1]
    n_node_sparse_feats = graph.ndata["sparse"].shape[-1] if args.use_sparse_fea else graph.ndata["feat"].shape[-1]
    n_edge_feats = graph.edata["feat"].shape[-1]

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = list(), list()
    version = str(int(time.time())) if args.log_file_name == "" else "%s_%d" % (args.log_file_name, int(time.time()))
    os.makedirs("%s/log" % args.root, exist_ok=True)
    for i in range(args.n_runs):
        log_f = open("%s/log/%s_part%d.log" % (args.root, version, i), mode='a')
        print_msg_and_write(args.__str__() + "\n", log_f)
        print_msg_and_write("Running for seeds %d" % (args.seed + i), log_f)
        seed(args.seed + i)
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1, log_f)
        val_scores.append(val_score)
        test_scores.append(test_score)
        log_f.close()

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")


if __name__ == "__main__":
    main()
