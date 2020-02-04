import torch
import numpy as np
import pickle
import os

import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam

from HMM import HMM
from cv_dataset import RWCVDataset
from utils import to_device, margin_loss, set_lr
from metrics import get_metrics


def train(train_loader, model, optimizer, params):
    device = params["device"]
    model.train()
    loss_sum = 0
    for i, data in enumerate(train_loader):
        x = to_device(data[0], device)
        y = to_device(data[1], device)
        f2, logits = model(*x)
        loss1 = F.cross_entropy(logits, y[0])
        loss2 = margin_loss(f2, y[1], params)*params["lambda"]
        loss = loss1 + loss2
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_sum /= len(train_loader.dataset)
    return loss_sum


def test(test_loader, model, params):
    device = params["device"]
    model.eval()
    loss_sum = 0.0
    y_true = []
    y_pred = []
    for i, data in enumerate(test_loader):
        x = to_device(data[0], device)
        y = to_device(data[1], device)
        f2, logits = model(*x)
        pred = torch.argmax(logits, dim=1)
        y_true.append(y[0])
        y_pred.append(pred)
        loss = F.cross_entropy(logits, y[0]).item()
        loss_sum += loss
    loss_sum /= len(test_loader.dataset)
    metric = get_metrics(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss_sum, metric


def run(params):
    with open(os.path.join(params["data_root"], "splits", "split_{}.pkl"
                           .format(params["train_ratio"])), "rb") as f:
        sp = pickle.load(f)
    train_idx = sp["train_idx"]
    vali_size = max(int(len(train_idx)*params["vali_ratio"]), 1)
    vali_idx = train_idx[:vali_size]
    train_idx = train_idx[vali_size:]
    test_idx = sp["test_idx"]
    train_dataset = RWCVDataset(params["data_root"],
                                params["v1_num"],
                                params["v2_num"],
                                params["rw_length"],
                                train_idx,
                                params["go_back_p"], is_test=False)

    vali_dataset = RWCVDataset(params["data_root"],
                               params["v1_num"],
                               params["v2_num"],
                               params["rw_length"],
                               vali_idx,
                               params["go_back_p"], is_test=True)

    test_dataset = RWCVDataset(params["data_root"],
                               params["v1_num"],
                               params["v2_num"],
                               params["rw_length"],
                               test_idx,
                               params["go_back_p"], is_test=True)
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True)
    vali_loader = DataLoader(
        vali_dataset, batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=False)

    model = HMM(v1_dim=params["v1_dim"],
                v2_dim=params["v2_dim"],
                v1_sample_num=params["v1_num"],
                v2_sample_num=params["v2_num"],
                hidden_dim=params["hidden_dim"],
                v1_edge_attr_dim=params["v1_edge_attr_dim"],
                v2_edge_attr_dim=params["v2_edge_attr_dim"],
                edge_layer_num=params["edge_layer_num"],
                edge_type_num=params["edge_type_num"],
                edge_head_num=params["edge_head_num"],
                edge_emb_dim=params["edge_emb_dim"],
                label_num=params["label_num"])
    model = model.to(params["device"])
    optimizer = Adam(model.parameters(),
                     lr=params["lr"],
                     weight_decay=params["weight_decay"], betas=(params["adam_alpha"], params["adam_beta"]))

    patience = params["patience"]
    best_vali_acc = 0
    best_vali_loss = 1e9
    best_vali_metric = None
    best_state_dict = None
    best_ep = None

    for ep in range(params["epoch_num"]):
        train_loss = train(train_loader, model, optimizer, params)
        vali_loss, vali_metric = test(vali_loader, model, params)
        vali_acc = vali_metric["acc"]
        if vali_loss < best_vali_loss:
            best_state_dict = model.state_dict()
            best_vali_acc = vali_acc
            best_vali_loss = vali_loss
            patience = params["patience"]
            best_ep = ep
            best_vali_metric = vali_metric
            print("best_vali_loss {} {}".format(vali_loss, vali_acc))
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state_dict)
    test_loss, test_metric = test(test_loader, model, params)
    test_acc = test_metric
    print("test_metric {}".format(test_metric))
    return {"test_loss": test_loss,
            "test_acc": test_acc,
            "test_metric": test_metric,
            "vali_acc": best_vali_acc,
            "best_ep": best_ep}


if __name__ == "__main__":
    params = {
        "data_root": "./data",
        "adam_alpha": 0.5,
        "adam_beta": 0.99,
        "train_ratio": 70,
        "batch_size": 64,
        "lr": 1e-3,
        "patience": 50,
        "weight_decay": 1e-5,
        "v1_num": 16,
        "v2_num": 16,
        "rw_length": 100,
        "go_back_p": 0.05,
        "vali_ratio": 0.1,
        "v1_dim": 132,
        "v2_dim": 50,
        "hidden_dim": 64,
        "v1_edge_attr_dim": 1,
        "v2_edge_attr_dim": 1,
        "edge_layer_num": 2,
        "edge_type_num": 22,
        "edge_head_num": 2,
        "edge_emb_dim": 32,
        "label_num": 4,
        # "device": "cuda:0",
        "device": "cpu",
        "epoch_num": 1000,
        "triplet_ratio": 1,
        "mu": 0.5,
        "lambda": 0.0,
    }
    run(params)
