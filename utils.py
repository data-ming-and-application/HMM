import numpy as np
import torch

import torch.nn.functional as F

def reshape(a, shape):
    new_a = np.zeros(shape, dtype=a.dtype)
    new_a[:a.shape[0], :a.shape[1]] = a
    return new_a

def to_device(x, device):
    if isinstance(x, tuple) or isinstance(x, list):
        return [to_device(_x, device) for _x in x]
    elif isinstance(x, torch.Tensor):
        if x.dtype == torch.float64:
            x = x.float()
        if x.dtype == torch.int32:
            x = x.long()
        return x.to(device)
    else:
        raise NotImplementedError()

def l2dis(x, y):
    return torch.sqrt(torch.mean((x-y)**2, dim=1))

def margin_loss(embeddings, labels, params):
    labels = labels.cpu().numpy()
    num = params["triplet_ratio"]*labels.shape[0]
    anchors = []
    pos = []
    neg = []
    triplets = []
    for i in range(labels.shape[0]):
        for j in range(i+1, labels.shape[0]):
            for k in range(j+1, labels.shape[0]):
                if labels[i] == labels[j] and labels[i] != labels[k]:
                    triplets.append((i, j, k))
    num = min(num, len(triplets))
    triplets_idx = np.random.choice(list(range(len(triplets))), size=(num,), replace=False)
    anchor_idx = [triplets[x][0] for x in triplets_idx]
    pos_idx = [triplets[x][1] for x in triplets_idx]
    neg_idx = [triplets[x][2] for x in triplets_idx]
    anchors = embeddings[anchor_idx, :]
    pos = embeddings[pos_idx, :]
    neg = embeddings[neg_idx, :]
    loss = torch.mean(F.relu(params["mu"]  + l2dis(anchors, pos)-l2dis(anchors, neg)))
    return loss

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
