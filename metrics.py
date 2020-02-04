from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np

def get_metrics(y_true, y_pred, mask=None):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.astype(np.bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    acc = np.mean(y_true == y_pred)
    p = precision_score(y_true, y_pred, average='weighted')
    r = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    m = {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }
    return m

if __name__ == "__main__":
    y_true = np.random.randint(0, 10, size=(1000,))
    y_pred = np.random.randint(0, 10, size=(1000,))
    mask = np.random.randint(0, 2, size=(1000,)).astype(np.bool)
    # mask = np.ones((1000,)).astype(np.bool)
    # print(mask)
    # print(y_true)
    # print(y_pred)
    m = get_metrics(y_true, y_pred, mask)
    print(m)
