import warnings
import torch.nn.functional as F


def head_loss(loss_func,logits,label):
    seg_logits = logits
    loss = loss_func(seg_logits,label)
    return loss