import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import numpy as np


def compute_kl_loss(alphas, target_concentration, epsilon=1e-8):
    # The KL loss is a commonly used regularization term, but we did not enable it when training the evidential teacher.
    target_alphas = torch.ones_like(alphas) * target_concentration

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss



class Evidential_Teacher(nn.Module):
    def __init__(self, student, cfg):
        super(Evidential_Teacher, self).__init__()
        self.student = student
        self.efunction = cfg.TEACHER.CLF_TYPE

        # The prior weight lamb can either be manually set or learned during training
        if cfg.TEACHER.LAMB == 0.0:
            self.lamb = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.lamb = torch.tensor(cfg.TEACHER.LAMB)

    def get_learnable_parameters(self):
        params_to_update = []
        if isinstance(self.lamb, nn.Parameter):
            params_to_update.append(self.lamb)
        params_to_update += [v for k, v in self.student.named_parameters()]
        return params_to_update

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)

        # compute evidential cross-entropy loss
        evidence = torch.exp(logits_student)
        # In evidential learning, various evidence activation functions can be used, and we choose the exponential function.
        alpha = evidence + torch.exp(self.lamb)
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S)-torch.digamma(alpha)), dim=-1).mean()

        losses_dict = {
            "loss_ce": loss_ce
        } 
                
        return logits_student, losses_dict

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        logits = self.student(image)[0]
        # During testing, the prediction depends only on the magnitude of the logits.
        evidence = torch.exp(logits)
        return evidence



