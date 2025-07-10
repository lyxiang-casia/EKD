from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ekd_loss(s_alpha, t_alpha):
    s_S = torch.sum(s_alpha, dim=1)
    t_S = torch.sum(t_alpha, dim=1)
    t_S_keep = torch.sum(t_alpha, dim=1, keepdim=True)

    loss_term1 = torch.lgamma(t_S) - torch.lgamma(s_S)
    loss_term2 = - torch.sum(torch.lgamma(t_alpha) - torch.lgamma(s_alpha), dim=1)
    # loss_term3 = torch.sum(
    #     (t_alpha - s_alpha) * (torch.log(t_alpha / t_S_keep) - 1 / (2 * t_alpha) + 1 / (2 * t_S_keep)),
    #     dim=1
    # )

    loss_term3 = torch.sum((t_alpha - s_alpha) * (torch.digamma(t_alpha) - torch.digamma(t_S_keep)), dim=1)
    loss = (loss_term1 + loss_term2 + loss_term3).mean()

    return loss


class EKD(nn.Module):
    def __init__(self):
        super(EKD, self).__init__()

    def forward(self, logits_student, logits_teacher, lamb, epoch, warmup):
        lamb = torch.tensor(lamb)
        alpha_student = torch.log1p(F.softplus(logits_student)) + torch.exp(lamb)
        alpha_teacher = torch.log1p(F.softplus(logits_teacher)) + torch.exp(lamb)

        EKD_loss = min( epoch / warmup, 1.0)  * compute_ekd_loss(alpha_student, alpha_teacher)

        return EKD_loss

class evidentialCE(nn.Module):
    def __init__(self):
        super(evidentialCE, self).__init__()

    def forward(self, logits, target, lamb):
        lamb = torch.tensor(lamb)
        evidence = torch.exp(logits)
        alpha = evidence + torch.exp(lamb)
        labels_1hot = torch.zeros_like(logits).scatter_(-1, target.unsqueeze(-1), 1)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S)-torch.digamma(alpha)), dim=-1).mean()

        return loss_ce

class evidentialDiv(nn.Module):
    def __init__(self):
        super(evidentialDiv, self).__init__()

    def forward(self, logits_student, logits_teacher, temp, lamb):
        lamb = torch.tensor(lamb)
        evidence_student = torch.exp(logits_student / temp)
        evidence_teacher = torch.exp(logits_teacher / temp)
        alpha_student = evidence_student + torch.exp(lamb)
        alpha_teacher = evidence_teacher + torch.exp(lamb)

        S_student = torch.sum(alpha_student, dim=1, keepdim=True)
        S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
        log_pred_student = torch.log(alpha_student / S_student)
        pred_teacher = alpha_teacher / S_teacher
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()

        loss_div = temp**2 * loss_kd

        return loss_div