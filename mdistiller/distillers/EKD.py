import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import numpy as np

def compute_ekd_loss(s_alpha, t_alpha):
    s_S = torch.sum(s_alpha, dim=1)
    t_S = torch.sum(t_alpha, dim=1)
    t_S_keep = torch.sum(t_alpha, dim=1, keepdim=True)

    loss_term1 = torch.lgamma(t_S) - torch.lgamma(s_S)
    loss_term2 = - torch.sum(torch.lgamma(t_alpha) - torch.lgamma(s_alpha), dim=1)
    loss_term3 = torch.sum((t_alpha - s_alpha) * (torch.digamma(t_alpha) - torch.digamma(t_S_keep)), dim=1)

    loss = (loss_term1 + loss_term2 + loss_term3).mean()
    return loss






def evidential_first_order_loss(alpha_student, alpha_teacher):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    log_pred_student = torch.log(alpha_student / S_student)
    pred_teacher = alpha_teacher / S_teacher
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return loss_kd




class EKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(EKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.EKD.LOSS.CE_WEIGHT
        self.first_loss_weight = cfg.EKD.LOSS.FIRST_WEIGHT
        self.second_loss_weight = cfg.EKD.LOSS.SECOND_WEIGHT

        if cfg.EKD.STUDENT.LAMB == 0.0:
            self.lamb_S = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.lamb_S = torch.tensor(cfg.EKD.STUDENT.LAMB)
        if cfg.EKD.TEACHER.LAMB == 0.0:
            self.lamb_T = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.lamb_T = torch.tensor(cfg.EKD.TEACHER.LAMB) 

        self.warmup = cfg.EKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # compute loss_ce
        evidence_student = torch.exp(logits_student)
        alpha_student = evidence_student + torch.exp(self.lamb_S)
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student) - torch.digamma(alpha_student)), dim=-1).mean()

        # compute first-order loss
        evidence_student = torch.exp(logits_student)
        evidence_teacher = torch.exp(logits_teacher)
        alpha_student = evidence_student + torch.exp(self.lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.lamb_T)
        loss_first =  self.first_loss_weight * evidential_first_order_loss(
                alpha_student, alpha_teacher
        )

        # compute second-order loss
        evidence_student = torch.exp(logits_student)
        evidence_teacher = torch.exp(logits_teacher)
        alpha_student = torch.log1p(evidence_student) + torch.exp(self.lamb_S)
        alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.lamb_T)
        loss_second = min(kwargs["epoch"] / self.warmup, 1.0) * self.second_loss_weight * compute_ekd_loss(alpha_student, alpha_teacher)


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_ekd": loss_first + loss_second,
        }
       
        return logits_student, losses_dict
    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        params_to_update = []
        if isinstance(self.lamb_T, nn.Parameter):
            params_to_update.append(self.lamb_T)
        if isinstance(self.lamb_S, nn.Parameter):
            params_to_update.append(self.lamb_S)
        params_to_update += [v for k, v in self.student.named_parameters()]
        return params_to_update



