import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import numpy as np

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    
    return logit - mean / (1e-7 + stdv)
    
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

def compute_kl_loss(alphas, target_concentration, epsilon=1e-8):
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



def get_evidence(logits, efunction):
        if efunction == "relu":
            evidence = F.relu(logits)
        elif efunction == "exp":
            evidence = torch.exp(logits)
        elif efunction == "softmax":
            evidence = F.softmax(logits, dim=1)
        else:
            evidence = F.softplus(logits)
        return evidence


def expected_kd_ce_loss(alpha_student, alpha_teacher):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_teacher = alpha_teacher / S_teacher
    temp = torch.digamma(S_student) - torch.digamma(alpha_student)
    loss_kd = torch.sum(pred_teacher * temp, dim=1).mean()
    return loss_kd

def evidential_kd_loss(alpha_student, alpha_teacher):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    log_pred_student = torch.log(alpha_student / S_student)
    pred_teacher = alpha_teacher / S_teacher
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()

    return loss_kd


def expected_dkd_loss(alpha_student, alpha_teacher, target, alpha, beta):
    gt_mask = _get_gt_mask(alpha_student, target)
    other_mask = _get_other_mask(alpha_student, target)

    # compute Target Class Loss
    alpha_student_t = cat_mask(alpha_student, gt_mask, other_mask)
    alpha_teacher_t = cat_mask(alpha_teacher, gt_mask, other_mask)
    tckd_loss = expected_kd_ce_loss(alpha_student_t, alpha_teacher_t)

    # compute Non-target Class Loss
    alpha_student_n = alpha_teacher[~gt_mask].view(alpha_teacher.size(0), alpha_teacher.size(1) - 1)
    alpha_teacher_n = alpha_student[~gt_mask].view(alpha_student.size(0), alpha_student.size(1) - 1)
    nckd_loss = expected_kd_ce_loss(alpha_student_n, alpha_student_n)

    # return alpha * tckd_loss + beta * nckd_loss
    return alpha * tckd_loss + beta * nckd_loss

def evidential_dkd_loss(alpha_student, alpha_teacher, target, alpha, beta):
    gt_mask = _get_gt_mask(alpha_student, target)
    other_mask = _get_other_mask(alpha_student, target)

    # compute Target Class Loss
    alpha_student_t = cat_mask(alpha_student, gt_mask, other_mask)
    alpha_teacher_t = cat_mask(alpha_teacher, gt_mask, other_mask)
    tckd_loss = evidential_kd_loss(alpha_student_t, alpha_teacher_t)

    # compute Non-target Class Loss
    alpha_teacher_n = alpha_teacher[~gt_mask].view(alpha_teacher.size(0), alpha_teacher.size(1) - 1)
    alpha_student_n = alpha_student[~gt_mask].view(alpha_student.size(0), alpha_student.size(1) - 1)
    nckd_loss = evidential_kd_loss(alpha_student_n, alpha_teacher_n)


    return alpha * tckd_loss + beta * nckd_loss
    # return alpha * tckd_loss
    # return beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
    

def Bal(bi, bc):
    return 1 - torch.abs(bi - bc) / (bi + bc)

def Dissonance(b):
    """
    计算一批次中的每个样本的 Diss(alpha) 的矢量化版本
    
    参数:
    b -- 包含信念质量的张量 (torch.Tensor) [batch_size, num_classes]
    
    返回:
    包含每个样本的 Diss(alpha) 的张量 (torch.Tensor) [batch_size]
    """
    batch_size, num_classes = b.shape
    b_expanded = b.unsqueeze(2)  # 变为 [batch_size, num_classes, 1]
    bal_matrix = Bal(b_expanded, b_expanded.transpose(1, 2))
    bal_matrix = bal_matrix * (1 - torch.eye(num_classes, device=b.device).unsqueeze(0))
    numerator = (b.unsqueeze(1) * bal_matrix).sum(dim=2)
    denominator = torch.sum(b, dim=1, keepdim=True) - b
    diss = torch.sum(b * (numerator / denominator), dim=1)
    
    return diss


class DKD4eModel(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DKD4eModel, self).__init__(student, teacher)
        self.temperature = cfg.EKD.LOSS.TEMPERATURE
        self.ce_loss_weight = cfg.EKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.EKD.LOSS.KD_WEIGHT
        self.ekd_loss_weight = cfg.EKD.LOSS.EKD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.efunction_student = cfg.EKD.STUDENT.CLF_TYPE
        self.efunction_teacher = cfg.EKD.TEACHER.CLF_TYPE
        self.alpha = cfg.EKD.LOSS.ALPHA
        self.beta = cfg.EKD.LOSS.BETA

        if cfg.EKD.STUDENT.LAMB == 0.0:
            self.ce_lamb_S = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.ce_lamb_S = torch.tensor(cfg.EKD.STUDENT.LAMB)
        if cfg.EKD.TEACHER.LAMB == 0.0:
            self.ce_lamb_T = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        else:
            self.ce_lamb_T = torch.tensor(cfg.EKD.TEACHER.LAMB) #first-order lamb
        self.lamb_2nd_order_S = nn.Parameter(torch.tensor(7.0, dtype=torch.float32)) #second-order lamb
        self.lamb_2nd_order_T = nn.Parameter(torch.tensor(7.0, dtype=torch.float32)) #second-order lamb

        self.expectation = cfg.KD.LOSS.EXPECTATION
        self.warmup = cfg.EKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        # compute loss_ce
        evidence_student = torch.exp(logits_student)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student)-torch.digamma(alpha_student)), dim=-1).mean()
        loss_ce = self.ce_loss_weight * loss_ce

        # compute first-order loss i.e. loss_dkd
        evidence_student = torch.exp(logits_student / self.temperature)
        evidence_teacher = torch.exp(logits_teacher / self.temperature)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.ce_lamb_T)
        if self.expectation:
            loss_dkd = (self.temperature**2) * expected_dkd_loss(
                alpha_student, alpha_teacher, target, self.alpha, self.beta
            )
        else:
            loss_dkd =  self.kd_loss_weight * (self.temperature**2) * evidential_dkd_loss(
                alpha_student, alpha_teacher, target, self.alpha, self.beta
            )


        # compute second-order loss i.e. loss_ekd
        evidence_student = get_evidence(logits_student, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher, self.efunction_teacher)
        # alpha_student = evidence_student + get_evidence(self.lamb_2nd_order_S, self.efunction_student)
        # alpha_teacher = evidence_teacher + get_evidence(self.lamb_2nd_order_T, self.efunction_teacher)
        alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S)
        alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T)
        loss_ekd =  min(kwargs["epoch"] / (self.warmup + 10 ), 1.0) * self.ekd_loss_weight  * compute_ekd_loss(alpha_student, alpha_teacher)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "loss_ekd": loss_ekd,
        }
        return logits_student, losses_dict
    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        params_to_update = [self.lamb_2nd_order_S, self.lamb_2nd_order_T]
        if isinstance(self.ce_lamb_T, nn.Parameter):
            params_to_update.append(self.ce_lamb_T)
        if isinstance(self.ce_lamb_S, nn.Parameter):
            params_to_update.append(self.ce_lamb_S)
        params_to_update += [v for k, v in self.student.named_parameters()]
        return params_to_update



