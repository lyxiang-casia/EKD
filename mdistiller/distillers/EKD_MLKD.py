import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import numpy as np

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    
    return logit - mean / (1e-7 + stdv)

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def projected_kd_loss(alpha_student, alpha_teacher):
    # 对于证据网络来说，此处的logits代表alpha
    # alpha_student = normalize(alpha_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    log_pred_student = torch.log(alpha_student / S_student)
    pred_teacher = alpha_teacher / S_teacher
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return loss_kd

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

def get_evidence(logits, efunction):
        if efunction == "relu":
            evidence = F.relu(logits)
        elif efunction == "exp":
            evidence = torch.exp(logits)
        elif efunction == "sigmoid":
            evidence = F.sigmoid(logits)
        elif efunction == "softmax":
            evidence = F.softmax(logits, dim=1)
        else:
            evidence = F.softplus(logits)
        return evidence

def compute_opinion_loss(evidence_s, evidence_t, lamb_s, lamb_t):
        W_s = lamb_s * 100
        W_t = lamb_t * 100
        belief_s = evidence_s / (W_s + torch.sum(evidence_s, dim=-1, keepdim=True))
        uncertainty_s = W_s / (W_s + torch.sum(evidence_s, dim=-1, keepdim=True))
        belief_t = evidence_t / (W_t + torch.sum(evidence_t, dim=-1, keepdim=True))
        uncertainty_t = W_t / (W_t + torch.sum(evidence_t, dim=-1, keepdim=True))

        opinion_student = torch.cat((belief_s, uncertainty_s), dim=1)
        opinion_teacher = torch.cat((belief_t, uncertainty_t), dim=1)

        loss = F.kl_div(torch.log(opinion_student), opinion_teacher, reduction="none").sum(1).mean()
        return loss

def compute_conflict(evidence_s, evidence_t, lamb_s, lamb_t):
        W_s = lamb_s * 100
        W_t = lamb_t * 100
        belief_s = evidence_s / (W_s + torch.sum(evidence_s, dim=-1, keepdim=True))
        belief_t = evidence_t / (W_t + torch.sum(evidence_t, dim=-1, keepdim=True))

        bs_expanded = belief_s.unsqueeze(2)
        bt_expanded = belief_t.unsqueeze(1) 
        product = bs_expanded * bt_expanded
        mask = torch.eye(100).bool().to(belief_s.device)
        masked_product = product.masked_fill(mask, 0)
        conflict = masked_product.sum(dim=[1, 2])

        return conflict.mean()

def dkd_loss(alpha_student, alpha_teacher, target, alpha, beta):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_student = alpha_student / S_student
    pred_teacher = alpha_teacher / S_teacher

    gt_mask = _get_gt_mask(alpha_student, target)
    other_mask = _get_other_mask(alpha_student, target)

    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student_t = torch.log(pred_student_t)

    tckd_loss = (
        F.kl_div(log_pred_student_t, pred_teacher_t, reduction='none').sum(1).mean()
    )
    pred_teacher_part2 = alpha_teacher[~gt_mask].view(alpha_student.size(0), alpha_student.size(1)-1)
    pred_teacher_part2 = pred_teacher_part2 / pred_teacher_part2.sum(1, keepdim=True)
    pred_student_part2 = alpha_student[~gt_mask].view(alpha_student.size(0), alpha_student.size(1)-1)
    pred_student_part2 = pred_student_part2 / pred_student_part2.sum(1, keepdim=True)
    log_pred_student_part2 = torch.log(pred_student_part2)
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1).mean()
    )
    return alpha * tckd_loss + beta * nckd_loss

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

class EKD(Distiller):
    def __init__(self, student, teacher, van_teacher, cfg, ce_weight, ekd_weight, student_clf, teacher_clf, student_lamb_ekd, teacher_lamb_ekd):
        super(EKD, self).__init__(student, teacher)
        self.lamb_student_ekd = student_lamb_ekd
        self.lamb_teacher_ekd = teacher_lamb_ekd
        self.efunction_teacher = teacher_clf
        self.efunction_student = student_clf
        self.ce_loss_weight = ce_weight
        self.ekd_loss_weight = ekd_weight * 1e-3
        self.KL = cfg.EKD.LOSS.KL
        self.warmup = 20 
        self.van_teacher = van_teacher

        # self.fc = nn.Linear(100,100)        
        # nn.init.eye_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            logits_van_teacher, _ = self.van_teacher(image)

        alpha4ce_student = get_evidence(logits_student, "exp") + 0.2

        # compute loss_ce
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha4ce_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student)-torch.digamma(alpha4ce_student)), dim=-1).mean()
        # if kwargs['epoch'] >= self.warmup:
        loss_ce = self.ce_loss_weight * loss_ce

        # compute loss_ekd
        evidence_student = get_evidence(logits_student, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher, self.efunction_teacher)
        alpha_student = evidence_student + 2.0
        alpha_teacher = evidence_teacher + 0.2
        loss_ekd =  self.ekd_loss_weight * compute_ekd_loss(alpha_student, alpha_teacher)

        # compute loss_kd
        # logits_student = normalize(self.fc(logits_student))
        # logits_student = normalize(logits_student)
        # logits_van_teacher = normalize(logits_van_teacher)
        alpha_student = get_evidence(logits_student, "exp") + 0.2
        alpha_teacher = get_evidence(logits_teacher, "exp") + 0.2
        # loss_kd = 2 * projected_kd_loss(alpha_student, alpha_teacher)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            alpha_student,
            alpha_teacher,
            target,
            1.0,
            8.0,
        )

        # loss_ekd =  self.ekd_loss_weight * compute_opinion_loss(evidence_student, evidence_teacher, \
        #     self.lamb2_student, self.lamb2_teacher)
        # loss_ekd = compute_conflict(evidence_student, evidence_teacher, \
        #      self.lamb2_student, self.lamb2_teacher) + compute_opinion_loss(evidence_student, evidence_teacher, \
        #                                                                     self.lamb2_student, self.lamb2_teacher)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_ekd + loss_dkd,
        }

        #compute loss_kl    
        if self.KL:
            kl_alpha = evidence_student * (1 - labels_1hot) + self.lamb2_student
            epoch = kwargs.get('epoch', 0)
            loss_kl = np.minimum(1.0, epoch / 10.) * compute_kl_loss(kl_alpha, self.lamb2_student)
            losses_dict["loss_kl"] = self.ce_loss_weight * loss_kl
                
        return logits_student, losses_dict
    # def get_learnable_parameters(self):
    #     # if the method introduces extra parameters, re-impl this function
    #     params = [v for k, v in self.student.named_parameters()]
    #     params += list(self.fc.parameters())
    #     return params

    # def get_extra_parameters(self):
    #     params = self.fc.weight.numel() + self.fc.bias.numel()
    #     return params



