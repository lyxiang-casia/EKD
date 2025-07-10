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
        elif efunction == "sigmoid":
            evidence = F.sigmoid(logits)
        elif efunction == "softmax":
            evidence = F.softmax(logits, dim=1)
        else:
            evidence = F.softplus(logits)
        return evidence



def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss, beta * nckd_loss
    # return alpha * tckd_loss + beta * nckd_loss

def evidential_dkd_loss(alpha_student, alpha_teacher, target, alpha, beta, temperature):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_student = alpha_student / S_student
    pred_teacher = alpha_teacher / S_teacher

    gt_mask = _get_gt_mask(alpha_student, target)
    other_mask = _get_other_mask(alpha_student, target)

    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student_t = torch.log(pred_student_t)

    tckd_loss = (temperature**2) * (
        F.kl_div(log_pred_student_t, pred_teacher_t, reduction='none').sum(1).mean()
    )
    pred_teacher_part2 = alpha_teacher[~gt_mask].view(alpha_teacher.size(0), alpha_teacher.size(1)-1)
    pred_teacher_part2 = pred_teacher_part2 / pred_teacher_part2.sum(1, keepdim=True)
    pred_student_part2 = alpha_student[~gt_mask].view(alpha_student.size(0), alpha_student.size(1)-1)
    pred_student_part2 = pred_student_part2 / pred_student_part2.sum(1, keepdim=True)
    log_pred_student_part2 = torch.log(pred_student_part2)
    nckd_loss = (temperature**2) * (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1).mean()
    )
    # return alpha * tckd_loss + beta * nckd_loss
    return alpha * tckd_loss, beta * nckd_loss


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
    
def test(logits_s, logits_t, lamb_student_ekd, lamb_teacher_ekd):
    evidence_student = get_evidence(logits_s, 'softplus')
    evidence_teacher = get_evidence(logits_t, 'softplus')
    alpha_student = evidence_student + lamb_student_ekd
    alpha_teacher = evidence_teacher + lamb_teacher_ekd
    return compute_ekd_loss(alpha_student, alpha_teacher)

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
        self.lamb_student_ekd = cfg.EKD.STUDENT.LAMB2
        self.lamb_teacher_ekd = cfg.EKD.TEACHER.LAMB2
        self.efunction_teacher = cfg.EKD.TEACHER.CLF_TYPE
        self.efunction_student = cfg.EKD.STUDENT.CLF_TYPE
        self.ce_loss_weight = cfg.EKD.LOSS.CE_WEIGHT
        self.ekd_loss_weight = cfg.EKD.LOSS.EKD_WEIGHT
        self.dkd_temperature = cfg.DKD.T
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.KL = cfg.EKD.LOSS.KL
        self.warmup = 20 

        # self.fc = nn.Linear(100,100)        
        # nn.init.eye_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        alpha4ce_student = get_evidence(logits_student, "exp") + 0.2

        # compute loss_ce
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha4ce_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student)-torch.digamma(alpha4ce_student)), dim=-1).mean()
        # if kwargs['epoch'] >= self.warmup:
        loss_ce = self.ce_loss_weight * loss_ce

        # Uncertainty: Dissonance
        # Diss = Dissonance(F.softplus(logits_teacher))
        # rate = torch.sigmoid(4*(Diss.mean() - 85)/(90 - 85) - 2)
        rate = 0.5

        # compute loss_ekd
        mask = _get_other_mask(logits_student, target)
        nt_logits_student = logits_student[mask].view(logits_student.size(0), logits_student.size(1) - 1)
        nt_logits_teacher = logits_teacher[mask].view(logits_teacher.size(0), logits_teacher.size(1) - 1)
        evidence_student = get_evidence(nt_logits_student, self.efunction_student)
        evidence_teacher = get_evidence(nt_logits_teacher, self.efunction_teacher)
        alpha_student = evidence_student + self.lamb_student_ekd
        alpha_teacher = evidence_teacher + self.lamb_teacher_ekd
        loss_ekd = 9 * rate * compute_ekd_loss(alpha_student, alpha_teacher)

        # compute loss_kd
        # logits_student = normalize(logits_student)
        # logits_teacher = normalize(logits_teacher)
        # loss_tckd, loss_nckd = dkd_loss(
        #     logits_student,
        #     logits_teacher,
        #     target,
        #     self.alpha,
        #     self.beta,
        #     self.dkd_temperature,
        #     False,
        # )

        # logits_student = normalize(logits_student)
        # logits_teacher = normalize(logits_teacher)
        alpha_student = torch.exp(logits_student / self.dkd_temperature) + 0.2
        alpha_teacher = torch.exp(logits_teacher / self.dkd_temperature) + 0.2
        loss_tckd, loss_nckd =  evidential_dkd_loss(
            alpha_student,
            alpha_teacher,
            target,
            self.alpha *(1 - rate),
            self.beta,
            self.dkd_temperature
        )

        # loss_ekd =  self.ekd_loss_weight * compute_opinion_loss(evidence_student, evidence_teacher, \
        #     self.lamb2_student, self.lamb2_teacher)
        # loss_ekd = compute_conflict(evidence_student, evidence_teacher, \
        #      self.lamb2_student, self.lamb2_teacher) + compute_opinion_loss(evidence_student, evidence_teacher, \
        #                                                                     self.lamb2_student, self.lamb2_teacher)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_tckd": min(kwargs["epoch"] / self.warmup, 1.0) * loss_tckd,
            "loss_nckd": min(kwargs["epoch"] / self.warmup, 1.0) * loss_nckd
        }
        if 'loss_ekd' in locals():
            losses_dict['loss_ekd'] = loss_ekd
        if 'loss_dkd' in locals():
            losses_dict['loss_dkd'] = loss_dkd

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



