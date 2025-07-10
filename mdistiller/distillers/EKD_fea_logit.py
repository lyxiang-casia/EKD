import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
import math
from .RKD import *

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv), stdv

def stdv_normalize(logit):
    stdv = logit.std(dim=-1, keepdims=True)
    return logit / (1e-7 + stdv), stdv


def evidential_kd_loss(alpha_student, alpha_teacher):
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    log_pred_student = torch.log(alpha_student / S_student)
    pred_teacher = alpha_teacher / S_teacher
    # total_U = -torch.sum(pred_teacher * torch.log(pred_teacher), dim=1)
    # data_U = torch.digamma(S_teacher) - (1 / S_teacher) * torch.sum((alpha_teacher - 1) * torch.digamma(alpha_teacher), dim=1, keepdim=True)
    # data_U = data_U.squeeze(-1)
    # knowl_U = total_U - data_U
    # loss_kd = ((torch.log(torch.tensor(num_class, dtype=torch.float32)) - knowl_U) * F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)).mean()
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return loss_kd

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def get_evidence(logits, efunction):
        if efunction == "relu":
            evidence = F.relu(logits)
        elif efunction == "exp":
            evidence = torch.exp(logits)
        # elif efunction == "softmax":
        #     evidence = F.softmax(logits, dim=1)
        else:
            evidence = F.softplus(logits)
        return evidence

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

def beta_function(alpha):
    """计算多维 Beta 函数 B(alpha)"""
    clamped = torch.clamp((torch.sum(torch.lgamma(alpha)) - torch.lgamma(torch.sum(alpha))), min=-20, max=20)
    return torch.exp(clamped)

def compute_renyi_loss(s_alpha, t_alpha, W):
    term1 = beta_function(W * t_alpha + (1-W) * s_alpha)
    term2 = beta_function(t_alpha)
    term3 = beta_function(s_alpha)
    
    divergence = (1/(W-1)) * torch.log(term1 / (term2 ** W * term3 ** (1-W)))
    return divergence

def test(logits_s, logits_t, lamb_student_ekd, lamb_teacher_ekd):
    evidence_student = get_evidence(logits_s, 'softplus')
    evidence_teacher = get_evidence(logits_t, 'softplus')
    alpha_student = evidence_student + lamb_student_ekd
    alpha_teacher = evidence_teacher + lamb_teacher_ekd
    return compute_ekd_loss(alpha_student, alpha_teacher)

def expected_kd_ce_loss(logits_student_in, logits_teacher_in, logit_stand, T):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    num_class = logits_teacher.size(1)

    evidence_student = torch.exp(logits_student / T)
    evidence_teacher = torch.exp(logits_teacher / T)

    alpha_student = evidence_student + 0.2
    alpha_teacher = evidence_teacher + 0.2

    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_teacher = alpha_teacher / S_teacher
    temp = torch.digamma(S_student) - torch.digamma(alpha_student)
    loss_kd = torch.sum(pred_teacher * temp, dim=1).mean()
    # loss_kd *= T
    return loss_kd

def evidential_kd_ce_loss(logits_student_in, logits_teacher_in, logit_stand, T):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    num_class = logits_teacher.size(1)

    evidence_student = torch.exp(logits_student / T)
    evidence_teacher = torch.exp(logits_teacher / T)

    alpha_student = evidence_student + 0.2
    alpha_teacher = evidence_teacher + 0.2

    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    log_pred_student = torch.log(alpha_student / S_student)
    pred_teacher = alpha_teacher / S_teacher
    loss_kd = - torch.sum(log_pred_student * pred_teacher, dim=1).mean()
    # loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= T**2
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

def compute_feature_loss(feature_s, feature_t):
    feature_loss = 0


    return feature_loss



def stable_digamma_sum_exp(z, lamb=None):
    """
    计算 digamma(sum(exp(z)) + K * exp(lamb))，其中 K 是 z 中类别的个数
    
    参数:
    - z (torch.Tensor): logits 向量或标量
    - lamb (float or torch.Tensor): 常数 lamb (如果需要的话)
    
    返回:
    - digamma 计算结果 (torch.Tensor)
    """
    if lamb is None:
        raise ValueError("lamb 参数不能为空")

    # 获取类别数 K
    K = z.size(-1)

    # 数值稳定化
    max_z = torch.max(z)
    max_val = torch.max(max_z, torch.tensor(lamb))
    
    # 计算 sum(exp(z_i)) + K * exp(lamb) 的数值稳定版本
    exp_sum_stable = torch.sum(torch.exp(z - max_val), dim=-1, keepdim=True) + K * torch.exp(torch.tensor(lamb) - max_val)
    
    # 恢复到原始的指数表达式 sum(exp(z_i)) + K * exp(lamb)
    exp_sum = torch.exp(max_val) * exp_sum_stable
    
    # 计算 digamma(sum(exp(z_i)) + K * exp(lamb))
    result = torch.digamma(exp_sum)
    
    return result

class EKD_fea_logit(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(EKD_fea_logit, self).__init__(student, teacher)
        self.temperature = cfg.EKD.LOSS.TEMPERATURE
        self.ce_loss_weight = cfg.EKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.EKD.LOSS.KD_WEIGHT
        self.ekd_loss_weight = cfg.EKD.LOSS.EKD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.efunction_student = cfg.EKD.STUDENT.CLF_TYPE
        self.efunction_teacher = cfg.EKD.TEACHER.CLF_TYPE

        if cfg.EKD.STUDENT.LAMB == 0.0:
            self.ce_lamb_S = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.ce_lamb_S = torch.tensor(cfg.EKD.STUDENT.LAMB)
        if cfg.EKD.TEACHER.LAMB == 0.0:
            self.ce_lamb_T = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.ce_lamb_T = torch.tensor(cfg.EKD.TEACHER.LAMB) #first-order lamb
        self.expectation = cfg.EKD.LOSS.EXPECTATION
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.warmup = cfg.EKD.WARMUP
        self.KL = cfg.EKD.LOSS.KL

        # self.temp_S = 0.0
        # self.temp_T = 0.0

        self.distance_weight = cfg.RKD.DISTANCE_WEIGHT
        self.angle_weight = cfg.RKD.ANGLE_WEIGHT
        # self.ce_loss_weight = cfg.RKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.RKD.LOSS.FEAT_WEIGHT
        self.eps = cfg.RKD.PDIST.EPSILON
        self.squared = cfg.RKD.PDIST.SQUARED



    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        
        # compute loss_ce
        # logits_student_ce = torch.clamp(logits_student, min=1e-6, max=80.0) 
        #  - torch.max(logits_student, dim=-1, keepdim=True)[0]
        evidence_student = torch.exp(logits_student)
        # evidence_student = torch.clamp(evidence_student, min=1e-6, max=1e6)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student) - torch.digamma(alpha_student)), dim=-1).mean()

        # logits_student_ce = logits_student + torch.log1p(torch.exp(self.ce_lamb_S - logits_student))
        # loss_ce = F.cross_entropy(logits_student_ce, target)
        # loss_ce = self.ce_loss_weight * loss_ce

        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # compute first-order loss i.e. loss_kd
        standed_logits_student = normalize(logits_student) if self.logit_stand else logits_student
        standed_logits_teacher = normalize(logits_teacher) if self.logit_stand else logits_teacher
        evidence_student = torch.exp(standed_logits_student / self.temperature)
        evidence_teacher = torch.exp(standed_logits_teacher / self.temperature)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.ce_lamb_T)

        # self.kd_loss_weight = 9.0 - (kwargs["epoch"] >= self.warmup) * self.ekd_loss_weight

        loss_kd =  self.kd_loss_weight * (self.temperature**2) * evidential_kd_loss(
                alpha_student, alpha_teacher
        )

        loss_kd = self.kd_loss_weight * kd_loss(
            standed_logits_student, standed_logits_teacher, self.temperature
        )
        # compute second-order loss i.e. loss_ekd
        evidence_student = get_evidence(logits_student / self.temperature, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher / self.temperature, self.efunction_teacher)
        # alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S) + get_evidence(self.lamb_2nd_order_S, self.efunction_student)
        # alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T) + get_evidence(self.lamb_2nd_order_T, self.efunction_teacher)
        alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S)
        alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T)
        loss_ekd =  min(kwargs["epoch"] / self.warmup, 1.0) * self.ekd_loss_weight * (self.temperature**2) * compute_ekd_loss(alpha_student, alpha_teacher)

        
        # loss_rkd = self.feat_loss_weight * rkd_loss(
        #     feature_student["pooled_feat"],
        #     feature_teacher["pooled_feat"],
        #     self.squared,
        #     self.eps,
        #     self.distance_weight,
        #     self.angle_weight,
        # )

        loss_ekd =  self.ekd_loss_weight * (logits_student - logits_teacher) ** 2
        
        #  min(kwargs["epoch"] / self.warmup, 1.0) *

        # import wandb

        # wandb.log({"ekd(evidence)": compute_ekd_loss(evidence_student, evidence_teacher).item()})

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_ekd": loss_ekd,
            # "loss_rkd": loss_rkd,
        }
        if self.KL:
            kl_alpha = evidence_student * (1 - labels_1hot) + torch.exp(self.ce_lamb_S)
            epoch = kwargs["epoch"]
            loss_kl = np.minimum(1.0, epoch / 30.) * compute_kl_loss(kl_alpha, torch.exp(self.ce_lamb_S))
            losses_dict["loss_kl"] = self.ce_loss_weight * loss_kl
        # if kwargs["epoch"] < self.warmup:
        #     losses_dict["loss_kd"] = loss_kd * 2
        # else:
        #     losses_dict["loss_ekd"] = loss_ekd
        return logits_student, losses_dict
    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        params_to_update = []
        if isinstance(self.ce_lamb_T, nn.Parameter):
            params_to_update.append(self.ce_lamb_T)
        if isinstance(self.ce_lamb_S, nn.Parameter):
            params_to_update.append(self.ce_lamb_S)
        params_to_update += [v for k, v in self.student.named_parameters()]
        return params_to_update


