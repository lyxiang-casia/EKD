from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def evidential_cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    alpha_student = torch.exp(logits_student / temperature) + 0.2
    alpha_teacher = torch.exp(logits_teacher / temperature) + 0.2
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_student = alpha_student / S_student
    pred_teacher = alpha_teacher / S_teacher
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def evidential_bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    alpha_student = torch.exp(logits_student / temperature) + 0.2
    alpha_teacher = torch.exp(logits_teacher / temperature) + 0.2
    S_student = torch.sum(alpha_student, dim=1, keepdim=True)
    S_teacher = torch.sum(alpha_teacher, dim=1, keepdim=True)
    pred_student = alpha_student / S_student
    pred_teacher = alpha_teacher / S_teacher
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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

def evidential_kd_loss(logits_student_in, logits_teacher_in, T, logit_stand):
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
    # total_U = -torch.sum(pred_teacher * torch.log(pred_teacher), dim=1)
    # data_U = torch.digamma(S_teacher) - (1 / S_teacher) * torch.sum((alpha_teacher - 1) * torch.digamma(alpha_teacher), dim=1, keepdim=True)
    # data_U = data_U.squeeze(-1)
    # knowl_U = total_U - data_U
    # loss_kd = ((torch.log(torch.tensor(num_class, dtype=torch.float32)) - knowl_U) * F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)).mean()
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= T**2
    return loss_kd

def evidential_ce_loss(logits_student, target):
    alpha = get_evidence(logits_student, "exp") + 0.2
    labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
    S_student = torch.sum(alpha, dim=-1, keepdim=True)
    loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student)-torch.digamma(alpha)), dim=-1).mean()
    return loss_ce

def compute_ekd_loss(s_logits, t_logits, s_efunction, t_efunction, s_lamb, t_lamb):
    s_alpha = get_evidence(s_logits, s_efunction) + s_lamb
    t_alpha = get_evidence(t_logits, t_efunction) + t_lamb
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




class MLKD4eModel(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MLKD4eModel, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.EKD.LOSS.CE_WEIGHT
        self.ekd_loss_weight = cfg.EKD.LOSS.EKD_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.lamb_student_ekd = cfg.EKD.STUDENT.LAMB2
        self.lamb_teacher_ekd = cfg.EKD.TEACHER.LAMB2
        self.efunction_teacher = cfg.EKD.TEACHER.CLF_TYPE
        self.efunction_student = cfg.EKD.STUDENT.CLF_TYPE


    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

 
        loss_ekd = self.ekd_loss_weight * (compute_ekd_loss(logits_student_weak, logits_teacher_weak, self.efunction_student, self.efunction_teacher, self.lamb_student_ekd, self.lamb_teacher_ekd) + \
                                            compute_ekd_loss(logits_student_strong, logits_teacher_strong, self.efunction_student, self.efunction_teacher, self.lamb_student_ekd, self.lamb_teacher_ekd))


        alpha_teacher_weak = torch.exp(logits_teacher_weak) + 0.2
        S_teacher_weak = torch.sum(alpha_teacher_weak, dim=1, keepdim=True)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = alpha_teacher_weak / S_teacher_weak

        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        # loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_ce = self.ce_loss_weight * (evidential_ce_loss(logits_student_weak, target) + evidential_ce_loss(logits_student_strong, target))
        

        loss_kd_weak = (self.kd_loss_weight - self.ekd_loss_weight) * ((evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + (self.kd_loss_weight - self.ekd_loss_weight) * ((evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + (self.kd_loss_weight - self.ekd_loss_weight) * ((evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + (self.kd_loss_weight - self.ekd_loss_weight) * ((evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + (self.kd_loss_weight - self.ekd_loss_weight) * ((evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean())

        loss_kd_strong = (self.kd_loss_weight - self.ekd_loss_weight) * evidential_kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + (self.kd_loss_weight - self.ekd_loss_weight) * evidential_kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
            logit_stand=self.logit_stand,
        ) + (self.kd_loss_weight - self.ekd_loss_weight) * evidential_kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
            logit_stand=self.logit_stand,
        ) + (self.kd_loss_weight - self.ekd_loss_weight) * evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
        ) + (self.kd_loss_weight - self.ekd_loss_weight) * evidential_kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
        )

        loss_cc_weak = self.kd_loss_weight * ((evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_cc_strong = self.kd_loss_weight * evidential_cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * evidential_cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * evidential_cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * evidential_cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )
        loss_bc_weak = self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        loss_bc_strong = self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((evidential_bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            6.0,
        ) * mask).mean())
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_ekd": loss_ekd,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }
        return logits_student_weak, losses_dict

