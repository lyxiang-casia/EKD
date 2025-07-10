import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

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


class Monitor(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(Monitor, self).__init__(student, teacher)
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
        self.lamb_2nd_order_S = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) #second-order lamb
        self.lamb_2nd_order_T = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) #second-order lamb

        self.expectation = cfg.EKD.LOSS.EXPECTATION
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
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

        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # compute first-order loss i.e. loss_kd
        standed_logits_student = normalize(logits_student) if self.logit_stand else logits_student
        standed_logits_teacher = normalize(logits_teacher) if self.logit_stand else logits_teacher
        evidence_student = torch.exp(standed_logits_student / self.temperature)
        evidence_teacher = torch.exp(standed_logits_teacher / self.temperature)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.ce_lamb_T)
        if self.expectation:
            loss_kd = self.kd_loss_weight * (self.temperature**2) * expected_kd_ce_loss(
                alpha_student, alpha_teacher
            )
        else:
            loss_kd = self.kd_loss_weight * (self.temperature**2) * evidential_kd_loss(
                    alpha_student, alpha_teacher
            )
            # loss_kd = self.kd_loss_weight * kd_loss(
            #     standed_logits_student, standed_logits_teacher, self.temperature
            # )

        # compute second-order loss i.e. loss_ekd
        evidence_student = get_evidence(logits_student, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher, self.efunction_teacher)
        # alpha_student = evidence_student + get_evidence(self.lamb_2nd_order_S, self.efunction_student)
        # alpha_teacher = evidence_teacher + get_evidence(self.lamb_2nd_order_T, self.efunction_teacher)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.ce_lamb_T)
        loss_ekd = min(kwargs["epoch"] / self.warmup, 1.0) * self.ekd_loss_weight  * compute_ekd_loss(alpha_student, alpha_teacher)

        # import wandb

        # wandb.log({"ekd(evidence)": compute_ekd_loss(evidence_student, evidence_teacher).item()})

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_ekd": loss_ekd,
        }
        # if kwargs["epoch"] < self.warmup:
        #     losses_dict["loss_kd"] = loss_kd * 2
        # else:
        #     losses_dict["loss_ekd"] = loss_ekd
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
