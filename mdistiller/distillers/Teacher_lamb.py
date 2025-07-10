import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import numpy as np

# def normalize(logit):
#     mean = logit.mean(dim=-1, keepdims=True)
#     stdv = logit.std(dim=-1, keepdims=True)
#     return (logit - mean) / (1e-7 + stdv)

# def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
#     logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
#     logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature**2
#     return loss_kd

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

class Teacher_lamb(nn.Module):
    def __init__(self, student, cfg, CLF_TYPE, LAMB2, KL):
        super(Teacher_lamb, self).__init__()
        self.student = student
        self.lamb1 = cfg.DISTILLER.LAMB1
        self.lamb2 = LAMB2
        self.efunction = CLF_TYPE
        self.KL = KL

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _, lamb2 = self.student(image)
        lamb2 = lamb2.unsqueeze(1)
        if self.efunction == "relu":
            evidence = F.relu(logits_student)
            lamb2 = F.relu(lamb2)
        elif self.efunction == "exp":
            evidence = torch.exp(logits_student)
            lamb2 = torch.exp(lamb2)
        elif self.efunction == "sigmoid":
            evidence = F.sigmoid(logits_student)
            lamb2 = F.sigmoid(lamb2)
        elif self.efunction == "softmax":
            evidence = F.softmax(logits_student,dim=1)
            lamb2 = F.softmax(lamb2, dim=1)
        else:
            evidence = F.softplus(logits_student)
            lamb2 = F.softplus(lamb2)
        alpha = evidence + lamb2
        # compute loss_mse
        # labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)

        # num_classes = evidence.shape[-1]

        # gap = labels_1hot - (evidence + self.lamb2) / \
        #       (evidence + self.lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + self.lamb2 * num_classes)

        # loss_mse = gap.pow(2).sum(-1).mean()

        # compute loss_ce
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S)-torch.digamma(alpha)), dim=-1).mean()

        losses_dict = {
            "loss_ce": loss_ce
        }
        #compute loss_kl    
        if self.KL:
            kl_alpha = evidence * (1 - labels_1hot) + self.lamb2
            epoch = kwargs.get('epoch', 0)
            loss_kl = np.minimum(1.0, epoch / 10.) * compute_kl_loss(kl_alpha, self.lamb2)
            losses_dict["loss_kl"] = loss_kl
                
        return logits_student, losses_dict

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]



