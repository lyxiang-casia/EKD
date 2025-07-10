import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

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



class CAT_KD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(CAT_KD, self).__init__(student, teacher)
        # self.ce_loss_weight = cfg.CAT_KD.LOSS.CE_WEIGHT
        self.CAT_loss_weight = cfg.CAT_KD.LOSS.CAT_loss_weight
        self.onlyCAT = cfg.CAT_KD.onlyCAT
        self.CAM_RESOLUTION = cfg.CAT_KD.LOSS.CAM_RESOLUTION
        self.relu = nn.ReLU()
        
        self.IF_NORMALIZE = cfg.CAT_KD.IF_NORMALIZE
        self.IF_BINARIZE = cfg.CAT_KD.IF_BINARIZE
        
        self.IF_OnlyTransferPartialCAMs = cfg.CAT_KD.IF_OnlyTransferPartialCAMs
        self.CAMs_Nums = cfg.CAT_KD.CAMs_Nums
        # 0: select CAMs with top x predicted classes
        # 1: select CAMs with the lowest x predicted classes
        self.Strategy = cfg.CAT_KD.Strategy


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
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)       
        tea = feature_teacher["feats"][-1]
        stu = feature_student["feats"][-1]
                        
        # perform binarization
        if self.IF_BINARIZE:
            n,c,h,w = tea.shape
            threshold = torch.norm(tea, dim=(2,3), keepdim=True, p=1)/(h*w)
            tea =tea - threshold
            tea = self.relu(tea).bool() * torch.ones_like(tea)
        
        
        # only transfer CAMs of certain classes
        if self.IF_OnlyTransferPartialCAMs:
            n,c,w,h = tea.shape
            with torch.no_grad():
                if self.Strategy==0:
                    l = torch.sort(logits_teacher, descending=True)[0][:, self.CAMs_Nums-1].view(n,1)
                    mask = self.relu(logits_teacher-l).bool()
                    mask = mask.unsqueeze(-1).reshape(n,c,1,1)
                elif self.Strategy==1:
                    l = torch.sort(logits_teacher, descending=True)[0][:, 99-self.CAMs_Nums].view(n,1)
                    mask = self.relu(logits_teacher-l).bool()
                    mask = ~mask.unsqueeze(-1).reshape(n,c,1,1)
            tea,stu = _mask(tea,stu,mask)

        loss_feat = self.CAT_loss_weight * CAT_loss(
            stu, tea, self.CAM_RESOLUTION, self.IF_NORMALIZE
        )
         
        evidence_student = torch.exp(logits_student)
        # evidence_student = torch.clamp(evidence_student, min=1e-6, max=1e6)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        labels_1hot = torch.zeros_like(logits_student).scatter_(-1, target.unsqueeze(-1), 1)
        S_student = torch.sum(alpha_student, dim=-1, keepdim=True)
        loss_ce = torch.sum(labels_1hot * (torch.digamma(S_student) - torch.digamma(alpha_student)), dim=-1).mean()
        standed_logits_student = normalize(logits_student) if self.logit_stand else logits_student
        standed_logits_teacher = normalize(logits_teacher) if self.logit_stand else logits_teacher
        evidence_student = torch.exp(standed_logits_student / self.temperature)
        evidence_teacher = torch.exp(standed_logits_teacher / self.temperature)
        alpha_student = evidence_student + torch.exp(self.ce_lamb_S)
        alpha_teacher = evidence_teacher + torch.exp(self.ce_lamb_T)

        loss_kd =  self.kd_loss_weight * (self.temperature**2) * evidential_kd_loss(
                alpha_student, alpha_teacher
        )
            # loss_kd = self.kd_loss_weight * kd_loss(
            #     standed_logits_student, standed_logits_teacher, self.temperature
            # )
        # compute second-order loss i.e. loss_ekd
        evidence_student = get_evidence(logits_student / self.temperature, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher / self.temperature, self.efunction_teacher)
        alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S)
        alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T)
        loss_ekd =  min(kwargs["epoch"] / self.warmup, 1.0) * self.ekd_loss_weight * (self.temperature**2) * compute_ekd_loss(alpha_student, alpha_teacher)

        if self.onlyCAT is False:
            # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            losses_dict = {
                "loss_CE": loss_ce,
                "loss_CAT": loss_feat,
                "loss_kd": loss_kd,
                "loss_ekd": loss_ekd,
            }
        else:
            losses_dict = {
                "loss_CAT": loss_feat,
            }

        return logits_student, losses_dict


def _Normalize(feat,IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat,dim=(2,3))
    return feat

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE):   
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE))
    return loss
    

def _mask(tea,stu,mask):
    n,c,w,h = tea.shape
    mid = torch.ones(n,c,w,h).cuda()
    mask_temp = mask.view(n,c,1,1)*mid.bool()
    t=torch.masked_select(tea, mask_temp)
    
    if (len(t))%(n*w*h)!=0:
        return tea, stu

    n,c,w_stu,h_stu = stu.shape
    mid = torch.ones(n,c,w_stu,h_stu).cuda()
    mask = mask.view(n,c,1,1)*mid.bool()
    stu=torch.masked_select(stu, mask)
    
    return t.view(n,-1,w,h), stu.view(n,-1,w_stu,h_stu)