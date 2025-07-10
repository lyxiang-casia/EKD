import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
import math
from ._common import *

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

def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1/dim*torch.ones_like(w1, device=w1.device, dtype=w1.dtype) # [batch,N,1]
    K = torch.exp(-cost / reg)
    Kt= K.transpose(2, 1)
    for i in range(max_iter):
        v=w2/(torch.bmm(Kt,u)+1e-8) #[batch,N,1]
        u=w1/(torch.bmm(K,v)+1e-8)  #[batch,N,1]

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow
        

def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)

    cost_matrix = F.relu(cost_matrix) + 1e-8
    cost_matrix = cost_matrix.to(pred_student.device)
    
    # flow shape [bxnxn]
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return ws_distance


def wkd_logit_loss_with_speration(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None, sinkhorn_lambda=0.05, sinkhorn_iter=10):
        
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

    # N*class
    N, c = logits_student.shape
    s_i = F.log_softmax(logits_student, dim=1)
    t_i = F.softmax(logits_teacher, dim=1)
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1)
    logits_teacher = logits_teacher[mask].reshape(N, -1)
    
    cost_matrix = cost_matrix.repeat(N, 1, 1)
    gd_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    cost_matrix = cost_matrix[gd_mask].reshape(N, c-1, c-1)
        
    # N*class
    loss_wkd = wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda, sinkhorn_iter)

    return loss_t + gamma * loss_wkd


def adaptive_avg_std_pool2d(input_tensor, out_size=(1, 1), eps=1e-5):
    def start_index(a, b, c):
        return int(np.floor(a * c / b))
    def end_index(a, b, c):
        return int(np.ceil((a+1) * c / b))

    b, c, isizeH, isizeW = input_tensor.shape
    if len(out_size) == 2:
        osizeH, osizeW = out_size
    else:
        osizeH = osizeW = out_size

    avg_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # cov_pooled_tensor = torch.zeros((b, c*c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    cov_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # block_list = []
    for oh in range(osizeH):
        istartH = start_index(oh, osizeH, isizeH)
        iendH = end_index(oh, osizeH, isizeH)
        kH = iendH - istartH
        for ow in range(osizeW):
            istartW = start_index(ow, osizeW, isizeW)
            iendW = end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            # avg pool2d
            input_block = input_tensor[:, :, istartH:iendH, istartW:iendW]
            avg_pooled_tensor[:, :, oh, ow] = input_block.mean(dim=(-1, -2))
            # diagonal cov pool2d
            cov_pooled_tensor[:, :, oh, ow] = torch.sqrt(input_block.var(dim=(-1, -2)) + eps)
    
    return avg_pooled_tensor, cov_pooled_tensor


def wkd_feature_loss(f_s, f_t, eps=1e-5, grid=1):
    if grid == 1:
        f_s_avg, f_t_avg = f_s.mean(dim=(-1,-2)), f_t.mean(dim=(-1,-2))
        f_s_std, f_t_std = torch.sqrt(f_s.var(dim=(-1,-2)) + eps), torch.sqrt(f_t.var(dim=(-1,-2)) + eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / f_s.size(0)
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / f_s.size(0)
    elif grid > 1:
        f_s_avg, f_s_std = adaptive_avg_std_pool2d(f_s, out_size=(grid, grid), eps=eps)
        f_t_avg, f_t_std = adaptive_avg_std_pool2d(f_t, out_size=(grid, grid), eps=eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / (grid**2 * f_s.size(0))
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / (grid**2 * f_s.size(0))

    return mean_loss, cov_loss


class EKDWKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(EKDWKD, self).__init__(student, teacher)
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
        self.cfg = cfg
        self.wkd_feature_loss_weight = cfg.WKD.LOSS.WKD_FEAT_WEIGHT
        self.loss_cosine_decay_epoch = cfg.WKD.LOSS.COSINE_DECAY_EPOCH
        self.wkd_feature_mean_cov_ratio = cfg.WKD.MEAN_COV_RATIO
        self.eps = cfg.WKD.EPS

        feat_s_shapes, feat_t_shapes = get_feat_shapes(
        self.student, self.teacher, cfg.WKD.INPUT_SIZE)

        self.hint_layer = cfg.WKD.HINT_LAYER
        self.projector = cfg.WKD.PROJECTOR
        self.spatial_grid = cfg.WKD.SPATIAL_GRID
        if self.projector == "bottleneck":
            self.conv_reg = ConvRegBottleNeck(
                feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], c_hidden=256, use_relu=True, use_bn=True
            )
        elif self.projector == "conv1x1":
            self.conv_reg = ConvReg(
                feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], use_relu=True, use_bn=True
            )
        else:
            raise NotImplementedError(f"Unknown projector type: {self.projector}")

        self.teacher = self.teacher.eval()
        self.student = self.student.eval()


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
        loss_ce = self.ce_loss_weight * loss_ce

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
            # loss_kd = self.kd_loss_weight * kd_loss(
            #     standed_logits_student, standed_logits_teacher, self.temperature
            # )
        # compute second-order loss i.e. loss_ekd
        evidence_student = get_evidence(logits_student / self.temperature, self.efunction_student)
        evidence_teacher = get_evidence(logits_teacher / self.temperature, self.efunction_teacher)
        # alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S) + get_evidence(self.lamb_2nd_order_S, self.efunction_student)
        # alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T) + get_evidence(self.lamb_2nd_order_T, self.efunction_teacher)
        alpha_student = torch.log1p(evidence_student) + torch.exp(self.ce_lamb_S)
        alpha_teacher = torch.log1p(evidence_teacher) + torch.exp(self.ce_lamb_T)
        loss_ekd =  min(kwargs["epoch"] / self.warmup, 1.0) * self.ekd_loss_weight * (self.temperature**2) * compute_ekd_loss(alpha_student, alpha_teacher)
        
       ##  Feature-Based WKD(F)
        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            self.wkd_feature_loss_weight_1 = 0.5*self.wkd_feature_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.cfg.SOLVER.EPOCHS-decay_start_epoch)*math.pi))
           
        else:
            self.wkd_feature_loss_weight_1 = self.wkd_feature_loss_weight

        loss_wkd = 0
        f_t = feature_teacher["feats"][self.hint_layer]
        f_s = feature_student["feats"][self.hint_layer]
        f_s = self.conv_reg(f_s)
        
        mean_loss, cov_loss = wkd_feature_loss(f_s, f_t, self.eps, grid=self.spatial_grid)

        loss_wkd_feat = self.wkd_feature_mean_cov_ratio * mean_loss + cov_loss
        loss_wkd += self.wkd_feature_loss_weight_1 * loss_wkd_feat
    

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_ekd": loss_ekd,
            "loss_wkd": loss_wkd,
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
        # params_to_update = []
        # if isinstance(self.ce_lamb_T, nn.Parameter):
        #     params_to_update.append(self.ce_lamb_T)
        # if isinstance(self.ce_lamb_S, nn.Parameter):
        #     params_to_update.append(self.ce_lamb_S)
        # params_to_update += [v for k, v in self.student.named_parameters()]
        student_params = [v for k, v in self.student.named_parameters()]
        return student_params + list(self.conv_reg.parameters())
        # return params_to_update


