import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import *

import math
import numpy as np


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


class WKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(WKD, self).__init__(student, teacher)
        self.cfg = cfg

        self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
        self.wkd_logit_loss_weight = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT
        self.wkd_feature_loss_weight = cfg.WKD.LOSS.WKD_FEAT_WEIGHT
        self.loss_cosine_decay_epoch = cfg.WKD.LOSS.COSINE_DECAY_EPOCH

        self.enable_wkdl = self.wkd_logit_loss_weight > 0
        self.enable_wkdf = self.wkd_feature_loss_weight > 0

        # WKD-L: WD for logits distillation
        if self.enable_wkdl:
            self.temperature = cfg.WKD.TEMPERATURE
            self.sinkhorn_lambda = cfg.WKD.SINKHORN.LAMBDA
            self.sinkhorn_iter = cfg.WKD.SINKHORN.ITER

            if cfg.WKD.COST_MATRIX == "fc":
                print("Using fc weight of teacher model as category prototype")
                self.prototype = self.teacher.fc.weight
                # caluate cosine similarity
                proto_normed = F.normalize(self.prototype, p=2, dim=-1)
                cosine_sim = proto_normed.matmul(proto_normed.transpose(-1, -2))
                self.dist = 1 - cosine_sim
            else:
                print("Using "+cfg.WKD.COST_MATRIX+" as cost matrix")
                path_gd = cfg.WKD.COST_MATRIX_PATH
                self.dist = torch.load(path_gd).cuda().detach()
                
            if cfg.WKD.COST_MATRIX_SHARPEN != 0:
                print("Sharpen ", cfg.WKD.COST_MATRIX_SHARPEN)
                sim = torch.exp(-cfg.WKD.COST_MATRIX_SHARPEN*self.dist)
                self.dist = 1 - sim

        # WKD-F: WD for feature distillation
        if self.enable_wkdf:
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

    def get_learnable_parameters(self):
        student_params = [v for k, v in self.student.named_parameters()]
        if self.enable_wkdf:
            return student_params + list(self.conv_reg.parameters())
        else:
            return student_params

    def get_extra_parameters(self):
        return 0

    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, feats_student = self.student(image)
            with torch.no_grad():
                logits_teacher, feats_teacher = self.teacher(image)

        logits_student = logits_student.to(torch.float32)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            self.wkd_logit_loss_weight_1 = 0.5*self.wkd_logit_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.cfg.SOLVER.EPOCHS-decay_start_epoch)*math.pi))
            self.wkd_feature_loss_weight_1 = 0.5*self.wkd_feature_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.cfg.SOLVER.EPOCHS-decay_start_epoch)*math.pi))
           
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
            self.wkd_feature_loss_weight_1 = self.wkd_feature_loss_weight

        loss_wkd = 0
        # WD for logits distillation
        if self.enable_wkdl:
            logits_teacher = logits_teacher.to(torch.float32)
            loss_wkd_logit = wkd_logit_loss_with_speration(logits_student, logits_teacher, target, self.temperature, self.wkd_logit_loss_weight_1, self.dist, self.sinkhorn_lambda, self.sinkhorn_iter)
            loss_wkd += loss_wkd_logit


        # WD for feature distillation
        if self.enable_wkdf:
            f_t = feats_teacher["feats"][self.hint_layer].to(torch.float32)
            f_s = feats_student["feats"][self.hint_layer].to(torch.float32)
            f_s = self.conv_reg(f_s)
            
            mean_loss, cov_loss = wkd_feature_loss(f_s, f_t, self.eps, grid=self.spatial_grid)

            loss_wkd_feat = self.wkd_feature_mean_cov_ratio * mean_loss + cov_loss
            loss_wkd += self.wkd_feature_loss_weight_1 * loss_wkd_feat


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_wkd,
        }

        return logits_student, losses_dict