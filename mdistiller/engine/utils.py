import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ekd_loss(s_logits, t_logits):
    evidence_student = F.softplus(s_logits)
    evidence_teacher = F.softplus(t_logits)
    s_alpha = evidence_student + 7.0
    t_alpha = evidence_teacher + 7.0
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


def evidential_kd_loss(logits_student_in, logits_teacher_in):
    logits_student =  logits_student_in
    logits_teacher =  logits_teacher_in

    # evidence_student = torch.exp(logits_student / T)
    # evidence_teacher = torch.exp(logits_teacher / T)
    evidence_student = F.softplus(logits_student)
    evidence_teacher = F.softplus(logits_teacher)

    alpha_student = evidence_student + 7.0
    alpha_teacher = evidence_teacher + 7.0

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


def ece_loss(probabilities, labels, n_bins=10):
    """
    计算 Expected Calibration Error (ECE) 损失。

    :param probabilities: 模型输出的概率张量，形状为 (N, C)，其中 N 是样本数，C 是类别数。
    :param labels: 真实标签，形状为 (N,)，每个样本的标签是整数。
    :param n_bins: 置信度分桶数，默认 10。
    :return: ECE 损失值。
    """
    N = probabilities.size(0)  # 样本数
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).cuda()  # 创建置信度分桶

    ece = 0.0  # 初始化 ECE 损失

    # 遍历每个置信度区间
    for i in range(n_bins):
        bin_start = bin_boundaries[i].item()
        bin_end = bin_boundaries[i+1].item()

        # 找出所有在该区间内的样本
        in_bin = (probabilities.max(dim=1)[0] >= bin_start) & (probabilities.max(dim=1)[0] < bin_end)
        bin_samples = probabilities[in_bin]
        bin_labels = labels[in_bin]

        if len(bin_samples) == 0:
            continue
        
        # 计算当前区间内的准确度和平均置信度
        accuracy = (bin_samples.argmax(dim=1) == bin_labels).float().mean()
        confidence = bin_samples.max(dim=1)[0].mean()

        # ECE 损失的加权贡献
        bin_weight = len(bin_samples) / N
        ece += bin_weight * abs(accuracy - confidence)

    return ece


def validate(val_loader, distiller):
    batch_time, losses, top1, top5, ece_meter = [AverageMeter() for _ in range(5)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    # student.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            ######ECE########
            probabilities = torch.exp(output) + torch.exp(torch.tensor(-1.22))
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            # probabilities = F.softmax(output, dim=1)
            ece = ece_loss(probabilities, target, n_bins=10)
            ece_meter.update(ece.item(), batch_size)

            ######Theorical Analysis########
            # logits_student, logits_teacher = distiller(image=image)
            # loss = 0.1 * loss
            # # loss_kd = 9.0 * compute_ekd_loss(logits_student, logits_teacher)
            # loss_kd = 9.0 * evidential_kd_loss(logits_student, logits_teacher)
            # losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            # losses_kd.update(loss_kd.cpu().detach().numpy().mean(), batch_size)
            ######End#######################

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}| ECE:{ece.avg:.3f}".format(
                top1=top1, top5=top5, ece=ece_meter
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg, ece_meter.avg


def validate_npy(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        start_eval = True
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            output = nn.Softmax()(output)
            if start_eval:
                all_image = image.float().cpu()
                all_output = output.float().cpu()
                all_label = target.float().cpu()
                start_eval = False
            else:
                all_image = torch.cat((all_image, image.float().cpu()), dim=0)
                all_output = torch.cat((all_output, output.float().cpu()), dim=0)
                all_label = torch.cat((all_label, target.float().cpu()), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    all_image, all_output, all_label = all_image.numpy(), all_output.numpy(), all_label.numpy()
    pbar.close()
    return top1.avg, top5.avg, losses.avg, all_image, all_output, all_label


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)
