import torch
from torchprof import Profiler

def compute_loss(s_alpha, t_alpha):
    s_S = torch.sum(s_alpha, dim=1)
    t_S = torch.sum(t_alpha, dim=1)
    loss_term1 = torch.lgamma(t_S) - torch.lgamma(s_S)
    loss_term2 = torch.sum(torch.lgamma(t_alpha) - torch.lgamma(s_alpha), dim=1)
    loss_term3 = torch.sum((t_alpha - s_alpha) * (torch.digamma(t_alpha) - torch.digamma(t_S)), dim=1)
    loss = (loss_term1 + loss_term2 + loss_term3).mean()
    return loss

B, K = 8, 100
s_alpha = torch.rand(B, K).cuda()
t_alpha = torch.rand(B, K).cuda()

with Profiler() as prof:
    compute_loss(s_alpha, t_alpha)

prof.print_table()