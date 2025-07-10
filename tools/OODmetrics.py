import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics

import wandb
import pandas as pd
from PIL import Image as im
from tqdm import tqdm


def compute_X_Y_alpha(model, loader, device, lamb):

    X_all, Y_all, model_pred_all = [], [], []

    for batch_index, (X, Y) in tqdm(enumerate(loader), total=len(loader), desc="Processing Batches"):
        X = X.float()
        X = X .to(device)
        Y = Y.to(device)

        if lamb < 0:
            model_pred = model(image=X) + torch.exp(torch.tensor(lamb))
        else:
            model_pred = model(image=X)

        X_all.append(X.to("cpu"))
        Y_all.append(Y.to("cpu"))
        model_pred_all.append(model_pred.to("cpu"))

    X_all = torch.cat(X_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    model_pred_all = torch.cat(model_pred_all, dim=0)

    return Y_all, X_all, model_pred_all

def our_anomaly_detection(alpha, ood_alpha, uncertainty_type='max_prob', lamb=0.0):
    if lamb == 0.0:
        if uncertainty_type == 'max_prob':
            p = torch.nn.functional.softmax(alpha, dim=-1)
            ood_p = torch.nn.functional.softmax(ood_alpha, dim=-1)
            scores = p.max(-1)[0].cpu().detach().numpy()
            ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()
        elif uncertainty_type == 'discrete_entropy':
            # "alpha" denotes the original model ouput, i.e., logits
            # shape of alpha: [n_models, n_samples, n_classes]
            alpha = torch.stack(alpha)
            eps = 1e-10
            id_probabilities = torch.nn.functional.softmax(alpha, dim=-1)
            scores = -torch.sum(id_probabilities * torch.log(id_probabilities + eps), dim=-1)
            scores = torch.mean(scores, dim=0).cpu().detach().numpy()
            ood_alpha = torch.stack(ood_alpha)
            ood_probabilities = torch.nn.functional.softmax(ood_alpha, dim=-1)
            ood_scores = -torch.sum(ood_probabilities * torch.log(ood_probabilities + eps), dim=-1)
            ood_scores = torch.mean(ood_scores, dim=0).cpu().detach().numpy()
    else:
        if uncertainty_type == 'alpha0':
            scores = alpha.sum(-1).cpu().detach().numpy()
            ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
        elif uncertainty_type == 'max_alpha':
            scores = alpha.max(-1)[0].cpu().detach().numpy()
            ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()
        elif uncertainty_type == 'max_prob':
            p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
            scores = p.max(-1)[0].cpu().detach().numpy()

            ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
            ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()
        elif uncertainty_type == 'max_modified_prob':
            num_classes = alpha.shape[-1]
            evidence = alpha - lamb
            S = evidence + (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb * num_classes
            p = alpha / S
            scores = p.max(-1)[0].cpu().detach().numpy()

            ood_evidence = ood_alpha - lamb
            ood_S = ood_evidence + (torch.sum(ood_evidence, dim=-1, keepdim=True) - ood_evidence) + lamb * num_classes
            ood_p = ood_alpha / ood_S
            ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

        elif uncertainty_type == 'differential_entropy':
            eps = 1e-6
            alpha = alpha + eps
            ood_alpha = ood_alpha + eps
            alpha0 = alpha.sum(-1)
            ood_alpha0 = ood_alpha.sum(-1)

            id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
            id_digamma_term = torch.sum((alpha - lamb) * (
                        torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), dim=-1)
            id_differential_entropy = id_log_term - id_digamma_term

            ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
            ood_digamma_term = torch.sum((ood_alpha - lamb) * (torch.digamma(ood_alpha) - torch.digamma(
                (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), dim=-1)
            ood_differential_entropy = ood_log_term - ood_digamma_term

            scores = - id_differential_entropy.cpu().detach().numpy()
            ood_scores = - ood_differential_entropy.cpu().detach().numpy()
        elif uncertainty_type == 'mutual_information':
            eps = 1e-6
            alpha = alpha + eps
            ood_alpha = ood_alpha + eps
            alpha0 = alpha.sum(-1)
            ood_alpha0 = ood_alpha.sum(-1)
            probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
            ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)

            id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
            id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
                alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
            id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
            id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
            id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

            ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=1)
            ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
                ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0)
            ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)
            ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=1)
            ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty

            scores = - id_distributional_uncertainty.cpu().detach().numpy()
            ood_scores = - ood_distributional_uncertainty.cpu().detach().numpy()
        else:
            raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    # if save_path is not None:
    #     if uncertainty_type in ['differential_entropy', 'mutual_information']:
    #         scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))
    #     else:
    #         scores_norm = (scores - min(scores)) / (max(scores) - min(scores))

    #     np.save(save_path, scores_norm)
    #     # results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
    #     # results_df = pd.DataFrame(results)
    #     # results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    return aupr, auroc, scores, ood_scores

