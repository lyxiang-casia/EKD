import os
import sys
import argparse
import yaml
import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import random
import numpy as np
from multiprocessing import Process

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from mdistiller.models.cifar.resnet_activ import SwiGLU, GEGLU

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

activation_mapping = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "geglu": GEGLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
}

def main(cfg, resume, opts, seed):
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    ####### Experiment Setting #######
    experiment_name = cfg.EXPERIMENT.NAME
    experiment_name += f'seed{seed}'
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    if cfg.LOG.WANDB:
        try:
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)


        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False
    
    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if 'MLKD' in cfg.DISTILLER.TYPE:
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)


    ###### Distiller Setting #########
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # Teacher("vanilla" for single evidential model)
    elif "Teacher" in cfg.DISTILLER.TYPE:
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, cfg)

    # Distillation with both student and teacher models
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            if 'activ' in cfg.DISTILLER.TEACHER:
                model_teacher = net(num_classes=num_classes, activation=activation_mapping.get(cfg.EKD.TEACHER.ACTIVATION, nn.ReLU))
            else:
                model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            # student model
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        # Loading Distiller
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE not in ["NONE", "Teacher", "Teacher_lamb"]:
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        cfg.EXPERIMENT.PROJECT, experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--sweep", type=str, default="", help="path to the sweep config file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("--base-temp", type=float, default=2)
    parser.add_argument("--kd-weight", type=float, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD','DKD','MLKD']:
        cfg.EXPERIMENT.LOGIT_STAND = True
        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'MLKD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
    cfg.freeze()

    main(cfg, args.resume, args.opts, args.seed)
