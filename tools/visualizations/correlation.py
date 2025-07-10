import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import sys, os
sys.path.append(os.path.join(os.getcwd(),'../..'))
print(os.getcwd())

from mdistiller.models import cifar_model_dict, cifar_emodel_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg

def get_output_metric(model, val_loader, num_classes=100):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            outputs, _ = model(data)
            preds = outputs
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))
    for p, l in zip(all_preds, all_labels):
        cnt[l, 0] += 1
        matrix[l] += p
    matrix /= cnt
    return matrix

def get_tea_stu_diff(tea, stu, mpath):
    cfg.defrost()
    cfg.DISTILLER.STUDENT = stu
    cfg.DISTILLER.TEACHER = tea
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = cifar_emodel_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    model.load_state_dict(load_checkpoint(mpath)["model"])
    tea_model = cifar_model_dict[cfg.DISTILLER.TEACHER][0](num_classes=num_classes)
    tea_model.load_state_dict(load_checkpoint(cifar_model_dict[cfg.DISTILLER.TEACHER][1])["model"])
    print("load model successfully!")
    ms = get_output_metric(model, val_loader)
    mt = get_output_metric(tea_model, val_loader)
    max_diff = np.max(np.abs(ms - mt))
    diff = np.abs((ms - mt)) / max_diff
    for i in range(100):
        diff[i, i] = 0
    print('max(diff):', diff.max())
    print('mean(diff):', diff.mean())
    seaborn.heatmap(diff, vmin=0, vmax=1.0, cmap="PuBuGn")
    plt.savefig("diff_heatmap.png", dpi=300, bbox_inches="tight")

mpath = "/data_SSD2/mmc_lyxiang/KD/output/Resnet32x4_Resnet8x4(226)/KDseed1/student_best"
diff = get_tea_stu_diff("resnet32x4", "resnet8x4", mpath, MAX_DIFF)

mpath = "/data/mmc_lyxiang/KD/logit-standardization-KD-master/output/Resnet32x4_Resnet8x4(226)/setting_63(random_train)_run0/student_best"
diff = get_tea_stu_diff("resnet32x4", "resnet8x4", mpath, MAX_DIFF)