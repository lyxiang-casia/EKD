import argparse
import torch
import torch.backends.cudnn as cudnn
import mmcv 
from mmengine.registry import Registry, build_from_cfg
cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.distillers import Teacher
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg
from OODmetrics import compute_X_Y_alpha
from OODdataset import OODDataset
from OODmetrics import our_anomaly_detection
import math

pipline =[
          dict(type='Collect', keys=['img', 'type'])
]
ood_data={
    "iNaturalist": dict(
        name='iNaturalist',
        type='FolderDataset',
        path='/data/mmc_lyxiang/KD/logit-standardization-KD-master/data/iNaturalist/images/',
        pipeline=pipline,
        # len_limit=1000 if quick_test else -1,
    ),
    "SUN": dict(
        name='SUN',
        type='FolderDataset',
        # path='/data/ood_data/SUN/images',
        path='/data/mmc_lyxiang/KD/logit-standardization-KD-master/data/SUN/images',
        pipeline=pipline,
        # len_limit=1000 if quick_test else -1,
    ),
    "Places": dict(
        name='Places',
        type='FolderDataset',
        # path='/data/ood_data/Places/images',
        path='/data/mmc_lyxiang/KD/logit-standardization-KD-master/data/Places/images',
        pipeline=pipline,
        # len_limit=1000 if quick_test else -1,
    )
    }
def get_ood_dataloader(ood_dataset_name, batch_size=64):
    DATASETS = Registry('dataset')
    ood_dataset = build_from_cfg(ood_data[ood_dataset_name], DATASETS)
    image_size = 32
    transform = transforms.Compose([
        transforms.Resize((imagesize, imagesize)),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    ])
    data_loader_ood = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            # sampler=sampler,
            num_workers=2,
            pin_memory=True,
            shuffle=True)
    return data_loader_ood

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "cifar10"],
    )
    parser.add_argument("-lamb", type=float, default=0.0)
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    # parser.add_argument("-smodel", "--student_model", type=str, default="")
    # parser.add_argument("-c_s", "--student_ckpt", type=str, default="pretrain")
    args = parser.parse_args()

    cfg.TEACHER.LAMB = args.lamb
    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    elif args.dataset == 'cifar10':
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
        # student_model, pretrain_model_path = cifar_model_dict[args.student_model]
        # student_model = student_model(num_classes=num_classes)
        # ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.student_ckpt
        # student_model.load_state_dict(load_checkpoint(ckpt)["model"])
    ood_dataset_name = 'SUN'
    ood_dataset = OODDataset('SUN', '/data/mmc_lyxiang/KD/logit-standardization-KD-master/data/iNaturalist/images')
    ood_test_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # student_model = Vanilla(student_model)
    # student_model = student_model.cuda()
    # student_model = torch.nn.DataParallel(student_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.lamb == 0.0: 
        model = Vanilla(model)
    else:
        model = Teacher(model, cfg)

    # path_list = ["/data_SSD2/mmc_lyxiang/KD/output/BKD/resnet32x4resnet8x4seed42/student_best",
    # "/data_SSD2/mmc_lyxiang/KD/output/BKD/resnet32x4resnet8x4seed111/student_best",
    # "/data_SSD2/mmc_lyxiang/KD/output/BKD/resnet32x4resnet8x4seed222/student_best",
    # "/data_SSD2/mmc_lyxiang/KD/output/BKD/resnet32x4resnet8x4seed666/student_best",
    # "/data_SSD2/mmc_lyxiang/KD/output/BKD/resnet32x4resnet8x4seed777/student_best",
    # ]
    # models=[]
    # for path in path_list:
    #     model, _ = cifar_model_dict['resnet8x4']
    #     model = model(num_classes=num_classes)
    #     model.load_state_dict(load_checkpoint(path)["model"])
    #     model = Vanilla(model)
    #     model = model.cuda()
    #     model = torch.nn.DataParallel(model)
    #     model.eval()
    #     models.append(model)

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
  
    # id_Y = []
    # id_X = []
    # id_alpha_pred = []
    # ood_Y = []
    # ood_X = []
    # ood_alpha_pred = []
    with torch.no_grad():
        id_Y_all, id_X_all, id_alpha_pred_all = compute_X_Y_alpha(model, val_loader, device, args.lamb)
        ood_Y_all, ood_X_all, ood_alpha_pred_all = compute_X_Y_alpha(model, ood_test_loader, device, args.lamb)

        if args.lamb == 0.0:
            # aupr, auroc, _, _ = our_anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
            #                                                     uncertainty_type='max_prob', lamb=0.0)

            aupr, auroc, _, _ = our_anomaly_detection(alpha=id_alpha_pred, ood_alpha=ood_alpha_pred,
                                uncertainty_type='discrete_entropy', lamb=0.0)                                   
            print("aupr:", aupr)
            print("auroc:", auroc)
        else:
            # for name in ['max_prob', 'max_modified_prob', 'max_alpha', 'alpha0', 'differential_entropy', 'mutual_information']:
            aupr, auroc, _, _ = our_anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
                                                                uncertainty_type='differential_entropy', lamb=torch.exp(torch.tensor(args.lamb)))
            print("aupr:", aupr)
            print("auroc:", auroc)
