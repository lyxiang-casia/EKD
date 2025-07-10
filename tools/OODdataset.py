import torchvision as tv
import os
import copy
from torch.utils.data import Dataset
from PIL import Image
# from .base_dataset import BaseDataset

class OODDataset(Dataset):
    def __init__(self, name, path):
        super().__init__()
        # pipline =[
        #   dict(type='Collect', keys=['img', 'type'])
        # ]
        # self.pipeline = Compose(pipeline)
        self.file_list = []
        self.data_prefix = path
        self.name = name
        # self.resize_size = input_size if input_size is not None else 256
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(32),
            tv.transforms.CenterCrop(32),
            tv.transforms.ToTensor(),
            # tv.transforms.Normalize([0.4914, 0.4822, 0.4465],
            #                         [0.2023, 0.1994, 0.2010]),

        ])
        images = os.listdir(path)
        for filename in images:
            self.file_list.append(filename)
        self.data_infos = []
        self.parse_datainfo()


    def parse_datainfo(self):
        # random.shuffle(self.file_list)
        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            sample = os.path.join(self.data_prefix, sample)
            info['img_info'] = {'filename': sample}
            info['filename'] = sample
            info['type'] = 3  # no type
            info['label'] = -1  # no label
            self.data_infos.append(info)

    def __len__(self):
        return len(self.data_infos)
        # return len(self.file_list)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        try:
            sample = Image.open(results['img_info']['filename'])
        except:
            print(results)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        sample = self.transform(sample)
        # results['img'] = sample
        # results['label'] = -1 #no label for OOD samples
        return (sample, -1)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

