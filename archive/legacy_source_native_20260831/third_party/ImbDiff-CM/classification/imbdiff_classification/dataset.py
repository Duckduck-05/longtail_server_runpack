import numpy as np
import torch
from torchvision import datasets


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return {"input": image, "target": target, "index": index}


class ImbalanceCIFAR100(datasets.CIFAR100):
    cls_num = 100

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = {}
        self.gen_imbalanced_data(img_num_list)
        self.img_num_list = img_num_list

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selected = idx[:the_img_num]
            new_data.append(self.data[selected, ...])
            new_targets.extend([the_class] * the_img_num)
        self.data = np.vstack(new_data)
        self.targets = new_targets

