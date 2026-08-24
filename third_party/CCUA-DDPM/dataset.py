import json
import os
import numpy as np
import math

import torch
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def brs(index, imb_data, imb_targets, cls_num, imb_factor=1.0): # batch resample
    prob = [imb_factor ** (cls_idx / (cls_num - 1)) for cls_idx in range(cls_num)]
    img_num_per_cls = [math.floor(len(imb_data) * prob[i] / sum(prob)) for i in range(cls_num)]
    # print(img_num_per_cls, len(imb_data))

    the_class = None
    for i in range(len(img_num_per_cls)):
        # print(i, index, (sum(img_num_per_cls[:i]), sum(img_num_per_cls[:i+1])), sum(img_num_per_cls[:i]) <= index <= sum(img_num_per_cls[:i+1]))
        if sum(img_num_per_cls[:i]) <= index <= sum(img_num_per_cls[:i+1]):
            the_class = i
        elif index > sum(img_num_per_cls[:i+1]):
            the_class = len(img_num_per_cls) - 1
    assert 0 <= the_class < cls_num
    # print(f"the_class: {the_class}")
    # the_class = np.floor(index / img_num_per_cls).astype(int)

    targets_np = np.array(imb_targets)
    idx_for_the_class = np.where(targets_np == the_class)[0]

    resample_index = np.random.choice(idx_for_the_class, 1, replace=False)[0]
    return resample_index



class ImbalanceCIFAR10(datasets.CIFAR10):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, brs=False, brs_factor=0.1, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

        self.brs, self.brs_factor = brs, brs_factor

    def __getitem__(self, index):
        if self.brs:
            index = brs(index, self.data, self.targets, self.cls_num, imb_factor=self.brs_factor)

        img, target = super().__getitem__(index)
        return img, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
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
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100(datasets.CIFAR100):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, brs=False, brs_factor=0.1, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

        self.brs, self.brs_factor = brs, brs_factor

    def __getitem__(self, index):
        if self.brs:
            index = brs(index, self.data, self.targets, self.cls_num, imb_factor=self.brs_factor)

        img, target = super().__getitem__(index)
        return img, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
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
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class ImageNet(ImageFolder):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    default_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # default_target_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    def __init__(self, root, split="train", transform=default_train_transform):
        super(ImageNet, self).__init__(root=os.path.join(root, split), transform=transform)
        print(self.class_to_idx, [sum(1 for l in self.targets if l==i) for i in self.class_to_idx.values()])
    def __getitem__(self, index):
        img, t = super().__getitem__(index)
        target = self.targets[index]
        # return img, target, index
        # print('img: ', img.shape, '\ttarget: ', target)
        return img, target


class ImbalanceImageNet(ImageNet):
    cls_num = 1000
    def __init__(self, root, split="train", imb_type="exp", imb_factor=1.0, rand_number=0, brs=False, brs_factor=0.1,
                 transform=ImageNet.default_train_transform):
        super(ImbalanceImageNet, self).__init__(root, split=split, transform=transform)
        # === classes, class_to_idx, samples, targets ==== #
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

        self.brs, self.brs_factor = brs, brs_factor

    def __getitem__(self, index):
        if self.brs:
            index = brs(index, self.samples, self.targets, self.cls_num, imb_factor=self.brs_factor)

        img, target = super().__getitem__(index)

        return img, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_samples = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_samples.append([self.samples[i] for i in select_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_samples = np.vstack(new_samples)
        self.samples = new_samples
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.classes)):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceTinyImageNet(ImageNet):
    cls_num = 200
    def __init__(self, root, split="train", imb_type="exp", imb_factor=1.0, rand_number=0, brs=False, brs_factor=0.1,
                 transform=ImageNet.default_train_transform):
        super(ImbalanceTinyImageNet, self).__init__(root, split=split, transform=transform)
        # === classes, class_to_idx, samples, targets ==== #
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

        self.brs, self.brs_factor = brs, brs_factor

    def __getitem__(self, index):
        if self.brs:
            index = brs(index, self.samples, self.targets, self.cls_num, imb_factor=self.brs_factor)

        img, target = super().__getitem__(index)

        return img, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_samples = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_samples.append([self.samples[i] for i in select_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_samples = np.vstack(new_samples)
        self.samples = new_samples
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.classes)):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class PlacesLT(datasets.Places365):
    cls_num = 365
    def __init__(self, root, split="train-standard", small=True,
                 imb_type="exp", imb_factor=1.0, rand_number=0, brs=False, brs_factor=0.1,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, split, small, download=download,
                         transform=transform, target_transform=target_transform)
        # === classes, class_to_idx, samples, targets ==== #
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

        self.brs, self.brs_factor = brs, brs_factor

    def __getitem__(self, index):
        if self.brs:
            index = brs(index, self.imgs, self.targets, self.cls_num, imb_factor=self.brs_factor)

        file, target = self.imgs[index]
        image = self.loader(file)
        target = int(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.imgs) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_imgs.append([self.imgs[i] for i in select_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_imgs = np.vstack(new_imgs)
        self.imgs = new_imgs
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.classes)):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class PlacesLD(datasets.Places365):
    cls_num = 365
    def __init__(self, root, split="train-standard", small=True,
                 img_num_per_cls=50, rand_number=0,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, split, small, download=download,
                         transform=transform, target_transform=target_transform)
        # === classes, class_to_idx, samples, targets ==== #
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(img_num_per_cls, len(self.classes))
        self.num_per_cls_dict = dict()
        self.gen_limited_data(img_num_list)
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

    def __getitem__(self, index):
        file, target = self.imgs[index]
        image = self.loader(file)
        target = int(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_img_num_per_cls(self, img_num_per_cls, cls_num):
        img_num_per_cls = [img_num_per_cls] * cls_num
        return img_num_per_cls

    def gen_limited_data(self, img_num_per_cls):
        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_imgs.append([self.imgs[i] for i in select_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_imgs = np.vstack(new_imgs)
        self.imgs = new_imgs
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.classes)):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


# class SubsetPerLabel(torch.utils.data.Dataset):
#     def __init__(self, dataset, label_indices):
#         self.dataset = dataset
#         targets = np.array(dataset.targets, dtype=np.int64)
#         indices = []
#         for label_id in label_indices:
#             indices.append(np.where(targets == int(label_id)))
#         self.indices = np.hstack(indices).reshape(-1).tolist()
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, index):
#         return self.dataset[self.indices[index]]


class SubsetPerLabel(torch.utils.data.Dataset):
    def __init__(self, dataset, label_indices):
        self.dataset = dataset
        targets = np.array(dataset.targets, dtype=np.int64)
        indices = []
        for label_id in label_indices:
            indices.append(np.where(targets == int(label_id)))
        self.indices = np.hstack(indices).reshape(-1).tolist()

        # Get subset targets
        sub_targets = targets[self.indices]
        class_idx = np.unique(sub_targets)
        self.cls_num = len(class_idx)

        class_to_idx = {}
        for key, value in self.dataset.class_to_idx.items():
            if value in class_idx: class_to_idx[key] = value
        print(class_to_idx, [sum(1 for l in sub_targets if l == i) for i in class_to_idx.values()])

        # Covert Mapping to new Indices
        mapping = {}
        for i, idx in enumerate(class_idx):
            mapping[idx] = i

        self.targets = sub_targets
        for key, value in mapping.items():
            self.targets[np.where(self.targets == key)] = value

        self.class_to_idx = {}
        for key, value in self.dataset.class_to_idx.items():
            if value in class_idx: self.class_to_idx[key] = mapping[value]
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, trgt = self.dataset[self.indices[index]]
        target = self.targets[index]
        return img, target


if __name__ == "__main__":
    tran_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Resize([64, 64], antialias=True)
    ])

    dataset = ImbalanceCIFAR100(
        root='.cache/.cifar100',
        imb_type='exp',
        imb_factor=0.01,
        rand_number=0,
        brs=True,
        train=True,
        transform=tran_transform,
        download=True,
    )


    # dataset = PlacesLT(root='/mnt/data/Places/', #/data/datasets/Places/',
    #                    split='train-standard', small=True,
    #                    imb_type="exp", imb_factor=0.01, rand_number=0,
    #                    brs=True,
    #                    transform=tran_transform,
    #                    download=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        # pin_memory=True,
        drop_last=False
    )

    # # subset = SubsetPerLabel(dataset, label_indices=[7,8,9])
    y_dist = {}
    for data in loader:
        x, y = data
        for cls in range(dataset.cls_num):
            if cls not in y_dist:
                y_dist[cls] = 0
            y_dist[cls] += int(torch.sum(y == cls).numpy())
    print(y_dist)

    # dataset = PlacesLT(root='/mnt/data/Places/',  # /data/datasets/Places/',
    #                    split='val', small=True,
    #                    imb_type="exp",
    #                    imb_factor=1.0,
    #                    rand_number=0,
    #                    transform=tran_transform,
    #                    download=True)


    # dataset = ImbalanceCIFAR100(root='.cache/data/',
    #                             imb_type="exp", imb_factor=0.01, rand_number=0,
    #                             transform=tran_transform,
    #                             download=True
    #                             )
    #
    # subset = SubsetPerLabel(dataset, label_indices=[7,8,9])
    #
    # for i in range(len(subset)):
    #     x, y = subset[i]
    #     print(i, y)


    # dataset = ImbalanceImageNet(root='/mnt/data2/imagenet2012/',
    #                             imb_type="exp", imb_factor=0.01, rand_number=0,
    #                             transform=tran_transform,
    #                             )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2,
    #     shuffle=True, num_workers=1, drop_last=True)
    #
    # for i, (x, y) in enumerate(dataloader):
    #     print(x.shape, y.shape)

    # # import kagglehub
    # # kagglejson = {"username":"cfun2orz","key":"4d88cd13d70bd66231712eb5dc89f35b"}
    #
    # # print(help(kagglehub.login))
    # # kagglehub.login()
    #
    #
    # # Download latest version
    # # path = kagglehub.dataset_download("imsparsh/flowers-dataset")
    #
    # # Download latest version
    # # path = kagglehub.dataset_download("andrewmvd/animal-faces") + 'afhq'
    #
    #
    # path = ".cache/data/AnimalFaces/"
    # # print("Path to dataset files:", path)
    #
    # tran_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     transforms.Resize([64, 64], antialias=True)
    # ])
    #
    # # dataset = ImbalanceImageNet(root=path,
    # #                             split="",
    # #                             imb_type='step',
    # #                             imb_factor=0.01,
    # #                             rand_number=0,
    # #                             transform=tran_transform
    # #                             )
    #
    # dataset = ImageNet(root=path, split="",)
    # print(len(dataset))
    #
    # build_animalfaceslt()

