import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import albumentations
import os
from torchvision.datasets import ImageFolder

class ImagePaths(Dataset):
    def __init__(self, image_dir, pkl_dir, image_size=112):
        self.data_path = image_dir
        self.size = image_size
        self.pkl_path = pkl_dir
        self.file_paths = glob.glob(self.data_path+'/*.jpg')

        self.aumat = pickle.load(open(self.pkl_path, 'rb'), encoding='iso-8859-1')

        self._length = len(self.file_paths)
        self.rescaler = albumentations.Resize(height=self.size, width=self.size)
        # # self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        # self.norm = albumentations.Normalize()
        self.preprocessor = albumentations.Compose([self.rescaler])

        # self.norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # self.preprocessor = transforms.Compose([
        #     transforms.Resize(112),
        #     # transforms.CenterCrop(112),
        #     transforms.ToTensor(),
        #     self.norm
        # ])

    # def get_labels(self):
    #     return self.labels

    # def get_aufeat(self):
    #     return self.au_feat

    def __len__(self):
        return len(self.file_paths)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        # image = self.preprocessor(image)
        image = (image/255.).astype(np.float32)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def get_au(self, image_path):
        name = os.path.basename(image_path[:-4])
        # name = name.split('_')[0]
        au_feat = self.aumat.get(name)
        # au_feat = (au_feat>=max(au_feat)).astype(int)
        au_feat = au_feat/5.0
        return au_feat

    def __getitem__(self, i):
        example = self.preprocess_image(self.file_paths[i])
        # label = self.labels[i]
        # au_feat = self.au_feat[i]/5.0
        au_feat = self.get_au(self.file_paths[i])
        # return example, label
        return example, au_feat


def get_train_loader(args):
    train_data = ImagePaths(args.images_dir, args.image_size)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size,
    #                           args.num_workers=2, pin_memory=True, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    return train_loader

def get_test_loader(args):
    transform = albumentations.Compose([ albumentations.Resize(height=args.image_size, width=args.image_size)])
    dataset = ImageFolder(args.images_dir, transform)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle= False,
                                  num_workers=args.num_workers)
    return data_loader, dataset.imgs