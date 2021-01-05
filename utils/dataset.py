from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale = 1):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img = transforms.CenterCrop((128,128))(pil_img.resize((newW, newH)))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]

        mask_file = glob(self.masks_dir + "mask" + idx[5:] + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.transform:
            aug = A.Compose([
                A.RandomCrop(128,128),
                A.Normalize((0.5), (0.5))
            ])
            augmented = aug(image = np.array(img), mask = np.array(mask))
            img_trans = augmented["image"].reshape((1,128,128))
            mask_trans = augmented["mask"].reshape((1,128,128))

            if img_trans.max() > 1:
                img_trans = img_trans / 255

            return{
                'image': torch.from_numpy(img_trans).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask_trans).type(torch.FloatTensor)
            }

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(torch.from_numpy(img).type(torch.FloatTensor)),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
