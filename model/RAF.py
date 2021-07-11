''' RAF Dataset class'''
# create data and label for RAF
# containing 12271 training samples and 3068 testing samples after aligned.
# 0: Surprise
# 1: Fear
# 2: Disgust
# 3: Happiness
# 4: Sadness
# 5: Anger
# 6: Neutral

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as tf


class RAF(data.Dataset):
    """`RAF Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('./data/RAF_data_gt.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            # ------------add------------
            self.train_gt = self.data['Training_gt']

            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 128, 128))
            # ------------add------------
            self.train_seg_data = np.asarray(self.train_gt)
            self.train_seg_data = self.train_seg_data.reshape((12271, 128, 128))


        elif self.split == 'Test':
            self.Test_data = self.data['Test_pixel']
            self.Test_labels = self.data['Test_label']
            # ------------add------------
            self.Test_gt = self.data['Test_gt']

            self.Test_data = np.asarray(self.Test_data)
            self.Test_data = self.Test_data.reshape((3068, 128, 128))
            # ------------add------------
            self.Test_seg_data = np.asarray(self.Test_gt)
            self.Test_seg_data = self.Test_seg_data.reshape((3068, 128, 128))


    def my_transform(self, image, mask):

        if self.split == 'Training':
            # print(self.split)
            # 拿到角度的随机数。angle是一个-180到180之间的一个数
            # angle = transforms.RandomRotation.get_params([-180, 180])
            # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
            # image = tf.rotate(img = image, angle = angle, resample=Image.NEAREST)
            # mask = tf.rotate(img = mask, angle = angle)
            # image = image.rotate(angle)
            # mask = mask.rotate(angle)
            # 自己写随机部分，50%的概率应用垂直，水平翻转。
            if random.random() > 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            if random.random() > 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)
            # 大小变换
            #image = tf.resize(image, 128)
            #mask = tf.resize(mask, 128)
            # 也可以实现一些复杂的操作
            # 50%的概率对图像放大后裁剪固定大小
            # 50%的概率对图像缩小后周边补0，并维持固定大小
            # 对图片进行随机裁剪，并将尺寸调整到128
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(1, 1))
            image = tf.resized_crop(image, i, j, h, w, 128)
            mask = tf.resized_crop(mask, i, j, h, w, 128)
            image = tf.to_tensor(image)
            mask = tf.to_tensor(mask)

        elif self.split == 'Test':

            # print(self.split)
            # i, j, h, w = transforms.RandomResizedCrop.get_params(
            #         image, scale=(0.80, 1.0), ratio=(1, 1))

            # image_crops = tf.five_crop(image, (h, w))
            # mask_crops = tf.five_crop(mask, (h, w))

            # image_crops = map(lambda crop: transforms.Resize(128)(crop), image_crops)
            # image = torch.stack(tuple(map(lambda crop: transforms.ToTensor()(crop), image_crops)))

            # mask_crops = map(lambda crop: transforms.Resize(128)(crop), mask_crops)
            # mask = torch.stack(tuple(map(lambda crop: transforms.ToTensor()(crop), mask_crops)))
            image = tf.resize(image, 128)
            mask = tf.resize(mask, 128)
            image = tf.to_tensor(image)
            mask = tf.to_tensor(mask)

        else:
            image = tf.resize(image, 128)
            mask = tf.resize(mask, 128)
            image = tf.to_tensor(image)
            mask = tf.to_tensor(mask)


        # if random.random() > 0.5:
        #     i, j, h, w = transforms.RandomResizedCrop.get_params(
        #             image, scale=(0.7, 1.0), ratio=(1, 1))
        #     image = tf.resized_crop(image, i, j, h, w, 128)
        #     mask = tf.resized_crop(mask, i, j, h, w, 128)
        # else:
        #     pad = random.randint(0, 192)
        #     image = tf.pad(image, pad)
        #     image = tf.resize(image, 128)
        #     mask = tf.pad(mask, pad)
        #     mask = tf.resize(mask, 128)
        # image = tf.resize(image, 128)
        # mask = tf.resize(mask, 128)
        # # 转换为tensor
        # image = tf.to_tensor(image)
        # 归一化
        # image = tf.normalize(image, [0.5], [0.5])
        # mask = tf.to_tensor(mask)
        # 归一化
        # mask = tf.normalize(mask, [0.5], [0.5])


        return image, mask


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target, gt = self.train_data[index], self.train_labels[index], self.train_seg_data[index]
            # img, target= self.train_data[index], self.train_labels[index]
        elif self.split == 'Test':
            img, target, gt = self.Test_data[index], self.Test_labels[index], self.Test_seg_data[index]
            # img, target= self.Test_data[index], self.Test_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]# 增加一维
        img = np.concatenate((img, img, img), axis=2)
        # print('img.shape:' + str(np.shape(img)))
        # print('img.type' + str(type(img)))
        img = Image.fromarray(img)
        # if self.transform is not None:
            # img = self.transform(img)

        # ------------add------------
        # gt = gt[:, :, np.newaxis] # 增加一维
        # gt = np.concatenate((gt), axis = 2)
        # print('gt.shape:' + str(np.shape(gt)))
        gt = Image.fromarray(gt)
        # if self.transform is not None:
            # img = self.transform(img)
            # gt = self.transform(gt)
        img, gt = self.my_transform(img, gt)

        return img, target, gt

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Test':
            return len(self.Test_data)
