import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf


def my_transform(tensors, split = 'train'):
    # print(self.split)
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    # angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    # image = tf.rotate(img = image, angle = angle, resample=Image.NEAREST)
    # image = image.rotate(angle)
    # 自己写随机部分，50%的概率应用垂直，水平翻转。
    # if random.random() > 0.5:
    #     image = tf.hflip(image)
    # if random.random() > 0.5:
    #     image = tf.vflip(image)
    # 大小变换
    # image = tf.resize(image, 128)
    # 也可以实现一些复杂的操作
    # 50%的概率对图像放大后裁剪固定大小
    # 50%的概率对图像缩小后周边补0，并维持固定大小
    # 对图片进行随机裁剪，并将尺寸调整到128
    # i, j, h, w = transforms.RandomResizedCrop.get_params(
    #     image, scale=(0.8, 1.0), ratio=(1, 1))
    # image = tf.resized_crop(image, i, j, h, w, 128)
    # image = tf.to_tensor(image)

    # print(self.split)

    tensor_list = list(tensors.split(1, 0))

    for k in range(len(tensor_list)):
        n, c, w, h = np.shape(tensor_list[k])
        pil_img = transforms.ToPILImage()(tensor_list[k].cpu().view(-1, w, h)).convert('L')

        if split == 'train':
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                pil_img, scale=(0.5, 0.8), ratio=(1, 1))
        elif split == 'test':
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                pil_img, scale=(0.6, 0.6), ratio=(1, 1))

        image_crops = list(tf.five_crop(pil_img, (h, w)))
        # pil_img = tf.resize(pil_img, 128)
        image_crops.append(pil_img)
        image_crops = map(lambda crop: transforms.Resize(128)(crop), image_crops)
        image = torch.stack(tuple(map(lambda crop: transforms.ToTensor()(crop), image_crops)), dim=-1)
        image = image.view(-1, 1, 128, 128, 6)
        image = torch.cat((image, image, image), 1)
        tensor_list[k] = image
    tensors = torch.cat(tuple(tensor_list), 0)
    # i, j, h, w = transforms.RandomResizedCrop.get_params(
    #     image, scale=(0.80, 1.0), ratio=(1, 1))
    #
    # image_crops = tf.five_crop(image, (h, w))
    #
    # image_crops = map(lambda crop: transforms.Resize(128)(crop), image_crops)
    # image = torch.stack(tuple(map(lambda crop: transforms.ToTensor()(crop), image_crops)))
    # tensors = tensors.cuda()
    return tensors
