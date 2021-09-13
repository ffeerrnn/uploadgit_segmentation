import os
import json
import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')
    return im.resize(size, resample)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, opt, **kwargs):
        self.cfg = opt
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = self.cfg.padding_constant

        # parse the input list
        # self.parse_input_list(self.cfg.list_train, **kwargs)
        self.normalize = transforms.Normalize(mean=self.cfg.mean,std=self.cfg.std)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        # 读取文件名列表
        print("odgt", odgt)

        if isinstance(odgt, list):
            list_sample = odgt
        elif isinstance(odgt, str):
            print("str")
            if os.path.isfile(odgt):
                list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
            elif os.path.isdir(odgt):
                list_sample = os.listdir(odgt)
        # print("list_sample", list_sample)



        # 取部分文件名列表
        if max_sample > 0:
            list_sample = list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            list_sample = list_sample[start_idx:end_idx]

        num_sample = len(list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))
        return list_sample




    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))    # chw
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to classes-1
        # segm = torch.from_numpy(np.array(segm)).long() - 1
        segm = torch.from_numpy(np.array(segm/2)).long()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, opt,  **kwargs):
        super(TrainDataset, self).__init__(opt, **kwargs)
        self.cfg = opt
        self.root_dataset = self.cfg.root_dataset
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        print("self.cfg.list_train", self.cfg.list_train)
        self.list_sample = self.parse_input_list(self.cfg.list_train)


    def __getitem__(self, index):
        batch_records = self.list_sample[index]

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)    #随机抽取
        else:
            this_short_size = self.imgSizes


        img_height, img_width = batch_records['height'], batch_records['width']
        this_scale = min(this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
        batch_widths = img_width * this_scale
        batch_heights = img_height * this_scale

        # load image and label
        image_path = os.path.join(self.root_dataset, batch_records['fpath_img'])
        segm_path = os.path.join(self.root_dataset, batch_records['fpath_segm'])
        # print(image_path)
        # print(segm_path)

        # img = Image.open(image_path).convert('RGB')
        # segm = Image.open(segm_path)
        # assert(segm.mode == "L")    # L 灰度图像
        # assert (img.size[0] == segm.size[0])
        # assert (img.size[1] == segm.size[1])

        img = cv2.imread(image_path)[:,:,::-1]    # bgr2rgb
        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        assert (img.shape[0] == segm.shape[0])
        assert (img.shape[1] == segm.shape[1])



        # random_flip 从数组中随机抽取元素
        # if self.cfg.random_flip:
        #     if np.random.choice([0, 1]):
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)    #左右翻转
        #         segm = segm.transpose(Image.FLIP_LEFT_RIGHT)    #左右翻转
        #
        # # note that each sample within a mini batch has different scale param
        # if self.cfg.img_resize:
        #     img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
        #     segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

        # image transform, to torch float tensor 3xHxW
        img = self.img_transform(img)

        # segm transform, to torch long tensor HxW


        segm = self.segm_transform(segm)

        output = dict()
        output['img_data'] = img
        output['seg_label'] = segm
        return output

    def __len__(self):
        return len(self.list_sample)


class ValDataset(BaseDataset):
    def __init__(self, opt, **kwargs):
        super(ValDataset, self).__init__(opt, **kwargs)
        self.cfg = opt
        self.root_dataset = self.cfg.root_dataset
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        print("self.cfg.list_val", self.cfg.list_val)
        self.list_sample = self.parse_input_list(self.cfg.list_val)

    def __getitem__(self, index):
        batch_records = self.list_sample[index]

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)  # 随机抽取
        else:
            this_short_size = self.imgSizes

        img_height, img_width = batch_records['height'], batch_records['width']
        this_scale = min(this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
        batch_widths = img_width * this_scale
        batch_heights = img_height * this_scale

        # load image and label
        image_path = os.path.join(self.root_dataset, batch_records['fpath_img'])
        segm_path = os.path.join(self.root_dataset, batch_records['fpath_segm'])

        img = cv2.imread(image_path)[:, :, ::-1]  # bgr2rgb
        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        assert (img.shape[0] == segm.shape[0])
        assert (img.shape[1] == segm.shape[1])

        img = cv2.flip(img, 1)  # 图像0垂直翻转
        segm = cv2.flip(segm, 1)  # 图像0垂直翻转

        img = self.img_transform(img)

        segm = self.segm_transform(segm)

        output = dict()
        output['img_data'] = img
        output['seg_label'] = segm
        return output

    def __len__(self):
        return len(self.list_sample)


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample

if __name__ == '__main__':
    class DATASET():
        root_dataset="../dataset/"
        list_train="../dataset/training.odgt"
        list_val="./dataset/validation.odgt"
        num_class=150
        imgSizes=(300, 375, 450, 525, 600)
        imgMaxSize=1000
        padding_constant=8
        segm_downsampling_rate=8
        random_flip=True
    cfg = DATASET()
    batch_size_per_gpu = 3

    dataset_train = TrainDataset(
        cfg.root_dataset,
        cfg.list_train,
        cfg,
        batch_size_per_gpu)

    gpus = [0]
    workers = 4
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=workers,
        drop_last=True,
        pin_memory=True)
    # create loader iterator
    iterator_train = iter(loader_train)
    # print(iterator_train)
    # batch_data = next(iterator_train)
    # print(batch_data)
    for i, input_target in enumerate(loader_train):
        img_data = input_target[0]['img_data']
        seg_data = input_target[0]['seg_label']
        print(i, img_data.shape, img_data.min(), img_data.max(), seg_data.shape, seg_data.min(), seg_data.max())