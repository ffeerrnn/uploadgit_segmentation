# import cv2
# if __name__ == '__main__':
#
#     img = cv2.imread("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes/image_RGB/GF2_PMS1__L1A0000564539-MSS1.tif")
#     print(img.shape)
#     img = img[:256,:256,:]
#     print(img.shape)
#     cv2.imwrite("tifs/2.tif", img)

import os

import cv2
import numpy as np
import torch
import time

from PIL import Image
from torchvision import transforms

from config.Config import cfg


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.cfg = cfg
        self.path = cfg.tif_path
        # self.classes = cfg.classes
        # self.out_path = cfg.out_path
        self.if_cuda = cfg.if_cuda
        self.files_name = os.listdir(self.path)
        self.do_files_name = []
        self.pass_file_name = []
        for name in self.files_name:
            file_path = os.path.join(self.path, name)
            f = open(file_path, "rb")
            size = len(f.read())
            # print(file_path, size)
            if size <= cfg.byte_range:    # byte
                self.pass_file_name.append(name)
            else:
                self.do_files_name.append(name)
                #

            # origin_img = cv2.imread(file_path)
            # if origin_img.min() == 255:
            #     self.pass_file_name.append(name)
            # else:
            #     self.do_files_name.append(name)

    # def norm(self, img):
    #     mean = np.array([0.192, 0.196, 0.177]).reshape(3, 1, 1)  # RGB
    #     std = np.array([0.491, 0.490, 0.481]).reshape(3, 1, 1)
    #     img = (img - mean) / std
    #     return img
    #
    def img_process(self, file_name):
        file_path = os.path.join(self.path, file_name)
        origin_img = cv2.imread(file_path)

        mask = np.sum((origin_img == 255), 2)
        mask = (mask != 3).astype(np.uint8)     # invalid mask

        image = origin_img[:, :, ::-1]
        image = image * mask[:, :, np.newaxis]
        image = np.float32(np.array(image)) / 255.
        image = image.transpose((2, 0, 1))  # chw

        mean = np.array(cfg.mean).reshape(3, 1, 1)  # RGB
        std = np.array(cfg.std).reshape(3, 1, 1)
        image = (image - mean) / std

        # image = torch.from_numpy(image).unsqueeze(0).float()
        image = torch.from_numpy(image).float()

        return [file_name, mask, image]

        # if origin_img.min() == 255:
        #     self.pass_file_name.append(file_name)
        #     img_dic = {}
        #     img_dic["name"] = file_path
        #     img_dic["image"] = origin_img
        #     img_dic["op"] = "pass"
        #     return img_dic
        # else:
        #     self.do_files_name.append(file_name)
        #     image = origin_img[:, :, ::-1]
        #     image = np.float32(np.array(image)) / 255.
        #     image = image.transpose((2, 0, 1))  # chw
        #     image = self.norm(image)
        #     image = torch.from_numpy(image).unsqueeze(0).float()
        #     if self.if_cuda:
        #         image = image.cuda()
        #
        #     img_dic = {}
        #     img_dic["name"] = file_path
        #     img_dic["image"] = image
        #     img_dic["op"] = "do"
        #     return img_dic
    #
    def __getitem__(self, index):
        # print("=> loaded dataset")
        file_name = self.do_files_name[index]
        img_dic = self.img_process(file_name)
        return img_dic

    def __len__(self):
        # print('# samples: {}, do: {}, pass: {}'.format(len(self.files_name), len(self.do_files_name), len(self.pass_file_name)))
        num_sample = len(self.do_files_name)
        return num_sample


def prepare_model():
    print("=> loaded model")
    layers = cfg.layers
    classes = cfg.classes
    zoom_factor = cfg.zoom_factor
    model_path = cfg.model_path
    if_cuda = cfg.if_cuda

    from model.pspnet import PSPNet
    model = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=False)

    if if_cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        if if_cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

    if if_cuda:
        model.cuda()
    model.eval()

    return model



def inference(model, image):
    with torch.no_grad():
        outputs = model(image)
        prediction = outputs.max(1)[1].cpu()
    return prediction
        # outputs = outputs.squeeze(0)
        # print("outputs", outputs.shape, outputs.min(), outputs.max())
        # grays = []
        # for output in outputs:
        #     prediction = output.max(0)[1].cpu()
        #     gray = np.uint8(prediction)
        #     grays.append(gray)

        # return grays


def colorize(img, color):
    color0 = np.zeros_like(color)
    color = np.concatenate((color0, color), 0).reshape(2,-1)    # color for RGB
    color_img = color[img].astype(np.uint8)
    color_img = color_img[...,[2,1,0,3]]    # RGBA2BGRA
    return color_img

# def colorize(gray, palette, transparent=False):
#     if not transparent:
#         # gray: numpy array of the label and 1*3N size list palette
#         color = Image.fromarray(gray.astype(np.uint8)).convert('P')
#         color.putpalette(palette)
#
#     else:
#         # transparent
#         color = Image.fromarray(gray.astype(np.uint8)).convert('P')
#         color.putpalette(palette)
#         color = color.convert("RGBA")
#         datas = color.getdata()
#         # print("datas.shape", datas.mode, datas.size)
#         # a_channel = np.zeros_like(datas)
#         # newdatas = np.concatenate((datas, a_channel), )
#         newData = []
#         for item in datas:
#             if item[0] == 0 and item[1] == 0 and item[2] == 0:
#                 newData.append((0, 0, 0, 0))
#             else:
#                 newData.append(item)
#         color.putdata(newData)
#     return color


def prepare_colors(transparent=True):
    colors_path = cfg.colors_path
    colors = np.loadtxt(colors_path).astype('uint8')
    print("=> loaded colors")
    if transparent:
        mask = (colors.sum(1) > 0).astype(np.uint8) * 255
        mask = mask.reshape(-1, 1)
        colors = np.concatenate((colors, mask), 1)
    return colors


def main(dataset_loader, model, colors):
    path = cfg.tif_path
    classes = cfg.classes
    out_path = cfg.out_path
    if_cuda = cfg.if_cuda
    save_class = cfg.save_class

    # pass_file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if cfg.run_pass:
        files = [os.path.join(path, file) for file in dataset.pass_file_name]
        print("pass_file_name start!", end="")
        s = time.time()
        for index, f in enumerate(files):
            print("[{}/{} {}]".format(index, len(files), f))
            for i in range(1, save_class):
                # pre = np.zeros((cfg.test_h, cfg.test_w), dtype=np.uint8)
                # color_img  = Image.fromarray(pre).convert("RGBA")
                color_img = Image.new("RGBA", (cfg.test_w, cfg.test_h))
                color_path = f.split("/")[-1].split(".")[0] + "_{}.{}".format(i, f.split("/")[-1].split(".")[-1])

                color_path = os.path.join(out_path, color_path.split("/")[-1])
                # print("save to:", color_path)
                color_img.save(color_path)
        print("pass_file_name done!")
        print("耗时:", time.time() - s)
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV


    # do_file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    for index, name_f in enumerate(dataset_loader):
        file_names, masks, images = name_f
        # f = name_f["name"]
        # images = name_f["image"]
        # op = name_f["op"]    # "pass" "do"
        # print("images.shape", images.shape)
        if cfg.if_cuda:
            images = images.cuda()

        s = time.time()
        predictions = inference(model, images)
        # print(predictions.shape, masks.shape, file_names)

        predictions = predictions * masks

        if cfg.merge_classes:
            if colors.shape[1] == 4:
                colors_3 = colors[:, :3]
            else:
                colors_3 = colors
                colors = Image.fromarray(predictions.numpy().astype(np.uint8)).convert('P')
                colors.putpalette(colors_3)
            for ind, color in enumerate(colors):
                f = file_names[ind]
                cv2.imwrite(cfg.result_classes + f.split("/")[-1].split(".")[0] + ".{}".format(f.split("/")[-1].split(".")[-1]), color)
                # color.save(
                #     cfg.result_classes + f.split("/")[-1].split(".")[0] + ".{}".format(f.split("/")[-1].split(".")[-1]))


        for ind, p in enumerate(predictions):
            p = p.squeeze(0).numpy()
            f = file_names[ind]
            for i in range(1, classes):
                pre = (p == i).astype(np.uint8)
                # print(pre)
                color_img = colorize(pre, colors[i, :])
                color_path = f.split("/")[-1].split(".")[0] + "_{}.tif".format(i)
                # print(color_path)
                color_path = os.path.join(out_path, color_path.split("/")[-1])
                # print("save to:", color_path)
                # color_img.save(color_path)
                cv2.imwrite(color_path, color_img)
        print(index, "/", (len(dataset_loader)), "耗时:", time.time() - s)





# load all 256*256
if __name__ == '__main__':
    global cfg
    cfg.if_cuda = torch.cuda.is_available()
    cfg.batch_size = 30
    # colors
    colors = prepare_colors()

    # dataset
    global dataset
    dataset = TestDataset()
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, drop_last=False)
    print('# samples: {}, do: {}, pass: {}'.format(len(dataset.files_name), len(dataset.do_files_name),
                                                   len(dataset.pass_file_name)))

    # model
    model = prepare_model()

    s = time.time()
    main(dataset_loader, model, colors)
    print("gpu总耗时:", time.time() - s)



