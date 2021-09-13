import os
import logging
import argparse
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import config


cv2.ocl.setUseOpenCL(False)

from PIL import Image
from torchvision import transforms

from util.utility import check, get_logger, colorize



# def main(cfg):
#     global logger
#     check(cfg)
#     logger = get_logger()
#     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)
#     logger.info(cfg)
#     logger.info("=> creating model ...")
#     logger.info("Classes: {}".format(cfg.classes))
#
#     # image = Image.open(cfg.image).convert('RGB')  # H * W * 3 ,RGB order
#
#     image = cv2.imread(cfg.image, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
#     print("image", image.shape)
#
#     image = np.float32(np.array(image)) / 255.
#     image = image.transpose((2, 0, 1))  # chw
#     norm = transforms.Normalize(mean=cfg.mean,std=cfg.mean)
#     image = norm(torch.from_numpy(image.copy()))
#     image = image.unsqueeze(0)
#
#
#     # value_scale = 255
#     # mean = [0.485, 0.456, 0.406]
#     # mean = [item * value_scale for item in mean]
#     # std = [0.229, 0.224, 0.225]
#     # std = [item * value_scale for item in std]
#     colors = np.loadtxt(cfg.colors_path).astype('uint8')
#
#     if cfg.arch == 'psp':
#         from model.pspnet import PSPNet
#         model = PSPNet(layers=cfg.layers, classes=cfg.classes, zoom_factor=cfg.zoom_factor, pretrained=False)
#
#
#     logger.info(model)
#     cudnn.benchmark = True
#     if os.path.isfile(cfg.model_path):
#         logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
#         checkpoint = torch.load(cfg.model_path)
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#         logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
#     else:
#         raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.model_path))
#
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#
#
#         output = F.softmax(output, dim=1)
#         output = output.squeeze(0)
#         output = output.data.cpu().numpy()
#         output = output.transpose(1, 2, 0)
#         print("output", output.shape, output.min(), output.max())
#
#         prediction = np.argmax(output, axis=2)
#         print("prediction", prediction.shape, prediction.min(), prediction.max())
#         gray = np.uint8(prediction)
#         print("gray", gray.shape, gray.min(), gray.max())
#         color = colorize(gray, colors)
#         image_name = cfg.image.split('/')[-1].split('.')[0]
#         gray_path = os.path.join('figure/demo/', image_name + '_gray.png')
#         color_path = os.path.join('figure/demo/', image_name + '_color.png')
#         cv2.imwrite(gray_path, gray)
#         color.save(color_path)
#         logger.info("=> Prediction saved in {}".format(color_path))

def img_process(origin_img):
    mask = np.sum((origin_img == 255), 2)
    mask = (mask != 3).astype(np.uint8)

    image = origin_img * mask[:,:,np.newaxis]
    image = image[:, :, ::-1]
    image = np.float32(np.array(image)) / 255.
    image = image.transpose((2, 0, 1))  # chw

    mean = np.array([0.192, 0.196, 0.177]).reshape(3, 1, 1)  # RGB
    std = np.array([0.491, 0.490, 0.481]).reshape(3, 1, 1)
    image = (image - mean) / std

    image = torch.from_numpy(image).unsqueeze(0).float()
    return image, mask

def run(cfg, image_name, model):
    global logger
    check(cfg)
    logger = get_logger()

    img_path = os.path.join(cfg.image, image_name)
    print("img_path", img_path)


    # try:
    #     img_as = os.path.join("figure/demo/" + image_name)
    #     img_origin = cv2.imread(img_path)
    #     cv2.imwrite(img_as, img_origin)
    #     label_path = os.path.join(cfg.root_dataset+"train_set/label_color", image_name)
    #     label = cv2.imread(label_path)
    #     label_as = os.path.join("figure/demo/" + image_name.split(".")[0])+"_label.tif"
    #     cv2.imwrite(label_as, label)
    #
    # except:
    #     pass


    image = cv2.imread(img_path)
    print("image.shape", image.shape)


    image, mask = img_process(image)
    colors = np.loadtxt(cfg.colors_path).astype('uint8')
    # print("colors", colors)


    with torch.no_grad():

        output = model(image)
        output = output.squeeze(0)
        # print("output", output.shape, output.min(), output.max())
        prediction = output.max(0)[1].cpu()
        prediction = prediction*mask
        print("prediction", prediction.shape, prediction.min(), prediction.max())

        gray = np.uint8(prediction)
        print("gray", gray.shape, gray.min(), gray.max())
        color = colorize(gray, colors)
        # image_name = cfg.image.split('/')[-1].split('.')[0]
        # gray_path = os.path.join('figure/demo/', image_name + '_gray.png')
        color_path = os.path.join('figure/demo/', image_name.split('.')[0] + '_color.png')
        # cv2.imwrite(gray_path, gray)

        # color = color.convert("RGBA")
        # datas = color.getdata()
        # newData = []
        # for item in datas:
        #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
        #         newData.append((0, 0, 0, 0))
        #     else:
        #         newData.append(item)
        #
        # color.putdata(newData)
        # image_origin.paste(color, (0, 0))
        # color.show()
        # color.save("img2.png", "PNG")


        color.save(color_path)
        # logger.info("=> Prediction saved in {}".format(color_path))
        return color, None

import time
def main():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/pspnet50.yaml', help='config file')
    # parser.add_argument('--image', type=str, default='figure/demo/GF2_PMS2__L1A0001119060-MSS2_15484.tif', help='input image folder')
    parser.add_argument('--image', type=str,
                        default="dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/rgb_bg/",
                        help='input image folder')

    parser.add_argument('opts', help='see config/pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.image = args.image
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    # model
    if cfg.arch == 'psp':
        from model.pspnet import PSPNet
        model = PSPNet(layers=cfg.layers, classes=cfg.classes, zoom_factor=cfg.zoom_factor, pretrained=False)
    # logger.info(model)
    cudnn.benchmark = True
    cfg.model_path = "exp/pspnet50/train_epoch_300.pth"
    if os.path.isfile(cfg.model_path):
        # logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location="cpu")  #
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.model_path))
    # model.cuda()
    model.eval()

    # cfg.image = "/home/liufang/project/DataSets/JSCZXB/data/geoserver/gwc/earth_R-CN-JSCZXB-202105-0_5M/EPSG_4326_17/426_173/"
    cfg.image = "dataset/CROP_GID_256/6classes/train_set/gf/rgb_bg/"
    # cfg.image = "/home/liufang/Files/2021fh_sub_tifs/1/"
    imgs_path = os.listdir(cfg.image)
    print(len(imgs_path), imgs_path[0])

    images = random.choices(imgs_path, k=10)
    # images = imgs_path[100:200]
    for img in images:
        st = time.time()
        run(cfg, img, model)
        print("耗时：", time.time() - st)


if __name__ == '__main__':
    main()
