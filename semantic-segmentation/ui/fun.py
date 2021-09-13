import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os

from PIL import Image

from util import config


class Config():
    config = '../config/pspnet50.yaml'

    if_cuda = False


args = Config()
cfg = config.load_cfg_from_cfg_file(args.config)
cfg.classes = 2
cfg.model_path = "../" + cfg.model_path
cfg.colors_path = "../" + cfg.colors_path

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
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def inference(cfg, image_name, model=None):
    #=============================================================================================================
    # model
    from model.pspnet import PSPNet
    model = PSPNet(layers=cfg.layers, classes=cfg.classes, zoom_factor=cfg.zoom_factor, pretrained=False)
    cudnn.benchmark = True
    if os.path.isfile(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location="cpu")  # map_location
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.model_path))
    model.eval()
    # prediction = inference(cfg, self.fname, model)
    #==============================================================================================================

    global logger
    img_path = image_name
    print("img_path", img_path)

    origin_image = cv2.imread(img_path)
    image = origin_image

    image, mask = img_process(image)
    colors = np.loadtxt(cfg.colors_path).astype('uint8')
    # print("colors", colors)


    with torch.no_grad():

        output = model(image)
        output = output.squeeze(0)
        # print("output", output.shape, output.min(), output.max())
        prediction = output.max(0)[1].cpu()
        prediction = prediction*mask
        # print("prediction", prediction.shape, prediction.min(), prediction.max())
        numel = np.count_nonzero(prediction)

        gray = np.uint8(prediction)

        color = colorize(gray, colors).convert(mode="RGB")
        # color_path = os.path.join('figure/demo/', image_name.split('.')[0] + '_color.png')
        #
        # color.save(color_path)
        # logger.info("=> Prediction saved in {}".format(color_path))
        color = np.array(color)
        print("color.shape, origin_image.shape", color.shape, origin_image.shape)
        color = cv2.addWeighted(color, 0.3, origin_image, 0.5, 0)
        return color, numel