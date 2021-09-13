import os

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from config.Config import cfg


def prepare_model():
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
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    if if_cuda:
        model.cuda()
    model.eval()
    return model

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def img_process(origin_img):

    mask = np.sum((origin_img == 255), 2)
    mask = (mask != 3).astype(np.uint8)

    image = origin_img[:, :, ::-1]
    image = image * mask[:,:,np.newaxis]
    image = np.float32(np.array(image)) / 255.
    image = image.transpose((2, 0, 1))  # chw


    mean = np.array(cfg.mean).reshape(3, 1, 1)  # RGB
    std = np.array(cfg.std).reshape(3, 1, 1)
    image = (image - mean) / std

    image = torch.from_numpy(image).unsqueeze(0).float()
    return [mask, image]


def colorize(img, color):
    color0 = np.zeros_like(color)
    color = np.concatenate((color0, color), 0).reshape(2,-1)    # color for RGB
    color_img = color[img].astype(np.uint8)
    color_img = color_img[...,[2,1,0,3]]    # RGBA2BGRA
    return color_img



def main(model, colors):
    path = cfg.tif_path
    classes = cfg.classes
    out_path = cfg.out_path
    if_cuda = cfg.if_cuda
    pic_num = cfg.pic_num
    save_class = cfg.save_class
    result_classes = cfg.result_classes

    if not os.path.exists(out_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_path)
    if not os.path.exists(result_classes):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(result_classes)


    file_name = os.listdir(path)
    print(len(file_name))
    # files = [os.path.join(path, file) for file in file_name]
    # print(files)

    # ======================
    do_file_name = file_name
    # ======================


    # ===============================================================================
    # pass_file_name = []
    # do_file_name = []
    # for name in file_name:
    #     file_path = os.path.join(path, name)
    #     f = open(file_path, "rb")
    #     size = len(f.read())
    #     # print(file_path, size)
    #     if size <= cfg.byte_range:  # byte  818    1784
    #         pass_file_name.append(name)
    #     else:
    #         do_file_name.append(name)
    #     f.close()
    # print('# samples: {}, do: {}, pass: {}'.format(len(file_name), len(do_file_name), len(pass_file_name)))
    print('# samples: {}, do: {}'.format(len(file_name), len(do_file_name)))

    # # pass_file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # if cfg.run_pass:
    #     files = [os.path.join(path, file) for file in pass_file_name]
    #     print("pass_file_name start!", end="")
    #     s = time.time()
    #     for index, f in enumerate(files):
    #         print("[{}/{} {}]".format(index, len(files), f))
    #         for i in range(1, save_class):
    #             # pre = np.zeros((cfg.test_h, cfg.test_w), dtype=np.uint8)
    #             # color_img  = Image.fromarray(pre).convert("RGBA")
    #             color_img = Image.new("RGBA", (cfg.test_w, cfg.test_h))
    #             color_path = f.split("/")[-1].split(".")[0] + "_{}.{}".format(i, f.split("/")[-1].split(".")[-1])
    #
    #             color_path = os.path.join(out_path, color_path.split("/")[-1])
    #             # print("save to:", color_path)
    #             color_img.save(color_path)
    #     print("pass_file_name done!")
    #     print("耗时:", time.time() - s)
    # # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    #
    #
    # # do_file^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    files = [os.path.join(path, file) for file in do_file_name]
    if pic_num == -1:
        pic_num = len(files)
    for index, f in enumerate(files[:pic_num]):
        print("[{}/{} {}]".format(index, len(files), f))
        origin_img = cv2.imread(f)
        # cv2.imshow("origin_img_{}".format(index), origin_img)
        image_out = img_process(origin_img)
        mask, image = image_out

        if cfg.if_cuda:
            image = image.cuda()

        with torch.no_grad():
            s = time.time()
            output = model(image)
            # print("output", output.shape)
            output = output.squeeze(0)
            prediction = output.max(0)[1].cpu().numpy()
            # print(prediction.shape, prediction.min(), prediction.max())

            prediction = prediction * mask
            print("耗时:", time.time() - s)

            if cfg.merge_classes:
                if colors.shape[1] == 4:
                    colors_3 = colors[:,:3]
                else:
                    colors_3 = colors
                color = Image.fromarray(prediction.astype(np.uint8)).convert('P')
                color.putpalette(colors_3)
                color.save(result_classes + f.split("/")[-1].split(".")[0] + ".{}".format(f.split("/")[-1].split(".")[-1]))

            # for i in range(1, save_class):
            i = 1 # water
            pre = (prediction == i).astype(np.uint8)
            color_img = colorize(pre, colors[i, :])
            color_path = f.split("/")[-1].split(".")[0] + "_{}.{}".format(i, f.split("/")[-1].split(".")[-1])

            color_path = os.path.join(out_path, color_path.split("/")[-1])
            print("save to:", color_path)
            # color_img.save(color_path)
            cv2.imwrite(color_path, color_img)


    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV



def prepare_colors(transparent=True):
    colors_path = cfg.colors_path
    colors = np.loadtxt(colors_path).astype('uint8')
    print("=> loaded colors")
    if transparent:
        mask = (colors.sum(1) > 0).astype(np.uint8) * 255
        mask = mask.reshape(-1, 1)
        colors = np.concatenate((colors, mask), 1)
    return colors


# run one 256*256 each time 
if __name__ == '__main__':
    global cfg
    cfg.if_cuda = torch.cuda.is_available() and cfg.if_cuda
    # colors
    colors = prepare_colors()
    # print("colors:", type(colors), colors.shape, colors)
    
    # model
    model = prepare_model()

    import time
    s = time.time()
    main(model, colors)
    if cfg.if_cuda:
        print("gpu总耗时:", time.time() - s)
    else:
        print("cpu总耗时:", time.time() - s)



