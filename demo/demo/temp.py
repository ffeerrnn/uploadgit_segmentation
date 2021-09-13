import os
import shutil

# img_path = "/home/liufang/project/EPSG_4326_17/425_173"
# label_path = "/home/liufang/project/EPSG_4326_17_label/425_173"
# img_names = os.listdir(img_path)
# label_names = os.listdir(label_path)
# print(len(img_names), len(label_names))
#
# inter = list(set(img_names) & set(label_names))
# print(len(inter))

# pass_file_name = []
# do_file_name = []
# for name in label_names:
#     file_path = os.path.join(label_path, name)
#     f = open(file_path, "rb")
#     size = len(f.read())
#     # print(file_path, size)
#     if size == 1784:  # byte
#         pass_file_name.append(name)
#     else:
#         do_file_name.append(name)
#     f.close()
#
# print(len(do_file_name), len(pass_file_name))
#
# # inter = list(set(img_names) & set(do_file_name))
# # print(len(inter))
#
# for name in pass_file_name:
#     file_path = os.path.join(label_path, name)
#     f = open(file_path, "rb")
#     size = len(f.read())
#     # print(file_path, size)
#     assert size == 1784, "wrong size"
#     f.close()
#     despath = os.path.join("/home/liufang/project/EPSG_4326_17/426_173_null/", name)
#
#     shutil.move(file_path, despath)

# for name in img_names:
#     if name not in label_names:
#         file_path = os.path.join(img_path, name)
#         despath = os.path.join("/home/liufang/project/EPSG_4326_17/null/425_173_null/", name)
#         shutil.move(file_path, despath)


# from torchvision import models
# # model = models.resnet18()
# model = models.segmentation.fcn_resnet50(pretrained=False)
# print(model)
# import torch
# inp = torch.randn([1,3,214,214])
# out = model(inp)
# print(out.shape)

# def colorize(img, color):
#     color_img = color[img].astype(np.uint8)
#     color_img = color_img[...,[2,1,0]]    # RGB2BGR
#     return color_img
# if __name__ == '__main__':
#     import os
#     import numpy as np
#     import cv2
#     path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/label_cls"
#     names = os.listdir(path)
#
#     colors = np.array([[0,0,0],[255,0,0],[0,255,0],[0,255,0],[255,0,0],[0,0,255],
#               [255,0,255],[255,0,0],[0,0,255],[0,0,255],[0,255,0],
#               [0,255,255],[0,255,255],[0,255,0],[255,255,0],[255,255,0]])
#
#     for i in names:
#         file = os.path.join(path, i)
#         img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#
#         color_img = colorize(img, colors)
#
#         color_path = os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/label_color1", i)
#
#         cv2.imwrite(color_path, color_img)

# if __name__ == '__main__':
#     import cv2
#     import numpy as np
#     path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/label_color"
#     names = os.listdir(path)
#
#     old = [0, 29, 76, 105, 150, 179, 226]
#     new = [0,  1,  2,   3,   4,  5,  6]
#     for i in names:
#         file = os.path.join(path, i)
#
#         label = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
#         print("label",label.shape, label.min(), label.max())
#
#
#         mask = np.zeros_like(label)
#         for o in range(len(old)):
#             temp = (label == old[o]).astype(np.uint8)
#             mask += temp*new[o]
#         print("mask", mask.min(), mask.max())
#
#
#
#
#
#
#
#         cv2.imwrite("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/label_cls/"+i, mask)


if __name__ == '__main__':
    import cv2

    # GF2_PMS2__L1A0001517494 - MSS2_6
    # label = cv2.imread("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/label_cls/GF2_PMS1__L1A0001680851-MSS1_0.tif", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", label)
    # cv2.waitKey(0)
    # path = "/home/liufang/project/DataSets/JSCZXB/data/geoserver/gwc/earth_R-CN-JSCZXB-202105-0_5M/EPSG_4326_17/426_173"
    # file_name = os.listdir(path)
    # files = [os.path.join(path, file) for file in file_name]
    # print(files)

    # pass_file_name = []
    # do_file_name = []
    # for name in file_name:
    #     file_path = os.path.join(path, name)
    #     f = open(file_path, "rb")
    #     size = len(f.read())
    #     # print(file_path, size)
    #     if size == 1784:  # byte
    #         pass_file_name.append(name)
    #     else:
    #         do_file_name.append(name)
    #     f.close()
    # print('# samples: {}, do: {}, pass: {}'.format(len(file_name), len(do_file_name), len(pass_file_name)))
    #
    # # pass_file=================================================================================================
    #
    # files = [os.path.join(path, file) for file in pass_file_name]
    # print("pass_file_name start!", end="")
    #
    # for index, n in enumerate(do_file_name):
    #     print("[{}/{} {}]".format(index, len(do_file_name), n))
    #     shutil.copyfile(os.path.join(path, n), os.path.join("/media/liufang/KINGSTON/samples/", n))
    #
    # print("pass_file_name done!")
    # path = "/home/liufang/project/DataSets/cz/EPSG_4326_15/213_058"
    # names = os.listdir(path)
    # print(len(names))
    # for name in names:
    #     if "_0.png" in name:
    #         os.remove(os.path.join(path,name))
    #
    # names = os.listdir(path)
    # print(len(names))


# import cv2
# import numpy as np
# from skimage import io
# # img = cv2.imread("/home/liufang/project/DataSets/cz2m/054528_022009.png")    #cz_202104_2m.tif
#
# img = io.imread("/home/liufang/project/DataSets/cz2m/cz_202104_2m.tif")
# # io.imsave("/home/liufang/project/DataSets/cz2m/cz_202104_2m.png", img)
# print(img.shape)
# h, w, _ = img.shape
#
# # h, w,_ = 67997, 58083, 0
# nh = 4
# nw = 4
# sh = int(np.floor(h/nh))
# sw = int(np.floor(w/nw))
# print(sh, sw)
# for i in range(nh):
#     for j in range(nw):
#         up = i*sh
#         down = (i+1)*sh
#         left = j*sw
#         right = (j+1)*sw
#         im = img[up:down, left:right, :]
#         name = "/home/liufang/project/DataSets/cz2m/" + "{}{}.tif".format(i,j)
#
#         io.imsave(name, im)
#
#         # print(up,"->", down, left, "->", right, name)
#         cv2.imwrite(name, im)

import os
path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/sub_tifs/1_json/"
files = os.listdir(path)
# print(files)

for f in files:
    print(path+f)




