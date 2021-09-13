"from labelme img.png & label.png to dateset"

import cv2
import numpy as np
import os
import json


def color2cls(color_path, gray_path, colors, lab):
    if not os.path.exists(gray_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(gray_path)

    names = os.listdir(color_path)

    for name in names:
        label_color = cv2.imread(os.path.join(color_path, name))
        print(label_color.shape)
        gray = cv2.cvtColor(label_color, cv2.COLOR_BGR2GRAY)    # small label
        print("set", set(list(gray.flatten())))

        label_gray = np.zeros_like(gray)
        for ic, c in enumerate(colors):
            m = (gray == c).astype(np.uint8) * lab[ic]
            label_gray += m    # small label cat

        # cv2.imshow("img", crop_img)
        # cv2.imshow("mask", mask.astype(np.uint8)*255)
        # cv2.imshow("color", crop_label_color)
        # cv2.waitKey(0)

        savelabelgray = gray_path + name.split("_")[0] + ".png"
        cv2.imwrite(savelabelgray, label_gray)


def color2cls_txt(train_txt, rgb_path, label_path, dest_rgb, dest_cls):

    img_list = os.listdir(rgb_path)
    label_list = os.listdir(label_path)
    print("len(img_list), len(label_list)", len(img_list), len(label_list))

    img_list = [i.split(".")[0]+".tif" for i in label_list]

    assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
    num = max(len(img_list), len(label_list))


    for i in range(num):
        print(i, img_list[i], label_list[i])
        assert img_list[i].split(".")[0] == label_list[i].split(".")[0]

    train_list = []
    for i in range(num):
        train_dic = {}
        train_dic["fpath_img"] = os.path.join(dest_rgb, img_list[i])
        train_dic["fpath_segm"] = os.path.join(dest_cls, label_list[i])
        train_dic["width"] = 256
        train_dic["height"] = 256
        train_list.append(train_dic)

    fh = open(train_txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()


if __name__ == '__main__':

    # color_path = "/home/liufang/Files/2021fh_sub_tifs/1_colors/"
    # gray_path = "/home/liufang/Files/2021fh_sub_tifs/1_cls/"
    # colors = [0, 38]
    # lab = [0, 2]
    # color2cls(color_path, gray_path, colors, lab)
    # # ==============
    # train_txt = "/home/liufang/Files/2021fh_sub_tifs/1.txt"
    # rgb_path = "/home/liufang/Files/2021fh_sub_tifs/1/"
    # label_path = "/home/liufang/Files/2021fh_sub_tifs/1_cls/"
    # dest_rgb = "/home/liufang/Files/2021fh_sub_tifs/1/"
    # dest_cls = "/home/liufang/Files/2021fh_sub_tifs/1_cls/"
    # color2cls_txt(train_txt, rgb_path, label_path, dest_rgb, dest_cls)
    # ===========================================================================================================

    color_path = "/home/liufang/Files/2021fh_sub_tifs/3_4_colors/"
    gray_path = "/home/liufang/Files/2021fh_sub_tifs/3_4_cls/"
    colors = [0, 38]
    lab = [0, 2]
    color2cls(color_path, gray_path, colors, lab)
    # =============
    train_txt = "/home/liufang/Files/2021fh_sub_tifs/3_4.txt"
    rgb_path = "/home/liufang/Files/2021fh_sub_tifs/3_4/"
    label_path = "/home/liufang/Files/2021fh_sub_tifs/3_4_cls/"
    dest_rgb = "/home/liufang/Files/2021fh_sub_tifs/3_4/"
    dest_cls = "/home/liufang/Files/2021fh_sub_tifs/3_4_cls/"
    color2cls_txt(train_txt, rgb_path, label_path, dest_rgb, dest_cls)

